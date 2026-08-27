"""HuggingFace transformers backend -- the reference implementation.

The decode loop here is written by hand rather than delegated to ``model.generate()``. That
is the whole point of this backend: ``generate()`` hides the per-token boundary inside
library code, and without that boundary there is nowhere to put the NVTX range that Nsight
Compute filters on. Driving the forward pass directly gives one clean range per generated
token, and the kernels inside it are exactly the ones a decode step executes.

The loop is greedy and deterministic by default so repeated runs are comparable; sampling
would add variance that shows up as noise in the memory numbers without telling us anything
about memory.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import ModelConfig, WorkloadConfig
from .base import Backend, GenerationState, register


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


@register
class HFTransformersBackend(Backend):
    """Runs a checkpoint through ``transformers`` with an explicit prefill/decode split."""

    name = "hf"

    def __init__(self, model_config: ModelConfig, workload_config: WorkloadConfig) -> None:
        super().__init__(model_config, workload_config)
        self.device = "cuda"
        self._load_error: str | None = None
        self._attn_impl_used: str = ""
        self._dtype_used: str = ""
        self._cache_class_used: str = ""
        #: None until resolved; "" means this model accepts no such keyword.
        self._logits_kwarg: str | None = None
        self._prefill_logits_full_vocab: bool = False

    # ---- lifecycle --------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.model_config
        # Offline by default: these checkpoints are already on the SSD, and an accidental
        # hub round-trip would both stall the run and risk pulling different weights than
        # the ones being benchmarked.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        tokenizer_path = cfg.tokenizer_path or cfg.path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=cfg.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
            "attn_implementation": cfg.attn_implementation,
        }
        # Pre-quantized checkpoints carry their own storage dtype; forcing one here either
        # errors or silently dequantizes the weights, which would defeat the purpose of
        # benchmarking the quantized model.
        if cfg.quantization is None:
            kwargs["dtype"] = _torch_dtype(cfg.dtype)
        if cfg.device_map:
            kwargs["device_map"] = cfg.device_map

        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.path, **kwargs)
        except (TypeError, ValueError) as exc:
            # attn_implementation is the usual casualty: not every architecture supports
            # sdpa or flash_attention_2. Retry on the default kernel and record that the
            # requested one was unavailable, rather than dropping the model from the sweep.
            if "attn_implementation" not in str(exc) and "attention" not in str(exc).lower():
                raise
            kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(cfg.path, **kwargs)
            self._load_error = (
                f"attn_implementation='{cfg.attn_implementation}' unsupported here; "
                "loaded with the architecture default"
            )

        if not cfg.device_map:
            self.model = self.model.to(self.device)
        self.model.eval()

        self._attn_impl_used = getattr(self.model.config, "_attn_implementation", "") or ""
        param = next(self.model.parameters(), None)
        self._dtype_used = str(param.dtype).replace("torch.", "") if param is not None else ""

        torch.manual_seed(self.workload_config.seed)

    def teardown(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ---- inputs -----------------------------------------------------------------------

    def prepare_inputs(self) -> Any:
        """Build a batch of exactly ``prompt_tokens`` tokens per sequence.

        Exactness matters: prefill cost scales with sequence length, so a prompt that is
        "about" 512 tokens makes runs incomparable. A synthetic prompt is generated by
        sampling token ids directly, which sidesteps the tokenizer's variable
        characters-per-token and guarantees the requested length.
        """
        import torch

        wl = self.workload_config

        if wl.prompt_source == "synthetic":
            generator = torch.Generator().manual_seed(wl.seed)
            vocab = int(getattr(self.tokenizer, "vocab_size", 0) or 32000)
            # Stay clear of the low id range, where special and reserved tokens live.
            low = min(1000, max(1, vocab // 10))
            input_ids = torch.randint(
                low, max(low + 1, vocab - 100),
                (wl.batch_size, wl.prompt_tokens),
                generator=generator, dtype=torch.long,
            )
        else:
            text = open(wl.prompt_source, encoding="utf-8").read()
            encoded = self.tokenizer(text, return_tensors="pt").input_ids[0]
            if encoded.numel() < wl.prompt_tokens:
                repeats = (wl.prompt_tokens // max(1, encoded.numel())) + 1
                encoded = encoded.repeat(repeats)
            encoded = encoded[: wl.prompt_tokens]
            input_ids = encoded.unsqueeze(0).repeat(wl.batch_size, 1)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": torch.ones_like(input_ids, device=self.device),
        }

    # ---- inference --------------------------------------------------------------------

    def _new_cache(self):
        """Create a fresh KV cache, tolerating the several APIs transformers has used."""
        try:
            from transformers import DynamicCache

            self._cache_class_used = "DynamicCache"
            return DynamicCache()
        except Exception:                                        # noqa: BLE001
            self._cache_class_used = "backend default"
            return None

    def _resolve_logits_to_keep(self) -> str:
        """Find the keyword this transformers version uses to limit the LM-head projection.

        Without it the prompt forward pass runs the LM head over **every** prompt position
        and materialises a ``[batch, prompt_tokens, vocab]`` logits tensor, of which only the
        final row is ever read. On a large-vocabulary model that is the single heaviest
        kernel in prefill -- for Qwen3-0.6B at 512 prompt tokens it was a 512x1024x151936
        GEMM moving 468 MB, about 11% of the phase's latency -- and it is work no inference
        stack actually performs. Measuring it makes prefill throughput read low for a reason
        that has nothing to do with the hardware.

        The keyword was ``num_logits_to_keep`` before transformers 4.49 and ``logits_to_keep``
        after, and a ``trust_remote_code`` model may accept neither, so it is resolved by
        inspecting the signature once rather than assumed.
        """
        import inspect

        if self._logits_kwarg is not None:
            return self._logits_kwarg

        self._logits_kwarg = ""
        try:
            parameters = inspect.signature(self.model.forward).parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                # **kwargs forwards to the base model, which accepts the modern spelling.
                self._logits_kwarg = "logits_to_keep"
            for name in ("logits_to_keep", "num_logits_to_keep"):
                if name in parameters:
                    self._logits_kwarg = name
                    break
        except (TypeError, ValueError):                          # pragma: no cover
            pass
        return self._logits_kwarg

    def prefill(self, inputs: Any) -> GenerationState:
        import torch

        cache = self._new_cache()
        kwargs: dict[str, Any] = {}
        keyword = self._resolve_logits_to_keep()
        if keyword:
            kwargs[keyword] = 1

        with torch.inference_mode():
            try:
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    past_key_values=cache,
                    use_cache=True,
                    **kwargs,
                )
            except (TypeError, ValueError):
                # The signature advertised the keyword but the implementation rejected it.
                # Fall back rather than dropping the model, and record that prefill included
                # the full-vocabulary projection so the report does not read as though it did
                # not.
                if not kwargs:
                    raise
                self._logits_kwarg = ""
                self._prefill_logits_full_vocab = True
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    past_key_values=cache,
                    use_cache=True,
                )
            else:
                self._prefill_logits_full_vocab = not kwargs
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        state = GenerationState(
            cache=outputs.past_key_values,
            last_token_ids=next_token,
            position=inputs["input_ids"].shape[1],
            generated_tokens=[next_token],
        )
        state.kv_bytes = self._measure_kv_bytes(state.cache)
        return state

    def decode_step(self, state: GenerationState) -> GenerationState:
        """One token. Exactly one forward pass, so the NVTX range wraps one step of work.

        Nothing here touches the host. The sampled token stays a device tensor and is fed
        straight back in as the next input, so the host can keep queueing steps instead of
        blocking on a D2H copy every token -- see :class:`GenerationState.generated_tokens`.
        """
        import torch

        with torch.inference_mode():
            outputs = self.model(
                input_ids=state.last_token_ids,
                past_key_values=state.cache,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        state.cache = outputs.past_key_values
        state.last_token_ids = next_token
        state.position += 1
        state.generated_tokens.append(next_token)
        return state

    # ---- introspection ----------------------------------------------------------------

    def _measure_kv_bytes(self, cache: Any) -> int | None:
        """Sum the real KV-cache tensors.

        Preferred over the analytic estimate because it captures what the implementation
        actually allocated -- padding, preallocation, and quantized cache dtypes included.
        Cache internals have moved around across transformers versions, so several shapes
        are tried and ``None`` is returned rather than a wrong number.
        """
        if cache is None:
            return None
        try:
            import torch

            total = 0
            layers = getattr(cache, "layers", None)
            if layers is not None:
                for layer in layers:
                    for attr in ("keys", "values", "key_cache", "value_cache"):
                        tensor = getattr(layer, attr, None)
                        if isinstance(tensor, torch.Tensor):
                            total += tensor.numel() * tensor.element_size()
                if total:
                    return total

            for attr in ("key_cache", "value_cache"):
                for tensor in getattr(cache, attr, []) or []:
                    if isinstance(tensor, torch.Tensor):
                        total += tensor.numel() * tensor.element_size()
            if total:
                return total

            if isinstance(cache, (tuple, list)):
                for layer in cache:
                    for tensor in layer:
                        if isinstance(tensor, torch.Tensor):
                            total += tensor.numel() * tensor.element_size()
            return total or None
        except Exception:                                        # noqa: BLE001
            return None

    def describe(self) -> dict:
        info: dict[str, Any] = {
            "backend": self.name,
            "model_path": self.model_config.path,
            "requested_dtype": self.model_config.dtype,
            "actual_dtype": self._dtype_used,
            "requested_attn_implementation": self.model_config.attn_implementation,
            "actual_attn_implementation": self._attn_impl_used,
            "quantization": self.model_config.quantization,
            # The cache implementation is a first-order influence on decode traffic, not a
            # detail: DynamicCache re-concatenates the whole K and V tensors on every step,
            # so a decode step moves the cache twice more than the analytic model predicts.
            # Recording it is what lets a high expected-versus-measured ratio be attributed
            # rather than guessed at.
            "kv_cache_class": self._cache_class_used,
            "prefill_logits_to_keep_kwarg": self._logits_kwarg or None,
            "prefill_computed_full_vocab_logits": self._prefill_logits_full_vocab,
        }
        if self._load_error:
            info["load_warning"] = self._load_error
        if self._prefill_logits_full_vocab:
            info["prefill_warning"] = (
                "this model accepted no logits_to_keep keyword, so prefill ran the LM head "
                "over every prompt position; its traffic and latency include a "
                "full-vocabulary projection that real inference does not perform"
            )

        try:
            import transformers

            info["transformers_version"] = transformers.__version__
        except Exception:                                        # noqa: BLE001
            pass

        if self.model is not None:
            try:
                params = sum(p.numel() for p in self.model.parameters())
                bytes_ = sum(p.numel() * p.element_size() for p in self.model.parameters())
                info["parameters"] = params
                info["parameter_bytes_resident"] = bytes_
                buffers = sum(b.numel() * b.element_size() for b in self.model.buffers())
                info["buffer_bytes_resident"] = buffers
            except Exception:                                    # noqa: BLE001
                pass
        return info
