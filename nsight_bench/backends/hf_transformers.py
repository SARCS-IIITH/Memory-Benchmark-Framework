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

import json
import os
from pathlib import Path
from typing import Any

from .. import compat
from ..config import ModelConfig, WorkloadConfig
from .base import Backend, GenerationState, register


def _read_config(path: str) -> dict:
    """The checkpoint's config.json, or an empty dict when it cannot be read."""
    try:
        return json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return {}


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
        #: Runtime adapters applied for remote code written against an older transformers.
        self._compat_shims: list[str] = []
        self._compat_cache_classes: list[str] = []
        self._attn_impl_replaced: str | None = None
        #: Whether the model builds its own cache (hybrid models) rather than taking ours.
        self._model_owns_cache: bool = False
        self._fp8_optimized: bool = False
        #: Decoding-state bytes by kind, from the most recent measurement.
        self._cache_breakdown: dict[str, int] = {}

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

        # Remote modelling and tokenizer code fails at import time when it reaches for a
        # symbol the installed transformers has moved, so the shims go in before either
        # is loaded. They only restore old names; they change nothing a model computes.
        if cfg.trust_remote_code:
            self._compat_shims = compat.apply_remote_code_shims()

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

        # An FP8 compressed-tensors checkpoint is only FP8 at runtime on the optimised path.
        # Left to the default, transformers keeps the weights packed until the first forward
        # pass and then unpacks them to bf16 -- doubling the resident footprint and turning
        # every later measurement into one of a bf16 model. The optimised path keeps them
        # in FP8 and runs real FP8 matmuls; transformers itself falls back, with a warning,
        # on a GPU below compute capability 8.9.
        if compat.is_fp8_compressed_tensors(_read_config(cfg.path)):
            from transformers.utils.quantization_config import CompressedTensorsConfig

            kwargs["quantization_config"] = CompressedTensorsConfig(use_optimized_inference=True)
            kwargs.pop("dtype", None)
            self._fp8_optimized = True

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

        if cfg.trust_remote_code:
            self._compat_cache_classes = compat.adapt_model_cache_api(self.model)
            self._attn_impl_replaced = compat.use_sdpa_if_flash_attn_missing(self.model)
        self._model_owns_cache = compat.model_manages_own_cache(self.model)

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
        """Create a fresh KV cache, tolerating the several APIs transformers has used.

        Returns ``None`` for a model that owns its cache. A hybrid model's cache holds more
        than keys and values -- Kimi-Linear's carries a recurrent and a convolution state per
        linear-attention layer -- and the model builds it when given ``None``, while a
        generic ``DynamicCache`` has no slot for that state and fails the model's own type
        check. The class actually used is read back from the prefill output.
        """
        if self._model_owns_cache:
            self._cache_class_used = "model-managed"
            return None
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
        if state.cache is not None:
            self._cache_class_used = type(state.cache).__name__
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

    #: Cache attributes by the kind of decoding state they hold. Keys and values grow with
    #: context; a linear-attention or state-space layer's recurrent and convolution state is
    #: a fixed size, but is read and rewritten on every decode step all the same -- which is
    #: why it belongs in the per-step traffic expectation alongside the KV cache.
    _CACHE_STATE_ATTRS: dict[str, tuple[str, ...]] = {
        "kv": ("keys", "values", "key_cache", "value_cache"),
        "recurrent_state": ("recurrent_states", "ssm_states"),
        "conv_state": ("conv_states",),
    }

    def _measure_kv_bytes(self, cache: Any) -> int | None:
        """Sum the real decoding-state tensors: the KV cache plus any recurrent state.

        Preferred over the analytic estimate because it captures what the implementation
        actually allocated -- padding, preallocation, and quantized cache dtypes included.
        Cache internals have moved around across transformers versions, so several shapes
        are tried and ``None`` is returned rather than a wrong number.

        A hybrid model's cache is more than keys and values: Kimi-Linear's 20 KDA layers each
        carry a 32x128x128 fp32 recurrent state, ~40 MiB in all, that a decode step reads and
        rewrites. Counting keys and values alone would leave it out of the expectation the
        measured traffic is checked against. The split by kind is kept for the manifest.
        """
        if cache is None:
            return None
        try:
            breakdown = self._cache_state_breakdown(cache)
        except Exception:                                        # noqa: BLE001
            return None
        self._cache_breakdown = breakdown
        return sum(breakdown.values()) or None

    def _cache_state_breakdown(self, cache: Any) -> dict[str, int]:
        import torch

        seen: set[tuple[int, int, str]] = set()
        totals = {kind: 0 for kind in self._CACHE_STATE_ATTRS}

        def add(kind: str, value: Any) -> None:
            # Tensors are deduplicated by storage, since one cache object can expose the
            # same tensor under two names (a layer list and a flat per-attribute list).
            if isinstance(value, torch.Tensor):
                key = (value.data_ptr(), value.numel(), str(value.dtype))
                if value.numel() and key not in seen:
                    seen.add(key)
                    totals[kind] += value.numel() * value.element_size()
            elif isinstance(value, (tuple, list)):
                for item in value:
                    add(kind, item)

        holders = [cache, *(getattr(cache, "layers", None) or [])]
        for holder in holders:
            for kind, attrs in self._CACHE_STATE_ATTRS.items():
                for attr in attrs:
                    add(kind, getattr(holder, attr, None))

        if not any(totals.values()) and isinstance(cache, (tuple, list)):
            add("kv", cache)                                     # legacy tuple-of-tuples
        return {kind: n for kind, n in totals.items() if n}

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
            "cache_managed_by_model": self._model_owns_cache,
            # Bytes by kind (kv / recurrent_state / conv_state) at the last measurement --
            # the end of generation. A hybrid model's recurrent state does not grow with
            # context, so this is what separates it from the KV cache in the report.
            "cache_state_bytes": dict(self._cache_breakdown) or None,
            "fp8_optimized_inference": self._fp8_optimized,
            "prefill_logits_to_keep_kwarg": self._logits_kwarg or None,
            "prefill_computed_full_vocab_logits": self._prefill_logits_full_vocab,
        }
        if self._compat_shims or self._compat_cache_classes:
            info["compat_adapters"] = {
                "import_shims": self._compat_shims,
                "cache_classes_adapted": self._compat_cache_classes,
            }
        if self._attn_impl_replaced:
            info["attn_warning"] = (
                f"the model forces {self._attn_impl_replaced}, which is not installed; its "
                "attention layers were switched to sdpa"
            )
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
