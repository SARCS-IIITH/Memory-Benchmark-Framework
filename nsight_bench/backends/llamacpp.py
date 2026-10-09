"""llama.cpp backend, through the ``llama-cpp-python`` bindings.

Runs a GGUF file with llama.cpp's own C++/CUDA engine, so the harness can put the same NVTX
ranges, nsys pass and ncu tiers around it as around the ``hf`` backend. The comparison this
exists for, and its phase 1 baseline from ``llama-bench``, is in docs/09-llamacpp-comparison.md.

The loop is driven through the low-level C API (``llama_decode`` on one batch per call), not
``Llama.eval()`` or ``create_completion()``: those copy the logits into numpy and sample on the
host every token, which is extra host work the harness would then measure as llama.cpp's.

How a decode step picks its input token is set by ``NSBENCH_LLAMACPP_TOKEN_MODE``:

* ``bench`` (default): what ``llama-bench``'s generation test does. Feed a predetermined
  token, never read the logits, and ``llama_synchronize`` after every token. This is the mode
  whose decode tok/s is checked against ``llama-bench`` -- the generated text is meaningless.
* ``greedy``: real greedy decoding. Read the last position's logits on the host and take the
  argmax. That read waits for the GPU every token, as real generation through llama.cpp does.

Either way every token waits on the GPU: llama.cpp hands logits back to the host as part of
each decode, so there is no equivalent of the ``hf`` backend keeping tokens on the device.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import ModelConfig, WorkloadConfig
from .base import Backend, GenerationState, register

#: The llama.cpp commit that llama-cpp-python 0.3.36 vendors, verified identical to the
#: ~/llama.cpp build used for phase 1 (docs/09 section 6.4).
VENDORED_LLAMA_CPP_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"

TOKEN_MODES = ("bench", "greedy")


@register
class LlamaCppBackend(Backend):
    """Runs a GGUF through llama.cpp with an explicit prefill/decode split."""

    name = "llamacpp"

    def __init__(self, model_config: ModelConfig, workload_config: WorkloadConfig) -> None:
        super().__init__(model_config, workload_config)
        self._llm: Any = None
        self._ctx: Any = None
        self._n_vocab: int = 0
        self._n_ctx: int = 0
        self._bench_tokens: list[int] = []
        self.token_mode = os.environ.get("NSBENCH_LLAMACPP_TOKEN_MODE", "bench").lower()
        if self.token_mode not in TOKEN_MODES:
            raise ValueError(
                f"NSBENCH_LLAMACPP_TOKEN_MODE={self.token_mode!r}; expected one of {TOKEN_MODES}"
            )

    # ---- lifecycle --------------------------------------------------------------------

    def load(self) -> None:
        wl = self.workload_config
        if wl.batch_size != 1:
            raise ValueError(
                f"the llamacpp backend runs one sequence; batch_size={wl.batch_size} is not supported"
            )
        if not str(self.model_config.path).endswith(".gguf"):
            raise ValueError(f"the llamacpp backend needs a .gguf file, got {self.model_config.path}")

        import llama_cpp

        # Just large enough for the workload, as llama-bench sizes its context
        # (n_prompt + n_gen + n_depth). llama.cpp may pad it; the value used is read back.
        n_ctx = wl.prompt_tokens + wl.generate_tokens
        self._llm = llama_cpp.Llama(
            model_path=str(self.model_config.path),
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_batch=max(512, wl.prompt_tokens),
            n_ubatch=max(512, wl.prompt_tokens),
            # The bindings default to flash attention off; llama-bench's default (auto) turns
            # it on for this GPU, so it is enabled here to run the same kernels.
            flash_attn=True,
            logits_all=False,
            seed=wl.seed,
            verbose=False,
        )
        self._ctx = self._llm._ctx.ctx
        vocab = llama_cpp.llama_model_get_vocab(self._llm._model.model)
        self._n_vocab = int(llama_cpp.llama_vocab_n_tokens(vocab))
        self._n_ctx = int(llama_cpp.llama_n_ctx(self._ctx))

        if self.token_mode == "bench":
            import torch

            generator = torch.Generator().manual_seed(wl.seed + 1)
            low = min(1000, max(1, self._n_vocab // 10))
            self._bench_tokens = torch.randint(
                low, max(low + 1, self._n_vocab - 100), (wl.generate_tokens + 1,),
                generator=generator, dtype=torch.long,
            ).tolist()

    def teardown(self) -> None:
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:                                    # noqa: BLE001
                pass
        self._llm = None
        self._ctx = None

    def synchronize(self) -> None:
        if self._ctx is not None:
            import llama_cpp

            llama_cpp.llama_synchronize(self._ctx)

    # ---- inputs -----------------------------------------------------------------------

    def prepare_inputs(self) -> Any:
        """The same token ids as the ``hf`` backend builds for this workload.

        Synthetic prompts are drawn with the same seeded generator and the same bound -- the
        Hugging Face tokenizer's ``vocab_size``, read from ``tokenizer_path`` -- so both
        engines prefill an identical sequence.
        """
        import numpy as np
        import torch

        wl = self.workload_config
        tokenizer = self._hf_tokenizer()
        if wl.prompt_source == "synthetic":
            generator = torch.Generator().manual_seed(wl.seed)
            vocab = int(getattr(tokenizer, "vocab_size", 0) or 32000)
            low = min(1000, max(1, vocab // 10))
            input_ids = torch.randint(
                low, max(low + 1, vocab - 100),
                (wl.batch_size, wl.prompt_tokens),
                generator=generator, dtype=torch.long,
            )
        else:
            text = open(wl.prompt_source, encoding="utf-8").read()
            encoded = tokenizer(text, return_tensors="pt").input_ids[0]
            if encoded.numel() < wl.prompt_tokens:
                repeats = (wl.prompt_tokens // max(1, encoded.numel())) + 1
                encoded = encoded.repeat(repeats)
            input_ids = encoded[: wl.prompt_tokens].unsqueeze(0)
        return np.ascontiguousarray(input_ids[0].numpy(), dtype=np.int32)

    def _hf_tokenizer(self):
        from transformers import AutoTokenizer

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        path = self.model_config.tokenizer_path
        if not path:
            raise ValueError(
                "the llamacpp backend needs tokenizer_path (the original Hugging Face snapshot) "
                "to build the same prompt token ids as the hf backend"
            )
        return AutoTokenizer.from_pretrained(path, trust_remote_code=self.model_config.trust_remote_code)

    # ---- inference --------------------------------------------------------------------

    def _decode(self, tokens) -> None:
        import ctypes

        import llama_cpp

        arr = (llama_cpp.llama_token * len(tokens))(*[int(t) for t in tokens])
        # batch_get_one asks for the last position's logits only, like logits_to_keep=1.
        rc = llama_cpp.llama_decode(self._ctx, llama_cpp.llama_batch_get_one(arr, ctypes.c_int32(len(tokens))))
        if rc != 0:
            raise RuntimeError(f"llama_decode returned {rc}")

    def _argmax_last_logits(self) -> int:
        import numpy as np

        import llama_cpp

        ptr = llama_cpp.llama_get_logits_ith(self._ctx, -1)
        return int(np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).argmax())

    def prefill(self, inputs: Any) -> GenerationState:
        import numpy as np

        import llama_cpp

        # A fresh sequence each repeat. data=False clears the bookkeeping only, so no memset
        # kernel lands inside the prefill range.
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(self._ctx), False)
        self._decode(inputs)
        if self.token_mode == "greedy":
            next_token = self._argmax_last_logits()
        else:
            llama_cpp.llama_synchronize(self._ctx)
            next_token = self._bench_tokens[0]
        token = np.array([[next_token]], dtype=np.int32)
        return GenerationState(
            cache=None,
            last_token_ids=token,
            position=len(inputs),
            generated_tokens=[token],
            # llama.cpp's KV cache is not visible from here; the report falls back to the
            # analytic estimate and says so.
            kv_bytes=None,
        )

    def decode_step(self, state: GenerationState) -> GenerationState:
        """One token: one ``llama_decode`` of a single position, then wait for it."""
        import numpy as np

        import llama_cpp

        self._decode([int(state.last_token_ids[0, 0])])
        if self.token_mode == "greedy":
            next_token = self._argmax_last_logits()
        else:
            llama_cpp.llama_synchronize(self._ctx)
            step = len(state.generated_tokens)
            next_token = self._bench_tokens[step % len(self._bench_tokens)]
        token = np.array([[next_token]], dtype=np.int32)
        state.last_token_ids = token
        state.position += 1
        state.generated_tokens.append(token)
        return state

    # ---- introspection ----------------------------------------------------------------

    def describe(self) -> dict:
        info: dict[str, Any] = {
            "backend": self.name,
            "model_path": self.model_config.path,
            "engine": "llama.cpp",
            "llama_cpp_commit": VENDORED_LLAMA_CPP_COMMIT,
            "token_mode": self.token_mode,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": -1,
            "flash_attn": "enabled",
            "kv_cache_class": "llama.cpp KV cache (not measured; analytic estimate used)",
        }
        if self.token_mode == "bench":
            info["token_warning"] = (
                "decode fed predetermined tokens without reading logits, as llama-bench does: "
                "the generated text is meaningless; only timing and traffic are representative"
            )
        try:
            import llama_cpp

            info["llama_cpp_python_version"] = llama_cpp.__version__
            if self._llm is not None:
                model = self._llm._model.model
                info["parameters"] = int(llama_cpp.llama_model_n_params(model))
                info["parameter_bytes_resident"] = int(llama_cpp.llama_model_size(model))
                info["n_threads"] = int(self._llm.n_threads)
                info["n_threads_batch"] = int(self._llm.n_threads_batch)
        except Exception:                                        # noqa: BLE001
            pass
        return info
