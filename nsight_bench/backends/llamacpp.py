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

Batch > 1 runs ``batch_size`` sequences side by side in one llama.cpp context, as the ``hf``
backend runs a ``[batch, prompt]`` tensor: the same per-row prompt ids, one ``llama_decode`` of
all ``batch x prompt`` tokens for the prefill (one ubatch, the last token of each row returning
logits), then one ``llama_decode`` of ``batch`` tokens (one per sequence) per decode step. The
context is the bindings' own context parameters with only ``n_ctx`` (``batch x (prompt +
generate)``), ``n_batch``/``n_ubatch`` and ``n_seq_max`` changed. Batch 1 keeps the original
single-sequence path untouched.

Forced MoE routing (workload ``routing: fixed|disjoint``) needs a llama.cpp built with the
nsbench patch (experiments/llamacpp-backend/forced-routing.patch, installed in
~/envs/nsbench-moeroute). The patch reads ``LLAMA_MOE_FORCE_ROUTING`` and replaces the router's
top-k with the same ids the ``hf`` hook forces (nsight_bench/compat/routing.py); this backend
sets that variable from the workload, and refuses forced routing on an unpatched build rather
than silently running natural routing under a forced label.
"""

from __future__ import annotations

import os
from typing import Any

from .. import compat
from ..config import ModelConfig, WorkloadConfig
from .base import Backend, GenerationState, register

#: The llama.cpp commit that llama-cpp-python 0.3.36 vendors, verified identical to the
#: ~/llama.cpp build used for phase 1 (docs/09 section 6.4).
VENDORED_LLAMA_CPP_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"

TOKEN_MODES = ("bench", "greedy")

#: Environment variable the forced-routing patch reads when it builds each graph.
FORCE_ROUTING_ENV = "LLAMA_MOE_FORCE_ROUTING"


def forced_routing_supported() -> bool:
    """Whether the loaded llama.cpp library carries the nsbench forced-routing patch."""
    import llama_cpp

    return hasattr(llama_cpp.llama_cpp._lib, "llama_nsbench_forced_routing")


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
        #: batch > 1 only: the context created with n_seq_max = batch, and its llama_batch
        self._own_ctx: Any = None
        self._batch: Any = None
        self.token_mode = os.environ.get("NSBENCH_LLAMACPP_TOKEN_MODE", "bench").lower()
        if self.token_mode not in TOKEN_MODES:
            raise ValueError(
                f"NSBENCH_LLAMACPP_TOKEN_MODE={self.token_mode!r}; expected one of {TOKEN_MODES}"
            )

    # ---- lifecycle --------------------------------------------------------------------

    def load(self) -> None:
        wl = self.workload_config
        if wl.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {wl.batch_size}")
        if not str(self.model_config.path).endswith(".gguf"):
            raise ValueError(f"the llamacpp backend needs a .gguf file, got {self.model_config.path}")

        import llama_cpp

        # Set (not inherited) every time, so a stray value can never force a natural run.
        if wl.routing != "natural" and not forced_routing_supported():
            raise ValueError(
                f"routing={wl.routing} needs llama.cpp with the nsbench forced-routing patch "
                "(~/envs/nsbench-moeroute, experiments/llamacpp-backend/forced-routing.patch); "
                f"this llama-cpp-python ({llama_cpp.__file__}) does not have it"
            )
        os.environ[FORCE_ROUTING_ENV] = wl.routing

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
        if wl.batch_size > 1:
            self._load_batch_context()
        self._n_ctx = int(llama_cpp.llama_n_ctx(self._ctx))

        if self.token_mode == "bench" and wl.batch_size > 1:
            import torch

            # Same generator and bounds as batch 1; one token per sequence per step.
            generator = torch.Generator().manual_seed(wl.seed + 1)
            low = min(1000, max(1, self._n_vocab // 10))
            self._bench_tokens = torch.randint(
                low, max(low + 1, self._n_vocab - 100), (wl.generate_tokens + 1, wl.batch_size),
                generator=generator, dtype=torch.long,
            ).tolist()
        elif self.token_mode == "bench":
            import torch

            generator = torch.Generator().manual_seed(wl.seed + 1)
            low = min(1000, max(1, self._n_vocab // 10))
            self._bench_tokens = torch.randint(
                low, max(low + 1, self._n_vocab - 100), (wl.generate_tokens + 1,),
                generator=generator, dtype=torch.long,
            ).tolist()

    def _load_batch_context(self) -> None:
        """A second context for ``batch_size`` sequences, alongside the bindings' own.

        ``llama_cpp.Llama`` has no ``n_seq_max`` argument, so the context the run uses is made
        here from a copy of the bindings' own context parameters (flash attention, threads,
        cache types...), changing only what the batch needs. The bindings' single-sequence
        context stays allocated but idle (its cache is a few tens of MB on Kimi).
        """
        import llama_cpp

        wl = self.workload_config
        params = llama_cpp.llama_context_params.from_buffer_copy(self._llm.context_params)
        # Not unified: each sequence gets n_ctx / n_seq_max cells, i.e. prompt + generate.
        params.n_ctx = wl.batch_size * (wl.prompt_tokens + wl.generate_tokens)
        params.n_batch = params.n_ubatch = max(512, wl.batch_size * wl.prompt_tokens)
        params.n_seq_max = wl.batch_size
        ctx = llama_cpp.llama_init_from_model(self._llm._model.model, params)
        if not ctx:
            raise RuntimeError(
                f"llama.cpp could not create a context for batch_size={wl.batch_size} "
                f"(n_ctx={params.n_ctx}, n_ubatch={params.n_ubatch})"
            )
        self._own_ctx = self._ctx = ctx
        self._batch = llama_cpp.llama_batch_init(params.n_batch, 0, 1)

    def teardown(self) -> None:
        if self._own_ctx is not None:
            import llama_cpp

            if self._batch is not None:
                llama_cpp.llama_batch_free(self._batch)
            llama_cpp.llama_free(self._own_ctx)
            self._own_ctx = self._batch = None
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
            input_ids = encoded[: wl.prompt_tokens].unsqueeze(0).expand(wl.batch_size, -1)
        if wl.batch_size > 1:
            return np.ascontiguousarray(input_ids.numpy(), dtype=np.int32)   # [batch, prompt]
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
        if self.model_config.trust_remote_code:
            compat.apply_remote_code_shims()
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

    def _decode_rows(self, rows, start_pos: int) -> None:
        """One ``llama_decode`` of several sequences: row s of ``rows`` is sequence s, its
        tokens at positions ``start_pos ..``; only each row's last token returns logits."""
        import llama_cpp

        b = self._batch
        i = 0
        for seq, row in enumerate(rows):
            for j, tok in enumerate(row):
                b.token[i] = int(tok)
                b.pos[i] = start_pos + j
                b.n_seq_id[i] = 1
                b.seq_id[i][0] = seq
                b.logits[i] = int(j == len(row) - 1)
                i += 1
        b.n_tokens = i
        rc = llama_cpp.llama_decode(self._ctx, b)
        if rc != 0:
            raise RuntimeError(f"llama_decode returned {rc}")

    def _argmax_rows(self, rows) -> list[int]:
        """Greedy next token of each row after :meth:`_decode_rows` (batch index of its last token)."""
        import numpy as np

        import llama_cpp

        out, i = [], 0
        for row in rows:
            i += len(row)
            ptr = llama_cpp.llama_get_logits_ith(self._ctx, i - 1)
            out.append(int(np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).argmax()))
        return out

    def _argmax_last_logits(self) -> int:
        import numpy as np

        import llama_cpp

        ptr = llama_cpp.llama_get_logits_ith(self._ctx, -1)
        return int(np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).argmax())

    def prefill(self, inputs: Any) -> GenerationState:
        import numpy as np

        import llama_cpp

        if self.workload_config.batch_size > 1:
            return self._prefill_rows(inputs)
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

    def _prefill_rows(self, inputs: Any) -> GenerationState:
        import numpy as np

        import llama_cpp

        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(self._ctx), False)
        rows = [list(r) for r in inputs]
        self._decode_rows(rows, 0)
        if self.token_mode == "greedy":
            next_tokens = self._argmax_rows(rows)
        else:
            llama_cpp.llama_synchronize(self._ctx)
            next_tokens = self._bench_tokens[0]
        token = np.array(next_tokens, dtype=np.int32).reshape(-1, 1)          # [batch, 1]
        return GenerationState(cache=None, last_token_ids=token, position=len(rows[0]),
                               generated_tokens=[token], kv_bytes=None)

    def _decode_step_rows(self, state: GenerationState) -> GenerationState:
        import numpy as np

        import llama_cpp

        rows = [[int(t)] for t in state.last_token_ids[:, 0]]
        self._decode_rows(rows, state.position)
        if self.token_mode == "greedy":
            next_tokens = self._argmax_rows(rows)
        else:
            llama_cpp.llama_synchronize(self._ctx)
            step = len(state.generated_tokens)
            next_tokens = self._bench_tokens[step % len(self._bench_tokens)]
        token = np.array(next_tokens, dtype=np.int32).reshape(-1, 1)
        state.last_token_ids = token
        state.position += 1
        state.generated_tokens.append(token)
        return state

    def decode_step(self, state: GenerationState) -> GenerationState:
        """One token: one ``llama_decode`` of a single position, then wait for it."""
        import numpy as np

        import llama_cpp

        if self.workload_config.batch_size > 1:
            return self._decode_step_rows(state)
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
            "routing": self.workload_config.routing,
            "batch_size": self.workload_config.batch_size,
            "n_seq_max": self.workload_config.batch_size,
        }
        if self.workload_config.routing != "natural":
            info["routing_warning"] = (
                f"MoE routing was forced ({self.workload_config.routing}) by the nsbench llama.cpp "
                "patch: the generated text is meaningless; only timing and traffic are "
                "representative. Expert weights are the router's own scores for the forced "
                "experts, renormalised (hf: uniform), so the router still runs"
            )
        if self.token_mode == "bench":
            info["token_warning"] = (
                "decode fed predetermined tokens without reading logits, as llama-bench does: "
                "the generated text is meaningless; only timing and traffic are representative"
            )
        try:
            import llama_cpp

            info["llama_cpp_python_version"] = llama_cpp.__version__
            info["forced_routing_patch"] = forced_routing_supported()
            if self._llm is not None:
                model = self._llm._model.model
                info["parameters"] = int(llama_cpp.llama_model_n_params(model))
                info["parameter_bytes_resident"] = int(llama_cpp.llama_model_size(model))
                info["n_threads"] = int(self._llm.n_threads)
                info["n_threads_batch"] = int(self._llm.n_threads_batch)
        except Exception:                                        # noqa: BLE001
            pass
        return info
