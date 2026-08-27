"""Text-generation workload: prefill and decode, measured separately.

The prefill/decode split is the primary axis of this whole harness, because the two phases
sit at opposite ends of the memory story:

* **Prefill** processes the entire prompt in one pass. Weights are read once and amortised
  across hundreds of tokens, so arithmetic intensity is high and the phase is compute-bound.
* **Decode** produces one token per forward pass. Every weight in the model is read to
  produce a single token, plus the whole KV cache. Arithmetic intensity is near its floor
  and the phase is bound almost entirely by memory bandwidth.

Averaging them together -- which any end-to-end tokens/second number does -- hides exactly
the effect we are trying to measure.

NVTX placement is mode-dependent, and the reason is practical. Nsight Compute replays each
kernel several times to collect a full metric set. Annotating all 64 decode steps with the
scope name would make it profile all 64, at no informational gain since a decode step's
kernel mix does not change. So in NCU mode exactly one step carries the scope name.
"""

from __future__ import annotations

from ..backends.base import Backend
from ..instrumentation import markers
from ..instrumentation.memory import MemorySampler, PhaseMemoryTracker
from ..instrumentation.nvtx import LayerAnnotator, Phase, mark, nvtx_range, phase
from .base import ProfileMode, Workload, WorkloadResult, register

#: NVTX name used for decode steps that are *not* the profiling target. Distinct from
#: ``Phase.DECODE_STEP`` so ncu's --nvtx-include matches one step and only one.
UNSCOPED_STEP = "nsbench.decode_step_other"


@register
class TextGenerationWorkload(Workload):
    """Prompt prefill followed by greedy token-by-token decode."""

    kind = "text-generation"

    def run(
        self,
        backend: Backend,
        mode: ProfileMode = ProfileMode.BASELINE,
        sampler: MemorySampler | None = None,
    ) -> WorkloadResult:
        cfg = self.config
        result = WorkloadResult(
            mode=mode.value,
            prompt_tokens=cfg.prompt_tokens,
            generated_tokens=cfg.generate_tokens,
            batch_size=cfg.batch_size,
        )

        annotator: LayerAnnotator | None = None
        if cfg.annotate_layers and backend.model is not None:
            annotator = LayerAnnotator(backend.model)
            result.layers_annotated = annotator.apply()
            if result.layers_annotated == 0:
                result.notes.append(
                    "layer annotation requested but no decoder block list was found on this "
                    "architecture; per-layer attribution will be unavailable"
                )

        try:
            with phase(Phase.TOKENIZE):
                inputs = backend.prepare_inputs()

            self._warmup(backend, inputs, result)

            # Nsight Compute is driven by the NVTX filter, not the capture range, and it
            # replays kernels -- so opening a cudaProfilerStart range here would have it
            # collect during warmup teardown as well.
            use_capture_range = mode is ProfileMode.NSYS

            if use_capture_range:
                markers.start_capture()
            try:
                if mode is ProfileMode.NCU:
                    self._run_once(backend, inputs, result, sampler, scoped_step=True)
                else:
                    for _ in range(max(1, cfg.repeat)):
                        self._run_once(backend, inputs, result, sampler, scoped_step=False)
            finally:
                if use_capture_range:
                    markers.stop_capture()
        finally:
            if annotator is not None:
                annotator.remove()

        return result

    # ---- internals --------------------------------------------------------------------

    def _warmup(self, backend: Backend, inputs, result: WorkloadResult) -> None:
        """Discarded iterations.

        These absorb cuBLAS autotuning, lazy module initialisation, allocator growth and
        the first-touch page faults that a unified-memory part pays on initial access.
        Measuring without them makes the first iteration an outlier by a wide margin.

        No prefill/decode_step range is emitted here on purpose -- an ncu NVTX filter would
        otherwise match warmup work and profile the wrong kernels.

        The final warmup iteration runs the **full** generation length. Earlier ones stop
        after a few steps, which is enough for autotuning and lazy init, but a short warmup
        never grows the KV cache to its measured depth -- so the first measured iteration
        still pays the allocator growth and first-touch faults for the deeper cache, which is
        exactly what warmup exists to absorb. Doing the full length once costs one extra
        generation and removes that.
        """
        cfg = self.config
        if cfg.warmup_iters <= 0:
            return

        full_length = max(1, cfg.generate_tokens)
        short_length = min(4, full_length)

        with nvtx_range(Phase.WARMUP.range_name):
            for iteration in range(cfg.warmup_iters):
                is_last = iteration == cfg.warmup_iters - 1
                state = backend.prefill(inputs)
                for _ in range(full_length if is_last else short_length):
                    state = backend.decode_step(state)
            backend.synchronize()
            del state
        result.notes.append(
            f"warmup: {cfg.warmup_iters} iterations discarded "
            f"(the last at the full {full_length}-token generation length, so the KV cache "
            "reaches its measured depth before measurement begins)"
        )

    def _run_once(
        self,
        backend: Backend,
        inputs,
        result: WorkloadResult,
        sampler: MemorySampler | None,
        scoped_step: bool,
    ) -> None:
        """One measured prefill plus decode loop.

        Args:
            scoped_step: When True (ncu mode), exactly one decode step carries the NVTX name
                the profiler filters on.
        """
        cfg = self.config

        # ---- prefill ----
        # The range name stays bare -- no ":batchxlen" suffix -- because Nsight Compute
        # filters on it by exact match. Shape detail goes in an instantaneous marker
        # instead, where it is still visible on the Nsight Systems timeline.
        mark(f"prefill {cfg.batch_size}x{cfg.prompt_tokens}")
        with PhaseMemoryTracker("prefill", sampler) as mem:
            with backend.measured_phase("prefill", tokens=cfg.prompt_tokens * cfg.batch_size) as t:
                with phase(Phase.PREFILL):
                    state = backend.prefill(inputs)
        result.timings.append(t)
        if mem.delta:
            result.memory_deltas.append(mem.delta)

        # ---- decode ----
        # The step chosen for ncu is the last one, where the KV cache is deepest. That is
        # the worst case for memory traffic and the one where the cache actually competes
        # with the weights for bandwidth -- an early step would flatter the model.
        target_index = max(0, cfg.generate_tokens - 1)

        with PhaseMemoryTracker("decode", sampler) as mem:
            with backend.measured_phase(
                "decode", tokens=cfg.generate_tokens * cfg.batch_size
            ) as t:
                with phase(Phase.DECODE, f"{cfg.generate_tokens}tok"):
                    for step in range(cfg.generate_tokens):
                        is_target = scoped_step and step == target_index
                        name = (
                            Phase.DECODE_STEP.range_name if (is_target or not scoped_step)
                            else UNSCOPED_STEP
                        )
                        if is_target:
                            result.profiled_step_context_len = state.position
                        with nvtx_range(name):
                            state = backend.decode_step(state)
        result.timings.append(t)
        if mem.delta:
            result.memory_deltas.append(mem.delta)

        # Materialising the sampled tokens costs a device-to-host copy, which is a full
        # synchronisation. It happens here, after the timer has closed, so it lands outside
        # the measured phase instead of stalling the host between every decode step.
        result.generated_token_ids = state.token_ids()

        if state.kv_bytes is None:
            state.kv_bytes = getattr(backend, "_measure_kv_bytes", lambda _c: None)(state.cache)
        measured_kv = backend._measure_kv_bytes(state.cache) if hasattr(
            backend, "_measure_kv_bytes"
        ) else state.kv_bytes
        if measured_kv:
            result.kv_cache_bytes = measured_kv

        # Per-token timing is the headline decode number, so record it as its own phase
        # rather than making every consumer divide.
        step_timing = type(t)(
            name="decode_step",
            start=t.start,
            end=t.start + (t.seconds / max(1, cfg.generate_tokens)),
            tokens=cfg.batch_size,
        )
        result.timings.append(step_timing)

        del state
