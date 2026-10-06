"""TensorRT-LLM backend -- scaffolded, not implemented.

This file exists so the engine backend can be added later without restructuring anything:
it implements the same :class:`~nsight_bench.backends.base.Backend` contract and is
registered under ``trtllm``, so ``nsbench run --backend trtllm`` resolves and fails with an
explanation rather than a stack trace.

Why it is not implemented yet
-----------------------------
Three things stood in the way when this harness was built (2026-08-22), all recorded here so
the decision does not have to be re-derived:

1. **No stable aarch64 release.** ``pypi.nvidia.com`` publishes only ``1.3.0rc*`` release
   candidates for ``linux_aarch64``/``manylinux_2_39_aarch64``. Plain PyPI carries an sdist
   stub that does not build. sm_121 (GB10) kernel coverage in those RCs is unverified.
2. **The container route is closed on this host.** NVIDIA's sanctioned path for DGX Spark is
   the NGC container, but docker here requires sudo and the user is not in the ``docker``
   group.
3. **It weakens the measurement, not just the setup.** TensorRT-LLM executes fused kernels
   behind CUDA graphs. Kernel-level ncu metrics still work, and ``nsys --cuda-graph-trace=node``
   resolves individual graph nodes -- but there is no way to place a Python-level NVTX range
   around one decode step or one transformer block, which is what the rest of this harness
   relies on for scoping and attribution. Comparing memory behaviour *across* models and
   quantization schemes is also harder, because each combination needs its own engine build
   (and ModelOpt conversion for the quantized ones) rather than just a different path.

Implementation notes for whoever picks this up
----------------------------------------------
* Provision the env first: ``bash setup/setup_trtllm_env.sh`` (separate venv, because
  tensorrt_llm pins its own torch and will otherwise replace the one the ``hf`` backend uses).
* Scope with ``cudaProfilerStart``/``Stop`` around a fixed number of decode steps instead of
  NVTX ranges, and set :attr:`NcuConfig.nvtx_scope` to empty so ncu falls back to
  ``--range-filter`` on the profiler API range.
* Keep ``--cuda-graph-trace=node`` on for the nsys pass, or every graph launch collapses into
  one opaque timeline entry.
* The memory-hierarchy analysis itself needs no changes -- it reads ncu counters per kernel
  and does not care who launched them.
"""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig, WorkloadConfig
from .base import Backend, GenerationState, register

_SETUP_HINT = (
    "The TensorRT-LLM backend is not implemented.\n"
    "  Environment:  bash setup/setup_trtllm_env.sh   (creates ~/envs/nsbench-trtllm)\n"
    "  Status:       aarch64 wheels are release-candidate only (tensorrt_llm 1.3.0rc*),\n"
    "                sm_121 support unverified, and the NGC container path needs docker\n"
    "                access this user does not have.\n"
    "  Use instead:  --backend hf   (HuggingFace transformers, fully instrumented)\n"
    "  See nsight_bench/backends/trtllm.py for implementation notes."
)


@register
class TensorRTLLMBackend(Backend):
    """Placeholder implementing the backend contract without a working engine path."""

    name = "trtllm"

    def __init__(self, model_config: ModelConfig, workload_config: WorkloadConfig) -> None:
        super().__init__(model_config, workload_config)

    @staticmethod
    def is_available() -> bool:
        """Whether ``tensorrt_llm`` can be imported in the current interpreter."""
        try:
            import tensorrt_llm  # noqa: F401

            return True
        except Exception:                                        # noqa: BLE001
            return False

    def load(self) -> None:
        raise NotImplementedError(_SETUP_HINT)

    def teardown(self) -> None:
        return None

    def prepare_inputs(self) -> Any:
        raise NotImplementedError(_SETUP_HINT)

    def prefill(self, inputs: Any) -> GenerationState:
        raise NotImplementedError(_SETUP_HINT)

    def decode_step(self, state: GenerationState) -> GenerationState:
        raise NotImplementedError(_SETUP_HINT)

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "implemented": False,
            "tensorrt_llm_importable": self.is_available(),
            "reason": _SETUP_HINT,
        }
