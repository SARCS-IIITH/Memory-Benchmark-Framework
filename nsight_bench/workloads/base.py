"""Workload interface and result container.

A workload drives a backend through the phases we want to measure and decides *where the
NVTX ranges go*. That placement is not cosmetic: it is what separates prefill from decode in
every report, and what gives Nsight Compute a single decode step to profile instead of the
whole generation loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum

from .. import stats
from ..backends.base import Backend, PhaseTiming
from ..config import WorkloadConfig
from ..instrumentation.memory import MemorySampler, PhaseMemoryDelta


class ProfileMode(str, Enum):
    """How the workload should annotate and scope itself for the collector in use.

    The three collectors want genuinely different things, and pretending otherwise produces
    either useless data or runs that never finish:

    * ``BASELINE`` -- no capture range, many repeats. Honest timing.
    * ``NSYS`` -- capture range around the measured region so the timeline is steady state
      only, with every phase annotated.
    * ``NCU`` -- exactly one range matching the scope filter, because Nsight Compute replays
      each kernel several times. Annotating every decode step here would multiply the
      collection cost by the number of generated tokens for no extra information.
    """

    BASELINE = "baseline"
    NSYS = "nsys"
    NCU = "ncu"


@dataclass
class WorkloadResult:
    """Everything one workload execution produced, minus the profiler artefacts."""

    mode: str = ProfileMode.BASELINE.value
    timings: list[PhaseTiming] = field(default_factory=list)
    memory_deltas: list[PhaseMemoryDelta] = field(default_factory=list)

    prompt_tokens: int = 0
    generated_tokens: int = 0
    batch_size: int = 0

    #: KV-cache bytes as measured from the live cache tensors at the end of generation.
    kv_cache_bytes: int | None = None
    #: Context length at the moment the ncu-scoped decode step ran, so its measured traffic
    #: can be compared against the analytic KV-cache size at that exact depth.
    profiled_step_context_len: int | None = None

    layers_annotated: int = 0
    #: Sampled tokens, materialised once after the timed region closes. Kept so a run can be
    #: shown to have generated real text rather than repeating one token, which is the usual
    #: symptom of a broken cache or mask.
    generated_token_ids: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def phase(self, name: str) -> PhaseTiming | None:
        for timing in self.timings:
            if timing.name == name:
                return timing
        return None

    def phases(self, name: str) -> list[PhaseTiming]:
        return [t for t in self.timings if t.name == name]

    def median_seconds(self, name: str) -> float | None:
        return stats.median([t.seconds for t in self.phases(name)])

    def iqr_seconds(self, name: str) -> float | None:
        """Interquartile range, reported alongside the median.

        Repeats on a thermally-managed box are not identically distributed; an IQR shows
        whether a headline number is stable or whether one iteration ran away.
        """
        return stats.iqr([t.seconds for t in self.phases(name)])

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "batch_size": self.batch_size,
            "kv_cache_bytes": self.kv_cache_bytes,
            "profiled_step_context_len": self.profiled_step_context_len,
            "layers_annotated": self.layers_annotated,
            "generated_token_ids": self.generated_token_ids,
            "distinct_generated_tokens": len(set(self.generated_token_ids)),
            "timings": [t.to_dict() for t in self.timings],
            "memory_deltas": [d.to_dict() for d in self.memory_deltas],
            "notes": self.notes,
        }


class Workload(ABC):
    """Abstract workload."""

    kind: str = "base"

    def __init__(self, config: WorkloadConfig) -> None:
        self.config = config

    @abstractmethod
    def run(
        self,
        backend: Backend,
        mode: ProfileMode = ProfileMode.BASELINE,
        sampler: MemorySampler | None = None,
    ) -> WorkloadResult:
        """Execute the workload and return its result."""

    def describe(self) -> dict:
        return {"kind": self.kind, **asdict(self.config)}


_REGISTRY: dict[str, type[Workload]] = {}


def register(cls: type[Workload]) -> type[Workload]:
    _REGISTRY[cls.kind] = cls
    return cls


def get_workload(kind: str) -> type[Workload]:
    if kind not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise KeyError(f"Unknown workload '{kind}'. Available: {available}")
    return _REGISTRY[kind]


def build_workload(config: WorkloadConfig) -> Workload:
    return get_workload(config.kind)(config)


def available_workloads() -> list[str]:
    return sorted(_REGISTRY)

