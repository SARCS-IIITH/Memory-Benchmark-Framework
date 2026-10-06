"""The backend interface.

A backend knows how to load a checkpoint and drive exactly two things: a prefill pass and a
single decode step. That narrow contract is deliberate -- it is the smallest surface that
still lets the harness put an NVTX range around one generated token, which is what makes
selective Nsight Compute profiling tractable.

Anything richer (full ``generate()`` with sampling, beam search, speculative decoding) would
bury the per-token boundary inside library code where no range can be placed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelConfig, WorkloadConfig


@dataclass
class GenerationState:
    """Carries the KV cache and the last token between decode steps."""

    cache: Any = None
    last_token_ids: Any = None
    position: int = 0

    #: Generated tokens as **device tensors**, not Python ints.
    #:
    #: Converting each token to an int inside the loop -- ``int(next_token[0, 0])`` -- forces
    #: a device-to-host copy, and a D2H copy on the default stream is a full synchronisation.
    #: That stalls the host between every step, so the decode loop can never run ahead and
    #: the launch of step N+1 waits on the completion of step N. It shows up directly as GPU
    #: idle time inside the measured phase, and it is pure bookkeeping: nothing in the
    #: forward pass needs the value on the host. The tensors are kept on-device and
    #: materialised once, after the timed region closes.
    generated_tokens: list[Any] = field(default_factory=list)

    #: Set when the backend cannot report a real KV cache size, so the report can fall back
    #: to the analytic estimate and say which it used.
    kv_bytes: int | None = None

    def token_ids(self) -> list[int]:
        """Materialise the generated tokens on the host.

        Call this only *outside* a timed or profiled region -- it is the D2H synchronisation
        the decode loop deliberately avoids.
        """
        ids: list[int] = []
        for token in self.generated_tokens:
            try:
                ids.append(int(token.reshape(-1)[0]))
            except Exception:                                    # noqa: BLE001
                continue
        return ids


@dataclass
class PhaseTiming:
    """Wall-clock for one phase, with the GPU synchronised at both ends.

    Without the synchronisation these numbers would measure queue submission rather than
    execution, and decode -- which submits far faster than it computes -- would look
    impossibly quick.
    """

    name: str
    start: float = 0.0
    end: float = 0.0
    tokens: int = 0

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 and self.tokens else 0.0

    def to_dict(self) -> dict:
        return {
            "phase": self.name,
            "seconds": self.seconds,
            "tokens": self.tokens,
            "tokens_per_second": self.tokens_per_second,
        }


class Backend(ABC):
    """Abstract inference backend."""

    #: Short identifier used in run directory names and on the CLI.
    name: str = "base"

    def __init__(self, model_config: ModelConfig, workload_config: WorkloadConfig) -> None:
        self.model_config = model_config
        self.workload_config = workload_config
        self.model: Any = None
        self.tokenizer: Any = None
        self._loaded = False
        #: The collector this process serves ("baseline", "nsys" or "ncu"), set by the worker
        #: before load. Lets a backend keep diagnostic hooks out of profiled runs.
        self.profile_mode: str = "baseline"

    # ---- lifecycle --------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the device and put the model in inference mode."""

    @abstractmethod
    def teardown(self) -> None:
        """Release the model and free device memory."""

    # ---- inference --------------------------------------------------------------------

    @abstractmethod
    def prepare_inputs(self) -> Any:
        """Build the input batch described by the workload config."""

    @abstractmethod
    def prefill(self, inputs: Any) -> GenerationState:
        """Run the prompt forward pass and return the state seeding decode."""

    @abstractmethod
    def decode_step(self, state: GenerationState) -> GenerationState:
        """Generate exactly one token. Must be a single forward pass, not a loop."""

    # ---- introspection ----------------------------------------------------------------

    @abstractmethod
    def describe(self) -> dict:
        """Backend and model facts worth recording in the run manifest."""

    def synchronize(self) -> None:
        """Block until all queued GPU work has completed."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:                                        # noqa: BLE001
            pass

    def measured_phase(self, name: str, tokens: int = 0) -> _PhaseTimer:
        return _PhaseTimer(self, name, tokens)

    # ---- context manager --------------------------------------------------------------

    def __enter__(self) -> Backend:
        if not self._loaded:
            self.load()
            self._loaded = True
        return self

    def __exit__(self, *_exc) -> None:
        self.teardown()
        self._loaded = False


class _PhaseTimer:
    """Times a phase with a GPU sync on entry and exit."""

    def __init__(self, backend: Backend, name: str, tokens: int) -> None:
        self.backend = backend
        self.timing = PhaseTiming(name=name, tokens=tokens)

    def __enter__(self) -> PhaseTiming:
        self.backend.synchronize()
        self.timing.start = time.perf_counter()
        return self.timing

    def __exit__(self, *_exc) -> None:
        self.backend.synchronize()
        self.timing.end = time.perf_counter()


_REGISTRY: dict[str, type[Backend]] = {}


def register(cls: type[Backend]) -> type[Backend]:
    """Register a backend under its ``name`` so the CLI can select it by string."""
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str) -> type[Backend]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")
    return _REGISTRY[name]


def available_backends() -> list[str]:
    return sorted(_REGISTRY)
