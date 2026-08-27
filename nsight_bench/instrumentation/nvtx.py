"""NVTX annotation: the scaffolding that makes selective profiling possible.

Both profilers are driven by these ranges. Nsight Systems projects GPU work onto them to
split prefill from decode; Nsight Compute uses ``--nvtx-include`` to profile *only* the
kernels inside one decode step, which is the difference between a five-minute collection and
an overnight one.

Range names are therefore a contract, not a cosmetic choice. They follow a flat, dotted
convention in the default NVTX domain::

    nsbench.load          model construction and weight load
    nsbench.warmup        discarded iterations
    nsbench.prefill       the prompt forward pass
    nsbench.decode        the whole generation loop
    nsbench.decode_step   exactly one generated token   <-- the ncu scoping target
    nsbench.layer.<i>     one transformer block, when layer annotation is enabled

The default domain is used deliberately: ``torch.cuda.nvtx`` pushes there, so custom-domain
ranges and torch's own annotations would not nest, and ncu's filter syntax stays simple.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from enum import Enum
from typing import Any

_PREFIX = "nsbench"


class Phase(str, Enum):
    """A benchmark phase. The value doubles as the NVTX range name."""

    LOAD = f"{_PREFIX}.load"
    WARMUP = f"{_PREFIX}.warmup"
    PREFILL = f"{_PREFIX}.prefill"
    DECODE = f"{_PREFIX}.decode"
    DECODE_STEP = f"{_PREFIX}.decode_step"
    TOKENIZE = f"{_PREFIX}.tokenize"
    CALIBRATE = f"{_PREFIX}.calibrate"

    @property
    def range_name(self) -> str:
        return self.value


#: Phases worth measuring separately in a report. Ordered as they occur.
MEASURED_PHASES: tuple[Phase, ...] = (Phase.PREFILL, Phase.DECODE, Phase.DECODE_STEP)


def layer_range_name(index: int, module_name: str = "") -> str:
    """NVTX name for one transformer block."""
    suffix = f".{module_name}" if module_name else ""
    return f"{_PREFIX}.layer.{index:03d}{suffix}"


def ncu_filter(range_name: str) -> str:
    """Convert a range name into the form ``ncu --nvtx-include`` expects.

    Nsight Compute's NVTX filter treats a bare name as a **start/end** range and a name with
    a ``/`` as a **push/pop** range. ``torch.cuda.nvtx.range_push`` produces push/pop ranges,
    so a bare name silently matches nothing -- ncu exits 0, writes no report, and prints only
    "No kernels were profiled", which reads like the workload never ran.

    The trailing slash means "any kernel launched anywhere inside this range", including
    nested deeper, which is what phase scoping needs.
    """
    return range_name if range_name.endswith("/") else f"{range_name}/"


def _nvtx():
    """Return torch's NVTX module, or None when torch/CUDA is unavailable.

    Annotation must never be the reason a run fails, so every entry point degrades to a
    no-op rather than raising.
    """
    try:
        import torch.cuda.nvtx as nvtx

        return nvtx
    except Exception:                                            # noqa: BLE001
        return None


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push an NVTX range for the duration of the block."""
    nvtx = _nvtx()
    if nvtx is None:
        yield
        return
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


@contextlib.contextmanager
def phase(which: Phase, detail: str = "") -> Iterator[None]:
    """Annotate a benchmark phase.

    ``detail`` is appended after a colon for human readability on the Nsight Systems
    timeline, where :attr:`NvtxRange.phase` strips it back off again for grouping.

    **Never pass ``detail`` for a range ncu filters on.** ``--nvtx-include`` matches the
    range name literally -- it is not prefix-based, and it does not treat a colon as a
    separator -- so ``nsbench.decode_step:12tok`` would not match the filter
    ``nsbench.decode_step/`` and the collection would come back empty. ncu reports that as a
    bare "No kernels were profiled" with a zero exit code, which reads like the workload
    never ran. Only the enclosing ``nsbench.decode`` range, which nothing filters on, carries
    detail today.
    """
    name = f"{which.range_name}:{detail}" if detail else which.range_name
    with nvtx_range(name):
        yield


def mark(message: str) -> None:
    """Emit an instantaneous NVTX marker, useful for pinning events on the timeline."""
    nvtx = _nvtx()
    if nvtx is not None:
        nvtx.mark(message)


class LayerAnnotator:
    """Wraps every transformer block in an NVTX range via forward hooks.

    Per-layer ranges turn an undifferentiated wall of kernels into something attributable:
    they are what lets the report say "attention reads 3x the bytes the MLP does" instead of
    just listing kernel names.

    The hooks are cheap but not free, so this is opt-in, and :meth:`remove` restores the
    model exactly. Applying it twice is a no-op rather than double-annotating.
    """

    def __init__(self, model: Any, layer_attr_candidates: tuple[str, ...] = ()) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.layer_count = 0
        self._candidates = layer_attr_candidates or (
            "model.layers",
            "model.decoder.layers",
            "transformer.h",
            "model.language_model.layers",
            "language_model.model.layers",
            "layers",
        )

    def _find_layers(self) -> list[Any]:
        """Locate the decoder block list across the common HF naming conventions."""
        for path in self._candidates:
            obj = self.model
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                try:
                    layers = list(obj)
                except TypeError:
                    continue
                if layers:
                    return layers
        return []

    def apply(self) -> int:
        """Install the hooks. Returns the number of layers annotated."""
        if self.handles:
            return self.layer_count

        layers = self._find_layers()
        nvtx = _nvtx()
        if nvtx is None or not layers:
            return 0

        for index, layer in enumerate(layers):
            name = layer_range_name(index, type(layer).__name__)

            def _pre(_module, _args, _name=name):
                nvtx.range_push(_name)

            def _post(_module, _args, _output):
                nvtx.range_pop()
                return None

            self.handles.append(layer.register_forward_pre_hook(_pre))
            self.handles.append(layer.register_forward_hook(_post))

        self.layer_count = len(layers)
        return self.layer_count

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.layer_count = 0

    def __enter__(self) -> LayerAnnotator:
        self.apply()
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()
