"""Profiler capture-range control.

Nsight Systems is launched with ``--capture-range=cudaProfilerApi``, which means it traces
nothing until the application calls ``cudaProfilerStart``. That is deliberate: a
transformers run spends most of its wall clock loading weights and warming up, and tracing
all of it would produce a report dominated by startup noise with the steady state squeezed
into a sliver of the timeline.

The workload calls :func:`start_capture` once warmup is done and :func:`stop_capture` when
the measured region ends, so the trace contains steady-state work and nothing else.

When the process is not running under a profiler these calls are harmless no-ops, so the
same code path serves profiled and unprofiled runs without branching.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


def _profiler():
    try:
        import torch.cuda.profiler as profiler

        return profiler
    except Exception:                                            # noqa: BLE001
        return None


def under_profiler() -> bool:
    """Best-effort detection of whether a profiler is attached.

    Both tools inject environment markers into the target process. This is used only to
    decide whether to log that a capture range was requested -- never to change what the
    workload computes, since that would make profiled and unprofiled runs incomparable.
    """
    markers = (
        "NSYS_PROFILING_SESSION_ID",
        "NSYS_INJECTION_SESSION_ID",
        "NVCOMPUTE_PROFILING_SESSION",
        "NV_COMPUTE_PROFILER",
    )
    return any(os.environ.get(m) for m in markers)


def start_capture() -> None:
    """Open the profiler capture range (``cudaProfilerStart``)."""
    profiler = _profiler()
    if profiler is not None:
        try:
            profiler.start()
        except Exception:                                        # noqa: BLE001
            pass


def stop_capture() -> None:
    """Close the profiler capture range (``cudaProfilerStop``)."""
    profiler = _profiler()
    if profiler is not None:
        try:
            profiler.stop()
        except Exception:                                        # noqa: BLE001
            pass


@contextlib.contextmanager
def capture_range() -> Iterator[None]:
    """Scope the profiler's capture range to this block.

    The stop is in a ``finally`` so an exception inside the measured region still closes the
    range; leaving it open makes nsys trace teardown and produces a misleading report.
    """
    start_capture()
    try:
        yield
    finally:
        stop_capture()
