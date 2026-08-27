"""Small summary statistics, shared by the workload result and the analysis layer.

Both need the same median-and-spread over a handful of repeats, and having two
implementations is how they drift apart. The interesting decisions are here.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Fewest repeats for which a spread figure is reported at all. Two points have no interior,
#: so a quartile range over them is just the range.
MIN_SAMPLES_FOR_IQR = 3


def median(values: Sequence[float]) -> float | None:
    """Median of an unsorted sequence, or None when empty."""
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Linearly interpolated percentile over an already-sorted sequence.

    Interpolated rather than nearest-rank because nearest-rank needs at least four samples
    before the first and third quartiles land on different elements. Benchmark repeats are
    expensive, so runs of three are normal, and a spread of "-" on every headline number is
    worse than an interpolated one -- it reads as "not measured" when the truth is "measured,
    over three points". The sample count is reported alongside so the reader can weigh it.
    """
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def iqr(values: Sequence[float]) -> float | None:
    """Interquartile range, or None when there are too few samples to mean anything."""
    ordered = sorted(values)
    if len(ordered) < MIN_SAMPLES_FOR_IQR:
        return None
    return _percentile(ordered, 0.75) - _percentile(ordered, 0.25)


def median_iqr(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    """``(median, iqr, sample_count)`` in one pass over the same data."""
    ordered = sorted(values)
    return median(ordered), iqr(ordered), len(ordered)
