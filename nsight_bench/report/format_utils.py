"""Formatting helpers shared by the Markdown and HTML reporters.

Both reporters render the same numbers, so the rules for how a number is presented live in
one place. Two of those rules matter more than they look:

``fmt_bytes`` and friends return an explicit ``"not measured"`` for ``None`` rather than
falling back to ``0``. A report that prints ``0 B`` for a counter this GPU cannot expose is
worse than one that says nothing -- it invents a measurement.

Byte quantities use decimal SI units (MB = 10^6). Sector counts, cache sizes and bandwidths
are all naturally decimal in Nsight's own output, and mixing MB with MiB across one report is
how factors of 1.05 creep into comparisons.
"""

from __future__ import annotations

NOT_MEASURED = "not measured"


def fmt_bytes(value: float | None, precision: int = 2) -> str:
    """Human-readable byte count in decimal SI units."""
    if value is None:
        return NOT_MEASURED
    magnitude = abs(value)
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if magnitude >= scale:
            return f"{value / scale:,.{precision}f} {unit}"
    return f"{value:,.0f} B"


def fmt_count(value: float | None, precision: int = 0) -> str:
    if value is None:
        return NOT_MEASURED
    return f"{value:,.{precision}f}"


def fmt_pct(value: float | None, precision: int = 1) -> str:
    if value is None:
        return NOT_MEASURED
    return f"{value:.{precision}f}%"


def fmt_ratio(value: float | None, precision: int = 2, suffix: str = "x") -> str:
    if value is None:
        return NOT_MEASURED
    return f"{value:.{precision}f}{suffix}"


def fmt_bandwidth(value: float | None, precision: int = 1) -> str:
    if value is None:
        return NOT_MEASURED
    return f"{value:,.{precision}f} GB/s"


def fmt_time_ns(value: float | None) -> str:
    """Duration from nanoseconds, scaled to whatever unit reads cleanly."""
    if value is None:
        return NOT_MEASURED
    if value >= 1e9:
        return f"{value / 1e9:,.3f} s"
    if value >= 1e6:
        return f"{value / 1e6:,.3f} ms"
    if value >= 1e3:
        return f"{value / 1e3:,.1f} us"
    return f"{value:,.0f} ns"


def fmt_seconds(value: float | None, precision: int = 3) -> str:
    if value is None:
        return NOT_MEASURED
    if value < 1e-3:
        return f"{value * 1e6:,.1f} us"
    if value < 1:
        return f"{value * 1e3:,.{precision}f} ms"
    return f"{value:,.{precision}f} s"


def fmt_flops(value: float | None) -> str:
    if value is None:
        return NOT_MEASURED
    for unit, scale in (("TFLOP", 1e12), ("GFLOP", 1e9), ("MFLOP", 1e6)):
        if abs(value) >= scale:
            return f"{value / scale:,.2f} {unit}"
    return f"{value:,.0f} FLOP"


def markdown_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Render a GitHub-flavoured Markdown table.

    Columns are padded to a uniform width so the raw Markdown stays readable in a terminal or
    a diff, not only after rendering.
    """
    if not rows:
        return "_No data._"

    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(str(cell)))

    align = align or ["left"] + ["right"] * (len(headers) - 1)

    def _sep(width: int, how: str) -> str:
        if how == "right":
            return "-" * (width + 1) + ":"
        if how == "center":
            return ":" + "-" * width + ":"
        return "-" * (width + 2)

    def _row(cells: list[str]) -> str:
        padded = []
        for index, cell in enumerate(cells):
            width = widths[index] if index < len(widths) else len(str(cell))
            how = align[index] if index < len(align) else "left"
            text = str(cell)
            padded.append(text.rjust(width) if how == "right" else text.ljust(width))
        return "| " + " | ".join(padded) + " |"

    lines = [_row(headers), "|" + "|".join(
        _sep(widths[i], align[i] if i < len(align) else "left") for i in range(len(headers))
    ) + "|"]
    lines.extend(_row([str(c) for c in row]) for row in rows)
    return "\n".join(lines)


def truncate(text: str, limit: int = 52) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"
