"""Inline SVG charts for the HTML report.

Charts are emitted as hand-built SVG rather than rendered images. That keeps the report a
single self-contained file with no image payload, keeps text selectable and crisp at any
zoom, and lets the marks inherit CSS custom properties so light and dark modes are one
variable swap rather than two rendered copies.

Colour follows the validated reference palette, restricted to its first three categorical
slots -- the documented subset that clears the all-pairs colour-vision-deficiency and
normal-vision separation gates in both modes. Nothing here invents a hue, and no chart uses
more than three series, so the gate holds by construction.

Form is chosen by the job the data has to do:

* **Hierarchy traffic** is magnitude across an ordered set of levels, and the story is a
  funnel -- what the SM asked for versus what survived to reach memory. Small-multiple
  horizontal bars, one panel per phase, on a linear scale with every bar directly labelled.
  A shared log axis was rejected: it would let a 30x difference between prefill and decode
  read as a modest one, which is precisely the finding being reported.
* **The bandwidth sweep** is change over a continuous variable, so it is a line with markers
  and a reference line at L2 capacity.
* **The roofline** is inherently log-log and is drawn as such, with the ceilings as lines and
  each phase as a labelled point.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

#: Validated categorical slots 1-3 from the reference palette. Light and dark are the same
#: hues re-stepped for their surface, not a separate palette.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70")


def esc(text) -> str:
    return html.escape(str(text), quote=True)


# --------------------------------------------------------------------------------------
# Layout primitives
#
# SVG has no text metrics and no flow layout: every glyph sits exactly where it is told, and
# nothing reflows or wraps to make room. Two failure modes follow, and both were visible in
# these charts:
#
#   * Stacked header text collides when a later element's baseline is computed independently
#     of what precedes it -- an axis label placed at a fixed offset above the plot lands on
#     top of a subtitle whenever a subtitle exists.
#   * Long category labels run past the viewBox. With ``overflow: visible`` they then paint
#     outside the card entirely, over whatever the page put next to them.
#
# The helpers below fix both by construction: headers are laid out sequentially so each row
# knows where the previous one ended, and labels are measured and truncated before they are
# ever emitted.
# --------------------------------------------------------------------------------------

#: Mean glyph advance as a fraction of font size, for the report's system sans stack. Real
#: advances vary per glyph and no metrics are available at render time, so this deliberately
#: overestimates: a label truncated slightly early is invisible, one truncated slightly late
#: overflows the panel.
_CHAR_WIDTH_RATIO = 0.56


def text_width(text: str, font_px: float) -> float:
    """Approximate rendered width of a string, in pixels."""
    return len(str(text)) * font_px * _CHAR_WIDTH_RATIO


def fit_label(text: str, max_px: float, font_px: float) -> str:
    """Truncate a label with an ellipsis so it fits inside ``max_px``.

    Truncation is from the left, keeping the tail, because these labels are kernel names:
    the distinguishing part of ``cutlass_80_wmma_tensorop_bf16_...`` is the end, and dozens
    of kernels share the first twenty characters. Cutting the head is what keeps a column of
    them readable.
    """
    text = str(text)
    if text_width(text, font_px) <= max_px:
        return text
    keep = max(4, int(max_px / (font_px * _CHAR_WIDTH_RATIO)) - 1)
    return "…" + text[-keep:] if keep < len(text) else text


def _header(title: str, subtitle: str, y_label: str = "") -> tuple[list[str], int]:
    """Emit the title/subtitle/axis-label stack and return it with the plot's top edge.

    Each row's baseline is derived from the previous row's, so adding a subtitle pushes the
    axis label down instead of colliding with it, and the plot area starts below whatever was
    actually drawn rather than below an assumed two rows.
    """
    parts: list[str] = []
    baseline = 16
    parts.append(f'<text class="chart-title" x="0" y="{baseline}">{esc(title)}</text>')

    if subtitle:
        baseline += 17
        parts.append(f'<text class="chart-sub" x="0" y="{baseline}">{esc(subtitle)}</text>')

    if y_label:
        baseline += 15
        parts.append(
            f'<text class="chart-axis-label" x="0" y="{baseline}">{esc(y_label)}</text>'
        )

    return parts, baseline + 12


def _nice_ceiling(value: float) -> float:
    """Round an axis maximum up to a readable 1/2/5 x 10^n step."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10 ** exponent
    for multiple in (1, 2, 2.5, 5, 10):
        if value <= multiple * base:
            return multiple * base
    return 10 * base


def fmt_bytes_short(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if magnitude >= scale:
            return f"{value / scale:,.1f} {unit}"
    return f"{value:,.0f} B"


@dataclass
class BarDatum:
    label: str
    value: float | None
    tooltip: str = ""


def hbar_panel(
    title: str,
    data: list[BarDatum],
    series_index: int = 0,
    width: int = 430,
    row_height: int = 34,
    value_formatter=fmt_bytes_short,
    subtitle: str = "",
) -> str:
    """One horizontal bar panel: ordered categories, magnitude, single series.

    No legend -- the title names the single series, which is the rule for one-series charts.
    Every bar is directly labelled, so the reader never has to measure against the axis, and
    unmeasured levels say so instead of drawing a zero-length bar that would read as "no
    traffic".
    """
    label_font = 11.5
    value_font = 11.5

    # The label gutter is sized to the labels rather than fixed. A fixed 118px worked for the
    # hierarchy panels ("L1 / TEX cache") and overflowed badly for the kernel panels, whose
    # demangled cutlass and cuBLAS names are several times longer. It is capped at a share of
    # the chart so the bars never get squeezed out by one long name, and anything still too
    # long is truncated to fit -- the full name stays in the tooltip and the accompanying
    # table.
    widest = max((text_width(d.label, label_font) for d in data), default=0.0)
    label_width = int(max(118, min(widest + 12, width * 0.34)))

    # The value sits to the right of its bar, so the right gutter must hold the longest one
    # plus its offset -- otherwise the number on the longest bar runs off the edge.
    widest_value = max(
        (text_width(value_formatter(d.value), value_font)
         for d in data if d.value is not None),
        default=0.0,
    )
    right_pad = int(max(64, widest_value + 16))

    plot_width = max(40, width - label_width - right_pad)
    header, top = _header(title, subtitle)
    height = top + len(data) * row_height + 14

    values = [d.value for d in data if d.value is not None and d.value > 0]
    maximum = _nice_ceiling(max(values)) if values else 1.0

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}" preserveAspectRatio="xMinYMin meet">',
        *header,
    ]

    colour = f"var(--series-{series_index + 1})"

    for index, datum in enumerate(data):
        y = top + index * row_height
        bar_y = y + 7
        bar_h = row_height - 16

        # Always carry the untruncated name in the tooltip, so shortening the visible label
        # never costs the reader the identity of the row.
        label = fit_label(datum.label, label_width - 8, label_font)
        label_tip = f' data-tip="{esc(datum.label)}"' if label != datum.label else ""
        parts.append(
            f'<text class="chart-cat" x="{label_width - 8}" y="{bar_y + bar_h - 2}" '
            f'text-anchor="end"{label_tip}>{esc(label)}</text>'
        )

        if datum.value is None:
            parts.append(
                f'<text class="chart-na" x="{label_width + 4}" y="{bar_y + bar_h - 2}">'
                f'not measured</text>'
            )
            continue

        bar_w = max(1.0, (datum.value / maximum) * plot_width) if datum.value > 0 else 1.0
        tooltip = datum.tooltip or f"{datum.label}: {value_formatter(datum.value)}"
        # 4px rounded data-end, square against the baseline.
        parts.append(
            f'<rect class="mark" x="{label_width}" y="{bar_y}" width="{bar_w:.2f}" '
            f'height="{bar_h}" rx="4" fill="{colour}" '
            f'data-tip="{esc(tooltip)}"><title>{esc(tooltip)}</title></rect>'
        )
        parts.append(
            f'<text class="chart-val" x="{label_width + bar_w + 8:.2f}" '
            f'y="{bar_y + bar_h - 2}">{esc(value_formatter(datum.value))}</text>'
        )

    parts.append(
        f'<line class="axis" x1="{label_width}" y1="{top + 2}" x2="{label_width}" '
        f'y2="{top + len(data) * row_height + 2}" />'
    )
    parts.append("</svg>")
    return "".join(parts)


def line_chart(
    title: str,
    points: list[tuple[float, float]],
    x_label: str = "",
    y_label: str = "",
    width: int = 720,
    height: int = 300,
    log_x: bool = False,
    reference_x: float | None = None,
    reference_label: str = "",
    series_index: int = 0,
    x_formatter=lambda v: f"{v:,.0f}",
    y_formatter=lambda v: f"{v:,.0f}",
    point_tooltip=None,
    subtitle: str = "",
) -> str:
    """Line with markers over a continuous x. Single series, so no legend box.

    ``log_x`` exists for the bandwidth sweep, whose working-set sizes are powers of two: on a
    linear axis every small size collapses into the origin and the knee -- the entire point
    of the chart -- becomes invisible.
    """
    if len(points) < 2:
        return ""

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    y_max = _nice_ceiling(max(ys))

    header, top = _header(title, subtitle, y_label)

    # The left gutter holds the y tick labels, so it is sized to the widest one. A fixed 62px
    # clipped five- and six-digit ticks against the axis.
    widest_tick = max(text_width(y_formatter(y_max * s / 4), 10.5) for s in range(5))
    left = int(max(46, widest_tick + 14))
    right, bottom = 22, 46
    plot_w = width - left - right
    # Adding an axis-label row moves the plot down; grow the canvas to match rather than
    # eating into the plot, which would squash the chart every time a label is supplied.
    height = height + max(0, top - (46 if subtitle else 32))
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        if log_x:
            lo, hi = math.log10(min(xs)), math.log10(max(xs))
            span = hi - lo or 1.0
            return left + (math.log10(value) - lo) / span * plot_w
        lo, hi = min(xs), max(xs)
        span = (hi - lo) or 1.0
        return left + (value - lo) / span * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    colour = f"var(--series-{series_index + 1})"
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}" preserveAspectRatio="xMinYMin meet">',
        *header,
    ]

    # Recessive gridlines with labelled y ticks.
    for step in range(5):
        value = y_max * step / 4
        y = sy(value)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" />'
        )
        parts.append(
            f'<text class="chart-tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">'
            f'{esc(y_formatter(value))}</text>'
        )

    if reference_x is not None and min(xs) <= reference_x <= max(xs):
        rx = sx(reference_x)
        parts.append(
            f'<line class="ref" x1="{rx:.1f}" y1="{top}" x2="{rx:.1f}" '
            f'y2="{top + plot_h}" />'
        )
        if reference_label:
            parts.append(
                f'<text class="chart-ref-label" x="{rx + 6:.1f}" y="{top + 12}">'
                f'{esc(reference_label)}</text>'
            )

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(x):.2f},{sy(y):.2f}"
        for i, (x, y) in enumerate(points)
    )
    parts.append(f'<path class="line" d="{path}" stroke="{colour}" />')

    for x, y in points:
        tip = point_tooltip(x, y) if point_tooltip else (
            f"{x_formatter(x)}: {y_formatter(y)}"
        )
        parts.append(
            f'<circle class="dot" cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="{colour}" '
            f'data-tip="{esc(tip)}"><title>{esc(tip)}</title></circle>'
        )

    # X ticks: thinned so labels never collide on a log axis of powers of two.
    step = max(1, len(points) // 7)
    for index in range(0, len(points), step):
        x = xs[index]
        parts.append(
            f'<text class="chart-tick" x="{sx(x):.1f}" y="{top + plot_h + 18}" '
            f'text-anchor="middle">{esc(x_formatter(x))}</text>'
        )

    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" />'
    )
    if x_label:
        parts.append(
            f'<text class="chart-axis-label" x="{left + plot_w / 2}" y="{height - 6}" '
            f'text-anchor="middle">{esc(x_label)}</text>'
        )
    # The y label is emitted by _header, above the plot and below the subtitle, so the two
    # can no longer share a baseline.
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class RooflinePoint:
    label: str
    intensity: float
    achieved_gflops: float


def roofline_chart(
    points: list[RooflinePoint],
    peak_bandwidth_gbps: float,
    peak_compute_gflops: float,
    width: int = 720,
    height: int = 340,
    title: str = "Roofline",
    subtitle: str = "",
) -> str:
    """Log-log roofline: the memory-bound slope, the compute ceiling, and each phase.

    Both ceilings are measured on this machine rather than taken from a datasheet, which is
    what makes a point's distance from the roof meaningful instead of aspirational.
    """
    usable = [p for p in points if p.intensity > 0 and p.achieved_gflops > 0]
    if not usable or peak_bandwidth_gbps <= 0 or peak_compute_gflops <= 0:
        return ""

    header, top = _header(title, subtitle, "GFLOP/s")
    left, right, bottom = 66, 100, 48
    plot_w = width - left - right
    height = height + max(0, top - (46 if subtitle else 32))
    plot_h = height - top - bottom

    ridge = peak_compute_gflops / peak_bandwidth_gbps
    x_min = min([p.intensity for p in usable] + [ridge]) / 8
    x_max = max([p.intensity for p in usable] + [ridge]) * 8
    y_min = min([p.achieved_gflops for p in usable]) / 8
    y_max = peak_compute_gflops * 2

    lx0, lx1 = math.log10(x_min), math.log10(x_max)
    ly0, ly1 = math.log10(y_min), math.log10(y_max)

    def sx(value: float) -> float:
        return left + (math.log10(value) - lx0) / (lx1 - lx0) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (math.log10(value) - ly0) / (ly1 - ly0) * plot_h

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}" preserveAspectRatio="xMinYMin meet">',
        *header,
    ]

    for exponent in range(math.floor(lx0), math.ceil(lx1) + 1):
        value = 10 ** exponent
        if not (x_min <= value <= x_max):
            continue
        x = sx(value)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                     f'y2="{top + plot_h}" />')
        parts.append(f'<text class="chart-tick" x="{x:.1f}" y="{top + plot_h + 18}" '
                     f'text-anchor="middle">{_pow_label(exponent)}</text>')

    for exponent in range(math.floor(ly0), math.ceil(ly1) + 1):
        value = 10 ** exponent
        if not (y_min <= value <= y_max):
            continue
        y = sy(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" '
                     f'x2="{left + plot_w}" y2="{y:.1f}" />')
        parts.append(f'<text class="chart-tick" x="{left - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{_pow_label(exponent)}</text>')

    # The roof: bandwidth-limited slope up to the ridge, flat compute ceiling after it.
    roof = []
    for value in (x_min, min(ridge, x_max)):
        roof.append((sx(value), sy(min(peak_bandwidth_gbps * value, peak_compute_gflops))))
    if ridge < x_max:
        roof.append((sx(x_max), sy(peak_compute_gflops)))
    roof_path = " ".join(
        f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}" for i, (x, y) in enumerate(roof)
    )
    parts.append(f'<path class="roof" d="{roof_path}" />')
    parts.append(
        f'<text class="chart-ref-label" x="{left + 6}" y="{sy(peak_bandwidth_gbps * x_min * 1.6) - 6:.1f}">'
        f'{peak_bandwidth_gbps:,.0f} GB/s LPDDR5X</text>'
    )
    parts.append(
        f'<text class="chart-ref-label" x="{left + plot_w - 4}" y="{sy(peak_compute_gflops) - 8:.1f}" '
        f'text-anchor="end">{peak_compute_gflops:,.0f} GFLOP/s peak</text>'
    )

    for index, point in enumerate(usable):
        colour = f"var(--series-{(index % 3) + 1})"
        x, y = sx(point.intensity), sy(point.achieved_gflops)
        ceiling = min(peak_bandwidth_gbps * point.intensity, peak_compute_gflops)
        pct = 100 * point.achieved_gflops / ceiling if ceiling else 0
        tip = (
            f"{point.label}: {point.intensity:,.2f} FLOP/byte, "
            f"{point.achieved_gflops:,.1f} GFLOP/s ({pct:.0f}% of the roof here)"
        )
        parts.append(
            f'<circle class="dot big" cx="{x:.2f}" cy="{y:.2f}" r="8" fill="{colour}" '
            f'data-tip="{esc(tip)}"><title>{esc(tip)}</title></circle>'
        )
        # A point near the right edge would push its label past the canvas, so the label
        # flips to the other side of the marker instead of overflowing.
        label_w = text_width(point.label, 11.5)
        if x + 13 + label_w > width - 4:
            anchor, label_x = "end", x - 13
        else:
            anchor, label_x = "start", x + 13
        parts.append(
            f'<text class="chart-point-label" x="{label_x:.2f}" y="{y + 4:.2f}" '
            f'text-anchor="{anchor}">{esc(point.label)}</text>'
        )

    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" '
                 f'x2="{left + plot_w}" y2="{top + plot_h}" />')
    parts.append(
        f'<text class="chart-axis-label" x="{left + plot_w / 2}" y="{height - 6}" '
        f'text-anchor="middle">Arithmetic intensity (FLOP / byte from LPDDR5X)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _pow_label(exponent: int) -> str:
    if -2 <= exponent <= 4:
        value = 10 ** exponent
        return f"{value:,.2f}".rstrip("0").rstrip(".") if exponent < 0 else f"{value:,.0f}"
    return f"1e{exponent}"


def chart_css() -> str:
    """Styles for every chart. Colours are roles, so a theme swap changes one block."""
    return """
/* overflow:hidden is the backstop, not the layout. Labels are measured and truncated to fit
   before they are emitted; this only guarantees that a bad width estimate on an unusual font
   clips inside the card instead of painting across the rest of the page. */
.chart { width: 100%; height: auto; overflow: hidden; font-family: inherit; display: block; }
.chart-title { fill: var(--text-primary); font-size: 13px; font-weight: 600; }
.chart-sub { fill: var(--text-secondary); font-size: 11px; }
.chart-cat { fill: var(--text-secondary); font-size: 11.5px; }
.chart-val { fill: var(--text-primary); font-size: 11.5px; font-variant-numeric: tabular-nums; }
.chart-na { fill: var(--text-muted); font-size: 11px; font-style: italic; }
.chart-tick { fill: var(--text-muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }
.chart-axis-label { fill: var(--text-muted); font-size: 10.5px; }
.chart-ref-label { fill: var(--text-muted); font-size: 10.5px; }
.chart-point-label { fill: var(--text-primary); font-size: 11.5px; font-weight: 500; }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--baseline); stroke-width: 1; }
.chart .ref { stroke: var(--text-muted); stroke-width: 1.5; stroke-dasharray: 4 3; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .roof { fill: none; stroke: var(--text-secondary); stroke-width: 2; stroke-linejoin: round; }
.chart .dot { stroke: var(--surface-1); stroke-width: 2; }
.chart .mark { transition: opacity .12s ease; }
.chart .mark:hover, .chart .dot:hover { opacity: .82; }
"""


def tooltip_script() -> str:
    """A single hover tooltip shared by every chart on the page.

    An HTML chart is interactive by default, so the marks carry a hover layer rather than
    relying on the browser's slow native ``<title>`` delay. The ``<title>`` elements stay in
    the markup as the accessible fallback and for anyone reading the SVG on its own.
    """
    return """
(function () {
  var tip = document.createElement('div');
  tip.className = 'viz-tip';
  tip.setAttribute('role', 'status');
  document.body.appendChild(tip);
  var visible = false;
  function show(e, text) {
    tip.textContent = text;
    tip.style.opacity = '1';
    visible = true;
    move(e);
  }
  function move(e) {
    if (!visible) return;
    var pad = 14;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
    tip.style.transform = 'translate(' + x + 'px,' + y + 'px)';
  }
  function hide() { tip.style.opacity = '0'; visible = false; }
  document.addEventListener('mouseover', function (e) {
    var target = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (target) show(e, target.getAttribute('data-tip'));
  });
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseout', function (e) {
    if (e.target.closest && e.target.closest('[data-tip]')) hide();
  });
  document.addEventListener('scroll', hide, true);
})();
"""
