"""Render a run as a self-contained, theme-aware HTML report.

Same content as the Markdown report, plus the charts that only make sense visually: the
memory funnel per phase, the calibration sweep with its L2 knee, and the roofline. Everything
is inline -- no external stylesheet, script, font or image -- so the file can be copied,
emailed or published as an artifact and still render exactly the same.

Every chart is accompanied by the table it was drawn from. That is the accessibility
guarantee, and it also happens to be what anyone actually wants when they stop skimming and
start checking a number.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..analysis.assemble import RunAnalysis
from ..analysis.deep_dive import STALL_REASONS
from ..metrics import DEVICE_LEVELS, Level
from .charts import (
    BarDatum,
    RooflinePoint,
    chart_css,
    esc,
    fmt_bytes_short,
    hbar_panel,
    line_chart,
    roofline_chart,
    tooltip_script,
)
from .format_utils import (
    NOT_MEASURED,
    fmt_bandwidth,
    fmt_bytes,
    fmt_count,
    fmt_flops,
    fmt_pct,
    fmt_ratio,
    fmt_time_ns,
    truncate,
)

PAGE_CSS = """
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --surface-2: #f9f9f7;
  --surface-3: #f2f1ed;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --good: #0ca30c;
  --good-text: #006300;
  --warning: #fab219;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --surface-2: #0d0d0d;
    --surface-3: #232322;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --good: #0ca30c;
    --good-text: #0ca30c;
    --warning: #fab219;
    --critical: #d03b3b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --surface-2: #0d0d0d;
  --surface-3: #232322;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #898781;
  --grid: #2c2c2a;
  --baseline: #383835;
  --border: rgba(255,255,255,0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --good: #0ca30c;
  --good-text: #0ca30c;
  --warning: #fab219;
  --critical: #d03b3b;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--surface-2);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 72px; }

h1 { font-size: 27px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.015em; }
h2 {
  font-size: 19px; margin: 44px 0 14px; padding-top: 20px;
  border-top: 1px solid var(--border); letter-spacing: -0.01em;
}
h3 { font-size: 15px; margin: 26px 0 10px; color: var(--text-secondary); font-weight: 600; }
p { margin: 0 0 12px; color: var(--text-secondary); max-width: 74ch; }
p.lead { color: var(--text-primary); }
code {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: .87em; background: var(--surface-3); padding: 1px 5px;
  border-radius: 4px; word-break: break-word;
}
a { color: var(--series-1); }

.sub { color: var(--text-muted); font-size: 13px; margin-bottom: 24px; }

.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 20px; margin: 16px 0;
}

.banner {
  border-radius: 10px; padding: 14px 18px; margin: 18px 0;
  border: 1px solid var(--border); background: var(--surface-1);
  border-left: 4px solid var(--text-muted);
}
.banner.pass { border-left-color: var(--good); }
.banner.fail { border-left-color: var(--critical); }
.banner.warn { border-left-color: var(--warning); }
.banner .banner-title { font-weight: 650; color: var(--text-primary); margin-bottom: 4px; }
.banner p { margin: 0; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; margin: 20px 0; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
}
.tile .k { font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); }
.tile .v { font-size: 25px; font-weight: 650; margin: 4px 0 2px; letter-spacing: -0.02em; }
.tile .n { font-size: 12px; color: var(--text-secondary); line-height: 1.4; }

.table-scroll { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
td { font-variant-numeric: tabular-nums; color: var(--text-primary); }
td.na { color: var(--text-muted); font-style: italic; font-variant-numeric: normal; }
td.desc, th.desc { text-align: left; white-space: normal; color: var(--text-secondary); font-variant-numeric: normal; }
tbody tr:hover { background: var(--surface-3); }
tr.total td { font-weight: 650; }

.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; margin: 16px 0; }

ul { color: var(--text-secondary); max-width: 74ch; padding-left: 20px; }
li { margin-bottom: 7px; }

.viz-tip {
  position: fixed; top: 0; left: 0; pointer-events: none; opacity: 0;
  transition: opacity .1s ease; z-index: 999;
  background: var(--text-primary); color: var(--surface-1);
  padding: 6px 10px; border-radius: 6px; font-size: 12.5px;
  max-width: 340px; line-height: 1.4;
  box-shadow: 0 4px 14px rgba(0,0,0,.18);
}
footer { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--border);
         color: var(--text-muted); font-size: 12.5px; }
"""


def _cell(value: str) -> str:
    css_class = ' class="na"' if value == NOT_MEASURED else ""
    return f"<td{css_class}>{esc(value)}</td>"


def _table(headers: list[str], rows: list[list[str]], desc_cols: set[int] | None = None) -> str:
    if not rows:
        return '<p class="sub">No data.</p>'
    desc_cols = desc_cols or set()
    head = "".join(
        f'<th{" class=\"desc\"" if i in desc_cols else ""}>{esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            if index in desc_cols:
                cells.append(f'<td class="desc">{esc(value)}</td>')
            else:
                cells.append(_cell(str(value)))
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>"
    )


def document(title: str, body: str) -> str:
    """Prefix rendered markup with the declarations a standalone file needs.

    Two things have to be true at once, and the shape below is what satisfies both.

    **Served as a file, the document must declare its own encoding.** These reports are full
    of characters outside ASCII -- em dashes, the multiplication sign in "0.43x", the arrows
    in "L2 -> LPDDR5X". A static file server sends ``Content-Type: text/html`` with no
    ``charset``, so the browser has nothing to go on and falls back to a locale default;
    UTF-8 bytes then decode as windows-1252 and every one of those characters turns to
    mojibake. It only ever looked right where something upstream supplied the charset. The
    ``<meta>`` must also sit inside the first 1024 bytes, which is where the encoding prescan
    stops looking -- hence before the stylesheet, which is large.

    **Published as an artifact, the file must not bring its own page skeleton**, because one
    is added at publish time and nesting two is not something to rely on.

    ``<html>``, ``<head>`` and ``<body>`` are optional tags in HTML5 -- the parser infers all
    three -- so omitting them costs nothing standalone while keeping the markup wrappable. The
    bare ``<!doctype html>`` is what keeps a standalone file out of quirks mode; when this
    markup is wrapped instead, a DOCTYPE token in body position is defined to be ignored, so
    it is inert rather than harmful.
    """
    return (
        "<!doctype html>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n"
        f"<style>{PAGE_CSS}{chart_css()}</style>\n"
        f"{body}\n"
    )


def render_html(analysis: RunAnalysis) -> str:
    model = analysis.model
    title = f"{model.name} memory profile" if model else analysis.run_id

    body = [
        '<div class="wrap">',
        _header(analysis),
        _trust(analysis),
        _tiles(analysis),
        _hierarchy(analysis),
        _roofline(analysis),
        _calibration(analysis),
        _expectation(analysis),
        _footprint(analysis),
        _kernels(analysis),
        _limits(analysis),
        _timeline(analysis),
        _method(analysis),
        _footer(analysis),
        "</div>",
        f"<script>{tooltip_script()}</script>",
    ]
    return document(title, "\n".join(p for p in body if p))


# --------------------------------------------------------------------------------------


def _header(analysis: RunAnalysis) -> str:
    model = analysis.model
    workload = analysis.workload
    gpu = analysis.platform.gpu if analysis.platform else None

    name = model.name if model else analysis.run_id
    bits = []
    if model:
        bits.append(f"{model.architecture or model.model_type or 'model'}")
        bits.append(f"{model.quantization or 'unquantized'}, {model.bits_per_weight}-bit weights")
    if workload:
        bits.append(
            f"{workload.prompt_tokens} prompt -> {workload.generate_tokens} generated, "
            f"batch {workload.batch_size}"
        )
    if gpu:
        bits.append(f"{gpu.name} sm_{gpu.compute_capability.replace('.', '')}")

    rows = []
    if model:
        rows.append(["Checkpoint", model.path])
        rows.append([
            "Shape",
            f"{model.num_layers} layers, hidden {model.hidden_size}, "
            f"{model.describe_attention()}",
        ])
        weights = fmt_bytes(model.weight_bytes_on_disk)
        # Reconcile against the resident figure here, since both appear in this report and
        # they differ for a tied-embedding checkpoint that stores both copies on disk.
        resident = analysis.footprint.model_weight_bytes_resident
        if resident and model.weight_bytes_on_disk and (
            abs(model.weight_bytes_on_disk - resident) / model.weight_bytes_on_disk > 0.02
        ):
            weights += f" on disk, {fmt_bytes(resident)} resident"
            if model.tie_word_embeddings:
                weights += " (embeddings tied, so one copy is held)"
        rows.append(["Weights", weights])
        rows.append(["Attention kernel", model.attn_implementation])
    if gpu:
        rows.append([
            "GPU",
            f"{gpu.name} [{gpu.chip}], {gpu.sm_count} SMs, L2 {fmt_bytes(gpu.l2_cache_bytes)}, "
            f"{fmt_bytes(gpu.total_memory_bytes)} "
            + ("unified memory" if gpu.unified_memory else "device memory"),
        ])
    if analysis.platform:
        rows.append([
            "Toolchain",
            f"nsys {analysis.platform.tools.nsys_version} / "
            f"ncu {analysis.platform.tools.ncu_version} / "
            f"CUDA {analysis.platform.tools.cuda_version} / "
            f"torch {analysis.platform.torch_version}",
        ])
    rows.append(["Run id", analysis.run_id])

    return (
        f"<h1>{esc(name)}</h1>"
        f'<div class="sub">{" · ".join(esc(b) for b in bits)}</div>'
        + _table(["", ""], rows, desc_cols={0, 1})
    )


def _trust(analysis: RunAnalysis) -> str:
    parts = ["<h2>Can these numbers be trusted?</h2>"]
    calibration = analysis.calibration

    if calibration is None:
        parts.append(
            '<div class="banner warn"><div class="banner-title">No calibration was run</div>'
            "<p>Every LPDDR5X figure below is derived from the L2 sysmem aperture without "
            "having been checked against a known quantity on this machine. Run "
            "<code>nsbench calibrate</code> before relying on them.</p></div>"
        )
    elif calibration.passed:
        gate = calibration.byte_accounting
        parts.append(
            '<div class="banner pass"><div class="banner-title">Calibration gate passed</div>'
            f"<p>Immediately before this run, a streaming kernel of known size was profiled "
            f"and the derivation reproduced {esc(fmt_bytes(gate.expected_bytes))} of traffic "
            f"as {esc(fmt_bytes(gate.measured_bytes))} — an error of "
            f"{gate.relative_error:+.2%}. The device and peer L2 apertures both read zero, "
            f"as they must on a part with no private VRAM.</p></div>"
        )
    else:
        parts.append(
            '<div class="banner fail"><div class="banner-title">Calibration gate failed</div>'
            f"<p>{esc(calibration.byte_accounting.summary())}</p>"
            "<p>The LPDDR5X figures below are <strong>unverified</strong>. Treat them as "
            "indicative only.</p></div>"
        )

    parts.append(
        "<p>LPDDR5X traffic on this GPU is not read from a <code>dram__*</code> counter "
        "— GB10 has none. It is derived from "
        "<code>lts__t_sectors_aperture_sysmem_lookup_miss × 32 B</code>, the L2 misses "
        "that reach the unified memory pool. The calibration above is what makes that "
        "substitution auditable rather than assumed.</p>"
    )

    tripped = [s for s, p in analysis.phases.items() if not p.hierarchy.sentinels_ok]
    if tripped:
        parts.append(
            '<div class="banner fail"><div class="banner-title">Sentinel tripped</div>'
            f"<p>In {esc(', '.join(tripped))}: the device or peer L2 aperture reported "
            "non-zero traffic, which should be impossible here. The sysmem-only derivation is "
            "incomplete and the LPDDR5X figures understate reality.</p></div>"
        )

    if analysis.warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in analysis.warnings)
        parts.append(f"<h3>Warnings</h3><ul>{items}</ul>")
    return "".join(parts)


def _tiles(analysis: RunAnalysis) -> str:
    """Headline numbers. Each states what it is, not just a bare figure."""
    decode = analysis.phase("decode_step")
    timing = analysis.timing
    tiles = []

    if timing.available and timing.decode_step_ms:
        tiles.append((
            "Decode latency",
            f"{timing.decode_step_ms:,.1f} ms",
            f"per token, unprofiled · {timing.decode_tokens_per_second:,.1f} tok/s"
            if timing.decode_tokens_per_second else "per token, unprofiled",
        ))

    if decode:
        dram = decode.hierarchy.level(Level.DRAM).bytes_total
        if dram is not None:
            tiles.append((
                "Memory read per token",
                fmt_bytes_short(dram),
                "from LPDDR5X in one decode step",
            ))
        l2 = decode.hierarchy.level(Level.L2)
        if l2.hit_rate_pct is not None:
            tiles.append((
                "L2 hit rate (decode)",
                fmt_pct(l2.hit_rate_pct),
                "share of L2 lookups served on chip",
            ))
        achieved = decode.hierarchy.achieved_dram_bandwidth_gbps
        ceiling = analysis.dram_ceiling_gbps
        if achieved and ceiling:
            tiles.append((
                "Bandwidth used",
                fmt_pct(100 * achieved / ceiling, 0),
                f"{fmt_bandwidth(achieved)} of a measured {fmt_bandwidth(ceiling)} ceiling",
            ))

    if not tiles:
        return ""
    cards = "".join(
        f'<div class="tile"><div class="k">{esc(k)}</div>'
        f'<div class="v">{v}</div><div class="n">{n}</div></div>'
        for k, v, n in tiles
    )
    return f'<div class="tiles">{cards}</div>'


def _hierarchy(analysis: RunAnalysis) -> str:
    if not analysis.phases:
        return "<h2>Memory hierarchy</h2><p>No Nsight Compute data in this run.</p>"

    parts = ["<h2>Memory hierarchy</h2>"]
    parts.append(
        '<p class="lead">Prefill and decode run the same weights through the same kernels, '
        "and behave nothing alike. Prefill amortises every weight read across the whole "
        "prompt; decode reads the entire model to produce a single token. These panels are "
        "that difference, measured.</p>"
    )

    scopes = [s for s in ("prefill", "decode_step") if s in analysis.phases]

    # Small multiples: one panel per phase, each on its own linear scale so the funnel
    # within a phase is readable. The tables below carry the cross-phase comparison.
    panels = []
    for index, scope in enumerate(scopes):
        phase = analysis.phases[scope]
        data = []
        for level in DEVICE_LEVELS:
            if level is Level.REGISTER:
                continue
            summary = phase.hierarchy.level(level)
            data.append(BarDatum(
                label=level.label,
                value=summary.bytes_total,
                tooltip=(
                    f"{phase.label} \u00b7 {level.label}: "
                    f"{fmt_bytes(summary.bytes_total)}"
                    + (f", hit rate {fmt_pct(summary.hit_rate_pct)}"
                       if summary.hit_rate_pct is not None else "")
                ),
            ))
        panels.append(
            '<div>' + hbar_panel(
                phase.label, data, series_index=index,
                subtitle=f"{phase.hierarchy.kernel_count} kernels, "
                         f"{fmt_time_ns(phase.hierarchy.gpu_time_ns)} GPU time",
            ) + "</div>"
        )
    parts.append(f'<div class="panels">{"".join(panels)}</div>')

    headers = ["Level"]
    for scope in scopes:
        headers += [f"{analysis.phases[scope].label}", "hit rate"]
    rows = []
    for level in DEVICE_LEVELS:
        if level is Level.REGISTER:
            continue
        row = [level.label]
        for scope in scopes:
            summary = analysis.phases[scope].hierarchy.level(level)
            row.append(fmt_bytes(summary.bytes_total))
            row.append(fmt_pct(summary.hit_rate_pct)
                       if summary.hit_rate_pct is not None else "-")
        rows.append(row)
    parts.append(_table(headers, rows))
    parts.append(
        "<p>Shared-memory bytes are estimated from wavefronts × 128 B — there is "
        "no shared-memory sector counter — so bank conflicts inflate that row, which is "
        "the cost worth seeing.</p>"
    )

    # ---- amplification ----
    parts.append("<h3>Traffic amplification</h3>")
    amp_rows = []
    for scope in scopes:
        amp = analysis.phases[scope].hierarchy.amplification()
        amp_rows.append([
            analysis.phases[scope].label,
            fmt_ratio(amp["sm_to_l2"]), fmt_ratio(amp["l2_to_dram"]),
            fmt_ratio(amp["sm_to_dram"]),
        ])
    parts.append(_table(["Phase", "SM → L2", "L2 → LPDDR5X", "SM → LPDDR5X"],
                        amp_rows))
    parts.append(
        "<p><strong>L2 → LPDDR5X</strong> is the fraction of what L1 asked of L2 that L2 "
        "could not supply. Near 1.0 means L2 is providing essentially no reuse; near 0 means "
        "the working set is being captured on chip.</p>"
    )

    # ---- bandwidth ----
    parts.append("<h3>Bandwidth and arithmetic intensity</h3>")
    ceiling = analysis.dram_ceiling_gbps
    bw_rows = []
    for scope in scopes:
        hierarchy = analysis.phases[scope].hierarchy
        achieved = hierarchy.achieved_dram_bandwidth_gbps
        bw_rows.append([
            analysis.phases[scope].label,
            fmt_time_ns(hierarchy.gpu_time_ns),
            fmt_bandwidth(achieved),
            fmt_pct(100 * achieved / ceiling) if achieved and ceiling else NOT_MEASURED,
            fmt_flops(hierarchy.total_flops),
            fmt_ratio(hierarchy.arithmetic_intensity, suffix=" FLOP/B"),
        ])
    parts.append(_table(
        ["Phase", "GPU time", "LPDDR5X bandwidth", "% of ceiling", "FLOPs", "Intensity"],
        bw_rows,
    ))

    # ---- was the GPU busy? ----
    # Every rate in the table above divides by kernel time. Whether kernel time is most of
    # the phase decides whether any of them describes the phase at all.
    busy_rows = []
    for scope in scopes:
        occupancy = analysis.phases[scope].occupancy
        if not occupancy:
            continue
        busy_rows.append([
            analysis.phases[scope].label,
            fmt_time_ns(occupancy["wall_ns_per_instance"]),
            fmt_time_ns(occupancy["gpu_busy_ns_per_instance"]),
            f"{occupancy['busy_pct']:.0f}%",
        ])
    if busy_rows:
        parts.append("<h3>Was the GPU actually busy?</h3>")
        parts.append(_table(
            ["Phase", "Wall (unprofiled trace)", "GPU executing", "Busy"], busy_rows,
        ))
        parts.append(
            "<p>Busy time is the union of kernel intervals inside the phase’s NVTX "
            "range, from the Nsight Systems timeline. The remainder is the gap between "
            "kernels — launch latency, or the host blocking on a synchronisation. It is "
            "neither bandwidth nor compute, and no figure above accounts for it: they all "
            "divide by kernel time, so they describe the busy fraction only.</p>"
        )
        for scope in scopes:
            phase = analysis.phases[scope]
            if phase.occupancy and phase.occupancy["busy_pct"] < 90:
                parts.append(
                    f'<div class="banner warn"><div class="banner-title">'
                    f"{esc(phase.label)}: the GPU was idle for "
                    f"{100 - phase.occupancy['busy_pct']:.0f}% of this phase</div>"
                    f"<p>{esc(phase.occupancy_verdict() or '')}</p></div>"
                )

    # ---- bytes over real latency: an upper bound, both phases ----
    real_rows = []
    for scope in scopes:
        bound = analysis.bandwidth_at_real_latency_gbps(scope)
        if not bound:
            continue
        util = analysis.bandwidth_utilisation_pct(scope)
        real_rows.append([
            analysis.phases[scope].label,
            fmt_bandwidth(bound),
            (fmt_pct(util, 0) if util else NOT_MEASURED)
            + (" — IMPOSSIBLE" if util and util > 100 else ""),
        ])

    if real_rows:
        parts.append("<h3>Bytes over real latency — an upper bound</h3>")
        parts.append(
            "<p>The table above divides measured bytes by summed kernel time from the ncu "
            "pass, where kernels run serialised — so it understates the rate. Dividing "
            "the same bytes by unprofiled wall time removes that distortion from the "
            "denominator. It does <strong>not</strong> fix the numerator: those bytes were "
            "counted with L2 flushed before every replay pass, so they exclude the "
            "cross-kernel reuse a real run gets, and they read high. What follows is an "
            "upper bound, not an achieved rate.</p>"
        )
        parts.append(_table(
            ["Phase", "Bytes / real latency", "% of measured ceiling"], real_rows,
        ))
        if analysis.bandwidth_bound_exceeded():
            parts.append(
                '<div class="banner warn"><div class="banner-title">This bound exceeds a '
                "ceiling measured on this same machine</div><p>A rate above the ceiling is "
                "impossible, so the bound has just demonstrated its own looseness: the "
                "cold-cache byte counts overstate what a warm run moves. Read every figure "
                "in this table as an upper bound, including the ones below 100%.</p></div>"
            )

    util = analysis.decode_bandwidth_utilisation_pct()
    if util and util < 60:
        decode_phase = analysis.phase("decode_step")
        busy = decode_phase.occupancy["busy_pct"] if (
            decode_phase and decode_phase.occupancy
        ) else None
        parts.append(
            f'<div class="banner warn"><div class="banner-title">Decode is not '
            f'bandwidth-saturated</div><p>Even at that upper bound, decode reaches only '
            f"{esc(fmt_pct(util, 0))} of the achievable rate, so the true figure is lower "
            "still. "
            + (
                f"The GPU was idle for {100 - busy:.0f}% of the step, which is where the "
                "time actually goes: many small kernels and the gaps between them."
                if busy is not None and busy < 90 else
                "Its cost is latency and occupancy — many small kernels, each too "
                "short to fill the memory pipeline."
            )
            + " Reducing bytes moved would help less here than fusing or enlarging the "
            "kernels, or cutting launch count.</p></div>"
        )

    # ---- registers & spill ----
    reg_rows = []
    for scope in scopes:
        detail = analysis.phases[scope].hierarchy.level(Level.REGISTER).detail
        local = analysis.phases[scope].hierarchy.level(Level.LOCAL)
        reg_rows.append([
            analysis.phases[scope].label,
            fmt_count(detail.get("max_registers_per_thread")),
            fmt_count(detail.get("mean_registers_per_thread"), 1),
            fmt_count(detail.get("kernels_at_register_limit")),
            fmt_bytes(local.bytes_total),
        ])
    if reg_rows:
        parts.append("<h3>Registers and spilling</h3>")
        parts.append(_table(
            ["Phase", "Max reg/thread", "Mean reg/thread", "Kernels at 255", "Spill traffic"],
            reg_rows,
        ))
        parts.append(
            "<p>Spill traffic is the register file overflowing into local memory, which is "
            "physically the same LPDDR5X as everything else — bandwidth spent moving "
            "data that never needed to leave the SM.</p>"
        )

    # ---- per token ----
    decode = analysis.phase("decode_step")
    if decode:
        parts.append("<h3>Bytes per generated token</h3>")
        parts.append(
            "<p>Normalised per token so models of different sizes and quantization schemes "
            "can be put side by side.</p>"
        )
        per_token = decode.hierarchy.bytes_per_token(
            analysis.workload.batch_size if analysis.workload else 1
        )
        rows = [
            [level.label, fmt_bytes(per_token.get(level.value))]
            for level in (Level.LOCAL, Level.SHARED, Level.L1TEX, Level.L2, Level.DRAM)
        ]
        parts.append(_table(["Level", "Bytes / token"], rows))
    return "".join(parts)


def _roofline(analysis: RunAnalysis) -> str:
    calibration = analysis.calibration
    if not calibration or not calibration.peak_compute_gflops:
        return ""

    points = []
    for scope in ("prefill", "decode_step"):
        phase = analysis.phase(scope)
        if phase is None:
            continue
        intensity = phase.hierarchy.arithmetic_intensity
        flops = phase.hierarchy.total_flops
        seconds = phase.hierarchy.gpu_time_s
        if not (intensity and flops and seconds > 0):
            continue
        points.append(RooflinePoint(
            label=phase.label,
            intensity=intensity,
            achieved_gflops=flops / seconds / 1e9,
        ))
    if not points:
        return ""

    svg = roofline_chart(
        points,
        peak_bandwidth_gbps=calibration.peak_dram_bandwidth_gbps,
        peak_compute_gflops=calibration.peak_compute_gflops,
        title="Roofline",
        subtitle="Both ceilings measured on this machine, not taken from a datasheet",
    )
    if not svg:
        return ""

    ridge = calibration.ridge_point
    note = ""
    if ridge:
        note = (
            f"<p>The roof turns at <strong>{ridge:,.0f} FLOP/byte</strong>. Any kernel below "
            "that intensity is memory-bound no matter how well it is written. A decode step "
            "reads the whole model to produce one token, which puts it far to the left of the "
            "ridge — the structural reason decode cannot be made fast by better "
            "arithmetic.</p>"
        )
    return (
        "<h2>Roofline</h2>"
        + f'<div class="card">{svg}</div>'
        + note
    )


def _calibration(analysis: RunAnalysis) -> str:
    calibration = analysis.calibration
    if not calibration or not calibration.sweep:
        return ""

    l2_mib = calibration.l2_cache_bytes / (1024 * 1024) if calibration.l2_cache_bytes else 0
    points = [(p.working_set_mib, p.bandwidth_gbps) for p in calibration.sweep]
    svg = line_chart(
        "Streaming bandwidth versus working-set size",
        points,
        log_x=True,
        reference_x=l2_mib or None,
        reference_label=f"L2 = {l2_mib:.0f} MiB" if l2_mib else "",
        x_formatter=lambda v: f"{v:,.0f}",
        y_formatter=lambda v: f"{v:,.0f}",
        point_tooltip=lambda x, y: f"{x:,.0f} MiB working set: {y:,.1f} GB/s",
        subtitle="Measured on this machine; the knee is where L2 stops holding the data",
        x_label="Working set (MiB, both arrays)",
        y_label="GB/s",
    )

    knee = calibration.knee_mib()
    rows = [[f"{p.working_set_mib:,.0f} MiB", f"{p.bandwidth_gbps:,.1f} GB/s"]
            for p in calibration.sweep]

    note_parts = []
    if knee and l2_mib:
        note_parts.append(
            f"Bandwidth falls off at a <strong>{knee:,.0f} MiB</strong> working set against "
            f"<strong>{l2_mib:.0f} MiB</strong> of L2."
        )
    if calibration.peak_l2_bandwidth_gbps and calibration.peak_dram_bandwidth_gbps:
        ratio = calibration.peak_l2_bandwidth_gbps / calibration.peak_dram_bandwidth_gbps
        note_parts.append(
            f"L2-resident data streams at "
            f"{fmt_bandwidth(calibration.peak_l2_bandwidth_gbps)} against "
            f"{fmt_bandwidth(calibration.peak_dram_bandwidth_gbps)} from LPDDR5X — "
            f"<strong>{ratio:.1f}×</strong>. That multiple is what an L2 hit is worth "
            "on this part, and why the hit-rate columns above carry so much weight."
        )

    return (
        "<h2>Calibration: the memory system itself</h2>"
        + f'<div class="card">{svg}</div>'
        + (f"<p>{' '.join(note_parts)}</p>" if note_parts else "")
        + _table(["Working set", "Bandwidth"], rows)
    )


def _expectation(analysis: RunAnalysis) -> str:
    decode = analysis.phase("decode_step")
    if decode is None or decode.expectation is None:
        return ""
    expectation = decode.expectation

    ratio = expectation.ratio
    tone = "pass" if (ratio and 0.75 <= ratio <= 1.6) else "warn"

    weights_label = "Weights"
    if expectation.is_moe:
        weights_label = (f"Weights read (routed-active, of "
                         f"{fmt_bytes(expectation.total_weight_bytes)} resident)")
    rows = [
        [weights_label, fmt_bytes(expectation.weight_bytes)
         + f" -- {expectation.weight_bytes_source}"],
        ["KV cache" + (" + recurrent state (read)" if expectation.state_write_bytes else ""),
         fmt_bytes(expectation.kv_bytes)
         + (f" (context {expectation.context_len})" if expectation.context_len else "")],
    ]
    if expectation.state_write_bytes:
        rows.append(["Recurrent state written back", fmt_bytes(expectation.state_write_bytes)])
    rows += [
        ["Expected total", fmt_bytes(expectation.expected_bytes)],
        ["Measured", fmt_bytes(expectation.measured_bytes)],
        ["Ratio", fmt_ratio(ratio)],
    ]
    return (
        "<h2>Does the measurement match the physics?</h2>"
        "<p>A decode step must read every weight and the whole KV cache to produce one token, "
        "so its memory traffic is predictable from first principles. Comparing prediction to "
        "measurement is the strongest available check that both the NVTX scoping and the byte "
        "derivation are correct.</p>"
        + _table(["", "Bytes"], rows)
        + f'<div class="banner {tone}"><div class="banner-title">Verdict</div>'
        f"<p>{esc(expectation.verdict())}</p></div>"
    )


def _footprint(analysis: RunAnalysis) -> str:
    footprint = analysis.footprint
    rows = [
        ["torch allocator (peak allocated)", fmt_bytes(footprint.torch_peak_allocated),
         "tensors torch allocated, run high-water mark; blind to library workspaces"],
        ["torch allocator (peak reserved)", fmt_bytes(footprint.torch_peak_reserved),
         "including cached blocks held for reuse"],
        ["CUDA driver (pool in use)", fmt_bytes(footprint.cuda_peak_used),
         footprint.cuda_used_label],
        ["nsys allocation timeline (peak)", fmt_bytes(footprint.nsys_peak_outstanding),
         "allocations made inside the traced region only"],
        ["Model weights resident", fmt_bytes(footprint.model_weight_bytes_resident),
         "parameters plus buffers, as loaded"],
        ["KV cache" + (" + recurrent state" if footprint.cache_state_bytes.get("recurrent_state")
                       else ""),
         fmt_bytes(footprint.kv_cache_bytes),
         "measured from the live cache tensors"
         + (" -- " + ", ".join(f"{k} {fmt_bytes(v)}" for k, v in footprint.cache_state_bytes.items())
            if len(footprint.cache_state_bytes) > 1 else "")],
    ]
    if footprint.mla_latent_kv_bytes:
        rows.append([
            "MLA latent equivalent", fmt_bytes(footprint.mla_latent_kv_bytes),
            "what an MLA-native engine would cache; transformers holds the expanded "
            f"per-head K/V instead ({fmt_bytes(footprint.mla_expanded_kv_bytes)})",
        ])
    if footprint.host_available_delta:
        rows.append([
            "Host pool drawn down by load",
            fmt_bytes(footprint.host_available_delta),
            "MemAvailable lost while loading weights — this process's own draw",
        ])
    parts = [
        "<h2>Memory footprint</h2>",
        "<p>NVML reports GPU memory as <code>N/A</code> on this part — there is no "
        "discrete VRAM to report — so footprint is reconstructed from three independent "
        "sources with different blind spots.</p>",
        _table(["Source", "Bytes", "What it sees"], rows, desc_cols={2}),
    ]
    agreement = footprint.agreement()
    if agreement:
        if footprint.unified_memory:
            parts.append(
                '<div class="banner warn"><div class="banner-title">The driver figure is '
                f"not this run's footprint</div><p>{esc(agreement)}</p></div>"
            )
        else:
            parts.append(f"<p>Driver versus allocator: {esc(agreement)}</p>")
    for note in footprint.notes:
        parts.append(f"<p>{esc(note)}</p>")
    return "".join(parts)


def _kernels(analysis: RunAnalysis) -> str:
    parts = ["<h2>Kernels by memory traffic</h2>",
             "<p>Ranked by bytes reaching LPDDR5X, not by duration — this harness is "
             "about memory, and the kernel that moves the most data is not always the one "
             "that takes longest.</p>"]
    any_data = False

    for index, scope in enumerate(("prefill", "decode_step")):
        phase = analysis.phase(scope)
        if phase is None or not phase.kernel_rows:
            continue
        any_data = True
        parts.append(f"<h3>{esc(phase.label)}</h3>")

        top = [r for r in phase.kernel_rows[:10] if r.get("dram_bytes")]
        if top:
            data = [
                BarDatum(
                    label=truncate(r["kernel_short"], 26),
                    value=r["dram_bytes"],
                    tooltip=f"{r['kernel_short']}: {fmt_bytes(r['dram_bytes'])} from LPDDR5X, "
                            f"{fmt_time_ns(r['duration_ns'])}, "
                            f"L2 hit {fmt_pct(r['l2_hit_rate_pct'])}",
                )
                for r in top
            ]
            parts.append(
                '<div class="card">'
                + hbar_panel(
                    f"Top kernels by LPDDR5X traffic — {phase.label}",
                    data, series_index=index, width=880, row_height=30,
                )
                + "</div>"
            )

        rows = []
        for row in phase.kernel_rows[:12]:
            rows.append([
                truncate(row["kernel_short"], 40),
                fmt_time_ns(row["duration_ns"]),
                fmt_bytes(row["dram_bytes"]),
                fmt_bytes(row["l2_bytes"]),
                fmt_pct(row["l2_hit_rate_pct"]),
                fmt_bandwidth(row["dram_bandwidth_gbps"]),
                fmt_count(row["registers_per_thread"]),
            ])
        parts.append(_table(
            ["Kernel", "Time", "LPDDR5X", "L2", "L2 hit", "Bandwidth", "Reg/thr"],
            rows, desc_cols={0},
        ))
        cross_check = phase.cross_check_kernel_counts()
        if cross_check:
            parts.append(f'<p class="sub">{esc(cross_check)}.</p>')

    if not any_data:
        parts.append("<p>No per-kernel data.</p>")
    return "".join(parts)


def _limits(analysis: RunAnalysis) -> str:
    """Tier-2 deep dive: what actually limits each phase."""
    phases = [
        (scope, analysis.phases[scope])
        for scope in ("prefill", "decode_step")
        if scope in analysis.phases
        and analysis.phases[scope].limits
        and analysis.phases[scope].limits.available
    ]
    if not phases:
        return ""

    parts = [
        "<h2>What limits each phase</h2>",
        "<p>Bytes moved say how much work the memory system did; they cannot say why a "
        "kernel took the time it did. These come from the tier-2 sections, which record why "
        "a warp that could have issued did not. <strong>Memory (long scoreboard)</strong> is "
        "a warp waiting on a global load — its share is what separates a genuinely "
        "memory-bound phase from one that simply lacks the parallelism to hide any latency "
        "at all.</p>",
    ]

    for index, (scope, phase) in enumerate(phases):
        limits = phase.limits
        parts.append(f"<h3>{esc(phase.label)}</h3>")

        verdict = limits.verdict()
        if verdict:
            memory_led = "Memory (long scoreboard)" in verdict
            parts.append(
                f'<div class="banner {"warn" if memory_led else ""}">'
                f'<div class="banner-title">Limiter</div><p>'
                + esc(verdict).replace("**", "")
                + "</p></div>"
            )

        profile = limits.weighted_stall_profile()
        if profile:
            data = [
                BarDatum(
                    label=STALL_REASONS.get(reason, (reason, ""))[0],
                    value=pct,
                    tooltip=f"{STALL_REASONS.get(reason, (reason, ''))[0]}: {pct:.1f}% of "
                            f"stall cycles"
                            + (f" -- {STALL_REASONS[reason][1]}"
                               if reason in STALL_REASONS and STALL_REASONS[reason][1] else ""),
                )
                for reason, pct in list(profile.items())[:7]
            ]
            parts.append(
                '<div class="card">'
                + hbar_panel(
                    f"Why warps stalled — {phase.label}", data,
                    series_index=index, width=880, row_height=30,
                    value_formatter=lambda v: f"{v:.1f}%",
                    subtitle="Weighted by each kernel's GPU time",
                )
                + "</div>"
            )
            parts.append(_table(
                ["Stall reason", "Share", "What it means"],
                [[STALL_REASONS.get(r, (r, ""))[0], fmt_pct(pct, 1),
                  STALL_REASONS.get(r, (r, ""))[1]]
                 for r, pct in list(profile.items())[:7]],
                desc_cols={0, 2},
            ))

        rows = []
        for kernel in limits.top(8):
            dominant = kernel.dominant_stall()
            label = STALL_REASONS.get(dominant[0], (dominant[0], ""))[0] if dominant else "-"
            rows.append([
                truncate(kernel.short_name, 34), fmt_count(kernel.launches),
                fmt_time_ns(kernel.duration_ns), fmt_pct(kernel.achieved_occupancy_pct),
                fmt_ratio(kernel.waves_per_sm, suffix=""),
                fmt_pct(kernel.memory_stall_pct, 0), label,
            ])
        parts.append(_table(
            ["Kernel", "Launches", "Time", "Occupancy", "Waves/SM", "Memory stalls",
             "Dominant stall"],
            rows, desc_cols={0, 6},
        ))
        parts.append(
            "<p>Waves per SM below 1.0 means the kernel cannot even fill the machine once "
            "— no amount of memory tuning fixes a kernel with nothing to overlap.</p>"
        )

    return "".join(parts)


def _timeline(analysis: RunAnalysis) -> str:
    nsys = analysis.nsys
    if nsys is None:
        return ""
    parts = ["<h2>Timeline</h2>"]

    durations = nsys.phase_durations_ns()
    counts = nsys.phase_instance_counts()
    interesting = {p: t for p, t in durations.items() if p.startswith("nsbench.")}
    if interesting:
        rows = [
            [phase.replace("nsbench.", ""), fmt_time_ns(total),
             fmt_count(counts.get(phase, 0)), fmt_count(len(nsys.kernels_in_phase(phase)))]
            for phase, total in sorted(interesting.items(), key=lambda kv: -kv[1])
        ]
        parts.append(_table(["Phase", "Wall time", "Instances", "Kernels"], rows))

    timeline = nsys.allocation_timeline()
    if len(timeline) > 3:
        timed = nsys.allocation_timestamps_available
        if timed:
            t0 = timeline[0]["timestamp_ns"]
            points = [
                ((r["timestamp_ns"] - t0) / 1e6, r["outstanding_bytes"] / 1e6)
                for r in timeline
            ]
            x_label, unit = "Time since trace start (ms)", "ms"
            subtitle = ("Reconstructed from CUDA allocation events; NVML reports nothing here")
        else:
            # This nsys build records every allocation event with start = 0. The order and
            # the sizes are sound, the timestamps are not -- so the x-axis is the event
            # sequence, labelled as such, rather than a time axis collapsed onto zero.
            points = [
                (float(r["event_index"]), r["outstanding_bytes"] / 1e6) for r in timeline
            ]
            x_label, unit = "Allocation event #", ""
            subtitle = ("Allocation events carry no timestamps on this nsys build, so the "
                        "x-axis is event order, not time")
        parts.append('<div class="card">' + line_chart(
            "GPU memory outstanding across the traced region",
            points, series_index=2,
            x_formatter=lambda v: f"{v:,.0f}",
            y_formatter=lambda v: f"{v:,.0f}",
            point_tooltip=lambda x, y: (
                f"{x:,.1f} {unit}: {y:,.1f} MB outstanding" if unit
                else f"event {x:,.0f}: {y:,.1f} MB outstanding"
            ),
            x_label=x_label, y_label="MB",
            subtitle=subtitle,
        ) + "</div>")
        if not timed:
            parts.append(
                "<p>Allocation events on this nsys build are recorded with a zero timestamp, "
                "so this shows the growth of the live allocation as events occur rather than "
                "against a clock. Sizes and ordering are unaffected.</p>"
            )

    clocks = nsys.clock_summary()
    if clocks:
        stable = clocks["gpc_clock_spread_pct"] < 15
        parts.append(
            f"<p><strong>GPC clock</strong> during the traced region: "
            f"{clocks['gpc_clock_mhz_min']:,.0f}–{clocks['gpc_clock_mhz_max']:,.0f} MHz "
            f"(mean {clocks['gpc_clock_mhz_mean']:,.0f}, spread "
            f"{clocks['gpc_clock_spread_pct']:.1f}%). "
            + ("Stable enough to compare against other runs.</p>" if stable else
               "This much variation softens any timing comparison against other runs.</p>")
        )

    if nsys.memcpy_bytes:
        rows = [[k, fmt_bytes(v)] for k, v in sorted(nsys.memcpy_bytes.items())]
        parts.append("<h3>Explicit host/device copies</h3>")
        parts.append(_table(["Direction", "Bytes"], rows))
        parts.append(
            "<p>On a unified-memory part these should be near zero in steady state. Copies "
            "during decode mean data is being staged that did not need to move.</p>"
        )

    if nsys.um_page_faults:
        rows = [[k.replace("_", " "), fmt_count(v)] for k, v in nsys.um_page_faults.items()]
        parts.append("<h3>Unified memory page faults</h3>")
        parts.append(_table(["Kind", "Count"], rows))
    return "".join(parts)


def _method(analysis: RunAnalysis) -> str:
    items = [
        "<strong>LPDDR5X traffic is derived, not counted.</strong> GB10 exposes no "
        "<code>dram__*</code> metrics. Bytes past L2 come from "
        "<code>lts__t_sectors_aperture_sysmem_lookup_miss × 32 B</code>, with the "
        "device and peer aperture counters carried as sentinels that must read zero.",
        "<strong>Profiled durations are not performance.</strong> Nsight Compute replays each "
        "kernel many times. Only the decode-latency tile and the timing table report real "
        "wall-clock, measured with no profiler attached.",
        "<strong>One decode step is profiled, not all of them.</strong> A decode step's kernel "
        "mix does not change between tokens, so one step gives full kernel coverage at a "
        "fraction of the cost. The step chosen is the last, where the KV cache is deepest.",
        "<strong>Hit rates are recomputed from raw counts</strong>, never averaged across "
        "kernels: <code>sum(hits) / sum(hits + misses)</code>. Averaging percentages would "
        "weight a tiny elementwise kernel equally with a large GEMM.",
        "<strong>Shared-memory bytes are an estimate</strong> from wavefront counts × "
        "128 B. Bank conflicts inflate the figure, which is intentional — that is the "
        "cost being shown.",
        "<strong>L2 hit rates are measured with a cold cache.</strong> ncu runs with "
        "<code>--cache-control all</code>, flushing L2 before each replay pass so every "
        "kernel is measured independently of what ran before it. In an un-profiled decode "
        "loop L2 may retain data across steps, so the reported hit rate is a "
        "<em>lower bound</em> on what the un-replayed workload sees. Set "
        "<code>ncu.cache_control: none</code> to measure warm behaviour instead.",
    ]
    parts = ["<h2>Method and caveats</h2><ul>"
             + "".join(f"<li>{i}</li>" for i in items) + "</ul>"]

    if analysis.platform and analysis.platform.notes:
        parts.append("<h3>Platform notes</h3><ul>"
                     + "".join(f"<li>{esc(n)}</li>" for n in analysis.platform.notes)
                     + "</ul>")
    if analysis.calibration and analysis.calibration.notes:
        parts.append("<h3>Calibration notes</h3><ul>"
                     + "".join(f"<li>{esc(n)}</li>" for n in analysis.calibration.notes)
                     + "</ul>")
    return "".join(parts)


def _footer(analysis: RunAnalysis) -> str:
    return (
        "<footer>"
        f"<div>Raw profiler reports: <code>{esc(analysis.root / 'raw')}</code></div>"
        f"<div>Tidy metric tables: <code>{esc(analysis.root / 'metrics')}</code></div>"
        f"<div>Full provenance: <code>{esc(analysis.root / 'manifest.json')}</code></div>"
        f"<div style=\"margin-top:8px\">Generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by nsight_bench.</div>"
        "</footer>"
    )


def write_html_report(analysis: RunAnalysis, path: str | Path | None = None) -> Path:
    out_path = Path(path) if path else analysis.root / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(analysis), encoding="utf-8")
    return out_path
