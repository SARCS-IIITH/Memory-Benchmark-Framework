"""Render a cross-run comparison as Markdown, HTML and CSV.

The matrix leads with per-token memory traffic rather than tokens per second. Throughput is
the number people ask for, but it conflates the model, the kernels and the machine; bytes
read per generated token isolates the thing this harness measures and is the quantity a
quantization change is actually supposed to move.

The most useful column is the last derived one: memory read per token expressed as a
multiple of the model's own weight bytes. It is dimensionless, so a 0.6B bf16 checkpoint and
a 30B 4-bit one are held to the same standard -- theory says a decode step reads the model
once, and the column says how close reality came.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..analysis.compare import Comparison, ComparisonRow
from .charts import BarDatum, chart_css, esc, hbar_panel, tooltip_script
from .format_utils import (
    NOT_MEASURED,
    fmt_bandwidth,
    fmt_bytes,
    fmt_pct,
    fmt_ratio,
    markdown_table,
)
from .html import PAGE_CSS, _table, document


def _label(row: ComparisonRow) -> str:
    """Row label carrying the axes that usually vary between runs."""
    bits = [row.model]
    if row.quantization and row.quantization != "none":
        bits.append(row.quantization)
    elif row.dtype:
        bits.append(row.dtype)
    if row.attn:
        bits.append(row.attn)
    return " / ".join(b for b in bits if b)


def _flag(row: ComparisonRow) -> str:
    if not row.calibration_passed:
        return " [uncalibrated]"
    if row.truncated:
        return " [partial]"
    return ""


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def render_markdown(comparison: Comparison) -> str:
    lines = [
        "# Cross-run comparison",
        "",
        f"_{len(comparison.runs)} runs, generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._",
        "",
    ]

    if comparison.dram_ceiling_gbps:
        lines.append(
            f"Machine ceilings used throughout: **{fmt_bandwidth(comparison.dram_ceiling_gbps)}** "
            "from LPDDR5X"
            + (f", **{fmt_bandwidth(comparison.l2_bandwidth_gbps)}** from L2"
               if comparison.l2_bandwidth_gbps else "")
            + ". Both measured on this machine, not taken from a datasheet."
        )
        lines.append("")

    if comparison.warnings:
        lines.append("## Comparability")
        lines.append("")
        for warning in comparison.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # ---- memory per token ----
    lines.append("## Memory traffic per generated token")
    lines.append("")
    lines.append(
        "Normalised per token, so runs with different generation lengths and batch sizes sit "
        "in the same table."
    )
    lines.append("")
    rows = []
    for row in comparison.runs:
        rows.append([
            _label(row) + _flag(row),
            "MoE" if row.is_moe else "dense",
            fmt_bytes(row.dram_bytes_per_token),
            fmt_bytes(row.l2_bytes_per_token),
            fmt_bytes(row.weight_bytes),
            fmt_bytes(row.active_weight_bytes) if row.is_moe else "same",
            fmt_ratio(row.bytes_per_token_vs_weights),
            fmt_ratio(row.bytes_per_token_vs_active),
        ])
    lines.append(markdown_table(
        ["Run", "Kind", "LPDDR5X / token", "L2 / token", "Stored", "Active",
         "x stored", "x active"], rows
    ))
    lines.append("")
    lines.append(
        "**x active** is the column to read. Memory moved per token divided by the weights a "
        "token actually routes through: for a dense model that is the whole checkpoint, for a "
        "mixture of experts it is attention plus the routed experts. Theory says a decode step "
        "reads that set once, so ~1.0 is the expected value across every row regardless of "
        "architecture."
    )
    lines.append("")
    lines.append(
        "**x stored** is the same traffic against the full checkpoint, and it is what makes "
        "the sparse models look extraordinary -- a top-8-of-128 model reads roughly a tenth of "
        "what it stores. That is not cache reuse. It never touched the other 120 experts. The "
        "two columns together are the dense-versus-sparse story: sparsity buys bandwidth, and "
        "pays for it in capacity."
    )
    lines.append("")

    # ---- hierarchy behaviour ----
    lines.append("## Hierarchy behaviour (decode)")
    lines.append("")
    rows = []
    for row in comparison.runs:
        rows.append([
            _label(row) + _flag(row),
            fmt_pct(row.decode_l1_hit_pct),
            fmt_pct(row.decode_l2_hit_pct),
            fmt_ratio(row.decode_l2_to_dram),
            fmt_bandwidth(row.decode_bandwidth_real_gbps),
            fmt_pct(row.decode_bandwidth_utilisation_pct, 0),
            fmt_pct(row.decode_gpu_busy_pct, 0),
            fmt_ratio(row.decode_arithmetic_intensity, suffix=" F/B"),
        ])
    lines.append(markdown_table(
        ["Run", "L1 hit", "L2 hit", "L2->DRAM", "Bandwidth", "% ceiling", "GPU busy",
         "Intensity"], rows
    ))
    lines.append("")
    lines.append(
        "**GPU busy** is the share of the decode step's wall time with a kernel actually "
        "executing, and it should be read before the bandwidth columns. Every rate here "
        "divides by kernel time, so on a row that is two-thirds busy they describe the "
        "kernels rather than the step. A low figure means the model is bounded by launch "
        "dispatch -- depth and MoE routing both push it down -- and moving fewer bytes will "
        "not help it."
    )
    lines.append("")
    lines.append(
        "_Bandwidth divides the measured bytes by the **unprofiled** per-token latency, "
        "because dividing by ncu's serialised kernel time would understate it. That fixes "
        "the denominator only: the bytes were counted with L2 flushed before every replay "
        "pass, so they exclude the reuse a warm run gets and read high. **Every bandwidth "
        "and utilisation figure in this table is an upper bound**, and comparisons between "
        "runs are sound only because the same bias applies to all of them._"
    )
    lines.append("")

    # ---- performance ----
    lines.append("## Performance (unprofiled)")
    lines.append("")
    rows = []
    for row in comparison.runs:
        rows.append([
            _label(row),
            f"{row.decode_ms_per_token:,.1f} ms" if row.decode_ms_per_token else NOT_MEASURED,
            f"{row.decode_tokens_per_second:,.1f}" if row.decode_tokens_per_second
            else NOT_MEASURED,
            f"{row.prefill_tokens_per_second:,.0f}" if row.prefill_tokens_per_second
            else NOT_MEASURED,
            fmt_bytes(row.peak_footprint_bytes),
            fmt_bytes(row.kv_cache_bytes),
        ])
    lines.append(markdown_table(
        ["Run", "ms / token", "Decode tok/s", "Prefill tok/s", "Peak footprint", "KV cache"],
        rows,
    ))
    lines.append("")
    lines.append(
        "_Measured with no profiler attached; the memory tables above come from profiled "
        "runs and their durations are not comparable to these._"
    )
    lines.append("")

    # ---- capacity vs throughput ----
    # The question an edge box actually poses. On a 128 GB unified part, capacity is the
    # binding constraint, and a sparse model spends a great deal of it to buy bandwidth.
    # Whether that trade is worth making is not visible in any single column above.
    if any(r.is_moe for r in comparison.runs):
        lines.append("## Capacity versus throughput")
        lines.append("")
        lines.append(
            "A mixture of experts occupies memory like a large model and moves bytes like a "
            "small one. On a shared 128 GB pool that trade has a price, and this is where it "
            "shows: **tok/s per GB resident** is throughput earned per gigabyte spent."
        )
        lines.append("")
        rows = []
        for row in comparison.runs:
            rows.append([
                _label(row),
                "MoE" if row.is_moe else "dense",
                f"{row.num_experts}x top-{row.experts_per_token}" if row.is_moe else "-",
                fmt_bytes(row.weight_bytes),
                fmt_pct(100 * row.sparsity_ratio, 0) if row.sparsity_ratio else "100%",
                f"{row.decode_tokens_per_second:,.1f}" if row.decode_tokens_per_second
                else NOT_MEASURED,
                f"{row.memory_efficiency:,.2f}" if row.memory_efficiency else NOT_MEASURED,
            ])
        lines.append(markdown_table(
            ["Run", "Kind", "Routing", "Resident", "Active %", "tok/s", "tok/s per GB"],
            rows, align=["left", "left", "left", "right", "right", "right", "right"],
        ))
        lines.append("")
        lines.append(
            "Read across a row: a sparse model should show a low **Active %** and a high "
            "**tok/s**, because it reads little to produce each token. If its **tok/s per GB** "
            "still lands below a dense model of the same speed, the sparsity is not paying for "
            "the capacity it consumes on this machine."
        )
        lines.append("")

    # ---- setup ----
    lines.append("## Run configurations")
    lines.append("")
    rows = []
    for row in comparison.runs:
        rows.append([
            _label(row),
            f"{row.parameters / 1e9:.2f} B" if row.parameters else "-",
            f"{row.bits_per_weight}-bit",
            f"{row.prompt_tokens} / {row.generate_tokens} / {row.batch_size}",
            "yes" if row.calibration_passed else "NO",
            "; ".join(row.caveats) or "-",
        ])
    lines.append(markdown_table(
        ["Run", "Params", "Weights", "prompt/gen/batch", "Calibrated", "Caveats"], rows,
        align=["left", "right", "right", "right", "right", "left"],
    ))
    lines.append("")
    lines.append(f"Run ids: " + ", ".join(f"`{r.run_id}`" for r in comparison.runs))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------


def render_html(comparison: Comparison) -> str:
    parts = [
        '<div class="wrap">',
        "<h1>Model memory comparison</h1>",
        f'<div class="sub">{len(comparison.runs)} runs · DGX Spark GB10 · '
        f'{esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}</div>',
    ]

    if comparison.dram_ceiling_gbps:
        parts.append(
            f'<div class="banner"><div class="banner-title">Machine ceilings</div>'
            f"<p>{esc(fmt_bandwidth(comparison.dram_ceiling_gbps))} from LPDDR5X"
            + (f", {esc(fmt_bandwidth(comparison.l2_bandwidth_gbps))} from L2"
               if comparison.l2_bandwidth_gbps else "")
            + ". Both measured on this machine, not taken from a datasheet, so a "
            "utilisation percentage against them is meaningful.</p></div>"
        )

    if comparison.warnings:
        parts.append(
            '<div class="banner warn"><div class="banner-title">Comparability</div><ul>'
            + "".join(f"<li>{esc(w)}</li>" for w in comparison.warnings)
            + "</ul></div>"
        )

    # ---- chart: bytes per token ----
    chart_rows = [r for r in comparison.runs if r.dram_bytes_per_token]
    if chart_rows:
        data = [
            BarDatum(
                label=_label(r),
                value=r.dram_bytes_per_token,
                tooltip=(
                    f"{_label(r)}: {fmt_bytes(r.dram_bytes_per_token)} read per token"
                    + (f", {r.bytes_per_token_vs_weights:.2f}x the model's weight bytes"
                       if r.bytes_per_token_vs_weights else "")
                ),
            )
            for r in chart_rows
        ]
        parts.append("<h2>Memory read per generated token</h2>")
        parts.append(
            "<p>The quantity this harness exists to measure, normalised so runs of different "
            "lengths and batch sizes are directly comparable.</p>"
        )
        parts.append(
            '<div class="card">'
            + hbar_panel("Bytes from LPDDR5X per token", data, width=880, row_height=32)
            + "</div>"
        )

    parts.append(_table(
        ["Run", "Kind", "LPDDR5X / token", "L2 / token", "Stored", "Active",
         "× stored", "× active"],
        [[_label(r) + _flag(r), "MoE" if r.is_moe else "dense",
          fmt_bytes(r.dram_bytes_per_token),
          fmt_bytes(r.l2_bytes_per_token),
          fmt_bytes(r.weight_bytes),
          fmt_bytes(r.active_weight_bytes) if r.is_moe else "same",
          fmt_ratio(r.bytes_per_token_vs_weights),
          fmt_ratio(r.bytes_per_token_vs_active)]
         for r in comparison.runs],
        desc_cols={0},
    ))
    parts.append(
        "<p><strong>× active</strong> is the column to read. Memory moved per token divided "
        "by the weights a token actually routes through: the whole checkpoint for a dense "
        "model, attention plus the routed experts for a mixture of experts. A decode step "
        "reads that set once, so ~1.0 is the expected value on every row regardless of "
        "architecture.</p>"
        "<p><strong>× stored</strong> is the same traffic against the full checkpoint, and it "
        "is what makes the sparse models look extraordinary — a top-8-of-128 model reads "
        "roughly a tenth of what it stores. That is not cache reuse; it never touched the "
        "other 120 experts. Together the two columns are the dense-versus-sparse trade: "
        "sparsity buys bandwidth and pays for it in capacity.</p>"
    )

    parts.append("<h2>Hierarchy behaviour (decode)</h2>")
    parts.append(_table(
        ["Run", "L1 hit", "L2 hit", "L2→DRAM", "Bandwidth", "% ceiling", "GPU busy",
         "Intensity"],
        [[_label(r) + _flag(r), fmt_pct(r.decode_l1_hit_pct), fmt_pct(r.decode_l2_hit_pct),
          fmt_ratio(r.decode_l2_to_dram), fmt_bandwidth(r.decode_bandwidth_real_gbps),
          fmt_pct(r.decode_bandwidth_utilisation_pct, 0),
          fmt_pct(r.decode_gpu_busy_pct, 0),
          fmt_ratio(r.decode_arithmetic_intensity, suffix=" F/B")]
         for r in comparison.runs],
        desc_cols={0},
    ))
    parts.append(
        "<p><strong>GPU busy</strong> is the share of the decode step&rsquo;s wall time with a "
        "kernel actually executing, and it should be read before the bandwidth columns. Every "
        "rate here divides by kernel time, so on a row that is two-thirds busy they describe "
        "the kernels rather than the step. A low figure means the model is bounded by launch "
        "dispatch — depth and MoE routing both push it down — and moving fewer bytes will not "
        "help it.</p>"
    )
    parts.append(
        "<p>Bandwidth divides the measured bytes by the <strong>unprofiled</strong> per-token "
        "latency, because dividing by ncu’s serialised kernel time would understate it. "
        "That fixes the denominator only: the bytes were counted with L2 flushed before every "
        "replay pass, so they exclude the reuse a warm run gets and read high. <strong>Every "
        "bandwidth and utilisation figure in this table is an upper bound</strong>, and "
        "comparisons between runs are sound only because the same bias applies to all of "
        "them.</p>"
    )

    parts.append("<h2>Performance (unprofiled)</h2>")
    parts.append(_table(
        ["Run", "ms / token", "Decode tok/s", "Prefill tok/s", "Peak footprint", "KV cache"],
        [[_label(r),
          f"{r.decode_ms_per_token:,.1f} ms" if r.decode_ms_per_token else NOT_MEASURED,
          f"{r.decode_tokens_per_second:,.1f}" if r.decode_tokens_per_second else NOT_MEASURED,
          f"{r.prefill_tokens_per_second:,.0f}" if r.prefill_tokens_per_second
          else NOT_MEASURED,
          fmt_bytes(r.peak_footprint_bytes), fmt_bytes(r.kv_cache_bytes)]
         for r in comparison.runs],
        desc_cols={0},
    ))
    parts.append(
        "<p>Measured with no profiler attached; the memory tables above come from profiled "
        "runs and their durations are not comparable to these.</p>"
    )

    parts.append("<h2>Run configurations</h2>")
    parts.append(_table(
        ["Run", "Params", "Weights", "prompt/gen/batch", "Calibrated", "Caveats"],
        [[_label(r), f"{r.parameters / 1e9:.2f} B" if r.parameters else "-",
          f"{r.bits_per_weight}-bit",
          f"{r.prompt_tokens} / {r.generate_tokens} / {r.batch_size}",
          "yes" if r.calibration_passed else "NO",
          "; ".join(r.caveats) or "-"]
         for r in comparison.runs],
        desc_cols={0, 5},
    ))

    parts.append(
        "<footer>Run ids: "
        + ", ".join(f"<code>{esc(r.run_id)}</code>" for r in comparison.runs)
        + "</footer>"
    )
    parts.append("</div>")
    parts.append(f"<script>{tooltip_script()}</script>")
    return document("Model memory comparison", "\n".join(parts))


# --------------------------------------------------------------------------------------


def write_comparison_reports(
    comparison: Comparison, out_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Write the Markdown, HTML and CSV forms of a comparison."""
    from ..runners.base import write_csv, write_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "comparison.md"
    md_path.write_text(render_markdown(comparison), encoding="utf-8")

    html_path = out_dir / "comparison.html"
    html_path.write_text(render_html(comparison), encoding="utf-8")

    csv_path = write_csv(out_dir / "comparison.csv", comparison.to_rows())
    write_json(out_dir / "comparison.json", comparison.to_dict())

    return md_path, html_path, csv_path
