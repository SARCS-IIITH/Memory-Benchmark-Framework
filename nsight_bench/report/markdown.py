"""Render a run as a Markdown report.

The report is ordered by what a reader needs to decide before reading further. Trust comes
first: if the calibration gate failed or a sentinel tripped, every memory figure below it is
suspect, and that belongs at the top rather than in a footnote. Performance comes from the
unprofiled baseline and is labelled as such, because the profiled durations elsewhere in the
report are not performance numbers.

The memory hierarchy table is the centrepiece, with prefill and decode side by side. That
juxtaposition is the point of the whole harness: the same model, the same weights, two
phases whose memory behaviour differs by an order of magnitude.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..analysis.assemble import RunAnalysis
from ..analysis.deep_dive import STALL_REASONS
from ..metrics import DEVICE_LEVELS, Level
from .format_utils import (
    NOT_MEASURED,
    fmt_bandwidth,
    fmt_bytes,
    fmt_count,
    fmt_flops,
    fmt_pct,
    fmt_ratio,
    fmt_seconds,
    fmt_time_ns,
    markdown_table,
    truncate,
)


def render_markdown(analysis: RunAnalysis) -> str:
    parts = [
        _header(analysis),
        _trust(analysis),
        _performance(analysis),
        _hierarchy(analysis),
        _per_token(analysis),
        _expectation(analysis),
        _footprint(analysis),
        _kernels(analysis),
        _limits(analysis),
        _timeline(analysis),
        _methodology(analysis),
    ]
    return "\n\n".join(p for p in parts if p).rstrip() + "\n"


# --------------------------------------------------------------------------------------


def _header(analysis: RunAnalysis) -> str:
    model = analysis.model
    workload = analysis.workload
    gpu = analysis.platform.gpu if analysis.platform else None

    lines = [f"# Memory profile: {model.name if model else analysis.run_id}", ""]

    rows = []
    if model:
        rows.append(["Model", f"{model.name} ({model.architecture or 'unknown arch'})"])
        rows.append(["Checkpoint", f"`{model.path}`"])
        rows.append([
            "Shape",
            f"{model.num_layers} layers, hidden {model.hidden_size}, "
            f"{model.describe_attention()}",
        ])
        weights = (
            f"{fmt_bytes(model.weight_bytes_on_disk)} on disk, "
            f"{model.quantization or 'unquantized'} ({model.bits_per_weight} bits/weight)"
        )
        # The on-disk figure and the resident figure differ for tied-embedding checkpoints,
        # and both appear in this report -- here and in the physics check. Reconciling them
        # at the point of first mention stops the two numbers reading as a contradiction.
        resident = analysis.footprint.model_weight_bytes_resident
        if resident and model.weight_bytes_on_disk and (
            abs(model.weight_bytes_on_disk - resident) / model.weight_bytes_on_disk > 0.02
        ):
            weights += f"; {fmt_bytes(resident)} resident once loaded"
            if model.tie_word_embeddings:
                weights += " (input and output embeddings are tied, so one copy is held)"
        rows.append(["Weights", weights])
    if workload:
        rows.append([
            "Workload",
            f"{workload.prompt_tokens} prompt tokens -> {workload.generate_tokens} generated, "
            f"batch {workload.batch_size}, {workload.repeat} repeats after "
            f"{workload.warmup_iters} warmup",
        ])
    if model:
        rows.append(["Attention", model.attn_implementation])
    if gpu:
        rows.append([
            "GPU",
            f"{gpu.name} [{gpu.chip}] sm_{gpu.compute_capability.replace('.', '')}, "
            f"{gpu.sm_count} SMs, L2 {fmt_bytes(gpu.l2_cache_bytes)}"
            + (", unified memory" if gpu.unified_memory else ""),
        ])
    if analysis.platform:
        rows.append([
            "Tools",
            f"nsys {analysis.platform.tools.nsys_version}, "
            f"ncu {analysis.platform.tools.ncu_version}, "
            f"CUDA {analysis.platform.tools.cuda_version}, "
            f"torch {analysis.platform.torch_version}",
        ])
    rows.append(["Run", f"`{analysis.run_id}`"])
    rows.append([
        "Generated",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    ])

    lines.append(markdown_table(["", ""], rows, align=["left", "left"]))
    return "\n".join(lines)


def _trust(analysis: RunAnalysis) -> str:
    """Whether the numbers below can be believed, stated before they are shown."""
    lines = ["## Can these numbers be trusted?", ""]

    calibration = analysis.calibration
    if calibration is None:
        lines.append(
            "> **No calibration was run.** Every DRAM figure below is derived from the L2 "
            "sysmem aperture without having been checked against a known quantity on this "
            "machine. Run `nsbench calibrate` before relying on them."
        )
    elif calibration.passed:
        gate = calibration.byte_accounting
        lines.append(
            f"**Calibration gate: PASSED.** Immediately before this run, a streaming kernel "
            f"of known size was profiled and the derivation reproduced "
            f"{fmt_bytes(gate.expected_bytes)} of traffic as {fmt_bytes(gate.measured_bytes)} "
            f"-- an error of {gate.relative_error:+.2%}. The device and peer L2 apertures both "
            f"read zero, as they must on a part with no private VRAM."
        )
    else:
        lines.append(
            "> **Calibration gate: FAILED.** "
            + calibration.byte_accounting.summary()
            + "\n>\n> The DRAM figures below are *unverified*. Treat them as indicative only."
        )

    lines.append("")
    lines.append(
        "DRAM traffic on this GPU is not read from a `dram__*` counter -- GB10 has none. "
        "It is derived from `lts__t_sectors_aperture_sysmem_lookup_miss x 32 B`, the L2 "
        "misses that reach the unified LPDDR5X pool. The calibration above is what makes "
        "that substitution auditable rather than assumed."
    )

    sentinel_problems = [
        f"`{scope}`" for scope, phase in analysis.phases.items()
        if not phase.hierarchy.sentinels_ok
    ]
    if sentinel_problems:
        lines.append("")
        lines.append(
            "> **Sentinel tripped** in " + ", ".join(sentinel_problems) + ". The device or "
            "peer L2 aperture reported non-zero traffic, which should be impossible here. "
            "The sysmem-only derivation is incomplete and DRAM figures understate reality."
        )

    if analysis.warnings:
        lines.append("")
        lines.append("### Warnings")
        lines.append("")
        for warning in analysis.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def _performance(analysis: RunAnalysis) -> str:
    timing = analysis.timing
    lines = ["## Performance", ""]

    if not timing.available:
        lines.append(
            "_No unprofiled baseline ran, so this report contains no valid latency or "
            "throughput figures. Durations elsewhere come from profiled runs, which are "
            "heavily perturbed -- Nsight Compute replays every kernel -- and must not be "
            "read as performance._"
        )
        return "\n".join(lines)

    lines.append(
        "_Measured with no profiler attached. Median of "
        f"{analysis.workload.repeat if analysis.workload else '?'} repeats after "
        f"{analysis.workload.warmup_iters if analysis.workload else '?'} discarded warmup "
        "iterations._"
    )
    lines.append("")

    rows = [
        ["Prefill", fmt_seconds(timing.prefill_seconds),
         fmt_seconds(timing.prefill_iqr) if timing.prefill_iqr else "-",
         f"{timing.prefill_tokens_per_second:,.0f} tok/s"
         if timing.prefill_tokens_per_second else NOT_MEASURED],
        ["Decode (all tokens)", fmt_seconds(timing.decode_seconds),
         fmt_seconds(timing.decode_iqr) if timing.decode_iqr else "-",
         f"{timing.decode_tokens_per_second:,.1f} tok/s"
         if timing.decode_tokens_per_second else NOT_MEASURED],
        ["Decode (per token)",
         fmt_seconds(timing.decode_step_ms / 1000 if timing.decode_step_ms else None),
         fmt_seconds(timing.decode_step_iqr_ms / 1000)
         if timing.decode_step_iqr_ms else "-", "-"],
    ]
    lines.append(markdown_table(
        ["Phase", "Median", "IQR", "Throughput"], rows,
        align=["left", "right", "right", "right"],
    ))

    # An IQR over three points is a real spread but a weak one, and the reader cannot tell
    # how many points it came from unless told. Saying so is what keeps a narrow IQR from
    # being read as evidence of stability it does not carry.
    if timing.sample_count and timing.sample_count < 4:
        lines.append("")
        lines.append(
            f"_The IQR is interpolated over only {timing.sample_count} repeats. It shows "
            "whether these particular iterations agreed, not that the figure is reproducible "
            "-- raise `--repeat` before treating a small spread as stability._"
        )

    prefill_rate = timing.prefill_tokens_per_second
    decode_rate = timing.decode_tokens_per_second
    if prefill_rate and decode_rate:
        lines.append("")
        lines.append(
            f"Prefill processes tokens {prefill_rate / decode_rate:,.0f}x faster than decode "
            "produces them: prefill reads each weight once and amortises it over the whole "
            "prompt, while decode reads the entire model to emit a single token."
        )
        # What that gap is *made of* is a question the tables answer, and the answer is not
        # always "the memory hierarchy". Asserting it here used to contradict the bandwidth
        # section further down, which on a launch-bound run says the opposite. Defer to the
        # measurement instead of pre-empting it.
        lines.append("")
        lines.append(
            "Whether the decode side of that gap is set by bandwidth or by something else is "
            "what the sections below establish -- see the GPU-busy column and the bandwidth "
            "utilisation figure before concluding either way."
        )
    return "\n".join(lines)


def _hierarchy(analysis: RunAnalysis) -> str:
    """The centrepiece: traffic at every rung, prefill against decode."""
    lines = ["## Memory hierarchy", ""]

    if not analysis.phases:
        lines.append("_No Nsight Compute data in this run._")
        return "\n".join(lines)

    scopes = [s for s in ("prefill", "decode_step") if s in analysis.phases]
    headers = ["Level"]
    for scope in scopes:
        headers += [f"{analysis.phases[scope].label} bytes", "hit rate"]

    rows = []
    for level in DEVICE_LEVELS:
        if level is Level.REGISTER:
            continue                                     # capacity, not traffic; shown below
        row = [level.label]
        for scope in scopes:
            summary = analysis.phases[scope].hierarchy.level(level)
            row.append(fmt_bytes(summary.bytes_total))
            row.append(fmt_pct(summary.hit_rate_pct) if summary.hit_rate_pct is not None
                       else "-")
        rows.append(row)

    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append(
        "Read top to bottom: bytes the SM requested, how much L1 and L2 absorbed, and what "
        "was left to fetch from LPDDR5X. Shared-memory bytes are estimated from wavefronts "
        f"(x128 B) -- there is no shared-memory sector counter -- so bank conflicts inflate "
        "that row, which is the effect worth seeing."
    )

    # ---- amplification ----
    lines.append("")
    lines.append("### Traffic amplification")
    lines.append("")
    amp_rows = []
    for scope in scopes:
        amp = analysis.phases[scope].hierarchy.amplification()
        amp_rows.append([
            analysis.phases[scope].label,
            fmt_ratio(amp["sm_to_l2"]),
            fmt_ratio(amp["l2_to_dram"]),
            fmt_ratio(amp["sm_to_dram"]),
        ])
    lines.append(markdown_table(
        ["Phase", "SM -> L2", "L2 -> LPDDR5X", "SM -> LPDDR5X"], amp_rows
    ))
    lines.append("")
    lines.append(
        "`L2 -> LPDDR5X` is the fraction of what L1 asked of L2 that L2 could not supply. "
        "Near 1.0 means L2 is providing essentially no reuse; near 0 means the working set "
        "is being captured on chip."
    )

    # ---- bandwidth & intensity ----
    lines.append("")
    lines.append("### Bandwidth and arithmetic intensity")
    lines.append("")
    ceiling = analysis.dram_ceiling_gbps
    bw_rows = []
    for scope in scopes:
        hierarchy = analysis.phases[scope].hierarchy
        achieved = hierarchy.achieved_dram_bandwidth_gbps
        utilisation = (
            fmt_pct(100 * achieved / ceiling) if achieved and ceiling else NOT_MEASURED
        )
        bw_rows.append([
            analysis.phases[scope].label,
            fmt_time_ns(hierarchy.gpu_time_ns)
            if analysis.phases[scope].traffic_collected else NOT_MEASURED,
            fmt_bandwidth(achieved),
            utilisation,
            fmt_flops(hierarchy.total_flops),
            fmt_ratio(hierarchy.arithmetic_intensity, suffix=" FLOP/B"),
        ])
    lines.append(markdown_table(
        ["Phase", "GPU time", "LPDDR5X bandwidth", "% of ceiling", "FLOPs", "Intensity"],
        bw_rows,
    ))

    if ceiling:
        lines.append("")
        lines.append(
            f"The ceiling is {fmt_bandwidth(ceiling)}, measured on this machine by the "
            "calibration sweep with a working set far beyond L2 -- an achievable rate, not a "
            "datasheet figure."
        )

    # Every rate above divides by kernel time. Whether that is most of the phase's time is a
    # separate question, and the one that decides if a bandwidth figure describes the phase
    # at all -- so it goes immediately after, not in a footnote.
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
            fmt_bytes(analysis.phases[scope].nsys_l2_bytes),
        ])

    if busy_rows:
        lines.append("")
        lines.append("#### Was the GPU actually busy?")
        lines.append("")
        lines.append(markdown_table(
            ["Phase", "Wall (unprofiled trace)", "GPU executing", "Busy", "L2 traffic (nsys)"],
            busy_rows, align=["left", "right", "right", "right", "right"],
        ))
        lines.append("")
        lines.append(
            "Busy time is the union of kernel intervals inside the phase's NVTX range, taken "
            "from the Nsight Systems timeline. The remainder is the gap between kernels: "
            "launch latency, or the host blocking on a synchronisation. It is neither "
            "bandwidth nor compute, and no figure above accounts for it -- they all divide "
            "by kernel time, so they describe the busy fraction only."
        )
        lines.append("")
        lines.append(
            "L2 traffic (nsys) is all L2 traffic per phase instance, sampled by nsys at no "
            "extra cost. It is an upper bound on DRAM traffic: on Qwen3-0.6B it ran 6-10% above "
            "ncu's DRAM bytes for decode, and about 1.9x for prefill, where activations are "
            "reused in L2."
        )
        for scope in scopes:
            verdict = analysis.phases[scope].occupancy_verdict()
            if verdict and (analysis.phases[scope].occupancy or {}).get("busy_pct", 100) < 90:
                lines.append("")
                lines.append(f"**{analysis.phases[scope].label}.** {verdict}")

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
            + (" -- IMPOSSIBLE" if util and util > 100 else ""),
        ])

    if real_rows:
        lines.append("")
        lines.append("#### Bytes over real latency -- an upper bound")
        lines.append("")
        lines.append(
            "The table above divides measured bytes by summed kernel time from the ncu pass, "
            "where kernels run serialised -- so it understates the rate. Dividing the same "
            "bytes by the unprofiled wall time removes that distortion from the denominator. "
            "It does **not** fix the numerator: those bytes were counted with L2 flushed "
            "before every replay pass, so they exclude the cross-kernel reuse a real run "
            "gets, and they read high. What follows is therefore an upper bound, not an "
            "achieved rate."
        )
        lines.append("")
        lines.append(markdown_table(
            ["Phase", "Bytes / real latency", "% of measured ceiling"], real_rows,
            align=["left", "right", "right"],
        ))

        if analysis.bandwidth_bound_exceeded():
            lines.append("")
            lines.append(
                "**One of these exceeds 100% of a ceiling measured on this same machine, "
                "which is impossible.** That is the bound proving its own looseness: the "
                "cold-cache byte counts overstate what a warm run moves. Read every figure "
                "in this table as an upper bound, including the ones below 100%."
            )

    util = analysis.decode_bandwidth_utilisation_pct()
    decode_phase = analysis.phase("decode_step")
    busy = decode_phase.occupancy["busy_pct"] if (
        decode_phase and decode_phase.occupancy
    ) else None
    if util and util < 60:
        lines.append("")
        lines.append(
            f"Even at that upper bound, decode reaches only {fmt_pct(util, 0)} of the "
            "achievable rate, so this decode step is **not bandwidth-saturated** -- the true "
            "figure is lower still."
            + (
                f" The GPU was idle for {100 - busy:.0f}% of the step, which is where the "
                "time actually goes: many small kernels and the gaps between them, not the "
                "memory system running out of headroom."
                if busy is not None and busy < 90 else
                " Its cost is latency and occupancy -- many small kernels, each too short to "
                "fill the memory pipeline."
            )
            + " Reducing bytes moved would help less here than fusing or enlarging the "
            "kernels, or cutting launch count."
        )
        if analysis.calibration and analysis.calibration.peak_l2_bandwidth_gbps:
            ratio = analysis.calibration.peak_l2_bandwidth_gbps / ceiling
            lines.append("")
            lines.append(
                f"For contrast, an L2-resident working set streams at "
                f"{fmt_bandwidth(analysis.calibration.peak_l2_bandwidth_gbps)} on this part -- "
                f"{ratio:.1f}x faster. That multiple is what an L2 hit is worth, and why the "
                "hit-rate column above carries so much weight."
            )

    # ---- registers & occupancy ----
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
        lines.append("")
        lines.append("### Registers and spilling")
        lines.append("")
        lines.append(markdown_table(
            ["Phase", "Max reg/thread", "Mean reg/thread", "Kernels at 255", "Spill traffic"],
            reg_rows,
        ))
        lines.append("")
        lines.append(
            "Spill traffic is the register file overflowing into local memory, which is "
            "physically the same LPDDR5X as everything else. Non-zero spill is bandwidth "
            "spent to move data that never needed to leave the SM."
        )

    return "\n".join(lines)


def _per_token(analysis: RunAnalysis) -> str:
    """Bytes per generated token -- the number that compares across models."""
    decode = analysis.phase("decode_step")
    if decode is None:
        return ""

    lines = ["## Bytes per generated token", ""]
    lines.append(
        "Normalised per token, so models of different sizes and quantization schemes can be "
        "put side by side. These come from one profiled decode step."
    )
    lines.append("")

    per_token = decode.hierarchy.bytes_per_token(
        (analysis.workload.batch_size if analysis.workload else 1)
    )
    rows = []
    for level in (Level.LOCAL, Level.SHARED, Level.L1TEX, Level.L2, Level.DRAM):
        value = per_token.get(level.value)
        rows.append([level.label, fmt_bytes(value)])
    lines.append(markdown_table(["Level", "Bytes / token"], rows))
    return "\n".join(lines)


def _expectation(analysis: RunAnalysis) -> str:
    """Measured decode traffic against what physics says it must be."""
    decode = analysis.phase("decode_step")
    if decode is None or decode.expectation is None:
        return ""

    expectation = decode.expectation
    lines = ["## Does the measurement match the physics?", ""]
    lines.append(
        "A decode step must read every weight and the whole KV cache to produce one token. "
        "That makes its DRAM traffic predictable, and comparing prediction to measurement is "
        "the strongest check that both the NVTX scoping and the byte derivation are correct."
    )
    lines.append("")

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
        ["**Expected total**", f"**{fmt_bytes(expectation.expected_bytes)}**"],
        ["**Measured**" + (" (nsys-sampled L2)" if expectation.measured_source == "nsys_l2"
                           else ""), f"**{fmt_bytes(expectation.measured_bytes)}**"],
        ["Ratio", fmt_ratio(expectation.ratio)],
    ]
    lines.append(markdown_table(["", "Bytes"], rows, align=["left", "right"]))
    lines.append("")
    lines.append(f"**Verdict:** {expectation.verdict()}")
    return "\n".join(lines)


def _footprint(analysis: RunAnalysis) -> str:
    footprint = analysis.footprint
    lines = ["## Memory footprint", ""]
    lines.append(
        "NVML reports GPU memory as `N/A` on this part -- there is no discrete VRAM to "
        "report -- so footprint is reconstructed from three independent sources with "
        "different blind spots."
    )
    lines.append("")

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
            "MemAvailable lost while loading weights -- this process's own draw",
        ])
    lines.append(markdown_table(["Source", "Bytes", "What it sees"], rows,
                                align=["left", "right", "left"]))

    agreement = footprint.agreement()
    if agreement:
        lines.append("")
        lines.append(
            ("Why the driver figure is not this run's footprint: " if footprint.unified_memory
             else "Driver versus allocator: ")
            + agreement
        )

    for note in footprint.notes:
        lines.append("")
        lines.append(f"- {note}")
    return "\n".join(lines)


def _kernels(analysis: RunAnalysis) -> str:
    """Per-kernel table, ranked by bytes reaching memory rather than by time."""
    lines = ["## Kernels by memory traffic", ""]
    any_rows = False

    for scope in ("prefill", "decode_step"):
        phase = analysis.phase(scope)
        if phase is None or not phase.kernel_rows:
            continue
        any_rows = True
        lines.append(f"### {phase.label}")
        lines.append("")

        rows = []
        for row in phase.kernel_rows[:12]:
            rows.append([
                truncate(row["kernel_short"], 44),
                fmt_time_ns(row["duration_ns"]),
                fmt_bytes(row["dram_bytes"]),
                fmt_bytes(row["l2_bytes"]),
                fmt_pct(row["l2_hit_rate_pct"]),
                fmt_bandwidth(row["dram_bandwidth_gbps"]),
                fmt_count(row["registers_per_thread"]),
            ])
        lines.append(markdown_table(
            ["Kernel", "Time", "LPDDR5X bytes", "L2 bytes", "L2 hit", "Bandwidth", "Reg/thr"],
            rows,
        ))

        cross_check = phase.cross_check_kernel_counts()
        if cross_check:
            lines.append("")
            lines.append(f"_{cross_check}._")
        lines.append("")

    if not any_rows:
        lines.append("_No per-kernel data._")
    lines.append(
        "Ranked by bytes reaching LPDDR5X, not by duration -- this harness is about memory, "
        "and the kernel that moves the most data is not always the one that takes longest."
    )
    return "\n".join(lines)


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

    lines = ["## What limits each phase", ""]
    lines.append(
        "Bytes moved say how much work the memory system did; they cannot say why a kernel "
        "took the time it did. These come from the tier-2 sections, which record why a warp "
        "that could have issued did not. `Memory (long scoreboard)` is a warp waiting on a "
        "global load -- its share is what separates a genuinely memory-bound phase from one "
        "that simply lacks the parallelism to hide any latency at all."
    )
    lines.append("")

    for scope, phase in phases:
        limits = phase.limits
        lines.append(f"### {phase.label}")
        lines.append("")
        verdict = limits.verdict()
        if verdict:
            lines.append(verdict)
            lines.append("")

        profile = limits.weighted_stall_profile()
        if profile:
            rows = []
            for reason, pct in list(profile.items())[:7]:
                label, meaning = STALL_REASONS.get(reason, (reason, ""))
                rows.append([label, fmt_pct(pct, 1), meaning])
            lines.append(markdown_table(
                ["Stall reason", "Share", "What it means"], rows,
                align=["left", "right", "left"],
            ))
            lines.append("")
            lines.append("_Weighted by each kernel's GPU time, so short kernels cannot "
                         "outvote the ones that dominate the phase._")
            lines.append("")

        rows = []
        for kernel in limits.top(8):
            dominant = kernel.dominant_stall()
            label = STALL_REASONS.get(dominant[0], (dominant[0], ""))[0] if dominant else "-"
            rows.append([
                truncate(kernel.short_name, 34),
                fmt_count(kernel.launches),
                fmt_time_ns(kernel.duration_ns),
                fmt_pct(kernel.achieved_occupancy_pct),
                fmt_ratio(kernel.waves_per_sm, suffix=""),
                fmt_pct(kernel.memory_stall_pct, 0),
                label,
            ])
        lines.append(markdown_table(
            ["Kernel", "Launches", "Time", "Occupancy", "Waves/SM", "Memory stalls",
             "Dominant stall"],
            rows,
        ))
        lines.append("")
        lines.append(
            "Waves per SM below 1.0 means the kernel cannot even fill the machine once -- "
            "there is no amount of memory tuning that fixes a kernel with nothing to overlap."
        )
        lines.append("")

    return "\n".join(lines).rstrip()


def _timeline(analysis: RunAnalysis) -> str:
    if analysis.nsys is None:
        return ""

    nsys = analysis.nsys
    lines = ["## Timeline", ""]

    durations = nsys.phase_durations_ns()
    counts = nsys.phase_instance_counts()
    interesting = {
        phase: total for phase, total in durations.items() if phase.startswith("nsbench.")
    }
    if interesting:
        rows = [
            [phase.replace("nsbench.", ""), fmt_time_ns(total), fmt_count(counts.get(phase, 0)),
             fmt_count(len(nsys.kernels_in_phase(phase)))]
            for phase, total in sorted(interesting.items(), key=lambda kv: -kv[1])
        ]
        lines.append(markdown_table(["Phase", "Wall time", "Instances", "Kernels"], rows))
        lines.append("")

    clocks = nsys.clock_summary()
    if clocks:
        lines.append(
            f"**GPC clock** during the traced region: "
            f"{clocks['gpc_clock_mhz_min']:,.0f}-{clocks['gpc_clock_mhz_max']:,.0f} MHz "
            f"(mean {clocks['gpc_clock_mhz_mean']:,.0f}, spread "
            f"{clocks['gpc_clock_spread_pct']:.1f}%). "
            + ("Stable enough to compare against other runs."
               if clocks["gpc_clock_spread_pct"] < 15
               else "This much variation softens any timing comparison against other runs.")
        )
        lines.append("")

    if nsys.memcpy_bytes:
        rows = [[kind, fmt_bytes(total)] for kind, total in sorted(nsys.memcpy_bytes.items())]
        lines.append("**Explicit host/device copies:**")
        lines.append("")
        lines.append(markdown_table(["Direction", "Bytes"], rows))
        lines.append("")
        lines.append(
            "On a unified-memory part these should be near zero in steady state. Copies "
            "during decode mean data is being staged that did not need to move."
        )
        lines.append("")

    if nsys.um_page_faults:
        rows = [[k.replace("_", " "), fmt_count(v)] for k, v in nsys.um_page_faults.items()]
        lines.append("**Unified memory page faults:**")
        lines.append("")
        lines.append(markdown_table(["Kind", "Count"], rows))
        lines.append("")

    peak = nsys.peak_allocated_bytes()
    if peak:
        lines.append(
            f"Peak outstanding GPU allocation inside the traced region: {fmt_bytes(peak)} "
            f"across {len(nsys.memory_events)} allocation events."
        )
    return "\n".join(lines)


def _methodology(analysis: RunAnalysis) -> str:
    lines = ["## Method and caveats", ""]

    lines.append(
        "- **DRAM traffic is derived, not counted.** GB10 exposes no `dram__*` metrics. "
        "Bytes past L2 come from `lts__t_sectors_aperture_sysmem_lookup_miss x 32 B`, with "
        "the device and peer aperture counters carried as sentinels that must read zero."
    )
    lines.append(
        "- **Profiled durations are not performance.** Nsight Compute replays each kernel "
        "many times. Only the baseline section reports real timing."
    )
    lines.append(
        "- **One decode step is profiled, not all of them.** A decode step's kernel mix does "
        "not change between tokens, so one step gives full kernel coverage at a fraction of "
        "the cost. The step chosen is the last one, where the KV cache is deepest."
    )
    lines.append(
        "- **Hit rates are recomputed from raw counts**, never averaged across kernels: "
        "`sum(hits) / sum(hits + misses)`. Averaging percentages would weight a tiny "
        "elementwise kernel equally with a large GEMM."
    )
    lines.append(
        "- **Shared-memory bytes are an estimate** from wavefront counts x 128 B. Bank "
        "conflicts inflate the figure, which is intentional -- that is the cost being shown."
    )
    lines.append(
        "- **L2 hit rates are measured with a cold cache.** ncu runs with "
        "`--cache-control all`, flushing L2 before each replay pass so every kernel is "
        "measured independently of whatever ran before it. In an un-profiled decode loop L2 "
        "may retain data across steps, so the reported hit rate is a *lower bound* on what "
        "the un-replayed workload sees. Set `ncu.cache_control: none` in the profile config "
        "to measure warm behaviour instead -- at the cost of each kernel's numbers depending "
        "on its predecessor."
    )

    if analysis.platform and analysis.platform.notes:
        lines.append("")
        lines.append("### Platform notes")
        lines.append("")
        for note in analysis.platform.notes:
            lines.append(f"- {note}")

    if analysis.calibration and analysis.calibration.notes:
        lines.append("")
        lines.append("### Calibration notes")
        lines.append("")
        for note in analysis.calibration.notes:
            lines.append(f"- {note}")

    lines.append("")
    lines.append("### Artefacts")
    lines.append("")
    lines.append(f"- Raw profiler reports: `{analysis.root / 'raw'}`")
    lines.append(f"- Tidy metric tables: `{analysis.root / 'metrics'}`")
    lines.append(f"- Full provenance: `{analysis.root / 'manifest.json'}`")
    lines.append("")
    lines.append(
        "Open `raw/timeline.nsys-rep` in Nsight Systems or `raw/ncu_*.ncu-rep` in Nsight "
        "Compute for the interactive views."
    )
    return "\n".join(lines)


def write_markdown_report(analysis: RunAnalysis, path: str | Path | None = None) -> Path:
    out_path = Path(path) if path else analysis.root / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(analysis), encoding="utf-8")
    return out_path
