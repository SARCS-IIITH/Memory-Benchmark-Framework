"""Tier-2 deep dive: why a kernel is slow, not just how much it moved.

Tier 1 answers "how many bytes crossed each level". That is the harness's main job, but it
cannot say *why* a kernel takes the time it does. A kernel moving little data can still be
slow, and on this machine a decode step turns out to run well below the memory ceiling --
which makes the reason it is slow the interesting question.

Tier 2's full sections answer it, through three families of metric that are unavailable in a
plain ``--metrics`` list:

* **Warp issue stall reasons.** For every cycle a warp could have issued and did not, the
  hardware records why. ``long_scoreboard`` is the memory one -- a warp waiting on a global
  load. Its share separates "stalled on memory" from "not enough parallelism to hide
  anything", which look identical in a bandwidth number.
* **Achieved occupancy and waves per SM.** A kernel with a fraction of a wave cannot fill the
  machine no matter how efficient its memory access is; it simply does not have enough work
  in flight to hide latency.
* **``derived__local_spilling_requests``**, which fails as a standalone metric request and is
  only obtainable through the MemoryWorkloadAnalysis section.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..parsers.ncu_parse import NcuReport, base_identifier

#: Warp issue stall reasons, mapped to a readable label and what each implies. Stored as
#: ratios of warps stalled per active issue cycle.
STALL_REASONS: dict[str, tuple[str, str]] = {
    "long_scoreboard": (
        "Memory (long scoreboard)",
        "waiting on a global or local memory load -- the memory-bound signature",
    ),
    "short_scoreboard": (
        "Shared / L1 (short scoreboard)",
        "waiting on shared memory or an L1 operation",
    ),
    "mio_throttle": (
        "MIO throttle",
        "the memory input/output pipe is saturated with requests",
    ),
    "lg_throttle": (
        "Local/global throttle",
        "the local/global instruction queue is full",
    ),
    "math_pipe_throttle": (
        "Math pipe throttle",
        "arithmetic pipes saturated -- the compute-bound signature",
    ),
    "barrier": ("Barrier", "waiting at a __syncthreads()"),
    "membar": ("Memory barrier", "waiting on a memory fence"),
    "not_selected": (
        "Not selected",
        "eligible but another warp was chosen -- a sign of healthy parallelism",
    ),
    "no_instruction": (
        "Instruction fetch",
        "waiting on instruction fetch, often a large unrolled loop",
    ),
    "wait": ("Fixed latency", "waiting on a fixed-latency arithmetic dependency"),
    "drain": ("Drain", "waiting for memory to drain at kernel exit"),
    "dispatch_stall": ("Dispatch", "the dispatcher could not issue"),
    "branch_resolving": ("Branch", "waiting on branch resolution"),
    "misc": ("Other", ""),
    "selected": ("Issued", "the warp issued this cycle"),
}

_STALL_METRIC = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"

_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"
_WAVES = "launch__waves_per_multiprocessor"
_SPILL_REQUESTS = "derived__local_spilling_requests"
_SPILL_PCT = "derived__local_spilling_requests_pct"
_ISSUE_ACTIVE = "sm__issue_active.avg.pct_of_peak_sustained_elapsed"


@dataclass
class KernelDeepDive:
    """One kernel's tier-2 view."""

    name: str = ""
    short_name: str = ""
    duration_ns: float = 0.0
    launches: int = 0
    achieved_occupancy_pct: float | None = None
    waves_per_sm: float | None = None
    issue_active_pct: float | None = None
    spill_requests: float | None = None
    spill_pct: float | None = None
    #: reason key -> warps stalled per issue-active cycle
    stalls: dict[str, float] = field(default_factory=dict)

    def dominant_stall(self) -> tuple[str, float] | None:
        """The largest stall reason, excluding warps that actually issued."""
        candidates = {
            key: value for key, value in self.stalls.items()
            if key not in ("selected", "not_selected") and value > 0
        }
        if not candidates:
            return None
        key = max(candidates, key=lambda k: candidates[k])
        return key, candidates[key]

    def stall_share_pct(self, reason: str) -> float | None:
        """One reason's share of all stall cycles, as a percentage."""
        total = sum(v for k, v in self.stalls.items() if k != "selected" and v > 0)
        if not total or reason not in self.stalls:
            return None
        return 100.0 * self.stalls[reason] / total

    @property
    def memory_stall_pct(self) -> float | None:
        """Share of stalls attributable to waiting on memory."""
        return self.stall_share_pct("long_scoreboard")

    def to_row(self) -> dict:
        dominant = self.dominant_stall()
        return {
            "kernel_short": self.short_name,
            "kernel_name": self.name,
            "launches": self.launches,
            "duration_ns": self.duration_ns,
            "achieved_occupancy_pct": self.achieved_occupancy_pct,
            "waves_per_sm": self.waves_per_sm,
            "issue_active_pct": self.issue_active_pct,
            "memory_stall_pct": self.memory_stall_pct,
            "dominant_stall": STALL_REASONS.get(dominant[0], (dominant[0], ""))[0]
            if dominant else None,
            "spill_requests": self.spill_requests,
            "spill_pct": self.spill_pct,
        }


@dataclass
class DeepDiveAnalysis:
    """Tier-2 findings for one phase."""

    scope: str = ""
    kernels: list[KernelDeepDive] = field(default_factory=list)
    available: bool = False
    #: kernel name -> total GPU time across the whole phase, from tier 1's complete
    #: collection or, without tier 1, from the nsys timeline. Used to correct tier 2's
    #: launch-ordered sampling bias.
    true_time_by_kernel: dict[str, float] = field(default_factory=dict)
    #: Which pass supplied :attr:`true_time_by_kernel`: "tier 1", "the nsys timeline" or "".
    time_source: str = ""
    #: Whether tier 2's sample covered every kernel tier 1 ranked as significant.
    sample_note: str = ""

    def top(self, n: int = 10) -> list[KernelDeepDive]:
        return sorted(self.kernels, key=lambda k: k.duration_ns, reverse=True)[:n]

    def weighted_stall_profile(self) -> dict[str, float]:
        """Stall reasons across the phase, weighted by each kernel's true share of GPU time.

        Weighting is essential and its *source* matters. An unweighted average would let a
        hundred microsecond-long elementwise kernels outvote the GEMM that dominates the
        phase. But weighting by tier 2's own sampled time is barely better: tier 2 profiles a
        capped, launch-ordered sample, which over-represents whatever launches earliest and
        most often -- in one measured case, 20 elementwise launches against a single launch
        of the GEMM that actually dominated the phase.

        So the weights come from tier 1, which profiled every kernel in the phase and
        therefore knows the true time distribution, while tier 2 supplies the per-kernel
        stall characteristics it alone can measure. Neither pass can produce this on its own.
        Without tier 1, the nsys timeline supplies the same distribution (see :func:`analyse`).
        """
        weights = self.true_time_by_kernel or {
            k.name: k.duration_ns for k in self.kernels
        }
        total_time = sum(
            weights.get(k.name, k.duration_ns) for k in self.kernels
        ) or 1.0

        profile: dict[str, float] = {}
        for kernel in self.kernels:
            weight = weights.get(kernel.name, kernel.duration_ns) / total_time
            for reason, value in kernel.stalls.items():
                if reason == "selected":
                    continue
                profile[reason] = profile.get(reason, 0.0) + value * weight
        total = sum(profile.values()) or 1.0
        return {k: 100.0 * v / total for k, v in sorted(
            profile.items(), key=lambda kv: -kv[1]
        )}

    def verdict(self) -> str | None:
        """A one-line reading of what limits this phase."""
        if not self.kernels:
            return None
        profile = self.weighted_stall_profile()
        if not profile:
            return None
        memory = profile.get("long_scoreboard", 0.0)
        occupancies = [
            k.achieved_occupancy_pct for k in self.kernels
            if k.achieved_occupancy_pct is not None
        ]
        mean_occupancy = sum(occupancies) / len(occupancies) if occupancies else None

        top_reason = next(iter(profile))
        label = STALL_REASONS.get(top_reason, (top_reason, ""))[0]

        parts = [
            f"Time-weighted, the dominant stall reason is **{label}** "
            f"({profile[top_reason]:.0f}% of stall cycles)."
        ]
        if memory >= 40:
            parts.append(
                f"Memory waits account for {memory:.0f}% of stalls, so this phase is genuinely "
                "latency-bound on memory rather than on arithmetic."
            )
        elif memory:
            parts.append(f"Memory waits are {memory:.0f}% of stalls.")
        if mean_occupancy is not None and mean_occupancy < 30:
            parts.append(
                f"Mean achieved occupancy is only {mean_occupancy:.0f}%, so there are too few "
                "warps in flight to hide the latency that does occur -- more parallelism "
                "would help more than fewer bytes."
            )
        return " ".join(parts)


def analyse(
    report: NcuReport | None,
    scope: str = "",
    tier1: NcuReport | None = None,
    nsys_time_by_base: dict[str, float] | None = None,
) -> DeepDiveAnalysis:
    """Reduce a tier-2 ncu report to a :class:`DeepDiveAnalysis`.

    Args:
        report: The tier-2 collection (full sections, capped sample).
        scope: Phase name.
        tier1: The tier-1 collection for the same phase. Supplies the true per-kernel time
            distribution, which corrects tier 2's launch-ordered sampling bias.
        nsys_time_by_base: Used only when ``tier1`` is None. The phase's total GPU time per
            base identifier, from the nsys timeline. It plays the same corrective role.
    """
    analysis = DeepDiveAnalysis(scope=scope or (report.scope if report else ""))
    if report is None or not report.kernels:
        return analysis

    if tier1 is not None:
        analysis.time_source = "tier 1"
        for kernel in tier1.kernels:
            analysis.true_time_by_kernel[kernel.name] = (
                analysis.true_time_by_kernel.get(kernel.name, 0.0) + kernel.duration_ns
            )

    # One entry per distinct kernel, aggregating its launches. Base-name filtering means
    # tier 2 profiles many instances of the same kernel; the per-kernel rows should reflect
    # the kernel, not each launch.
    grouped: dict[str, list] = {}
    for kernel in report.kernels:
        grouped.setdefault(kernel.name, []).append(kernel)

    for name, launches in grouped.items():
        entry = KernelDeepDive(
            name=name,
            short_name=launches[0].short_name,
            launches=len(launches),
            duration_ns=sum(k.duration_ns for k in launches),
        )
        entry.achieved_occupancy_pct = _mean(launches, _OCCUPANCY)
        entry.waves_per_sm = _mean(launches, _WAVES)
        entry.issue_active_pct = _mean(launches, _ISSUE_ACTIVE)
        entry.spill_requests = _sum(launches, _SPILL_REQUESTS)
        entry.spill_pct = _mean(launches, _SPILL_PCT)

        for reason in STALL_REASONS:
            value = _mean(launches, _STALL_METRIC.format(reason))
            if value is not None:
                entry.stalls[reason] = value

        analysis.kernels.append(entry)

    analysis.available = bool(analysis.kernels)

    if tier1 is None and nsys_time_by_base:
        analysis.time_source = "the nsys timeline"
        analysis.true_time_by_kernel = _split_by_instantiation(
            analysis.kernels, nsys_time_by_base
        )

    if analysis.true_time_by_kernel:
        sampled = {k.name for k in analysis.kernels}
        ranked = sorted(
            analysis.true_time_by_kernel.items(), key=lambda kv: -kv[1]
        )[:len(sampled) + 2]
        missed = [name for name, _ in ranked if name not in sampled]
        covered_time = sum(
            analysis.true_time_by_kernel.get(k.name, 0.0) for k in analysis.kernels
        )
        total_time = sum(analysis.true_time_by_kernel.values()) or 1.0
        analysis.sample_note = (
            f"Tier 2 sampled {len(analysis.kernels)} distinct kernels covering "
            f"{100 * covered_time / total_time:.0f}% of the phase's GPU time; stall shares "
            f"are weighted by {analysis.time_source}'s complete timing, not by the sample."
        )
        if missed:
            analysis.sample_note += (
                f" {len(missed)} significant kernel(s) fell outside the sample."
            )

    return analysis


def _split_by_instantiation(
    kernels: list[KernelDeepDive], time_by_base: dict[str, float]
) -> dict[str, float]:
    """Spread nsys phase time per base identifier over tier 2's full kernel names.

    nsys and ncu demangle differently, so the two only agree at the base identifier, and one
    base name (``device_kernel``, say) can cover several template instantiations. Each one
    sampled gets a share of its base name's phase time in proportion to its sampled time.
    Base names tier 2 never sampled are kept under the base name, so coverage and "fell
    outside the sample" still count them.
    """
    sampled_by_base: dict[str, float] = {}
    for kernel in kernels:
        base = base_identifier(kernel.name)
        sampled_by_base[base] = sampled_by_base.get(base, 0.0) + kernel.duration_ns

    weights: dict[str, float] = {}
    for kernel in kernels:
        base = base_identifier(kernel.name)
        if base not in time_by_base:
            continue
        share = kernel.duration_ns / sampled_by_base[base] if sampled_by_base[base] else 0.0
        weights[kernel.name] = time_by_base[base] * share
    for base, time_ns in time_by_base.items():
        if base not in sampled_by_base:
            weights[base] = time_ns
    return weights


def _mean(launches: list, metric: str) -> float | None:
    values = [k.metrics[metric] for k in launches if metric in k.metrics]
    return sum(values) / len(values) if values else None


def _sum(launches: list, metric: str) -> float | None:
    values = [k.metrics[metric] for k in launches if metric in k.metrics]
    return sum(values) if values else None
