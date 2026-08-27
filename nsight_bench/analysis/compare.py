"""Cross-run comparison.

A single run tells you what one model does. The reason to build a harness rather than run
`ncu` by hand is to put runs beside each other -- this model against that one, 4-bit against
16-bit, sdpa against eager -- and have the numbers mean the same thing in every column.

Making that valid takes more than tabulating. Three things are enforced here:

* **Normalised quantities only.** Raw byte totals depend on how many tokens were generated
  and at what batch size. The comparison columns are per-token and per-phase rates, so a run
  that generated 32 tokens and one that generated 256 are directly comparable.
* **Comparability is checked, not assumed.** Runs that used different workload shapes, or
  whose calibration gate failed, or whose collections were truncated, are flagged. A number
  from a truncated run sitting silently in a matrix is worse than no number.
* **Missing stays missing.** A metric a run never collected renders as absent, never as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..metrics import Level
from .assemble import RunAnalysis, assemble


@dataclass
class ComparisonRow:
    """One run, reduced to the quantities that compare across runs."""

    run_id: str = ""
    label: str = ""
    model: str = ""
    quantization: str = "none"
    bits_per_weight: int = 16
    dtype: str = ""
    attn: str = ""
    parameters: int = 0
    weight_bytes: int = 0

    # ---- sparsity -----------------------------------------------------------------------
    #: Whole checkpoint versus the subset a single token routes through. For a dense model
    #: these are equal; for a mixture of experts they differ by an order of magnitude, and
    #: that gap is the single most important thing this comparison has to show. A model that
    #: occupies memory like a 30B and moves bytes like a 3B is not well described by either
    #: number alone.
    is_moe: bool = False
    num_experts: int = 0
    experts_per_token: int = 0
    active_weight_bytes: int = 0
    #: GPU-busy share of the decode step's wall time. Separates memory-bound from
    #: launch-bound, which for these deep models is the difference that decides what to fix.
    decode_gpu_busy_pct: float | None = None

    prompt_tokens: int = 0
    generate_tokens: int = 0
    batch_size: int = 1

    # ---- performance (unprofiled) ----
    decode_ms_per_token: float | None = None
    decode_tokens_per_second: float | None = None
    prefill_tokens_per_second: float | None = None

    # ---- memory, per generated token ----
    dram_bytes_per_token: float | None = None
    l2_bytes_per_token: float | None = None
    l1_bytes_per_token: float | None = None

    # ---- hierarchy behaviour ----
    decode_l1_hit_pct: float | None = None
    decode_l2_hit_pct: float | None = None
    decode_l2_to_dram: float | None = None
    decode_bandwidth_gbps: float | None = None
    #: Same measured bytes divided by the *unprofiled* per-token latency. The profiled
    #: figure above uses ncu's serialised, cache-flushed kernel time and reads low.
    decode_bandwidth_real_gbps: float | None = None
    decode_bandwidth_utilisation_pct: float | None = None
    decode_arithmetic_intensity: float | None = None

    prefill_l2_hit_pct: float | None = None
    prefill_bandwidth_gbps: float | None = None
    prefill_arithmetic_intensity: float | None = None

    # ---- footprint ----
    peak_footprint_bytes: int | None = None
    kv_cache_bytes: int | None = None

    # ---- trust ----
    calibration_passed: bool = False
    truncated: bool = False
    expectation_ratio: float | None = None
    caveats: list[str] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        """Whether this row's memory figures can sit in a matrix without a caveat."""
        return self.calibration_passed and not self.truncated

    @property
    def bytes_per_token_vs_weights(self) -> float | None:
        """Memory read per token as a multiple of the model's own weight bytes.

        The most transferable number in the whole comparison: it is dimensionless, so a
        0.6B bf16 model and a 30B 4-bit model can be judged by the same standard. Close to
        1.0 means the decode step reads the model once, as theory says it must. Well below
        means real cache reuse; well above means something is moving data it need not.
        """
        if not (self.dram_bytes_per_token and self.weight_bytes):
            return None
        return self.dram_bytes_per_token / self.weight_bytes

    @property
    def bytes_per_token_vs_active(self) -> float | None:
        """Memory read per token against the weights a token actually routes through.

        This is the column that makes dense and sparse models comparable at all. Measured
        against the whole checkpoint, a top-8-of-128 MoE reads about a tenth of "the model",
        which looks like spectacular cache reuse and is nothing of the sort -- it simply never
        touched the other 120 experts. Against the active subset, every model in the table is
        held to the same standard: roughly 1.0 means the step read what it had to, once.
        """
        basis = self.active_weight_bytes or self.weight_bytes
        if not (self.dram_bytes_per_token and basis):
            return None
        return self.dram_bytes_per_token / basis

    @property
    def sparsity_ratio(self) -> float | None:
        """How much smaller the active weight set is than the stored one."""
        if not (self.active_weight_bytes and self.weight_bytes):
            return None
        return self.active_weight_bytes / self.weight_bytes

    @property
    def memory_efficiency(self) -> float | None:
        """Tokens per second per gigabyte resident.

        The figure of merit for an edge box, where capacity is the binding constraint. A
        sparse model earns its footprint only if the throughput it buys justifies the memory
        it occupies, and this is the ratio that says whether it does.
        """
        if not (self.decode_tokens_per_second and self.weight_bytes):
            return None
        return self.decode_tokens_per_second / (self.weight_bytes / 1e9)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "model": self.model,
            "quantization": self.quantization,
            "bits_per_weight": self.bits_per_weight,
            "dtype": self.dtype,
            "attn": self.attn,
            "parameters": self.parameters,
            "weight_bytes": self.weight_bytes,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "active_weight_bytes": self.active_weight_bytes or None,
            "sparsity_ratio": self.sparsity_ratio,
            "bytes_per_token_vs_active": self.bytes_per_token_vs_active,
            "memory_efficiency_tok_s_per_gb": self.memory_efficiency,
            "decode_gpu_busy_pct": self.decode_gpu_busy_pct,
            "prompt_tokens": self.prompt_tokens,
            "generate_tokens": self.generate_tokens,
            "batch_size": self.batch_size,
            "decode_ms_per_token": self.decode_ms_per_token,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "dram_bytes_per_token": self.dram_bytes_per_token,
            "l2_bytes_per_token": self.l2_bytes_per_token,
            "l1_bytes_per_token": self.l1_bytes_per_token,
            "bytes_per_token_vs_weights": self.bytes_per_token_vs_weights,
            "decode_l1_hit_pct": self.decode_l1_hit_pct,
            "decode_l2_hit_pct": self.decode_l2_hit_pct,
            "decode_l2_to_dram": self.decode_l2_to_dram,
            "decode_bandwidth_gbps": self.decode_bandwidth_gbps,
            "decode_bandwidth_real_gbps": self.decode_bandwidth_real_gbps,
            "decode_bandwidth_utilisation_pct": self.decode_bandwidth_utilisation_pct,
            "decode_arithmetic_intensity": self.decode_arithmetic_intensity,
            "prefill_l2_hit_pct": self.prefill_l2_hit_pct,
            "prefill_bandwidth_gbps": self.prefill_bandwidth_gbps,
            "prefill_arithmetic_intensity": self.prefill_arithmetic_intensity,
            "peak_footprint_bytes": self.peak_footprint_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "calibration_passed": self.calibration_passed,
            "truncated": self.truncated,
            "comparable": self.comparable,
            "expectation_ratio": self.expectation_ratio,
            "caveats": self.caveats,
        }


@dataclass
class Comparison:
    """A set of runs reduced to one comparable table."""

    runs: list[ComparisonRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dram_ceiling_gbps: float | None = None
    l2_bandwidth_gbps: float | None = None

    def comparable_runs(self) -> list[ComparisonRow]:
        return [r for r in self.runs if r.comparable]

    def workload_shapes(self) -> set[tuple[int, int, int]]:
        return {(r.prompt_tokens, r.generate_tokens, r.batch_size) for r in self.runs}

    def check(self) -> None:
        """Populate warnings about anything that undermines the comparison."""
        shapes = self.workload_shapes()
        if len(shapes) > 1:
            described = ", ".join(
                f"{p} prompt / {g} generated / batch {b}" for p, g, b in sorted(shapes)
            )
            self.warnings.append(
                "These runs used different workload shapes (" + described + "). Per-token "
                "columns remain comparable because they normalise by token count, but "
                "absolute times and footprints do not."
            )

        failed = [r.run_id for r in self.runs if not r.calibration_passed]
        if failed:
            self.warnings.append(
                "Calibration did not pass for: " + ", ".join(failed)
                + ". Their memory figures are unverified and should not be compared."
            )

        truncated = [r.run_id for r in self.runs if r.truncated]
        if truncated:
            self.warnings.append(
                "Truncated ncu collections in: " + ", ".join(truncated)
                + ". Their memory totals are prefixes of the real traffic, not samples, so "
                "they read low. Re-run with a higher --max-kernels."
            )

        off_expectation = [
            f"{r.run_id} ({r.expectation_ratio:.2f}x)"
            for r in self.runs
            if r.expectation_ratio is not None and not (0.75 <= r.expectation_ratio <= 1.6)
        ]
        if off_expectation:
            self.warnings.append(
                "Measured decode traffic diverges from the analytic prediction in: "
                + ", ".join(off_expectation)
                + ". Worth understanding before drawing conclusions from those rows."
            )

    def to_dict(self) -> dict:
        return {
            "runs": [r.to_dict() for r in self.runs],
            "warnings": self.warnings,
            "dram_ceiling_gbps": self.dram_ceiling_gbps,
            "l2_bandwidth_gbps": self.l2_bandwidth_gbps,
        }

    def to_rows(self) -> list[dict]:
        return [r.to_dict() for r in self.runs]


def _row_from_analysis(analysis: RunAnalysis) -> ComparisonRow:
    model = analysis.model
    workload = analysis.workload
    decode = analysis.phase("decode_step")
    prefill = analysis.phase("prefill")

    row = ComparisonRow(
        run_id=analysis.run_id,
        label=model.name if model else analysis.run_id,
        model=model.name if model else "",
        quantization=(model.quantization or "none") if model else "none",
        bits_per_weight=model.bits_per_weight if model else 16,
        dtype=model.dtype if model else "",
        attn=model.attn_implementation if model else "",
        parameters=model.param_count if model else 0,
        weight_bytes=model.estimated_weight_bytes() if model else 0,
        is_moe=bool(model and model.is_moe),
        num_experts=model.num_experts if model else 0,
        experts_per_token=model.num_experts_per_token if model else 0,
        active_weight_bytes=model.active_weight_bytes() if model else 0,
        prompt_tokens=workload.prompt_tokens if workload else 0,
        generate_tokens=workload.generate_tokens if workload else 0,
        batch_size=workload.batch_size if workload else 1,
        calibration_passed=analysis.calibration_passed,
    )

    timing = analysis.timing
    if timing.available:
        row.decode_ms_per_token = timing.decode_step_ms
        row.decode_tokens_per_second = timing.decode_tokens_per_second
        row.prefill_tokens_per_second = timing.prefill_tokens_per_second

    row.peak_footprint_bytes = analysis.footprint.cuda_peak_used
    row.kv_cache_bytes = analysis.footprint.kv_cache_bytes

    if decode:
        row.truncated = row.truncated or decode.truncated
        hierarchy = decode.hierarchy
        # One profiled decode step produces batch_size tokens, so per-token normalisation
        # divides by batch, not by the total generated across the whole run.
        tokens = max(1, row.batch_size)
        per_token = hierarchy.bytes_per_token(tokens)
        row.dram_bytes_per_token = per_token.get(Level.DRAM.value)
        row.l2_bytes_per_token = per_token.get(Level.L2.value)
        row.l1_bytes_per_token = per_token.get(Level.L1TEX.value)
        row.decode_l1_hit_pct = hierarchy.level(Level.L1TEX).hit_rate_pct
        row.decode_l2_hit_pct = hierarchy.level(Level.L2).hit_rate_pct
        row.decode_l2_to_dram = hierarchy.amplification().get("l2_to_dram")
        row.decode_bandwidth_gbps = hierarchy.achieved_dram_bandwidth_gbps
        row.decode_arithmetic_intensity = hierarchy.arithmetic_intensity
        ceiling = analysis.dram_ceiling_gbps
        if row.decode_bandwidth_gbps and ceiling:
            row.decode_bandwidth_utilisation_pct = 100 * row.decode_bandwidth_gbps / ceiling
        if decode.expectation:
            row.expectation_ratio = decode.expectation.ratio
        row.decode_bandwidth_real_gbps = analysis.decode_bandwidth_at_real_latency_gbps()
        if decode.occupancy:
            row.decode_gpu_busy_pct = decode.occupancy["busy_pct"]

    if prefill:
        row.truncated = row.truncated or prefill.truncated
        row.prefill_l2_hit_pct = prefill.hierarchy.level(Level.L2).hit_rate_pct
        row.prefill_bandwidth_gbps = prefill.hierarchy.achieved_dram_bandwidth_gbps
        row.prefill_arithmetic_intensity = prefill.hierarchy.arithmetic_intensity

    if not analysis.calibration_passed:
        row.caveats.append("calibration gate did not pass")
    if row.truncated:
        row.caveats.append("ncu collection truncated at the launch cap")
    if not timing.available:
        row.caveats.append("no unprofiled baseline; timing columns absent")

    return row


def build_comparison(run_dirs: list[str | Path]) -> Comparison:
    """Assemble several runs into one comparison."""
    comparison = Comparison()
    ceilings: list[float] = []
    l2_bandwidths: list[float] = []

    for run_dir in run_dirs:
        try:
            analysis = assemble(run_dir)
        except Exception as exc:                                 # noqa: BLE001
            # Both operands are clipped. This takes whatever the caller passed, and an
            # accidental object rather than a path stringifies to its entire repr -- a parsed
            # run is hundreds of megabytes of it, which turns one bad argument into an
            # unreadable report rather than a one-line warning.
            comparison.warnings.append(
                f"could not read {str(run_dir)[:200]}: {str(exc)[:300]}"
            )
            continue
        comparison.runs.append(_row_from_analysis(analysis))
        if analysis.dram_ceiling_gbps:
            ceilings.append(analysis.dram_ceiling_gbps)
        if analysis.calibration and analysis.calibration.peak_l2_bandwidth_gbps:
            l2_bandwidths.append(analysis.calibration.peak_l2_bandwidth_gbps)

    # The ceiling is a property of the machine, not of a run. Taking the median across runs
    # smooths out a single collection that happened to land during a thermal excursion.
    if ceilings:
        comparison.dram_ceiling_gbps = sorted(ceilings)[len(ceilings) // 2]
    if l2_bandwidths:
        comparison.l2_bandwidth_gbps = sorted(l2_bandwidths)[len(l2_bandwidths) // 2]

    # Utilisation is recomputed against the shared ceiling so the column is consistent even
    # if one run's own calibration measured a slightly different peak.
    if comparison.dram_ceiling_gbps:
        for row in comparison.runs:
            # Prefer the real-latency figure: the profiled one divides by ncu's serialised,
            # cache-flushed kernel time and would understate utilisation across the board.
            achieved = row.decode_bandwidth_real_gbps or row.decode_bandwidth_gbps
            if achieved:
                row.decode_bandwidth_utilisation_pct = (
                    100 * achieved / comparison.dram_ceiling_gbps
                )

    comparison.runs.sort(key=lambda r: (r.model, r.quantization, r.run_id))
    comparison.check()
    return comparison
