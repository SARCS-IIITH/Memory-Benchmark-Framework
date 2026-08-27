"""Assemble a finished run directory into one analysis object.

This is the join point. A run leaves behind four independent artefact sets -- a calibration
result, an unprofiled timing pass, an nsys timeline, and a set of ncu collections -- and
none of them individually answers the question the harness exists to answer. This module
pulls them together, cross-checks them against each other, and produces the single structure
that both reporters render.

The cross-checks are the valuable part. Three independent sources report memory footprint
here, and each has a different blind spot; where they disagree is a finding. The analytic
prediction of decode-phase DRAM traffic is compared against the measured figure, which is
the strongest available test that the NVTX scoping and the byte derivation are both right.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import stats
from ..calibration import CalibrationResult
from ..config import ModelConfig, RunConfig, WorkloadConfig
from ..metrics import Level
from ..parsers.ncu_parse import NcuReport, parse_csv
from ..parsers.nsys_parse import NsysReport, parse_sqlite
from ..platform import PlatformProfile
from .deep_dive import DeepDiveAnalysis, analyse as analyse_deep_dive
from .derive import (
    DecodeExpectation,
    HierarchySummary,
    check_decode_expectation,
    kernel_traffic_rows,
    summarize,
)

#: NVTX phase name -> ncu scope key. The ncu collections are filed by scope; the nsys
#: timeline uses the raw NVTX range names.
PHASE_TO_SCOPE = {"nsbench.prefill": "prefill", "nsbench.decode_step": "decode_step"}


@dataclass
class PhaseAnalysis:
    """Everything known about one phase."""

    scope: str
    hierarchy: HierarchySummary
    kernel_rows: list[dict] = field(default_factory=list)
    ncu_report: NcuReport | None = None
    deep_dive: NcuReport | None = None
    #: Tier-2 findings: why the phase is slow, as opposed to how much it moved.
    limits: DeepDiveAnalysis | None = None
    expectation: DecodeExpectation | None = None
    nsys_kernel_time_ns: int = 0
    nsys_kernel_count: int = 0
    #: How many times this NVTX range occurred in the nsys trace. In nsys mode every decode
    #: step is annotated, so this is the number of steps; ncu profiles exactly one.
    nsys_instances: int = 0
    #: Wall-versus-busy breakdown from the nsys timeline; see NsysReport.phase_occupancy.
    occupancy: dict | None = None
    truncated: bool = False
    launch_cap: int = 0

    @property
    def label(self) -> str:
        return {"prefill": "Prefill", "decode_step": "Decode (one token)"}.get(
            self.scope, self.scope
        )

    def occupancy_verdict(self) -> str | None:
        """What the busy fraction says about where this phase's time actually goes."""
        if not self.occupancy:
            return None
        busy = self.occupancy["busy_pct"]
        idle_ms = self.occupancy["gpu_idle_ns"] / self.occupancy["instances"] / 1e6
        if busy >= 90:
            return (
                f"The GPU was executing for {busy:.0f}% of this phase, so its wall time is "
                "genuinely device work and a bandwidth or compute figure describes it well."
            )
        return (
            f"The GPU was idle for {100 - busy:.0f}% of this phase "
            f"({idle_ms:.2f} ms per instance). That time is not bandwidth and not compute -- "
            "it is the gap between kernels, from launch latency or a host synchronisation. "
            "Bandwidth utilisation below describes only the busy fraction, so treat it as a "
            "statement about the kernels rather than about the phase."
        )

    @property
    def nsys_kernels_per_instance(self) -> float | None:
        """nsys kernel count normalised to one instance of the range.

        Necessary because the two tools deliberately scope differently: the nsys pass
        annotates every decode step so the timeline is complete, while ncu profiles exactly
        one. Comparing the raw totals would report a mismatch on every healthy run.
        """
        if not self.nsys_instances:
            return None
        return self.nsys_kernel_count / self.nsys_instances

    def cross_check_kernel_counts(self) -> str | None:
        """Compare, per range instance, the kernel counts nsys and ncu saw.

        A genuine disagreement means the two tools are not measuring the same work, which
        would invalidate putting their figures side by side. Truncation is called out
        separately, because it has a different cause and a different fix.
        """
        per_instance = self.nsys_kernels_per_instance
        ncu_count = self.hierarchy.kernel_count
        if not per_instance or not ncu_count:
            return None

        if self.truncated:
            return (
                f"PARTIAL: ncu stopped at its {self.launch_cap}-launch cap, but nsys counted "
                f"{per_instance:,.0f} kernels in one instance of this phase. The memory "
                f"totals above cover only the first {ncu_count:,} kernels and understate the "
                "phase. Raise --max-kernels and re-run."
            )

        ratio = ncu_count / per_instance
        if 0.7 <= ratio <= 1.4:
            return (
                f"nsys and ncu agree on scope: {per_instance:,.0f} kernels per instance "
                f"versus {ncu_count:,} profiled"
            )
        return (
            f"SCOPE MISMATCH: nsys counted {per_instance:,.0f} kernels per instance of this "
            f"phase but ncu profiled {ncu_count:,}. The two tools are not measuring the same "
            "work, so their figures should not be compared directly."
        )

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "label": self.label,
            "hierarchy": self.hierarchy.to_dict(),
            "expectation": self.expectation.to_dict() if self.expectation else None,
            "nsys_kernel_time_ns": self.nsys_kernel_time_ns,
            "nsys_kernel_count": self.nsys_kernel_count,
            "occupancy": self.occupancy,
            "occupancy_verdict": self.occupancy_verdict(),
            "cross_check": self.cross_check_kernel_counts(),
            "top_kernels_by_dram_bytes": self.kernel_rows[:15],
            "limits": {
                "verdict": self.limits.verdict(),
                "stall_profile_pct": self.limits.weighted_stall_profile(),
                "kernels": [k.to_row() for k in self.limits.top(12)],
            } if self.limits and self.limits.available else None,
        }


@dataclass
class FootprintAnalysis:
    """GPU memory footprint, from the three sources that can see it on this part."""

    torch_peak_allocated: int | None = None
    torch_peak_reserved: int | None = None
    cuda_peak_used: int | None = None
    nsys_peak_outstanding: int | None = None
    model_weight_bytes_resident: int | None = None
    kv_cache_bytes: int | None = None
    host_available_delta: int | None = None
    #: True on a part where the GPU and the host share one physical pool, as on GB10. It
    #: changes what ``cuda_peak_used`` means, so it must be known before that figure is
    #: described in words.
    unified_memory: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def cuda_used_label(self) -> str:
        """What ``cudaMemGetInfo`` is actually reporting on this part."""
        if self.unified_memory:
            return "whole unified pool in use, by every process on the machine"
        return "everything CUDA has taken on this device"

    def agreement(self) -> str | None:
        """Compare the driver's view against the allocator's.

        On a discrete GPU the gap between these two is this process's non-torch CUDA memory:
        the context and the library workspaces. On a unified-memory part it is not, and
        saying so would be a fabrication dressed as a finding.

        ``cudaMemGetInfo`` returns free and total for the **entire** LPDDR5X pool that the
        Grace CPU and the GPU share, so ``total - free`` counts every host process, the page
        cache and the kernel's own allocations alongside anything CUDA took. A 1.6 GB process
        routinely sits inside a 25 GB system-wide figure, and attributing that 23 GB to
        cuBLAS workspaces would be off by more than an order of magnitude. The honest bound
        on this process's non-torch CUDA memory is the host-visible delta measured across
        weight loading, not this subtraction.
        """
        if not (self.cuda_peak_used and self.torch_peak_reserved):
            return None

        if self.unified_memory:
            return (
                f"The driver reports {self.cuda_peak_used / 1e9:,.1f} GB in use against "
                f"{self.torch_peak_reserved / 1e9:,.2f} GB reserved by the torch allocator, "
                "but on this part those are not comparable quantities: cudaMemGetInfo covers "
                "the whole unified pool, so its figure includes every other process and the "
                "page cache. It is a system-wide reading, not this run's footprint, and the "
                "difference must not be read as CUDA context or library workspace."
            )

        gap = self.cuda_peak_used - self.torch_peak_reserved
        if gap <= 0:
            return (
                "the torch allocator reserved at least as much as the driver reports in use "
                "-- consistent, with caching accounting for the difference"
            )
        return (
            f"{gap / 1e6:,.0f} MB is held outside the torch allocator (CUDA context, cuBLAS "
            f"and cuDNN workspaces): {gap / self.cuda_peak_used:.0%} of the peak footprint. "
            "A tensor-level accounting alone would miss this."
        )

    def process_footprint_bytes(self) -> int | None:
        """The best available bound on what *this run* took from the pool.

        Preferred over the driver figure on a unified-memory part, where the driver cannot
        distinguish this process from the rest of the machine. The host-visible delta across
        weight loading is measured against this process's own activity, so it attributes.
        """
        candidates = [
            v for v in (self.host_available_delta, self.torch_peak_reserved) if v and v > 0
        ]
        return max(candidates) if candidates else None

    def to_dict(self) -> dict:
        return {
            "torch_peak_allocated_bytes": self.torch_peak_allocated,
            "torch_peak_reserved_bytes": self.torch_peak_reserved,
            "cuda_peak_used_bytes": self.cuda_peak_used,
            "cuda_peak_used_is_system_wide": self.unified_memory,
            "nsys_peak_outstanding_bytes": self.nsys_peak_outstanding,
            "model_weight_bytes_resident": self.model_weight_bytes_resident,
            "kv_cache_bytes": self.kv_cache_bytes,
            "host_available_delta_bytes": self.host_available_delta,
            "process_footprint_bytes": self.process_footprint_bytes(),
            "agreement": self.agreement(),
            "notes": self.notes,
        }


@dataclass
class TimingAnalysis:
    """Unprofiled timing. The only performance numbers in the report."""

    prefill_seconds: float | None = None
    prefill_iqr: float | None = None
    decode_seconds: float | None = None
    decode_iqr: float | None = None
    decode_step_ms: float | None = None
    decode_step_iqr_ms: float | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    batch_size: int = 1
    available: bool = False
    #: How many measured repeats each median is drawn from. Reported alongside the spread,
    #: because an IQR over three points and one over twenty are not the same claim.
    sample_count: int = 0

    def phase_seconds(self, scope: str) -> float | None:
        """Unprofiled wall time for a phase, keyed by the ncu scope name."""
        if scope == "prefill":
            return self.prefill_seconds
        if scope == "decode_step":
            return self.decode_step_ms / 1000.0 if self.decode_step_ms else None
        return None

    @property
    def prefill_tokens_per_second(self) -> float | None:
        if not self.prefill_seconds or not self.prompt_tokens:
            return None
        return self.prompt_tokens * self.batch_size / self.prefill_seconds

    @property
    def decode_tokens_per_second(self) -> float | None:
        if not self.decode_seconds or not self.generated_tokens:
            return None
        return self.generated_tokens * self.batch_size / self.decode_seconds

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "sample_count": self.sample_count,
            "prefill_seconds": self.prefill_seconds,
            "prefill_iqr_seconds": self.prefill_iqr,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_seconds": self.decode_seconds,
            "decode_iqr_seconds": self.decode_iqr,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "decode_step_ms": self.decode_step_ms,
            "decode_step_iqr_ms": self.decode_step_iqr_ms,
        }


@dataclass
class RunAnalysis:
    """The complete, rendered-ready analysis of one run."""

    run_id: str = ""
    root: Path = Path()
    manifest: dict = field(default_factory=dict)
    model: ModelConfig | None = None
    workload: WorkloadConfig | None = None
    platform: PlatformProfile | None = None

    calibration: CalibrationResult | None = None
    timing: TimingAnalysis = field(default_factory=TimingAnalysis)
    footprint: FootprintAnalysis = field(default_factory=FootprintAnalysis)
    phases: dict[str, PhaseAnalysis] = field(default_factory=dict)
    nsys: NsysReport | None = None

    warnings: list[str] = field(default_factory=list)

    @property
    def calibration_passed(self) -> bool:
        return bool(self.calibration and self.calibration.passed)

    @property
    def dram_ceiling_gbps(self) -> float | None:
        """Measured LPDDR5X ceiling, for the roofline and for utilisation percentages."""
        if self.calibration and self.calibration.peak_dram_bandwidth_gbps:
            return self.calibration.peak_dram_bandwidth_gbps
        return None

    def bandwidth_at_real_latency_gbps(self, scope: str) -> float | None:
        """Phase DRAM bytes divided by the *unprofiled* wall time for that phase.

        This is an **upper bound on the achieved rate, not a measurement of it**, and the
        distinction is load-bearing enough to be worth stating precisely.

        The denominator is honest: ncu serialises kernels and replays each one, so summed
        kernel duration from the ncu pass is longer than the phase really takes, and dividing
        by it understates the rate. Substituting the baseline's wall time fixes that.

        The numerator is not. Those bytes were counted under ``--cache-control all``, which
        flushes L2 before every replay pass, so no kernel can inherit anything its
        predecessor left in the 25 MB L2. A real un-profiled phase does inherit some -- which
        is exactly why the reported L2 hit rate is documented as a lower bound. Cold bytes
        over warm time therefore reads *high*, and how high depends on how much reuse the
        flush destroyed.

        Both phases are computed, deliberately. Applying this to prefill on a healthy run
        exceeds the machine's own measured ceiling -- which is impossible, and is the
        clearest available evidence that the cold-byte numerator is inflated. Reporting only
        the phase where the answer happens to land below 100% would hide the very check that
        bounds the method's error.
        """
        phase = self.phase(scope)
        if phase is None or not self.timing.available:
            return None
        dram_bytes = phase.hierarchy.dram_bytes
        seconds = self.timing.phase_seconds(scope)
        if not dram_bytes or not seconds:
            return None
        return dram_bytes / seconds / 1e9

    def bandwidth_utilisation_pct(self, scope: str) -> float | None:
        """Real-latency bandwidth for a phase as a share of the measured ceiling."""
        achieved = self.bandwidth_at_real_latency_gbps(scope)
        ceiling = self.dram_ceiling_gbps
        if not achieved or not ceiling:
            return None
        return 100.0 * achieved / ceiling

    def bandwidth_bound_exceeded(self) -> list[str]:
        """Phases whose upper-bound rate exceeds the machine's measured ceiling.

        A rate above the ceiling is not a fast phase -- it is proof that the numerator
        overcounts, because no phase can move bytes faster than the memory system can carry
        them. Surfacing it keeps the cold-cache caveat above from being decorative.
        """
        exceeded = []
        for scope in self.phases:
            utilisation = self.bandwidth_utilisation_pct(scope)
            if utilisation is not None and utilisation > 100.0:
                exceeded.append(scope)
        return exceeded

    def decode_bandwidth_at_real_latency_gbps(self) -> float | None:
        """Convenience wrapper for the decode phase. See the general method above."""
        return self.bandwidth_at_real_latency_gbps("decode_step")

    def decode_bandwidth_utilisation_pct(self) -> float | None:
        return self.bandwidth_utilisation_pct("decode_step")

    def phase(self, scope: str) -> PhaseAnalysis | None:
        return self.phases.get(scope)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "model": self.model.to_dict() if self.model else None,
            "workload": self.workload.to_dict() if self.workload else None,
            "calibration_passed": self.calibration_passed,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "dram_ceiling_gbps": self.dram_ceiling_gbps,
            "decode_bandwidth_at_real_latency_gbps": (
                self.decode_bandwidth_at_real_latency_gbps()
            ),
            "decode_bandwidth_utilisation_pct": self.decode_bandwidth_utilisation_pct(),
            "bandwidth_at_real_latency_gbps": {
                scope: self.bandwidth_at_real_latency_gbps(scope) for scope in self.phases
            },
            "bandwidth_utilisation_pct": {
                scope: self.bandwidth_utilisation_pct(scope) for scope in self.phases
            },
            "bandwidth_bound_exceeded": self.bandwidth_bound_exceeded(),
            "timing": self.timing.to_dict(),
            "footprint": self.footprint.to_dict(),
            "phases": {k: v.to_dict() for k, v in self.phases.items()},
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _exists(path: Path) -> bool:
    """``Path.exists`` that cannot raise.

    The stdlib version propagates ``PermissionError`` when a parent directory is unreadable,
    which is exactly what happens to a run directory copied off the machine that produced it:
    the manifest still names the original absolute path, and merely *testing* for it takes
    down the whole analysis.
    """
    try:
        return path.exists()
    except OSError:
        return False


def _artifact(root: Path, recorded: str | None, *subdirs: str) -> Path | None:
    """Locate an artefact named in the manifest, tolerating a relocated run directory.

    Paths are recorded absolute, which is right for provenance and wrong for portability --
    a run archived, copied to a workstation, or analysed inside a container has a valid
    manifest full of paths that no longer resolve. The recorded location wins when it is
    readable; otherwise the file is looked up by name under this run directory, which is
    where the harness put it in the first place.
    """
    if not recorded:
        return None
    candidate = Path(recorded)
    if _exists(candidate):
        return candidate
    for subdir in subdirs or ("",):
        relocated = (root / subdir / candidate.name) if subdir else (root / candidate.name)
        if _exists(relocated):
            return relocated
    return None


def assemble(run_dir: str | Path) -> RunAnalysis:
    """Read a completed run directory and build its :class:`RunAnalysis`."""
    root = Path(run_dir)
    manifest = _read_json(root / "manifest.json")

    analysis = RunAnalysis(
        run_id=manifest.get("run_id", root.name),
        root=root,
        manifest=manifest,
        warnings=list(manifest.get("warnings", [])),
    )

    config_data = manifest.get("config", {})
    if config_data:
        try:
            run_config = RunConfig.load(root / "run_config.json")
            analysis.model = run_config.model
            analysis.workload = run_config.workload
        except Exception:                                        # noqa: BLE001
            analysis.warnings.append("run_config.json missing or unreadable")

    if manifest.get("platform"):
        try:
            raw = manifest["platform"]
            profile = PlatformProfile(**{
                k: v for k, v in raw.items() if k not in ("gpu", "tools", "permissions")
            })
            from ..platform import GpuInfo, PermissionInfo, ToolInfo

            profile.gpu = GpuInfo(**raw.get("gpu", {}))
            profile.tools = ToolInfo(**raw.get("tools", {}))
            profile.permissions = PermissionInfo(**raw.get("permissions", {}))
            analysis.platform = profile
        except Exception:                                        # noqa: BLE001
            pass

    _load_calibration(analysis, root, manifest)
    _load_timing_and_footprint(analysis, root, manifest)
    _load_nsys(analysis, manifest)
    _load_phases(analysis, manifest)
    _cross_check(analysis)

    return analysis


def _load_calibration(analysis: RunAnalysis, root: Path, manifest: dict) -> None:
    data = manifest.get("calibration") or _read_json(root / "metrics" / "calibration.json")
    if not data:
        analysis.warnings.append(
            "No calibration result. The DRAM figures rest on the L2 sysmem-aperture "
            "derivation, which was not verified for this run."
        )
        return

    from ..calibration import ByteAccountingResult, SweepPoint

    result = CalibrationResult(
        l2_cache_bytes=data.get("l2_cache_bytes", 0),
        peak_l2_bandwidth_gbps=data.get("peak_l2_bandwidth_gbps", 0.0),
        peak_dram_bandwidth_gbps=data.get("peak_dram_bandwidth_gbps", 0.0),
        peak_compute_gflops=data.get("peak_compute_gflops", 0.0),
        compute_peak_detail=data.get("compute_peak_detail", {}) or {},
        notes=list(data.get("notes", [])),
    )
    gate = data.get("byte_accounting", {})
    result.byte_accounting = ByteAccountingResult(**{
        k: v for k, v in gate.items()
        if k in ByteAccountingResult.__dataclass_fields__
    })
    result.sweep = [
        SweepPoint(**{k: v for k, v in p.items() if k in SweepPoint.__dataclass_fields__})
        for p in data.get("sweep", [])
    ]
    analysis.calibration = result

    if not result.passed:
        analysis.warnings.append(
            "Calibration gate did not pass -- treat every DRAM figure below as unverified."
        )


def _load_timing_and_footprint(analysis: RunAnalysis, root: Path, manifest: dict) -> None:
    baseline = manifest.get("baseline") or {}
    result_path = _artifact(root, baseline.get("result_path"), "metrics")
    payload = _read_json(result_path) if result_path else {}

    if not payload.get("ok"):
        analysis.timing.available = False
        analysis.footprint.notes.append(
            "no unprofiled baseline run, so no untainted timing is available"
        )
        return

    analysis.timing.available = True
    result = payload.get("result", {})
    workload = analysis.workload

    timings = result.get("timings", [])

    def _phase_stats(name: str) -> tuple[float | None, float | None, int]:
        return stats.median_iqr([t["seconds"] for t in timings if t.get("phase") == name])

    analysis.timing.prefill_seconds, analysis.timing.prefill_iqr, samples = _phase_stats(
        "prefill"
    )
    analysis.timing.decode_seconds, analysis.timing.decode_iqr, _ = _phase_stats("decode")
    analysis.timing.sample_count = samples
    analysis.timing.prompt_tokens = result.get("prompt_tokens", 0)
    analysis.timing.generated_tokens = result.get("generated_tokens", 0)
    analysis.timing.batch_size = result.get("batch_size", 1) or 1
    if analysis.timing.decode_seconds and analysis.timing.generated_tokens:
        analysis.timing.decode_step_ms = (
            analysis.timing.decode_seconds / analysis.timing.generated_tokens * 1000
        )
        # The workload records a per-step timing of its own, so the spread across repeats is
        # available for this row too rather than being left blank.
        _, step_iqr, _ = _phase_stats("decode_step")
        analysis.timing.decode_step_iqr_ms = step_iqr * 1000 if step_iqr else None

    # ---- footprint ----
    allocator = payload.get("allocator_stats", {})
    # Prefer the run-level high-water marks. The ``.peak`` fields are peaks since the last
    # per-phase counter reset, which on a completed run means "since the final decode phase
    # began" -- and the run's real peak is normally set during prefill, several resets back.
    analysis.footprint.torch_peak_allocated = (
        allocator.get("run_peak_allocated_bytes")
        or allocator.get("allocated_bytes.all.peak")
    )
    analysis.footprint.torch_peak_reserved = (
        allocator.get("run_peak_reserved_bytes")
        or allocator.get("reserved_bytes.all.peak")
    )
    analysis.footprint.cuda_peak_used = payload.get("memory_peak_cuda_used")
    analysis.footprint.kv_cache_bytes = result.get("kv_cache_bytes")
    if analysis.platform is not None:
        analysis.footprint.unified_memory = analysis.platform.gpu.unified_memory

    backend = payload.get("backend", {})
    resident = backend.get("parameter_bytes_resident")
    if resident:
        analysis.footprint.model_weight_bytes_resident = resident + (
            backend.get("buffer_bytes_resident") or 0
        )

    load_mem = payload.get("load_memory") or {}
    if load_mem.get("host_delta_bytes"):
        analysis.footprint.host_available_delta = load_mem["host_delta_bytes"]
        analysis.footprint.notes.append(
            "On this unified-memory part, loading weights draws down the same pool the host "
            "allocates from -- the host-visible delta is the honest cost of resident weights."
        )

    if allocator.get("num_alloc_retries"):
        analysis.footprint.notes.append(
            f"{allocator['num_alloc_retries']} allocator retries occurred: the caching "
            "allocator had to release cached blocks to satisfy a request, which costs time "
            "and indicates memory pressure."
        )
    if allocator.get("num_ooms"):
        analysis.warnings.append(
            f"{allocator['num_ooms']} out-of-memory events during the run"
        )


def _load_nsys(analysis: RunAnalysis, manifest: dict) -> None:
    nsys = manifest.get("nsys") or {}
    sqlite_path = _artifact(analysis.root, nsys.get("sqlite_path"), "raw")
    if sqlite_path is None:
        analysis.warnings.append(
            "no nsys database, so there is no timeline, allocation history or clock record"
        )
        return

    report = parse_sqlite(sqlite_path)
    analysis.nsys = report
    analysis.warnings.extend(report.warnings)

    peak = report.peak_allocated_bytes()
    if peak:
        analysis.footprint.nsys_peak_outstanding = peak

    clocks = report.clock_summary()
    if clocks.get("gpc_clock_spread_pct", 0) > 15:
        analysis.warnings.append(
            f"GPC clock varied by {clocks['gpc_clock_spread_pct']:.0f}% during the traced "
            f"region ({clocks['gpc_clock_mhz_min']:.0f}-{clocks['gpc_clock_mhz_max']:.0f} "
            "MHz). Timing comparisons against other runs are correspondingly softer."
        )


def _load_phases(analysis: RunAnalysis, manifest: dict) -> None:
    ncu = manifest.get("ncu") or {}
    collections = ncu.get("collections", [])

    by_scope_tier: dict[tuple[str, int], NcuReport] = {}
    for record in collections:
        if not record.get("ok"):
            continue
        path = _artifact(analysis.root, record.get("csv_path"), "metrics")
        if path is None:
            continue
        scope = record.get("scope", "")
        tier = int(record.get("tier", 1))
        by_scope_tier[(scope, tier)] = parse_csv(path, scope=scope, tier=tier)

    truncation = {
        (r.get("scope", ""), int(r.get("tier", 1))): (
            bool(r.get("truncated")), int(r.get("launch_cap") or 0)
        )
        for r in collections
    }

    for (scope, tier), report in sorted(by_scope_tier.items()):
        if tier != 1:
            continue
        hierarchy = summarize(report, scope=scope)
        truncated, cap = truncation.get((scope, 1), (False, 0))
        phase = PhaseAnalysis(
            scope=scope,
            hierarchy=hierarchy,
            kernel_rows=kernel_traffic_rows(report),
            ncu_report=report,
            deep_dive=by_scope_tier.get((scope, 2)),
            truncated=truncated,
            launch_cap=cap,
        )
        phase.limits = analyse_deep_dive(phase.deep_dive, scope, tier1=phase.ncu_report)
        analysis.phases[scope] = phase
        analysis.warnings.extend(
            w for w in hierarchy.warnings if w not in analysis.warnings
        )

    if not analysis.phases:
        analysis.warnings.append(
            "no Nsight Compute collections succeeded, so there are no memory-hierarchy "
            "figures in this run"
        )


def _cross_check(analysis: RunAnalysis) -> None:
    """Compare independent measurements against each other and against theory."""
    # nsys kernel counts per phase, for the scope cross-check.
    if analysis.nsys:
        instances = analysis.nsys.phase_instance_counts()
        for nvtx_phase, scope in PHASE_TO_SCOPE.items():
            phase = analysis.phases.get(scope)
            if phase is None:
                continue
            kernels = analysis.nsys.kernels_in_phase(nvtx_phase)
            phase.nsys_kernel_count = len(kernels)
            phase.nsys_kernel_time_ns = sum(k.duration_ns for k in kernels)
            phase.nsys_instances = instances.get(nvtx_phase, 0)
            phase.occupancy = analysis.nsys.phase_occupancy(nvtx_phase)

            # A phase that spends a large share of its wall time with an idle GPU is not
            # described by any bandwidth or compute figure, and every such figure in this
            # report divides by kernel time. Say so once, loudly, rather than letting the
            # tables imply the phase was device-bound throughout.
            if phase.occupancy and phase.occupancy["busy_pct"] < 75:
                idle_ms = (
                    phase.occupancy["gpu_idle_ns"] / phase.occupancy["instances"] / 1e6
                )
                analysis.warnings.append(
                    f"{phase.label}: the GPU was idle for "
                    f"{100 - phase.occupancy['busy_pct']:.0f}% of this phase "
                    f"({idle_ms:.2f} ms per instance, over "
                    f"{phase.nsys_kernels_per_instance or 0:,.0f} kernel launches). That time "
                    "is launch latency or host synchronisation, not memory and not compute, "
                    "so this phase is bounded by dispatch rather than by the hardware. "
                    "Bandwidth utilisation figures describe only the busy fraction."
                )

    for scope, phase in analysis.phases.items():
        if phase.truncated:
            analysis.warnings.append(
                f"PARTIAL DATA for {scope}: ncu stopped at its {phase.launch_cap}-launch cap, "
                "so every memory total for that phase is a prefix of the real traffic, not a "
                "sample of it. Raise --max-kernels and re-run before quoting these figures."
            )

    # The physics check: a decode step must read the weights plus the KV cache. Skipped when
    # the collection was truncated, since the comparison would only measure the truncation.
    decode = analysis.phases.get("decode_step")
    if decode and decode.truncated:
        analysis.warnings.append(
            "Skipping the expected-versus-measured decode check: the collection was "
            "truncated, so the comparison would report the launch cap rather than the model."
        )
    elif decode and analysis.model:
        context_len = None
        kv_bytes = analysis.footprint.kv_cache_bytes or 0
        if not kv_bytes and analysis.workload:
            context_len = analysis.workload.prompt_tokens + analysis.workload.generate_tokens
            kv_bytes = analysis.model.kv_cache_bytes(
                context_len, analysis.workload.batch_size
            )
        # Which weight figure this uses changes the verdict, so it is recorded rather than
        # left implicit. The resident measurement counts each tensor once, so a checkpoint
        # that stores embed_tokens and lm_head separately but ties them at load is counted
        # correctly -- a decode step reads that matrix once, for the output projection. The
        # on-disk fallback counts both copies and would overstate the expectation by a whole
        # embedding matrix, which on a large-vocabulary model is easily a quarter of the
        # total and enough to move the ratio across a verdict boundary.
        resident = analysis.footprint.model_weight_bytes_resident
        if resident:
            weight_bytes, source = resident, "resident"
        else:
            weight_bytes, source = analysis.model.estimated_weight_bytes(), "on-disk"

        # A mixture of experts reads only the experts a token routes to, so the resident
        # total is the wrong yardstick -- it would predict roughly num_experts/top_k times
        # the traffic a step actually moves and turn a correctly-working model into a
        # scoping alarm. Scale to the active subset, and keep the total so the report can
        # show both.
        total_weight_bytes = 0
        activation = analysis.model.expert_activation_ratio
        if activation is not None:
            total_weight_bytes = weight_bytes
            dense_part = analysis.model.embedding_matrix_bytes()
            weight_bytes = int(
                dense_part + max(0, weight_bytes - dense_part) * activation
            )
            source += " (routed-active)"
            if analysis.model.tie_word_embeddings:
                analysis.warnings.append(
                    "The decode expectation falls back to the checkpoint's on-disk size "
                    "because the resident parameter count is unavailable. This model ties "
                    "its input and output embeddings, so if the checkpoint stores both "
                    "copies the expectation overstates the weights by one embedding matrix "
                    f"(~{analysis.model.embedding_matrix_bytes() / 1e6:,.0f} MB) and the "
                    "measured/expected ratio below reads correspondingly low."
                )

        decode.expectation = check_decode_expectation(
            decode.hierarchy, weight_bytes, kv_bytes, context_len,
            weight_bytes_source=source,
            total_weight_bytes=total_weight_bytes,
            expert_activation_ratio=activation,
        )

    # A rate above the machine's own measured ceiling is impossible, so when the
    # real-latency substitution produces one it has bounded its own error for us.
    exceeded = analysis.bandwidth_bound_exceeded()
    if exceeded:
        labels = ", ".join(
            analysis.phases[s].label for s in exceeded if s in analysis.phases
        )
        analysis.warnings.append(
            f"Bytes-over-real-latency exceeds the measured LPDDR5X ceiling for: {labels}. "
            "That rate is not achievable, which confirms the ncu byte counts read high "
            "against un-profiled wall time -- they were collected with L2 flushed before "
            "every replay pass, so they exclude the cross-kernel reuse a real run gets. "
            "Treat every real-latency bandwidth figure in this report as an upper bound, "
            "including the ones that land below 100%."
        )

    # Footprint sources: flag a genuine disagreement, not a routine one.
    footprint = analysis.footprint
    # On a unified-memory part the driver figure is system-wide, so comparing it against a
    # per-process allocation timeline explains nothing -- the gap is dominated by other
    # processes, not by tracing scope. Compare against the allocator instead, which is the
    # only other view scoped to this run.
    reference = (
        footprint.torch_peak_reserved if footprint.unified_memory else footprint.cuda_peak_used
    )
    if reference and footprint.nsys_peak_outstanding:
        ratio = footprint.nsys_peak_outstanding / reference
        if ratio < 0.5 or ratio > 2.0:
            footprint.notes.append(
                f"The nsys allocation timeline peaks at "
                f"{footprint.nsys_peak_outstanding / 1e9:.2f} GB against "
                f"{reference / 1e9:.2f} GB "
                + ("reserved by the torch allocator" if footprint.unified_memory
                   else "reported in use by the CUDA driver")
                + ". nsys only counts allocations made inside its capture range, so a gap "
                "this large usually means the weights were loaded before tracing began -- "
                "which is intended."
            )

    # L2 effectiveness, the headline finding for a decode step on this part.
    if decode:
        l2 = decode.hierarchy.level(Level.L2)
        if l2.hit_rate_pct is not None and analysis.calibration:
            speedup = (
                analysis.calibration.peak_l2_bandwidth_gbps
                / analysis.calibration.peak_dram_bandwidth_gbps
                if analysis.calibration.peak_dram_bandwidth_gbps else 0
            )
            if l2.hit_rate_pct < 20 and speedup > 2:
                analysis.warnings.append(
                    f"Decode L2 hit rate is {l2.hit_rate_pct:.1f}%, while L2-resident data "
                    f"moves {speedup:.1f}x faster than LPDDR5X on this machine. The decode "
                    "step is reading essentially everything from main memory."
                )


def write_metric_tables(analysis: RunAnalysis) -> dict[str, Path]:
    """Write the tidy CSV/JSON extracts alongside the raw profiler artefacts."""
    from ..runners.base import write_csv, write_json

    out_dir = analysis.root / "metrics"
    written: dict[str, Path] = {}

    for scope, phase in analysis.phases.items():
        if phase.kernel_rows:
            written[f"kernels_{scope}"] = write_csv(
                out_dir / f"kernels_{scope}.csv", phase.kernel_rows
            )

    hierarchy_rows = []
    for scope, phase in analysis.phases.items():
        tokens = 1 if scope == "decode_step" else (
            analysis.workload.prompt_tokens if analysis.workload else 0
        )
        per_token = phase.hierarchy.bytes_per_token(tokens) if tokens else {}
        for level, summary in phase.hierarchy.levels.items():
            row = {"scope": scope, **summary.to_dict()}
            row["bytes_per_token"] = per_token.get(level.value)
            hierarchy_rows.append(row)
    if hierarchy_rows:
        written["hierarchy"] = write_csv(out_dir / "hierarchy.csv", hierarchy_rows)

    if analysis.nsys:
        timeline = analysis.nsys.allocation_timeline()
        if timeline:
            written["alloc_events"] = write_csv(out_dir / "alloc_events.csv", timeline)
        kernel_rows = [k.to_row() for k in analysis.nsys.kernels]
        if kernel_rows:
            written["nsys_kernels"] = write_csv(out_dir / "nsys_kernels.csv", kernel_rows)

    written["summary"] = write_json(out_dir / "summary.json", analysis.to_dict())
    return written
