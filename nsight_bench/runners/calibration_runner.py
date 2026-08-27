"""Orchestrate the calibration microbenchmarks.

Two collections, deliberately different in kind:

* The **byte-accounting gate** runs under Nsight Compute, because it needs the L2 aperture
  counters that only ncu exposes.
* The **bandwidth sweep** runs with no profiler attached, because it measures achievable
  bandwidth and ncu's kernel replay would time a serialised re-execution instead.

Running the gate before every benchmark is not ceremony. It is the difference between a
report that says "this decode step moved 3.1 GB" and one that says "this decode step moved
3.1 GB, and the same derivation reproduced a known 537 MB to within 0.1% ten seconds ago".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import metrics as M
from ..calibration import (
    CalibrationResult,
    SweepPoint,
    parse_byte_accounting,
)
from ..instrumentation.nvtx import Phase, ncu_filter
from ..platform import PlatformProfile
from .base import RunPaths, run_command


class CalibrationRunner:
    """Runs and evaluates the calibration microbenchmarks."""

    name = "calibration"

    def __init__(
        self,
        paths: RunPaths,
        profile: PlatformProfile,
        python_executable: str | None = None,
        megabytes: int = 256,
        clock_control: str = "base",
    ) -> None:
        self.paths = paths
        self.profile = profile
        self.python = python_executable or sys.executable
        self.megabytes = megabytes
        # Matched to the benchmark's own ncu setting rather than hardcoded. The gate itself
        # only reads sector counters, which do not care about clocks -- but a gate run under
        # a different clock policy than the collections it vouches for is one more difference
        # to reason about for no benefit.
        self.clock_control = clock_control
        self.repo_root = Path(__file__).resolve().parents[2]

    def run(self, include_sweep: bool = True) -> CalibrationResult:
        result = CalibrationResult(l2_cache_bytes=self.profile.gpu.l2_cache_bytes)
        result.byte_accounting = self._run_byte_accounting()

        if include_sweep:
            result.sweep, compute = self._run_sweep()
            result.classify_sweep()
            result.compute_peak_detail = compute
            result.peak_compute_gflops = compute.get("gflops", 0.0)

        self._add_notes(result)
        return result

    # ---- byte accounting --------------------------------------------------------------

    def _run_byte_accounting(self) -> "object":
        expected = 2 * 4 * ((self.megabytes * 1024 * 1024) // 4)
        stem = self.paths.raw / "calibration_stream"
        available = set(self.profile.metric_available) or None

        argv = [
            "ncu",
            "--target-processes", "all",
            "--nvtx",
            # Scoped to the NVTX range so the setup allocation kernels are excluded and the
            # gate measures exactly the kernel whose byte count we know.
            "--nvtx-include", ncu_filter(Phase.CALIBRATE.range_name),
            "--kernel-name-base", "function",
            "--metrics", M.tier1_metrics_arg(available),
            "--clock-control", self.clock_control,
            "--cache-control", "all",
            "--print-summary", "none",
            "--force-overwrite",
            "-o", str(stem),
            self.python, "-m", "nsight_bench.calibration",
            "--task", "stream", "--megabytes", str(self.megabytes),
        ]

        command = run_command(
            argv, log_dir=self.paths.logs, log_name="calibration_stream",
            timeout=1800, cwd=self.repo_root,
        )
        report_path = stem.with_suffix(".ncu-rep")

        from ..calibration import ByteAccountingResult

        if not report_path.exists():
            return ByteAccountingResult(
                expected_bytes=expected,
                error=f"ncu produced no report (exit {command.returncode}). "
                      f"{command.stderr_tail[-400:]}",
            )

        csv_path = self.paths.metrics / "calibration_stream.csv"
        export = run_command(
            ["ncu", "--import", str(report_path), "--csv", "--page", "raw"],
            log_dir=self.paths.logs, log_name="calibration_export", timeout=600,
        )
        stdout_path = Path(export.stdout_path)
        if not export.ok or not stdout_path.exists():
            return ByteAccountingResult(
                expected_bytes=expected, error="CSV export of the calibration report failed"
            )
        csv_path.write_text(stdout_path.read_text(errors="replace"))

        return parse_byte_accounting(csv_path, expected)

    # ---- bandwidth sweep --------------------------------------------------------------

    def _run_sweep(self) -> tuple[list[SweepPoint], dict]:
        out_path = self.paths.metrics / "calibration_sweep.json"
        command = run_command(
            [self.python, "-m", "nsight_bench.calibration",
             "--task", "sweep", "--out", str(out_path)],
            log_dir=self.paths.logs, log_name="calibration_sweep",
            timeout=1800, cwd=self.repo_root,
        )
        if not command.ok or not out_path.exists():
            return [], {}
        try:
            payload = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            return [], {}
        return (
            [SweepPoint(**point) for point in payload.get("sweep", [])],
            payload.get("compute", {}),
        )

    # ---- interpretation ---------------------------------------------------------------

    def _add_notes(self, result: CalibrationResult) -> None:
        gate = result.byte_accounting
        if gate.passed:
            result.notes.append(
                "Byte-accounting gate PASSED: the L2 sysmem-aperture derivation reproduced a "
                f"known {gate.expected_bytes / 1e6:,.1f} MB of traffic to within "
                f"{abs(gate.relative_error or 0):.2%}. DRAM figures in this run are trustworthy."
            )
        else:
            result.notes.append(
                "Byte-accounting gate FAILED. Every DRAM figure derived from the L2 sysmem "
                "aperture in this run should be treated as unverified. " + gate.summary()
            )

        knee = result.knee_mib()
        l2_mib = result.l2_cache_bytes / (1024 * 1024) if result.l2_cache_bytes else 0
        if knee and l2_mib:
            result.notes.append(
                f"Bandwidth knee measured at a {knee:.0f} MiB working set against "
                f"{l2_mib:.1f} MiB of L2 -- consistent with L2 capacity governing reuse."
                if knee <= l2_mib * 3 else
                f"Bandwidth knee measured at {knee:.0f} MiB, well beyond the {l2_mib:.1f} MiB "
                "L2; reuse is being limited by something other than L2 capacity."
            )
        elif result.sweep:
            result.notes.append(
                "No clear bandwidth knee in the sweep -- the working sets may all sit on the "
                "same side of L2, or the streaming kernel is not capacity-limited here."
            )

        if result.peak_dram_bandwidth_gbps:
            result.notes.append(
                f"Peak measured LPDDR5X streaming bandwidth "
                f"{result.peak_dram_bandwidth_gbps:,.1f} GB/s (working set beyond L2). This "
                "is the memory-bound ceiling used on the roofline -- an achievable figure, "
                "not a datasheet one."
            )
        if result.peak_compute_gflops:
            detail = result.compute_peak_detail
            note = (
                f"Peak measured dense {detail.get('dtype', 'bf16')} GEMM throughput "
                f"{result.peak_compute_gflops / 1000:,.1f} TFLOP/s "
                f"({detail.get('matrix_size', '?')}^2). This is the roofline's compute ceiling."
            )
            ridge = result.ridge_point
            if ridge:
                note += (
                    f" The ridge point sits at {ridge:,.0f} FLOP/byte -- any kernel below that "
                    "intensity is memory-bound no matter how well it is written, and LLM "
                    "decode sits far below it."
                )
            result.notes.append(note)

        if result.peak_l2_bandwidth_gbps:
            ratio = (
                result.peak_l2_bandwidth_gbps / result.peak_dram_bandwidth_gbps
                if result.peak_dram_bandwidth_gbps else 0
            )
            result.notes.append(
                f"Peak L2-resident bandwidth {result.peak_l2_bandwidth_gbps:,.1f} GB/s"
                + (f", {ratio:.1f}x the LPDDR5X rate. " if ratio else ". ")
                + "That multiple is the prize for keeping a working set inside the 25 MB L2, "
                "and the reason L2 hit rate is the headline metric for decode."
            )
