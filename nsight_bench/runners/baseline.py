"""Unprofiled timing pass.

This is the only run in the harness whose latency and throughput numbers mean anything.
Both profilers perturb execution, and Nsight Compute does so severely -- it serialises kernel
launches and replays each one several times to collect a full metric set, so a decode step
measured under ncu can be an order of magnitude slower than the same step running normally.

Reporting profiler-run wall clock as "performance" is a common and quietly wrong thing to
do. Keeping the baseline as a separate collection makes the distinction structural rather
than a footnote.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import RunConfig
from ..gpu_state import GpuStateRecorder
from .base import CommandResult, RunPaths, run_command


class BaselineRunner:
    """Runs the workload with no profiler attached."""

    name = "baseline"

    def __init__(
        self,
        run_config: RunConfig,
        paths: RunPaths,
        python_executable: str | None = None,
    ) -> None:
        self.run_config = run_config
        self.paths = paths
        self.python = python_executable or sys.executable

    def build_command(self, config_path: Path, result_path: Path) -> list[str]:
        return [
            self.python, "-m", "nsight_bench.worker",
            "--config", str(config_path),
            "--mode", "baseline",
            "--out", str(result_path),
            # Matched to the nsys pass rather than raised above it. This is the run whose
            # timing has to be untainted, so it should carry the *least* instrumentation, not
            # the most -- and each sample makes a cudaMemGetInfo call plus two /proc reads
            # from a background thread, contending for the GIL with the loop being timed.
            # Sampling faster here was buying allocation-peak resolution that the allocator's
            # own exact high-water counters already provide for free.
            "--memory-sample-hz", "20",
            "--allocator-history",
        ]

    def run(self, config_path: Path) -> dict:
        result_path = self.paths.metrics / "baseline_result.json"
        argv = self.build_command(config_path, result_path)

        with GpuStateRecorder() as gpu:
            command: CommandResult = run_command(
                argv,
                log_dir=self.paths.logs,
                log_name="baseline",
                timeout=self.run_config.profile.nsys.timeout_s,
                cwd=Path(__file__).resolve().parents[2],
            )

        record = {
            "runner": self.name,
            "command": command.to_dict(),
            "result_path": str(result_path) if result_path.exists() else None,
            "gpu_state": gpu.window.to_dict() if gpu.window else None,
        }
        if not command.ok:
            record["error"] = (
                "Baseline run failed. Without it the report has no untainted timing "
                "reference, so profiled durations must not be quoted as performance.\n"
                + command.stderr_tail
            )
        return record
