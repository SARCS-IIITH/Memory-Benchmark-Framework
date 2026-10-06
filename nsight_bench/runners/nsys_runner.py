"""Nsight Systems collection: the timeline and the allocation history.

What nsys contributes on this platform is narrower than the marketing suggests, and knowing
which parts are real saves a lot of confusion:

* **CUDA/NVTX trace** -- kernel and memory-op timeline, projected onto the phase ranges. This
  is the source of the prefill/decode split and of the kernel ranking that tier 2 of the ncu
  pass deep-dives.
* **``CUDA_GPU_MEMORY_USAGE_EVENTS``** -- every allocation and free with size and timestamp.
  On GB10 this is the closest thing to a GPU memory footprint timeline, because NVML reports
  nothing at all on this part.
* **Unified-memory page faults** -- meaningful here in a way they are not on a discrete GPU,
  since CPU and GPU share one coherent pool.
* **GPU metric sampling** -- clocks, engine activity and warp occupancy. Worth having for
  throttle detection. It does **not** include memory bandwidth: the ``gb20b`` metric set has
  no bandwidth rows, so all traffic numbers come from the ncu pass instead.

CPU sampling is disabled automatically when the kernel's ``perf_event_paranoid`` forbids it,
which it does on this host. GPU tracing is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import RunConfig
from ..gpu_state import GpuStateRecorder
from ..platform import PlatformProfile
from .base import CommandResult, RunPaths, run_command


#: Stock nsys metric set -> an extended set in configs/nsys/ that adds ``lts__t_sectors``, so
#: the timeline carries sampled L2 traffic. Only chips where the extension was verified.
L2_METRIC_SETS: dict[str, str] = {"gb20b": "gb20b_l2.config"}


class NsysRunner:
    """Builds and executes the Nsight Systems collection."""

    name = "nsys"

    def __init__(
        self,
        run_config: RunConfig,
        paths: RunPaths,
        profile: PlatformProfile,
        python_executable: str | None = None,
    ) -> None:
        self.run_config = run_config
        self.paths = paths
        self.profile = profile
        self.python = python_executable or sys.executable

    # ---- command construction ---------------------------------------------------------

    def gpu_metric_set(self) -> str:
        """The ``--gpu-metrics-set`` value: the L2-extended file when enabled and known."""
        stock = self.profile.nsys_gpu_metric_set
        extended = L2_METRIC_SETS.get(stock)
        if self.run_config.profile.nsys.sample_l2_traffic and extended:
            path = Path(__file__).resolve().parents[2] / "configs" / "nsys" / extended
            if path.exists():
                return f"file:{path}"
        return stock

    def build_command(self, config_path: Path, result_path: Path, report_stem: Path) -> list[str]:
        cfg = self.run_config.profile.nsys

        argv = [
            "nsys", "profile",
            "--force-overwrite=true",
            "-o", str(report_stem),
            f"--trace={cfg.trace}",
            f"--cuda-memory-usage={_bool(cfg.cuda_memory_usage)}",
            f"--cuda-um-cpu-page-faults={_bool(cfg.cuda_um_cpu_page_faults)}",
            f"--cuda-um-gpu-page-faults={_bool(cfg.cuda_um_gpu_page_faults)}",
            f"--cuda-graph-trace={cfg.cuda_graph_trace}",
        ]

        # Without these nsys spends its startup attempting perf_event_open and then warns
        # for every thread. Driven by the platform probe rather than hardcoded, so the same
        # config works on a host where sampling is permitted.
        argv += self.profile.permissions.nsys_sampling_flags

        if cfg.python_sampling and self.profile.permissions.nsys_cpu_sampling_available:
            argv.append("--python-sampling=true")

        # GPU hardware counter sampling needs the metric set that matches the chip; passing
        # a set for the wrong chip is a hard error, so it is skipped when unmapped.
        if cfg.gpu_metrics and self.profile.nsys_gpu_metric_set:
            argv += [
                "--gpu-metrics-devices=0",
                f"--gpu-metrics-set={self.gpu_metric_set()}",
                f"--gpu-metrics-frequency={cfg.gpu_metrics_frequency}",
            ]

        # Restrict the trace to the region between cudaProfilerStart and Stop, which the
        # workload opens after warmup. Otherwise weight loading -- by far the longest part
        # of the run -- dominates the timeline and the steady state is a sliver.
        if cfg.capture_range and cfg.capture_range != "none":
            argv += [
                f"--capture-range={cfg.capture_range}",
                f"--capture-range-end={cfg.capture_range_end}",
            ]

        argv += [
            self.python, "-m", "nsight_bench.worker",
            "--config", str(config_path),
            "--mode", "nsys",
            "--out", str(result_path),
            "--memory-sample-hz", "20",
        ]
        return argv

    # ---- execution --------------------------------------------------------------------

    def run(self, config_path: Path) -> dict:
        cfg = self.run_config.profile.nsys
        report_stem = self.paths.raw / "timeline"
        result_path = self.paths.metrics / "nsys_worker_result.json"
        argv = self.build_command(config_path, result_path, report_stem)

        with GpuStateRecorder() as gpu:
            command: CommandResult = run_command(
                argv,
                log_dir=self.paths.logs,
                log_name="nsys",
                timeout=cfg.timeout_s,
                cwd=Path(__file__).resolve().parents[2],
            )

        report_path = report_stem.with_suffix(".nsys-rep")
        record = {
            "runner": self.name,
            "command": command.to_dict(),
            "report_path": str(report_path) if report_path.exists() else None,
            "worker_result_path": str(result_path) if result_path.exists() else None,
            "gpu_state": gpu.window.to_dict() if gpu.window else None,
            "gpu_metric_set": (self.gpu_metric_set() if cfg.gpu_metrics else None) or None,
            "cpu_sampling": self.profile.permissions.nsys_cpu_sampling_available,
        }

        if not report_path.exists():
            record["error"] = (
                "nsys produced no report. Common causes: the target process crashed before "
                "cudaProfilerStart was reached, or the capture range never opened.\n"
                + command.stderr_tail
            )
            return record

        record["sqlite_path"] = self.export_sqlite(report_path)
        return record

    def export_sqlite(self, report_path: Path) -> str | None:
        """Convert the .nsys-rep into a SQLite database.

        Everything downstream queries the database rather than shelling out to ``nsys stats``
        per report type: one export, then arbitrary joins across kernels, NVTX ranges and
        allocation events. ``nsys stats`` can only emit its fixed set of summaries.
        """
        sqlite_path = report_path.with_suffix(".sqlite")
        command = run_command(
            ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
             "--output", str(sqlite_path), str(report_path)],
            log_dir=self.paths.logs,
            log_name="nsys_export",
            timeout=1800,
        )
        return str(sqlite_path) if sqlite_path.exists() and command.ok else None


def _bool(value: bool) -> str:
    return "true" if value else "false"
