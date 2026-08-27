"""Run one complete benchmark: calibrate, measure, profile, analyse, report.

The order of the four collections is not arbitrary.

1. **Calibration** first, because if the L2 sysmem-aperture derivation cannot reproduce a
   known byte count on this machine right now, nothing measured afterwards can be trusted.
   The gate result is stamped into the manifest so a reader knows which regime the numbers
   came from.
2. **Baseline** second, with no profiler attached. This is the only honest timing, and it
   runs before the profilers have touched the GPU.
3. **Nsight Systems** third. It is cheap, and its kernel ranking tells us what the phases
   actually spend their time on.
4. **Nsight Compute** last, because it is by far the most expensive and benefits from
   everything already known.

Each collection runs in its own subprocess. A model that OOMs or segfaults takes down one
collection, not the run -- the others still produce results, and the manifest records what
failed and why.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .gpu_state import sample as sample_gpu
from .platform import PlatformProfile
from .runners.base import RunPaths, model_provenance, utc_stamp, write_json
from .runners.baseline import BaselineRunner
from .runners.calibration_runner import CalibrationRunner
from .runners.ncu_runner import NcuRunner
from .runners.nsys_runner import NsysRunner


@dataclass
class RunRecord:
    """The manifest: everything needed to reproduce and audit one run."""

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    root: str = ""

    config: dict = field(default_factory=dict)
    platform: dict = field(default_factory=dict)
    model_provenance: dict = field(default_factory=dict)

    calibration: dict | None = None
    baseline: dict | None = None
    nsys: dict | None = None
    ncu: dict | None = None

    gpu_state_start: dict = field(default_factory=dict)
    gpu_state_end: dict = field(default_factory=dict)

    stages_ok: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(self.stages_ok.values()) if self.stages_ok else False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ok"] = self.ok
        return data


class BenchmarkOrchestrator:
    """Executes all collections for one :class:`RunConfig`."""

    def __init__(
        self,
        run_config: RunConfig,
        profile: PlatformProfile,
        python_executable: str | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        self.run_config = run_config
        self.profile = profile
        self.python = python_executable or profile.python_executable
        self.output_root = Path(output_root or run_config.output_root)

    def run(
        self,
        skip_calibration: bool = False,
        skip_baseline: bool = False,
        skip_nsys: bool = False,
        skip_ncu: bool = False,
        calibration_sweep: bool = True,
    ) -> RunRecord:
        stamp = utc_stamp()
        run_id = self.run_config.run_id(stamp)
        paths = RunPaths(self.output_root / run_id).create()

        record = RunRecord(
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            root=str(paths.root),
            config=self.run_config.to_dict(),
            platform=self.profile.to_dict(),
            gpu_state_start=sample_gpu().to_dict(),
        )

        # The config is written to disk because the worker subprocesses read it -- they are
        # separate processes and cannot be handed a Python object.
        config_path = paths.root / "run_config.json"
        config_path.write_text(json.dumps(self.run_config.to_dict(), indent=2, default=str))

        record.model_provenance = model_provenance(
            self.run_config.model.path, self.run_config.model.safetensors_files
        )

        self._preflight_warnings(record)

        # ---- 1. calibration ----
        if not skip_calibration:
            calibration = CalibrationRunner(
                paths, self.profile, self.python,
                clock_control=self.run_config.profile.ncu.clock_control,
            ).run(include_sweep=calibration_sweep)
            record.calibration = calibration.to_dict()
            record.stages_ok["calibration"] = calibration.passed
            write_json(paths.metrics / "calibration.json", calibration.to_dict())
            if not calibration.passed:
                record.warnings.append(
                    "CALIBRATION GATE FAILED -- DRAM figures in this run are unverified. "
                    + calibration.byte_accounting.summary()
                )

        # ---- 2. baseline (unprofiled timing) ----
        if not skip_baseline and self.run_config.profile.baseline:
            baseline = BaselineRunner(self.run_config, paths, self.python).run(config_path)
            record.baseline = baseline
            record.stages_ok["baseline"] = bool(baseline["command"]["ok"])
            if not record.stages_ok["baseline"]:
                record.warnings.append(
                    "Baseline failed, so this run has no untainted timing reference. "
                    "Profiled durations must not be quoted as performance."
                )

        # ---- 3. nsight systems ----
        if not skip_nsys and self.run_config.profile.nsys.enabled:
            nsys = NsysRunner(self.run_config, paths, self.profile, self.python).run(config_path)
            record.nsys = nsys
            record.stages_ok["nsys"] = nsys.get("report_path") is not None
            if nsys.get("error"):
                record.warnings.append(f"nsys: {nsys['error'].splitlines()[0]}")

        # ---- 4. nsight compute ----
        if not skip_ncu and self.run_config.profile.ncu.enabled:
            ncu = NcuRunner(self.run_config, paths, self.profile, self.python).run(config_path)
            record.ncu = ncu
            collections = ncu.get("collections", [])
            record.stages_ok["ncu"] = any(c.get("ok") for c in collections)
            for collection in collections:
                if collection.get("error"):
                    record.warnings.append(
                        f"ncu {collection['scope']} tier {collection['tier']}: "
                        + collection["error"].splitlines()[0]
                    )
            record.warnings.extend(ncu.get("warnings", []))

        record.gpu_state_end = sample_gpu().to_dict()
        record.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        self._collection_state_warnings(record)
        self._thermal_warnings(record)
        write_json(paths.manifest_path, record.to_dict())
        return record

    # ---- checks -----------------------------------------------------------------------

    def _preflight_warnings(self, record: RunRecord) -> None:
        """Surface platform limitations before they show up as puzzling gaps in the report."""
        if not self.profile.permissions.ncu_counters_allowed:
            record.warnings.append(
                "GPU performance counters are not readable by this user; the Nsight Compute "
                "stage will produce nothing. " + self.profile.permissions.ncu_permission_detail
            )
        if not self.profile.availability.ok:
            record.warnings.append(
                "Required metrics are unavailable on this GPU: "
                + ", ".join(self.profile.availability.missing_required())
            )
        unprobed = self.profile.unprobed_metrics()
        if unprobed:
            record.warnings.append(
                f"STALE PLATFORM PROFILE: {len(unprobed)} metric(s) in the registry were "
                "never probed on this machine, so they are filtered out of every ncu "
                "collection and their columns will be blank in the report "
                f"({', '.join(unprobed[:4])}"
                + (", ..." if len(unprobed) > 4 else "")
                + "). Run 'nsbench preflight' to refresh it."
            )
        if not self.profile.nsys_gpu_metric_set:
            record.warnings.append(
                f"No nsys GPU metric set is mapped for chip '{self.profile.gpu.chip}'; "
                "hardware counter sampling is disabled for the timeline."
            )
        if self.run_config.model.quantization:
            record.warnings.append(
                f"Model is quantized ({self.run_config.model.quantization}). Weight bytes are "
                "taken from on-disk size, which includes scales and zero-points, so the "
                "expected-vs-measured decode check accounts for them."
            )

    def _collection_state_warnings(self, record: RunRecord) -> None:
        """Promote each collection's own telemetry window into the run's warnings.

        Every runner already brackets its work with a :class:`GpuStateRecorder` and that
        window computes its own warnings -- throttling, a large temperature rise -- but until
        now nothing read them, so a collection that ran while throttled reported clean unless
        the condition happened to persist to the end of the whole run. Whole-run start/end
        sampling cannot see a throttle that began and ended inside the ncu pass; these
        per-collection windows can.
        """
        windows: list[tuple[str, dict]] = []
        for name in ("baseline", "nsys"):
            stage = getattr(record, name) or {}
            if stage.get("gpu_state"):
                windows.append((name, stage["gpu_state"]))

        # The ncu runner records one window per scope rather than one for the stage.
        for scope, window in ((record.ncu or {}).get("gpu_state") or {}).items():
            if window:
                windows.append((f"ncu {scope}", window))

        for name, window in windows:
            for warning in window.get("warnings") or []:
                message = f"During the {name} collection: {warning}"
                if message not in record.warnings:
                    record.warnings.append(message)

    def _thermal_warnings(self, record: RunRecord) -> None:
        """Flag conditions that make this run less comparable to others."""
        start_temp = record.gpu_state_start.get("temperature_c")
        end_temp = record.gpu_state_end.get("temperature_c")
        if start_temp is not None and end_temp is not None and end_temp - start_temp > 20:
            record.warnings.append(
                f"GPU temperature rose {end_temp - start_temp:.0f} C over the run "
                f"({start_temp:.0f} -> {end_temp:.0f} C). Later collections may have run at "
                "lower clocks than earlier ones; compare against the sampled GPC clock in the "
                "nsys timeline before treating small differences as real."
            )
        throttles = record.gpu_state_end.get("throttle_reasons") or []
        if throttles:
            record.warnings.append(
                "GPU was throttling when the run finished: " + ", ".join(throttles)
            )
