"""GPU telemetry sampling around a run.

Clock and thermal state is provenance, not a metric. A decode step that reports lower
bandwidth than the run before it may have been throttled rather than changed, and without a
before/after record there is no way to tell those apart.

On GB10 the useful subset of NVML is narrower than usual: ``memory.used`` and
``memory.total`` return ``N/A`` because there is no discrete VRAM to report, and
``clocks.current.memory`` is likewise absent. Power, temperature, SM clock and utilisation do
work, and those are what this module collects. Anything that reports ``N/A`` is stored as
``None`` rather than being silently coerced to zero -- a missing reading and a reading of
zero mean very different things.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

#: nvidia-smi query fields that return real values on GB10, verified on this host.
QUERY_FIELDS = (
    "clocks.current.sm",
    "clocks.current.graphics",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "pstate",
)

#: Fields that exist on discrete GPUs but return N/A here. Queried anyway so the run record
#: shows they were checked and found unavailable, rather than leaving a silent gap.
KNOWN_UNAVAILABLE = ("memory.used", "memory.total", "clocks.current.memory")


@dataclass
class GpuState:
    """One telemetry reading."""

    timestamp: str = ""
    sm_clock_mhz: float | None = None
    graphics_clock_mhz: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    utilization_gpu_pct: float | None = None
    utilization_memory_pct: float | None = None
    pstate: str = ""
    throttle_reasons: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse(value: str) -> float | None:
    value = value.strip()
    if not value or value in ("N/A", "[N/A]", "Not Supported"):
        return None
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return None


def sample() -> GpuState:
    """Take one telemetry reading. Never raises."""
    state = GpuState(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= len(QUERY_FIELDS):
                state.sm_clock_mhz = _parse(parts[0])
                state.graphics_clock_mhz = _parse(parts[1])
                state.temperature_c = _parse(parts[2])
                state.power_w = _parse(parts[3])
                state.utilization_gpu_pct = _parse(parts[4])
                state.utilization_memory_pct = _parse(parts[5])
                state.pstate = parts[6]
    except Exception:                                            # noqa: BLE001
        pass

    state.throttle_reasons = _throttle_reasons()
    return state


def _throttle_reasons() -> list[str]:
    """Active clock-throttle reasons.

    A run that hit a power cap mid-collection produces numbers that are real but not
    representative, and this is the only way to find that out after the fact.
    """
    reasons: list[str] = []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-q", "-d", "PERFORMANCE"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if proc.returncode != 0:
            return reasons
        in_block = False
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Clocks Event Reasons") or stripped.startswith(
                "Clocks Throttle Reasons"
            ):
                in_block = "Counters" not in stripped
                continue
            if in_block:
                if not stripped or ":" not in stripped:
                    if stripped and not stripped[0].isspace():
                        in_block = False
                    continue
                name, _, value = stripped.partition(":")
                if value.strip() == "Active":
                    reasons.append(name.strip())
    except Exception:                                            # noqa: BLE001
        pass
    return reasons


@dataclass
class GpuStateWindow:
    """Telemetry captured either side of a collection."""

    before: GpuState
    after: GpuState

    @property
    def temperature_rise_c(self) -> float | None:
        if self.before.temperature_c is None or self.after.temperature_c is None:
            return None
        return self.after.temperature_c - self.before.temperature_c

    @property
    def clock_drop_mhz(self) -> float | None:
        """SM clock change across the run. A large negative value indicates throttling."""
        if self.before.sm_clock_mhz is None or self.after.sm_clock_mhz is None:
            return None
        return self.after.sm_clock_mhz - self.before.sm_clock_mhz

    def warnings(self) -> list[str]:
        """Conditions that make the run's numbers less trustworthy."""
        issues: list[str] = []
        if self.after.throttle_reasons:
            issues.append(
                "GPU was throttling at the end of the run: "
                + ", ".join(self.after.throttle_reasons)
            )
        rise = self.temperature_rise_c
        if rise is not None and rise > 20:
            issues.append(
                f"GPU temperature rose {rise:.0f} C during the run; later iterations may "
                "have run at lower clocks than earlier ones"
            )
        return issues

    def to_dict(self) -> dict:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "temperature_rise_c": self.temperature_rise_c,
            "sm_clock_delta_mhz": self.clock_drop_mhz,
            "warnings": self.warnings(),
        }


class GpuStateRecorder:
    """Context manager recording telemetry either side of a block."""

    def __init__(self) -> None:
        self.window: GpuStateWindow | None = None
        self._before: GpuState | None = None

    def __enter__(self) -> GpuStateRecorder:
        self._before = sample()
        return self

    def __exit__(self, *_exc) -> None:
        assert self._before is not None
        self.window = GpuStateWindow(before=self._before, after=sample())
