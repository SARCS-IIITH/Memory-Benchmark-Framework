"""GPU energy per phase, from NVML's cumulative energy counter, in an unprofiled pass.

The baseline stage times prefill and decode but records power only as single readings
before and after (gpu_state.py), so it says nothing about the energy of the work. This module
runs the workload's phases again, unprofiled, inside measurement windows long enough for the
counter to resolve them, and reports joules per prefill, per decode step and per token.

What it measures, and what it cannot (GB10, checked 2026-10-13):

* **Scope: the GPU only.** ``nvmlDeviceGetTotalEnergyConsumption`` (mJ, cumulative) and
  ``power.draw`` cover the GPU rail. "GPU Memory Power" and "Module Power" read N/A on GB10,
  so the LPDDR5X, the CPU and the rest of the module are not included. For memory-bound decode
  that leaves out a large share of the system's energy: these are lower bounds on it.
* **Windows.** A 16-token prefill takes ~80 ms, shorter than the counter's update interval
  (measured here and recorded). Each phase is therefore repeated until its window lasts at
  least ``target_s`` seconds and divided by the repetitions. Decode cannot run longer than
  the context allows, so it is measured in segments (prefill outside the window, then the
  workload's decode steps inside it) and the segments are summed.
* **Cross-check.** ``power.draw`` is sampled every ``sample_ms`` in a thread and integrated over
  the same windows; the two figures are reported side by side.
* **Idle.** The GPU draws power with nothing running. Idle power is measured with the model
  loaded and reported; ``above_idle`` energy subtracts it over each window's duration.

The result is a dict (written as ``metrics/energy.json`` by scripts/measure_energy.py).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


class Nvml:
    """The few NVML calls used here, on device 0."""

    def __init__(self) -> None:
        import pynvml

        self._n = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def energy_mj(self) -> int:
        return int(self._n.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def power_w(self) -> float:
        return self._n.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def state(self) -> dict:
        n, h = self._n, self.handle
        out: dict[str, Any] = {}
        for key, fn in (("sm_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                        ("temperature_c", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                        ("power_w", self.power_w)):
            try:
                out[key] = fn()
            except Exception:                                    # noqa: BLE001
                out[key] = None
        return out

    def counter_resolution(self, seconds: float = 2.0) -> dict:
        """How often the energy counter changes and by how much, polled every ~1 ms."""
        changes, last, t_end = [], self.energy_mj(), time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            e = self.energy_mj()
            if e != last:
                changes.append((time.perf_counter(), e - last))
                last = e
            time.sleep(0.001)
        gaps = [b[0] - a[0] for a, b in zip(changes, changes[1:])]
        gaps.sort()
        return {
            "polled_s": seconds,
            "changes": len(changes),
            "median_update_interval_ms": round(gaps[len(gaps) // 2] * 1e3, 2) if gaps else None,
            "median_step_mj": sorted(c[1] for c in changes)[len(changes) // 2] if changes else None,
        }


@dataclass
class _Sampler:
    """power.draw sampled in a thread; integrated over the windows it is told about."""

    nvml: Nvml
    period_s: float
    samples: list[tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> _Sampler:
        def run() -> None:
            while not self._stop.is_set():
                self.samples.append((time.perf_counter(), self.nvml.power_w()))
                time.sleep(self.period_s)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def energy_j(self, windows: list[tuple[float, float]]) -> float:
        """Trapezoid integral of the samples inside each window, summed."""
        total = 0.0
        for t0, t1 in windows:
            pts = [(t, p) for t, p in self.samples if t0 <= t <= t1]
            total += sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
        return total


def _phase(windows: list[tuple[float, float]], e_mj: float, sampler_j: float, instances: int,
           idle_w: float, tokens_per_instance: int) -> dict:
    seconds = sum(t1 - t0 for t0, t1 in windows)
    energy = e_mj / 1e3
    above = energy - idle_w * seconds
    return {
        "instances": instances,
        "window_s": round(seconds, 3),
        "energy_j": round(energy, 3),
        "sampled_power_energy_j": round(sampler_j, 3),
        "counter_vs_sampled": round(energy / sampler_j, 4) if sampler_j else None,
        "avg_power_w": round(energy / seconds, 2) if seconds else None,
        "above_idle_energy_j": round(above, 3),
        "seconds_per_instance": seconds / instances if instances else None,
        "energy_j_per_instance": energy / instances if instances else None,
        "above_idle_j_per_instance": above / instances if instances else None,
        "energy_j_per_token": energy / (instances * tokens_per_instance) if instances else None,
        "above_idle_j_per_token": above / (instances * tokens_per_instance) if instances else None,
        "tokens_per_instance": tokens_per_instance,
    }


def measure(backend, workload, target_s: float = 10.0, idle_s: float = 8.0,
            sample_ms: float = 20.0, max_decode_wall_s: float = 240.0) -> dict:
    """Energy of one prefill and one decode step of ``workload`` on a loaded ``backend``."""
    nvml = Nvml()
    batch = workload.batch_size
    inputs = backend.prepare_inputs()
    steps = max(1, workload.generate_tokens - 1)

    def one_prefill():
        state = backend.prefill(inputs)
        backend.synchronize()
        return state

    def decode(state):
        for _ in range(steps):
            state = backend.decode_step(state)
        backend.synchronize()

    for _ in range(max(1, workload.warmup_iters)):
        decode(one_prefill())

    resolution = nvml.counter_resolution()
    state_before = nvml.state()
    with _Sampler(nvml, sample_ms / 1e3) as sampler:
        # Idle, model resident.
        time.sleep(1.0)
        e0, t0 = nvml.energy_mj(), time.perf_counter()
        time.sleep(idle_s)
        e1, t1 = nvml.energy_mj(), time.perf_counter()
        idle_w = (e1 - e0) / 1e3 / (t1 - t0)
        idle = {"window_s": round(t1 - t0, 3), "energy_j": (e1 - e0) / 1e3,
                "avg_power_w": round(idle_w, 2),
                "sampled_power_energy_j": round(sampler.energy_j([(t0, t1)]), 3)}

        # Prefill: one window, repeated until it lasts target_s (at least 5 repetitions).
        n_prefill = 0
        e0, t0 = nvml.energy_mj(), time.perf_counter()
        while n_prefill < 5 or time.perf_counter() - t0 < target_s:
            one_prefill()
            n_prefill += 1
        e1, t1 = nvml.energy_mj(), time.perf_counter()
        prefill = _phase([(t0, t1)], e1 - e0, sampler.energy_j([(t0, t1)]), n_prefill, idle_w,
                         workload.prompt_tokens * batch)

        # Decode: segments of `steps` steps, the prefill before each one outside the window.
        windows, e_dec, segments, wall0 = [], 0, 0, time.perf_counter()
        e_all0 = nvml.energy_mj()
        while (sum(b - a for a, b in windows) < target_s and segments < 10_000
               and (time.perf_counter() - wall0 < max_decode_wall_s or segments < 3)):
            state = one_prefill()
            e0, t0 = nvml.energy_mj(), time.perf_counter()
            decode(state)
            e1, t1 = nvml.energy_mj(), time.perf_counter()
            windows.append((t0, t1))
            e_dec += e1 - e0
            segments += 1
        e_all1, wall1 = nvml.energy_mj(), time.perf_counter()
        decode_res = _phase(windows, e_dec, sampler.energy_j(windows), segments * steps, idle_w, batch)
        decode_res["segments"] = segments
        decode_res["steps_per_segment"] = steps
        # Edge-free cross-check: one window over every segment (prefills included), minus the
        # prefills' energy. Immune to the counter's ~100 ms steps at each segment's edges.
        sub = (e_all1 - e_all0) / 1e3 - segments * prefill["energy_j_per_instance"]
        decode_res["by_subtraction"] = {
            "window_s": round(wall1 - wall0, 3),
            "energy_j": round(sub, 3),
            "energy_j_per_token": sub / (segments * steps * batch),
            "segments_vs_subtraction": round(e_dec / 1e3 / sub, 4) if sub > 0 else None,
        }
    state_after = nvml.state()

    return {
        "scope": "GPU rail only (NVML total energy counter). GB10 reports GPU-memory and module "
                 "power as N/A, so LPDDR5X, CPU and the rest of the module are NOT included: a "
                 "lower bound on system energy.",
        "method": f"unprofiled; each phase repeated to >= {target_s:.0f} s; decode in segments "
                  "(prefill outside the window); power.draw sampled every "
                  f"{sample_ms:.0f} ms as a cross-check",
        "batch_size": batch,
        "counter_resolution": resolution,
        "idle": idle,
        "prefill": prefill,
        "decode_step": decode_res,
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
