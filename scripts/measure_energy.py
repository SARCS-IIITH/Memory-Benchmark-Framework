"""Measure GPU energy for an existing run folder and write <run>/metrics/energy.json.

Loads the run's own run_config.json (same model, workload, backend), runs the phases again
unprofiled with nsight_bench/energy.py, and adds energy per DRAM byte from the run's ncu tier-1
bytes. Read the "scope" field: GPU rail only, a lower bound on system energy.

    ~/envs/nsbench/bin/python scripts/measure_energy.py RUN_DIR [RUN_DIR ...] [--force]

Forced-routing runs need the patched llama.cpp; the script re-runs itself with
~/envs/nsbench-moeroute/bin/python when the workload's routing is not natural. Skips runs that
already have energy.json (unless --force) and waits for an idle GPU before each run (the GPU is
shared). One model load per run (~52 GB), about 1-2 minutes each (the 16k run longer).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOE_PY = Path.home() / "envs" / "nsbench-moeroute" / "bin" / "python"


def gpu_busy() -> bool:
    """Another process has a GPU context. This process's own (left by an earlier run's
    measurement in the same invocation) does not count, or it would wait for itself."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    return any(pid.strip() and int(pid) != os.getpid() for pid in out)


def measure_one(run_dir: Path) -> int:
    from nsight_bench.backends import hf_transformers, llamacpp, trtllm  # noqa: F401  (registers)
    from nsight_bench.backends.base import get_backend
    from nsight_bench.config import RunConfig
    from nsight_bench.energy import measure

    run = RunConfig.load(run_dir / "run_config.json")
    if run.workload.routing != "natural":
        from nsight_bench.backends.llamacpp import forced_routing_supported
        if not forced_routing_supported():
            # Compare environments, not interpreters: both venvs' python resolve to the same
            # system binary.
            if Path(sys.prefix).resolve() != MOE_PY.parents[1].resolve() and MOE_PY.exists():
                return subprocess.run([str(MOE_PY), __file__, str(run_dir), "--force"]).returncode
            print(f"{run_dir.name}: routing={run.workload.routing} needs the patched llama.cpp")
            return 1
    while gpu_busy():
        print("GPU busy; waiting 60 s", flush=True)
        time.sleep(60)

    backend = get_backend(run.backend)(run.model, run.workload)
    backend.load()
    try:
        result = measure(backend, run.workload)
    finally:
        backend.teardown()

    summary = json.loads((run_dir / "metrics" / "summary.json").read_text())
    for phase in ("prefill", "decode_step"):
        dram = ((((summary.get("phases") or {}).get(phase) or {}).get("hierarchy") or {})
                .get("levels") or {}).get("dram", {}).get("bytes_total")
        r = result[phase]
        r["dram_bytes_per_instance"] = dram
        if dram:
            r["j_per_dram_gb"] = r["energy_j_per_instance"] / (dram / 1e9)
            r["pj_per_dram_bit"] = r["energy_j_per_instance"] / (dram * 8) * 1e12
    result["python"] = sys.executable
    result["routing"] = run.workload.routing
    out = run_dir / "metrics" / "energy.json"
    out.write_text(json.dumps(result, indent=2))
    d, p = result["decode_step"], result["prefill"]
    print(f"{run_dir.name}: decode {d['energy_j_per_token']:.3f} J/token ({d['avg_power_w']} W, "
          f"{d.get('pj_per_dram_bit', 0):.1f} pJ/DRAM bit), prefill {p['energy_j_per_instance']:.2f} J "
          f"({p['avg_power_w']} W), idle {result['idle']['avg_power_w']} W -> {out}", flush=True)
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    status = 0
    for a in args:
        run_dir = Path(a).resolve()
        if not (run_dir / "metrics" / "summary.json").exists():
            print(f"{a}: no metrics/summary.json, skipped")
            continue
        if (run_dir / "metrics" / "energy.json").exists() and not force:
            print(f"{run_dir.name}: energy.json exists, skipped")
            continue
        try:
            status |= measure_one(run_dir)
        except Exception as e:                                   # noqa: BLE001
            print(f"{run_dir.name}: FAILED: {e!r}", flush=True)
            status = 1
    return status


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
