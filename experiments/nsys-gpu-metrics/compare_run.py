"""Compare nsys-sampled L2 traffic against ncu tier 1 for one run.

The run must have been made with the l2_sectors metric set (platform_profile_l2.json) and with
ncu tier 1 on. Per phase it prints:

* nsys L2 GB per instance: sampled ``lts__t_sectors`` x 32 B, summed over the GPU windows of
  the phase and divided by the number of instances;
* ncu L2 GB and ncu DRAM GB, scaled to one instance by the ratio of nsys kernels per instance
  to ncu kernels profiled.

ncu runs with caches flushed before every pass, so its DRAM figure is an upper bound on real
DRAM traffic. If L2 is close to ncu's DRAM, it is at least as close to real DRAM.

Usage: python compare_run.py runs/<run-dir>
"""

import sqlite3
import sys
from pathlib import Path

from nsight_bench.analysis.assemble import PHASE_TO_SCOPE, assemble
from nsight_bench.metrics import Level

MERGE_GAP_NS = 1_000_000  # kernels of one phase closer than this form one GPU window


def gpu_windows(kernels):
    """Merge one phase's kernel intervals into contiguous GPU-busy windows."""
    spans = sorted((k.start_ns, k.end_ns) for k in kernels)
    windows = []
    for start, end in spans:
        if windows and start - windows[-1][1] <= MERGE_GAP_NS:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    return windows


def main(run_dir: str) -> None:
    analysis = assemble(run_dir)
    nsys = analysis.nsys
    db = sqlite3.connect(Path(run_dir) / "raw" / "timeline.sqlite")
    ids = [i for i, n in db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS")
           if n == "L2 Sectors [Sectors]"]
    if not ids:
        sys.exit("no 'L2 Sectors' metric in this run -- was it made with platform_profile_l2.json?")
    samples = db.execute(
        "SELECT timestamp, value FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp", (ids[0],)
    ).fetchall()
    period = (samples[-1][0] - samples[0][0]) / max(1, len(samples) - 1)
    print(f"{len(samples):,} L2 samples, {period / 1e3:.0f} us apart, "
          f"{sum(v for _, v in samples) * 32 / 1e9:.2f} GB over the whole capture\n")

    instances = nsys.phase_instance_counts()
    weights = analysis.footprint.model_weight_bytes_resident
    for nvtx_phase, scope in PHASE_TO_SCOPE.items():
        kernels = nsys.kernels_in_phase(nvtx_phase)
        n = instances.get(nvtx_phase, 0)
        if not kernels or not n:
            continue
        windows = gpu_windows(kernels)
        # A sample is the counter delta over the interval ending at its timestamp.
        l2 = 0.0
        for start, end in windows:
            l2 += sum(v for t, v in samples if start < t <= end + period)
        nsys_l2 = l2 * 32 / n

        print(f"== {scope}: {n} instances, {len(kernels) / n:,.0f} kernels each")
        print(f"   nsys sampled L2   {nsys_l2 / 1e9:8.3f} GB per instance")
        phase = analysis.phases.get(scope)
        if phase is None or not phase.traffic_collected or not phase.hierarchy.kernel_count:
            print("   (no ncu tier 1 for this phase)")
            continue
        scale = (len(kernels) / n) / phase.hierarchy.kernel_count
        ncu_l2 = (phase.hierarchy.level(Level.L2).bytes_total or 0) * scale
        ncu_dram = (phase.hierarchy.dram_bytes or 0) * scale
        print(f"   ncu L2            {ncu_l2 / 1e9:8.3f} GB per instance "
              f"(ncu saw {phase.hierarchy.kernel_count:,} kernels, scaled x{scale:.2f})")
        print(f"   ncu DRAM (cold)   {ncu_dram / 1e9:8.3f} GB per instance")
        if ncu_l2:
            print(f"   nsys L2 / ncu L2      {nsys_l2 / ncu_l2:5.2f}   <- do the tools agree?")
        if ncu_dram:
            print(f"   ncu L2 / ncu DRAM     {ncu_l2 / ncu_dram:5.2f}   <- how far L2 overstates DRAM")
            print(f"   nsys L2 / ncu DRAM    {nsys_l2 / ncu_dram:5.2f}")
        if weights and scope == "decode_step":
            print(f"   resident weights  {weights / 1e9:8.3f} GB (decode must read ~all of it)")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
