"""Compare per-phase kernel sequences (name, grid, block) across runs, from their nsys timelines.

ncu application replay is only valid if every process launches the same kernels in the same
order, so a forced-routing build has to show (a) one distinct sequence per phase within a run,
across instances whose routing differs (disjoint rotates every pass), and (b) the same sequence
in every run of the same workload. Comparing a patched natural run with an unpatched one also
shows that the patch leaves the natural graph alone.

    ~/envs/nsbench/bin/python experiments/llamacpp-backend/compare_kernel_sequences.py RUN_DIR [RUN_DIR ...]

The first run is the reference. Exit 1 if any check fails.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nsight_bench.parsers.nsys_parse import parse_sqlite  # noqa: E402

PHASES = ("nsbench.prefill", "nsbench.decode_step")


def sequences(run_dir: Path) -> dict[str, list[tuple]]:
    """Per phase: the distinct per-instance kernel sequences, most common first."""
    report = parse_sqlite(run_dir / "raw" / "timeline.sqlite")
    counts = report.phase_instance_counts()
    out = {}
    for phase in PHASES:
        kernels = sorted(report.kernels_in_phase(phase), key=lambda k: k.start_ns)
        n = counts.get(phase, 0)
        if not n or len(kernels) % n:
            out[phase] = []
            print(f"  {run_dir.name} {phase}: {len(kernels)} kernels do not split into {n} instances")
            continue
        per = len(kernels) // n
        seqs = Counter(
            tuple((k.short_name, k.grid, k.block) for k in kernels[i * per:(i + 1) * per])
            for i in range(n)
        )
        out[phase] = [s for s, _ in seqs.most_common()]
        print(f"  {run_dir.name} {phase}: {n} instances x {per} kernels, "
              f"{len(seqs)} distinct sequence(s)")
    return out


def main() -> int:
    runs = [Path(a) for a in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 2
    ok = True
    ref = sequences(runs[0])
    for phase in PHASES:
        if len(ref[phase]) != 1:
            ok = False
    for run in runs[1:]:
        other = sequences(run)
        for phase in PHASES:
            if len(other[phase]) != 1:
                ok = False
                continue
            if not ref[phase]:
                continue
            a, b = ref[phase][0], other[phase][0]
            if a == b:
                print(f"  {phase}: identical to reference ({len(a)} kernels)")
            else:
                ok = False
                diff = Counter(a) - Counter(b), Counter(b) - Counter(a)
                print(f"  {phase}: DIFFERS from reference ({len(a)} vs {len(b)} kernels); "
                      f"only in reference: {dict(list(diff[0].items())[:5])}; "
                      f"only here: {dict(list(diff[1].items())[:5])}")
    print("ALL SEQUENCES CONSISTENT" if ok else "SEQUENCE CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
