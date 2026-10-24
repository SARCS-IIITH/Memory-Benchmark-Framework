"""GPU validation of the forced-routing llama.cpp build, from finished runs. Gate for the
forced-routing queue (scripts/run_forced_routing_after_queue.sh starts it only on exit 0).

    ~/envs/nsbench/bin/python experiments/llamacpp-backend/check_forced_routing_gpu.py \\
        --reference RUN  --natural RUN  --fixed RUN RUN  --disjoint RUN RUN  [--out FILE.md]

* reference: the unpatched natural run of the same workload (the sweep's moe-decode-natural-b1).
* natural:   the same workload on the patched build, with ncu tier 1.
* fixed / disjoint: two nsys-only runs each on the patched build.

Checks:
1. The patched build was used where it should be, with the routing the workload asked for.
2. Patch off = no change: the patched natural run has the reference's kernel sequences and
   its tier-1 DRAM bytes within 1% per phase.
3. Replay safety: each forced mode has one kernel sequence per phase within a run (disjoint
   rotates experts every pass) and the same one in both runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_kernel_sequences import PHASES, sequences  # noqa: E402

lines: list[str] = []
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    line = f"- [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else "")
    print(line)
    lines.append(line)
    if not ok:
        failures.append(name)


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def backend_info(run: Path) -> dict:
    return load(run / "metrics" / "baseline_result.json").get("backend") or {}


def dram(run: Path, phase: str):
    s = load(run / "metrics" / "summary.json")
    levels = ((((s.get("phases") or {}).get(phase) or {}).get("hierarchy") or {}).get("levels") or {})
    return (levels.get("dram") or {}).get("bytes_total")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--natural", type=Path, required=True)
    ap.add_argument("--fixed", type=Path, nargs=2, required=True)
    ap.add_argument("--disjoint", type=Path, nargs=2, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    lines.append("## 1. Build and routing used")
    ref_info = backend_info(a.reference)
    check("reference ran unpatched llama.cpp", ref_info.get("forced_routing_patch") in (False, None),
          f"forced_routing_patch={ref_info.get('forced_routing_patch')}")
    for mode, runs in (("natural", [a.natural]), ("fixed", a.fixed), ("disjoint", a.disjoint)):
        for run in runs:
            info = backend_info(run)
            check(f"{run.name}: patched build, routing={mode}",
                  info.get("forced_routing_patch") is True and info.get("routing") == mode,
                  f"patch={info.get('forced_routing_patch')} routing={info.get('routing')}")

    lines.append("\n## 2. Patch off = no change (natural, patched vs unpatched)")
    ref_seq, nat_seq = sequences(a.reference), sequences(a.natural)
    for phase in PHASES:
        same = bool(ref_seq[phase]) and len(nat_seq[phase]) == 1 and nat_seq[phase][0] == ref_seq[phase][0]
        check(f"{phase}: same kernel sequence", same,
              f"{len(ref_seq[phase][0]) if ref_seq[phase] else '?'} kernels")
    for phase in ("prefill", "decode_step"):
        r, n = dram(a.reference, phase), dram(a.natural, phase)
        ok = bool(r and n) and abs(n / r - 1) < 0.01
        check(f"{phase}: tier-1 DRAM within 1%", ok,
              f"{(r or 0) / 1e9:.3f} vs {(n or 0) / 1e9:.3f} GB" + (f" ({n / r:.4f}x)" if r and n else ""))

    for mode, (r1, r2) in (("fixed", a.fixed), ("disjoint", a.disjoint)):
        lines.append(f"\n## 3. {mode}: replay safety")
        s1, s2 = sequences(r1), sequences(r2)
        for phase in PHASES:
            one = len(s1[phase]) == 1 and len(s2[phase]) == 1
            check(f"{phase}: one kernel sequence per run", one,
                  f"{len(s1[phase])} and {len(s2[phase])} distinct")
            check(f"{phase}: identical in both runs", one and s1[phase][0] == s2[phase][0],
                  f"{len(s1[phase][0]) if s1[phase] else '?'} kernels"
                  + (f", natural has {len(ref_seq[phase][0])}" if ref_seq[phase] else ""))

    verdict = "ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"
    print(verdict)
    if a.out:
        a.out.write_text("# Forced-routing build: GPU validation\n\n" + "\n".join(lines)
                         + f"\n\n**{verdict}**\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
