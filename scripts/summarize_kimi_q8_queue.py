"""Summarise the Kimi Q8 llama.cpp queue (scripts/run_kimi_q8_llamacpp_queue.sh) into
<queue dir>/results.md.

Same idea as scripts/summarize_moe_queue.py (the HF FP8 queue), but these runs have ncu tier 1,
so the byte columns are exact DRAM traffic from ncu rather than nsys-sampled L2. Reads
<queue dir>/queue.log for finished runs, then each run's metrics/summary.json,
manifest.json and the run script's /proc/meminfo samples. The queue calls it after every
start and finish; safe to rerun at any time, since it rewrites results.md from scratch.

    ~/envs/nsbench/bin/python scripts/summarize_kimi_q8_queue.py [queue dir]

The queue dir defaults to results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/natural-b1/queue (the natural-routing set),
or its earlier locations when that is the one that exists. After writing the set's table it
also refreshes results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/results.md, the table of all 22 workloads
(scripts/summarize_kimi_q8_all.py).

Findings are written up in docs/10-kimi-llamacpp-moe-sweep.md, not here.
"""

import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from summarize_moe_queue import gb, load

REPO = Path(__file__).resolve().parents[1]
QDIR = next((d for d in (REPO / "results" / "kimi-linear-48b-a3b-q8_0-gguf__llamacpp" / "natural-b1" / "queue",
                         REPO / "runs" / "kimi-q8-llamacpp-sweep" / "queue",
                         REPO / "runs" / "kimi-q8-queue") if d.exists()),
            REPO / "results" / "kimi-linear-48b-a3b-q8_0-gguf__llamacpp" / "natural-b1" / "queue")
LINE = re.compile(r"^(\S+ \S+) (DONE|FAIL) (\S+) exit=(\d+) (\d+)min run=(\S+) log=(\S+)")


def queue_length(lines: list[str]) -> str:
    """Number of workloads in the queue, from its latest QUEUE START line (runs=N)."""
    for line in reversed(lines):
        m = re.search(r"QUEUE START .*runs=(\d+)", line)
        if m:
            return m.group(1)
    return "?"


def level_bytes(phase: dict, level: str):
    return (((phase.get("hierarchy") or {}).get("levels") or {}).get(level) or {}).get("bytes_total")


def peak_memory(run_log: Path) -> str:
    """Peak in-use system memory over the whole run, and over ncu, from the 5-s samples (GiB)."""
    try:
        rows = list(csv.DictReader(run_log.with_suffix(".mem.csv").open()))
    except OSError:
        return "–"
    if not rows:
        return "–"

    def used(r):
        return (int(r["mem_total_mb"]) - int(r["mem_available_mb"])) / 1024

    peak = max(used(r) for r in rows)
    ncu = [used(r) for r in rows if r["stage"] == "ncu"]
    return f"{peak:.1f}" + (f" ({max(ncu):.1f})" if ncu else "")


def stage_minutes(manifest: dict) -> str:
    """Tier-1 ncu wall minutes per phase, from the manifest's per-phase GPU-state window."""
    windows = ((manifest.get("ncu") or {}).get("gpu_state") or {})
    out = []
    for scope in ("prefill", "decode_step"):
        w = windows.get(scope) or {}
        try:
            t0 = datetime.fromisoformat(w["before"]["timestamp"])
            t1 = datetime.fromisoformat(w["after"]["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(f"{'P' if scope == 'prefill' else 'D'} {(t1 - t0).total_seconds() / 60:.0f}")
    return ", ".join(out) or "–"


def summarise_run(name: str, run_dir: Path, run_log: Path) -> tuple[dict, list[str]]:
    summary = load(run_dir / "metrics" / "summary.json")
    manifest = load(run_dir / "manifest.json")
    wl = summary.get("workload") or {}
    timing = summary.get("timing") or {}
    phases = summary.get("phases") or {}
    decode = phases.get("decode_step") or {}
    prefill = phases.get("prefill") or {}
    expectation = decode.get("expectation") or {}
    cache = (summary.get("footprint") or {}).get("analytic_cache_state_bytes") or {}
    ncu = manifest.get("ncu") or {}

    row = {
        "run": name,
        "prompt": wl.get("prompt_tokens", "?"),
        "prefill_ms": f"{timing['prefill_seconds'] * 1e3:.1f}" if timing.get("prefill_seconds") else "–",
        "prefill_tps": f"{timing['prefill_tokens_per_second']:.0f}"
                       if timing.get("prefill_tokens_per_second") else "–",
        "decode_ms": f"{timing['decode_step_ms']:.2f}" if timing.get("decode_step_ms") else "–",
        "decode_tps": f"{timing['decode_tokens_per_second']:.1f}"
                      if timing.get("decode_tokens_per_second") else "–",
        "busy": "/".join(f"{p['occupancy']['busy_pct']:.0f}%" if p.get("occupancy") else "–"
                         for p in (prefill, decode)),
        "decode_dram": gb(level_bytes(decode, "dram")),
        "expected": gb(expectation.get("expected_bytes")),
        "ratio": f"{expectation['ratio']:.2f}" if expectation.get("ratio") else "–",
        "decode_l2": gb(level_bytes(decode, "l2")),
        "decode_bw": (f"{summary['decode_bandwidth_utilisation_pct']:.0f}%"
                      if summary.get("decode_bandwidth_utilisation_pct") else "–"),
        "prefill_dram": gb(level_bytes(prefill, "dram")),
        "prefill_l2": gb(level_bytes(prefill, "l2")),
        "cache": f"{sum(cache.values()) / 1e6:,.1f}" if cache else "–",
        "mem": peak_memory(run_log),
        "tiers": ",".join(str(t) for t in ncu.get("tiers", [])) or "–",
        "ncu_min": stage_minutes(manifest),
        "stages": ",".join(k for k, ok in (manifest.get("stages_ok") or {}).items() if not ok)
                  or "all ok",
    }
    notes = list(summary.get("warnings", []))
    if expectation.get("verdict"):
        notes.insert(0, f"decode check: {expectation['verdict']}")
    for phase in (prefill, decode):
        if phase.get("cross_check"):
            notes.append(f"{phase.get('label', phase.get('scope'))}: {phase['cross_check']}")
    return row, notes


def main() -> None:
    global QDIR
    if len(sys.argv) > 1:
        QDIR = (REPO / sys.argv[1]).resolve()
    qlog = QDIR / "queue.log"
    lines = qlog.read_text().splitlines() if qlog.exists() else []
    finished = [m for m in map(LINE.match, lines) if m]
    total = queue_length(lines)
    out = ["# Kimi-Linear Q8_0 GGUF via llama.cpp: MoE queue results", ""]
    out.append(f"{len(finished)} of {total} finished ({sum(m[2] == 'FAIL' for m in finished)} failed). "
               f"Latest queue line: `{lines[-1] if lines else 'none'}`")
    done_names = {m[3] for m in finished}
    started = [next(w for w in l.split()[3:] if w.startswith("moe-"))
               for l in lines if l.split()[2:3] == ["START"]]
    unfinished = [n for n in started if n not in done_names]
    queue_alive = subprocess.run(
        ["pgrep", "-f", r"^(/usr/bin/)?bash (\./)?scripts/run_kimi_q8_llamacpp_queue\.sh"],
        capture_output=True,
    ).returncode == 0
    if unfinished and queue_alive:
        out += ["", f"Running now: **{unfinished[-1]}**."]
    elif unfinished:
        out += ["", f"**⚠️ INTERRUPTED: {', '.join(unfinished)} started but never finished, and "
                    "the queue is not running.**"]
    out += ["", "Every run: model `kimi-linear-48b-q8-gguf`, backend llamacpp, profile "
            "`configs/profiles/moe.yaml`, ncu tier 1 + 2 with application replay. DRAM and L2 GB "
            "are exact ncu tier-1 totals per decode step / per prefill (not nsys-sampled). "
            "Expected = the routing-aware decode prediction (weights + cache read + state "
            "write-back). Cache = llama.cpp-layout analytic size (latent MLA K + KDA state), "
            "llama.cpp does not expose it. Peak mem = system memory in use, whole run (during "
            "ncu in brackets), from 5-s /proc/meminfo samples; includes the ~52 GB of weights. "
            "Tier-1 ncu min = wall minutes of the tier-1 collection per phase (P = prefill, D = decode). Busy = "
            "GPU busy share prefill/decode from nsys. Timings come from the unprofiled baseline.",
            ""]
    header = ["run", "prompt", "prefill_ms", "prefill_tps", "decode_ms", "decode_tps", "busy",
              "decode_dram", "expected", "ratio", "decode_l2", "decode_bw", "prefill_dram",
              "prefill_l2", "cache", "mem", "tiers", "ncu_min", "stages"]
    titles = ["Run", "Prompt", "Prefill ms", "Prefill tok/s", "Decode ms/step", "Decode tok/s",
              "Busy P/D", "Decode DRAM GB", "Expected GB", "DRAM/exp", "Decode L2 GB",
              "Decode BW of ceiling", "Prefill DRAM GB", "Prefill L2 GB", "Cache MB",
              "Peak mem GiB", "ncu tiers", "Tier-1 ncu min", "Failed stages"]
    out.append("| " + " | ".join(titles) + " |")
    out.append("|" + "---|" * len(titles))
    all_notes = []
    for m in finished:
        when, status, name, code, minutes, run, log = m.groups()
        run_dir = REPO / run
        if status == "FAIL" or not (run_dir / "metrics" / "summary.json").exists():
            out.append(f"| {name} | **{status} exit {code}** after {minutes} min | " +
                       " | ".join("" for _ in titles[2:]) + " |")
            all_notes.append((name, [f"{status}: see `{log}`"]))
            continue
        row, notes = summarise_run(name, run_dir, REPO / log)
        out.append("| " + " | ".join(str(row[k]) for k in header) + " |")
        all_notes.append((f"{name} ({minutes} min, `{run}`, log `{log}`)", notes))
    out += ["", "Findings: docs/10-kimi-llamacpp-moe-sweep.md.", "", "## Per-run notes", ""]
    for name, notes in all_notes:
        out.append(f"**{name}**")
        out += [f"- {n}" for n in notes] or ["- none"]
        out.append("")
    (QDIR / "results.md").write_text("\n".join(out))
    print("\n".join(out[:12 + len(finished)]))
    try:                                       # the all-22 table; never fails this script
        import summarize_kimi_q8_all
        summarize_kimi_q8_all.main()
    except Exception as e:                     # noqa: BLE001
        print(f"(all-workloads table not refreshed: {e})")


if __name__ == "__main__":
    main()
