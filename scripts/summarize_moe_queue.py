"""Summarise the MoE queue (scripts/run_moe_queue.sh) into runs/moe-queue/results.md.

Reads runs/moe-queue/queue.log for finished runs, then each run's metrics/summary.json and
metrics/baseline_result.json. Safe to rerun at any time; it rewrites results.md from scratch.

    ~/envs/nsbench/bin/python scripts/summarize_moe_queue.py
"""

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QDIR = REPO / "runs" / "moe-queue"
LINE = re.compile(r"^(\S+ \S+) (DONE|FAIL) (\S+) exit=(\d+) (\d+)min run=(\S+)")


def gb(value):
    return f"{value / 1e9:.2f}" if value else "–"


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def cache_cell(footprint: dict) -> str:
    """Measured cache at the end of generation: MLA K/V + KDA state (incl. conv), in MB."""
    parts = footprint.get("cache_state_bytes") or {}
    if not parts:
        total = footprint.get("kv_cache_bytes")
        return f"{total / 1e6:,.0f}" if total else "–"
    mla = parts.get("kv", 0)
    state = parts.get("recurrent_state", 0) + parts.get("conv_state", 0)
    return f"{(mla + state) / 1e6:,.0f} ({mla / 1e6:,.0f} + {state / 1e6:,.0f})"


def summarise_run(name: str, run_dir: Path) -> tuple[dict, list[str]]:
    summary = load(run_dir / "metrics" / "summary.json")
    baseline = load(run_dir / "metrics" / "baseline_result.json")
    manifest = load(run_dir / "manifest.json")
    wl = summary.get("workload") or {}
    timing = summary.get("timing") or {}
    phases = summary.get("phases") or {}
    decode = phases.get("decode_step") or {}
    prefill = phases.get("prefill") or {}
    expectation = decode.get("expectation") or {}
    routing = (baseline.get("backend") or {}).get("routing") or {}
    observed = routing.get("phases") or {}

    def experts(phase: str) -> str:
        p = observed.get(phase) or {}
        got = (p.get("distinct_experts_per_pass") or {}).get("mean")
        want = p.get("expected_distinct_experts_per_pass")
        if got is None:
            return "–"
        return f"{got:.0f}" + (f" (exp {want:.0f})" if want is not None else "")

    row = {
        "run": name,
        "routing": wl.get("routing", "?"),
        "batch": wl.get("batch_size", "?"),
        "prompt": wl.get("prompt_tokens", "?"),
        "prefill_s": f"{timing['prefill_seconds']:.2f}" if timing.get("prefill_seconds") else "–",
        "decode_ms": f"{timing['decode_step_ms']:.0f}" if timing.get("decode_step_ms") else "–",
        "decode_busy": (f"{decode['occupancy']['busy_pct']:.0f}%"
                        if decode.get("occupancy") else "–"),
        "decode_l2": gb(decode.get("nsys_l2_bytes_per_instance")),
        "expected": gb(expectation.get("expected_bytes")),
        "ratio": f"{expectation['ratio']:.2f}" if expectation.get("ratio") else "–",
        "prefill_l2": gb(prefill.get("nsys_l2_bytes_per_instance")),
        "cache": cache_cell(summary.get("footprint") or {}),
        "experts_decode": experts("decode"),
        "experts_prefill": experts("prefill"),
        "stages": ",".join(k for k, ok in (manifest.get("stages_ok") or {}).items() if not ok)
                  or "all ok",
    }
    notes = [w for w in summary.get("warnings", []) if not w.startswith("ncu tier 1 was not")]
    routing_warning = (baseline.get("backend") or {}).get("routing_warning")
    if routing_warning:
        notes.append(f"routing: {routing_warning}")
    if expectation.get("verdict"):
        notes.insert(0, f"decode check: {expectation['verdict']}")
    return row, notes


def main() -> None:
    lines = (QDIR / "queue.log").read_text().splitlines() if (QDIR / "queue.log").exists() else []
    finished = [m for m in map(LINE.match, lines) if m]
    started = [l for l in lines if " START " in l]
    out = ["# MoE queue results", ""]
    out.append(f"{len(finished)} of 22 finished ({sum(m[2] == 'FAIL' for m in finished)} failed). "
               f"Latest queue line: `{lines[-1] if lines else 'none'}`")
    # A run that STARTed but never logged DONE/FAIL while no queue process is alive was cut
    # off (crash, reboot, kill). Say so, rather than letting the count read as "0 failed".
    done_names = {m[3] for m in finished}
    unfinished = [l.split()[4] for l in started
                  if l.split()[2] == "START" and l.split()[4] not in done_names]
    # Anchored so it matches only the queue itself (`bash ./scripts/run_moe_queue.sh`), not a
    # shell that merely mentions the script name in its command line.
    queue_alive = subprocess.run(
        ["pgrep", "-f", r"^(/usr/bin/)?bash (\./)?scripts/run_moe_queue\.sh"],
        capture_output=True,
    ).returncode == 0
    if unfinished and not queue_alive:
        out.append("")
        out.append(f"**⚠️ INTERRUPTED: {', '.join(unfinished)} started but never finished, and "
                   "the queue is not running** (see Findings below).")
    out += ["", "Bytes are per decode step (L2 traffic sampled by nsys; an upper bound on DRAM, "
            "+6-10% on Qwen decode) or per prefill (L2, ~1.9x DRAM on Qwen). Expected = the "
            "routing-aware prediction (docs/07 section 3). Experts = mean distinct experts per "
            "layer per pass observed in the baseline (expected in brackets). Cache = measured size of the "
            "live cache tensors at the end of generation (all sequences): MLA K/V plus KDA recurrent + "
            "conv state; see the cache section below.", ""]
    header = ["run", "routing", "batch", "prompt", "prefill_s", "decode_ms", "decode_busy",
              "decode_l2", "expected", "ratio", "prefill_l2", "cache", "experts_decode",
              "experts_prefill", "stages"]
    titles = ["Run", "Routing", "B", "Prompt", "Prefill s", "Decode ms/step", "Decode busy",
              "Decode L2 GB", "Expected GB", "L2/exp", "Prefill L2 GB",
              "Cache MB (MLA K/V + KDA state)", "Experts/pass decode",
              "Experts/pass prefill", "Failed stages"]
    out.append("| " + " | ".join(titles) + " |")
    out.append("|" + "---|" * len(titles))
    all_notes = []
    for m in finished:
        when, status, name, code, minutes, run = m.groups()
        run_dir = REPO / run
        if status == "FAIL" or not (run_dir / "metrics" / "summary.json").exists():
            out.append(f"| {name} | **{status} exit {code}** after {minutes} min | " +
                       " | ".join("" for _ in titles[2:]) + " |")
            all_notes.append((name, [f"{status}: see runs/moe-queue/*-{name}.log"]))
            continue
        row, notes = summarise_run(name, run_dir)
        out.append("| " + " | ".join(str(row[k]) for k in header) + " |")
        all_notes.append((f"{name} ({minutes} min, `{run}`)", notes))
    findings = QDIR / "findings.md"
    if findings.exists():
        out += ["", findings.read_text().rstrip()]
    out += ["", "## Per-run notes", ""]
    for name, notes in all_notes:
        out.append(f"**{name}**")
        out += [f"- {n}" for n in notes] or ["- none"]
        out.append("")
    (QDIR / "results.md").write_text("\n".join(out))
    print("\n".join(out[:8 + len(finished)]))


if __name__ == "__main__":
    main()
