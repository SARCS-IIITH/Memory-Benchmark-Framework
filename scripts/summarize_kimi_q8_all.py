"""One results table for all 22 MoE workloads on Kimi-Linear Q8 GGUF through llama.cpp:
results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/results.md.

The workloads run in sets (one queue each, scripts/run_kimi_q8_llamacpp_queue.sh), and each set
keeps its own <set>/queue/results.md with every column and per-run notes. This file puts the
headline columns of every set side by side, in the order of the 22 workloads, with the status
of the ones not run yet. It reads each set's queue/queue.log, so it is safe to rerun at any time;
scripts/summarize_kimi_q8_queue.py calls it after rewriting a set's own table, so it stays
current while a queue runs.

    ~/envs/nsbench/bin/python scripts/summarize_kimi_q8_all.py

Findings are written up in docs/10-kimi-llamacpp-moe-sweep.md.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_kimi_q8_queue import LINE, summarise_run  # noqa: E402
from summarize_moe_queue import load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "kimi-linear-48b-a3b-q8_0-gguf__llamacpp"

#: (set name, where its queue folder is now or was before the set moved under ROOT)
SETS = [
    ("natural-b1", [ROOT / "natural-b1" / "queue", REPO / "runs" / "kimi-q8-llamacpp-sweep" / "queue"]),
    ("forced-b1", [ROOT / "forced-b1" / "queue", REPO / "runs" / "kimi-q8-llamacpp-forced-routing" / "queue"]),
    ("batch", [ROOT / "batch" / "queue"]),
]

#: The 22 workloads in configs/workloads/moe/, in the order of the original HF FP8 queue.
WORKLOADS = [
    f"moe-decode-{r}-b{b}" for b in (1, 8, 16, 32) for r in ("natural", "fixed", "disjoint")
] + [
    f"moe-prefill-{r}-p{p}" for p in (16, 64, 512) for r in ("natural", "fixed", "disjoint")
] + ["moe-longctx-natural-p16384"]


def workload(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / "workloads" / "moe" / f"{name}.yaml").read_text())


EXPECTED_HOW = """## How "Expected GB" is calculated

The bytes one **decode step** must move from DRAM if nothing is wasted, computed by the harness
(`nsight_bench/analysis/assemble.py`, `ModelConfig.decode_read_weight_bytes` and
`kv_cache_breakdown` in `nsight_bench/config.py`, `DecodeExpectation` in
`nsight_bench/analysis/derive.py`). There is no prefill expectation; prefill bytes are compared
with the expert counts in docs/10 section 5.

**Expected = weights read + cache read + recurrent-state write-back**

1. **Weights read**, from the GGUF's own tensor sizes by role
   (`configs/models/kimi-linear-48b-q8-gguf.yaml`, `weight_bytes_by_role`, read with gguf-py):
   - sizes as stored, so the Q8_0 block scales are included;
   - everything that is not a routed expert, in full: attention, shared expert, dense first
     layer, routers, norms (`other` 1.384 GB) and the output head (`lm_head` 0.401 GB);
   - routed experts (50.055 GB) scaled by the fraction of the 256 experts the step touches:
     distinct experts / 256, where for a step of B tokens (B = batch size) natural routing
     assumes uniform independent picks, 256 x (1 - (1 - 8/256)^B), fixed is 8, and disjoint is
     min(256, 8B). At batch 1 every mode touches 8, so 50.055 x 8/256 = 1.564 GB;
   - the input embedding (0.401 GB) excluded: a step gathers one row per token, not the matrix.
   Batch 1: 1.384 + 0.401 + 1.564 = **3.350 GB**.
2. **Cache read**: the whole decoding state, read once per step, sized analytically in
   llama.cpp's layout (llama.cpp does not expose its cache): latent MLA K only, 576 values x
   2 B x 7 MLA layers = 8,064 B per token of context, plus the KDA recurrent state (f32,
   41.9 MB) and conv window (f32, 2.9 MB). Context = prompt + generated tokens, i.e. the
   state at the *end* of generation (an upper bound for the average step). Batch 1, 160
   tokens: **0.046 GB**; at 16,416 tokens: 0.177 GB. Multiplied by the batch size.
3. **State write-back**: a linear-attention (KDA) layer rewrites its whole recurrent state
   every step, unlike a KV cache that appends one token: **0.042 GB** x batch size.

Batch 1, 160-token context: 3.350 + 0.046 + 0.042 = **3.44 GB**. **DRAM/exp** = measured
tier-1 DRAM per decode step / expected. Every batch-1 run reads 1.05x (+0.16 GB). Not the Q8_0 block scales:
the GGUF tensor sizes above already include them (34 bytes per 32 weights).

**Where the excess comes from (per-kernel ncu, 2026-10-13):** the KDA state is moved about
four times per step, not once. `k_get_rows_float_vec` gathers each sequence's state into the
graph (read + write) and `gated_delta_net_cuda` reads it and writes the new one: 0.084 + 0.086
= 0.17 GB at batch 1 against the 0.042 GB counted, which is most of the +0.16 GB. At batch 8
the same two kernels move 0.72 + 0.68 = 1.40 GB against 0.34 GB counted, the +1.07 GB seen in
fixed-b8 (1.26x) and disjoint-b8 (1.07x), whose expert counts are exact. The formula is left
as it is so the ratio keeps showing this; natural-b8 reads 0.98x because its uniform-routing
expert count (56 per layer) is too high (router skew), which hides the same +1.07 GB.
"""


BANDWIDTH_HOW = """## Bandwidth

Real bandwidth = exact DRAM bytes from ncu tier 1 / the phase's time in the **unprofiled
baseline** (decode: per step; prefill: per prefill). ncu's own kernel durations are not used:
they come from replay at locked clocks and are not performance. Ceiling = the DRAM bandwidth
this run's calibration measured with a streaming kernel just before it (~225 GB/s on GB10).
L2 hit % and L2/DRAM (bytes through L2 per byte from DRAM) are ncu tier-1 totals. Per-kernel
bytes are in each run's `metrics/kernels_*.csv`.
"""

POWER_HOW = """## Power and energy

Measured separately from the profiling, in an unprofiled pass per run
(`scripts/measure_energy.py`, `nsight_bench/energy.py`, result in the run's
`metrics/energy.json`): the run's model and workload are loaded again, each phase is repeated
until its measurement window lasts >= 10 s, and NVML's cumulative GPU energy counter is read
around it (decode in segments of the workload's decode steps, each segment's prefill outside
the window).

How accurate:
- **GPU rail only.** GB10 reports GPU-memory and module power as N/A, so the LPDDR5X memory,
  the CPU and the rest of the module are **not** included. For memory-bound decode that is a
  large share of the system's energy: read these as **lower bounds** on system energy, and
  use them to compare workloads, not as wall-plug figures.
- The counter updates only every ~100 ms (in steps of ~0.8 J, measured each pass:
  `counter_resolution`), hence the >= 10 s windows. Two cross-checks per run: power.draw
  sampled every 20 ms and integrated over the same windows (Counter/sampled), and decode energy
  from one continuous window minus the prefills (Seg/sub), which the counter's step size at
  window edges cannot bias. Both near 1.00 = consistent.
- Above-idle = energy minus idle power (model resident, nothing running) over the window: the
  energy the work itself adds. NVIDIA does not publish GB10's power-telemetry accuracy; the
  cross-checks of the valid measurements range 0.84-1.08, so quote absolute values as
  +/-10-15%, better for comparisons (same sensor, same method).
- **Decode energy is not valid ("n/v") for the six moe-prefill-* workloads.** They generate 4
  tokens, so each decode segment is 3 steps, ~55 ms: shorter than the counter's ~100 ms update,
  and right after a prefill whose power has not settled. Their cross-checks are 0.82-1.68 and
  their J/token rises with prompt length although the decode work is identical. Batch-1 decode
  energy comes from the 32-token runs (moe-decode-*-b1, longctx). Their prefill energy is valid.
- Step-by-step method (idle, prefill and decode windows, cross-checks):
  docs/10-kimi-llamacpp-moe-sweep.md section 2.1.
- pJ/bit = GPU energy per decode step / DRAM bits that step moved: all GPU energy (compute
  included) per bit of DRAM traffic, not DRAM energy.
"""


def bandwidth(run_dir: Path, row: dict) -> dict:
    """Real-latency DRAM bandwidth and L2 behaviour per phase, from summary.json."""
    s = load(run_dir / "metrics" / "summary.json")
    t = s.get("timing") or {}
    ceiling = s.get("dram_ceiling_gbps")
    out = {"ceiling": f"{ceiling:.1f}" if ceiling else "–"}
    for phase, secs in (("decode_step", (t.get("decode_step_ms") or 0) / 1e3),
                        ("prefill", t.get("prefill_seconds") or 0)):
        lv = ((((s.get("phases") or {}).get(phase) or {}).get("hierarchy") or {}).get("levels") or {})
        dram = (lv.get("dram") or {}).get("bytes_total")
        l2 = lv.get("l2") or {}
        gbs = dram / secs / 1e9 if dram and secs else None
        key = "d" if phase == "decode_step" else "p"
        out[f"{key}_gbs"] = f"{gbs:.1f}" if gbs else "–"
        out[f"{key}_pct"] = f"{100 * gbs / ceiling:.0f}%" if gbs and ceiling else "–"
        out[f"{key}_hit"] = f"{l2['hit_rate_pct']:.1f}%" if l2.get("hit_rate_pct") is not None else "–"
        out[f"{key}_l2x"] = f"{l2['bytes_total'] / dram:.2f}" if l2.get("bytes_total") and dram else "–"
    return out


def energy(run_dir: Path) -> dict | None:
    e = load(run_dir / "metrics" / "energy.json")
    return e or None


def decode_valid(d: dict) -> bool:
    """Decode segments long enough for the ~100 ms energy counter (>= 0.5 s each on average)."""
    return bool(d.get("segments")) and d["window_s"] / d["segments"] >= 0.5


def fmt(v, spec: str) -> str:
    if isinstance(v, str):
        return v
    return format(v, spec) if isinstance(v, (int, float)) else "–"


def fmt_gb(v) -> str:
    return f"{v / 1e9:.3f}" if v else "–"


def main() -> int:
    runs: dict[str, tuple[str, object]] = {}      # workload -> (set, LINE match)
    running: dict[str, str] = {}
    for set_name, candidates in SETS:
        qdir = next((d for d in candidates if (d / "queue.log").exists()), None)
        if qdir is None:
            continue
        lines = (qdir / "queue.log").read_text().splitlines()
        for m in map(LINE.match, lines):
            if m:
                runs[m[3]] = (set_name, m)
        ended = any("QUEUE END" in l for l in lines[-1:])
        alive = subprocess.run(
            ["pgrep", "-f", r"^(/usr/bin/)?bash (\./)?scripts/run_kimi_q8_llamacpp_queue\.sh"],
            capture_output=True).returncode == 0
        for l in lines:
            parts = l.split()
            if parts[2:3] == ["START"]:
                name = next(w for w in parts[3:] if w.startswith("moe-"))
                if name not in runs and not ended and alive:
                    running[name] = set_name

    done = [w for w in WORKLOADS if w in runs and runs[w][1][2] == "DONE"]
    failed = [w for w in WORKLOADS if w in runs and runs[w][1][2] == "FAIL"]
    out = ["# Kimi-Linear Q8_0 GGUF via llama.cpp: all 22 MoE workloads", "",
           f"**{len(done)} of {len(WORKLOADS)} done**" + (f", {len(failed)} failed" if failed else "")
           + (f", running now: {', '.join(running)}" if running else "") + ".", "",
           "One row per workload in `configs/workloads/moe/`. Each set has its own table with every "
           "column (L2 hit rates, cache, ncu minutes) and per-run notes in `<set>/queue/results.md`; "
           "findings are in docs/10-kimi-llamacpp-moe-sweep.md. Model `kimi-linear-48b-q8-gguf`, "
           "profile `configs/profiles/moe.yaml`, ncu tier 1 + 2 with application replay. DRAM and "
           "L2 GB are exact ncu tier-1 totals per decode step / per prefill. Expected = the "
           "routing-aware decode prediction; DRAM/exp is measured over expected. Timings come from "
           "the unprofiled baseline; Busy = GPU busy share prefill / decode from nsys. Peak mem = "
           "system memory in use over the whole run (includes the ~52 GB of weights).", ""]
    titles = ["#", "Workload", "Routing", "Batch", "Prompt", "Status", "Prefill ms",
              "Decode ms/step", "Decode tok/s", "Busy P/D", "Decode DRAM GB", "Expected GB",
              "DRAM/exp", "Decode L2 GB", "Prefill DRAM GB", "Prefill L2 GB", "Peak mem GiB",
              "Run folder"]
    keys = ["prefill_ms", "decode_ms", "decode_tps", "busy", "decode_dram", "expected", "ratio",
            "decode_l2", "prefill_dram", "prefill_l2", "mem"]
    out.append("| " + " | ".join(titles) + " |")
    out.append("|" + "---|" * len(titles))
    for i, name in enumerate(WORKLOADS, 1):
        wl = workload(name)
        lead = [str(i), name, wl.get("routing", "natural"), str(wl.get("batch_size", 1)),
                str(wl.get("prompt_tokens", "?"))]
        if name in runs:
            set_name, m = runs[name]
            _, status, _, code, minutes, run, log = m.groups()
            run_dir = REPO / run
            rel = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else Path(run)
            if status == "DONE" and (run_dir / "metrics" / "summary.json").exists():
                row, _ = summarise_run(name, run_dir, REPO / log)
                cells = [f"done ({set_name}, {minutes} min)"] + [str(row[k]) for k in keys]
            else:
                cells = [f"**{status} exit {code}** ({set_name})"] + [""] * len(keys)
            cells.append(f"`{rel}`")
        elif name in running:
            cells = [f"running ({running[name]})"] + [""] * len(keys) + [""]
        elif wl.get("batch_size", 1) > 1:
            started = (ROOT / "batch" / "queue" / "queue.log").exists()
            cells = ["queued (batch)" if started else
                     "queued (batch set starts after the energy pass)"] + [""] * len(keys) + [""]
        else:
            cells = ["queued"] + [""] * len(keys) + [""]
        out.append("| " + " | ".join(lead + cells) + " |")
    done_runs = [(i, n, runs[n]) for i, n in enumerate(WORKLOADS, 1)
                 if n in runs and runs[n][1][2] == "DONE"
                 and (REPO / runs[n][1][6] / "metrics" / "summary.json").exists()]
    out += ["", BANDWIDTH_HOW,
            "| # | Workload | Decode GB/s | Decode % of ceiling | Decode L2 hit | Decode L2/DRAM | "
            "Prefill GB/s | Prefill % of ceiling | Prefill L2 hit | Prefill L2/DRAM | Ceiling GB/s |",
            "|" + "---|" * 11]
    for i, name, (set_name, m) in done_runs:
        b = bandwidth(REPO / m[6], {})
        out.append(f"| {i} | {name} | {b['d_gbs']} | {b['d_pct']} | {b['d_hit']} | {b['d_l2x']} | "
                   f"{b['p_gbs']} | {b['p_pct']} | {b['p_hit']} | {b['p_l2x']} | {b['ceiling']} |")
    out += ["", POWER_HOW]
    measured = [(i, n, m, energy(REPO / m[6])) for i, n, (_, m) in done_runs]
    if not any(e for *_, e in measured):
        out += ["**Pending:** no run has `metrics/energy.json` yet. The energy pass runs after the "
                "forced-routing queue (see progress.md).", ""]
    else:
        out += ["| # | Workload | Idle W | Decode W | Decode J/token | Decode above-idle J/token | "
                "Decode pJ/DRAM bit | Prefill W | Prefill J | Prefill J/token | Counter/sampled (P, D) | "
                "Seg/sub | Counter step |", "|" + "---|" * 13]
        for i, name, m, e in measured:
            if not e:
                out.append(f"| {i} | {name} | pending |" + " |" * 11)
                continue
            d, p_ = e["decode_step"], e["prefill"]
            res = e.get("counter_resolution") or {}
            if not decode_valid(d):
                d = {k: "n/v" for k in ("avg_power_w", "energy_j_per_token", "above_idle_j_per_token",
                                        "pj_per_dram_bit", "counter_vs_sampled")}
                d["by_subtraction"] = {"segments_vs_subtraction": "n/v"}
            out.append(
                f"| {i} | {name} | {fmt(e['idle']['avg_power_w'], '.1f')} | {fmt(d.get('avg_power_w'), '.1f')} | "
                f"{fmt(d.get('energy_j_per_token'), '.3f')} | {fmt(d.get('above_idle_j_per_token'), '.3f')} | "
                f"{fmt(d.get('pj_per_dram_bit'), '.1f')} | {fmt(p_['avg_power_w'], '.1f')} | "
                f"{fmt(p_['energy_j_per_instance'], '.2f')} | {fmt(p_['energy_j_per_token'], '.3f')} | "
                f"{fmt(p_.get('counter_vs_sampled'), '.2f')}, {fmt(d.get('counter_vs_sampled'), '.2f')} | "
                f"{fmt((d.get('by_subtraction') or {}).get('segments_vs_subtraction'), '.2f')} | "
                f"{res.get('median_update_interval_ms', '–')} ms |")
        out.append("")
    out += ["", EXPECTED_HOW, "## Per run", "",
            "Every finished run, in the table's order: all columns of its set table, the expected-"
            "bytes breakdown, its notes, and where its files are.", ""]
    for i, name in enumerate(WORKLOADS, 1):
        if name not in runs:
            continue
        set_name, m = runs[name]
        _, status, _, code, minutes, run, log = m.groups()
        run_dir = REPO / run
        out.append(f"### {i}. {name}")
        out.append("")
        if status != "DONE" or not (run_dir / "metrics" / "summary.json").exists():
            out += [f"**{status} exit {code}** after {minutes} min. Log: `{log}`", ""]
            continue
        row, notes = summarise_run(name, run_dir, REPO / log)
        e = (((load(run_dir / "metrics" / "summary.json").get("phases") or {})
              .get("decode_step") or {}).get("expectation") or {})
        out += [
            f"- **Set** `{set_name}`, {minutes} min, all stages: {row['stages']}; ncu tiers {row['tiers']}, "
            f"tier-1 ncu minutes {row['ncu_min']}",
            f"- **Timing** (baseline): prefill {row['prefill_ms']} ms ({row['prefill_tps']} tok/s), "
            f"decode {row['decode_ms']} ms/step ({row['decode_tps']} tok/s); GPU busy prefill/decode {row['busy']}",
            f"- **Decode step**: DRAM {row['decode_dram']} GB, L2 {row['decode_l2']} GB, "
            f"{row['decode_bw']} of the DRAM ceiling; expected {row['expected']} GB, ratio {row['ratio']}",
            f"- **Expected breakdown**: weights {fmt_gb(e.get('weight_bytes'))} + cache read "
            f"{fmt_gb(e.get('kv_bytes'))} + state write-back {fmt_gb(e.get('state_write_bytes'))} = "
            f"{fmt_gb(e.get('expected_bytes'))} GB (context {e.get('context_len', '?')} tokens; "
            f"weights: {e.get('weight_bytes_source', '?')})",
            f"- **Prefill**: DRAM {row['prefill_dram']} GB, L2 {row['prefill_l2']} GB",
            f"- **Cache** (analytic, llama.cpp layout): {row['cache']} MB; **peak memory** {row['mem']} GiB "
            "(whole run, ncu in brackets)",
            f"- **Bandwidth**: decode {bandwidth(run_dir, row)['d_gbs']} GB/s, prefill "
            f"{bandwidth(run_dir, row)['p_gbs']} GB/s (ceiling {bandwidth(run_dir, row)['ceiling']} GB/s)",
            f"- **Files**: run `{run}`, log `{log}`, set table `{Path(log).parent}/results.md`",
        ]
        en = energy(run_dir)
        if en:
            d, p_ = en["decode_step"], en["prefill"]
            dec = (f"decode {fmt(d['energy_j_per_token'], '.3f')} J/token at {fmt(d['avg_power_w'], '.1f')} W "
                   f"({fmt(d.get('pj_per_dram_bit'), '.1f')} pJ per DRAM bit), " if decode_valid(d) else
                   "decode not valid (3-step segments, shorter than the counter's update; see Power and energy), ")
            out.append(
                f"- **Energy** (GPU rail only, lower bound on system): {dec}"
                f"prefill {fmt(p_['energy_j_per_instance'], '.2f')} J at {fmt(p_['avg_power_w'], '.1f')} W, idle "
                f"{fmt(en['idle']['avg_power_w'], '.1f')} W; measured {en.get('measured_at', '?')}")
        else:
            out.append("- **Energy**: pending (no metrics/energy.json yet)")
        out += [f"- note: {n}" for n in notes]
        out.append("")
    out += ["", "Sets: " + ", ".join(f"`{s}/`" for s, _ in SETS) + " (run folders + `queue/` with "
            "queue.log, results.md and per-run logs). Validation runs: `validation-forced-routing/`, "
            "`validation-batch/`.", ""]
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results.md").write_text("\n".join(out))
    print("\n".join(out[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
