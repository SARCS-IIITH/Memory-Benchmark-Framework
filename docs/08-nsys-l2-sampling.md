# Measuring memory traffic with nsys L2 sampling

How the harness gets per-phase byte figures without Nsight Compute tier 1, why it was needed, how the method was validated, and where it can and cannot be trusted.

> **Status (2026-10-06):** built into the harness and **on by default** (`nsys.sample_l2_traffic: true`). It was validated on Qwen3-0.6B only. It is in use for the 22 Kimi MoE runs ([07-moe-routing.md](07-moe-routing.md)).

---

## 1. Why this was needed

The MoE experiments in [07-moe-routing.md](07-moe-routing.md) exist to answer one question: **how many bytes does Kimi read from memory per step under each routing mode and batch size?** For example, does a batch-32 decode step with `disjoint` routing really read about 49 GB, against 3.5 GB with `fixed`?

Until now the only source of memory bytes was **ncu tier 1**. It replays every kernel about 15 times to read its counters. That is tolerable for a dense model (about 1,600 launches per decode step) but not for Kimi:

| Kimi-Linear phase | Kernel launches | ncu tier 1 |
|---|---|---|
| One decode step | ~12,000 | ~13 h per run, at ~15 kernels/min |
| One 512-token prefill | ~190,000 | not feasible; truncates at the launch cap |

So tier 1 was turned off by default (see [02-metric-reference.md](02-metric-reference.md)). Without a replacement, the MoE runs would have produced timing and stall data but **no byte figures**, which are the whole point of those experiments.

## 2. The idea

**ncu** pauses the program before each kernel and replays it to read exact counters. That is precise per kernel, but slow.

**nsys** never pauses anything. While the model runs at normal speed, nsys reads a few GPU counters about **10,000 times a second**, like a speedometer. Line those samples up against the timeline, add them up over a phase, and you get that phase's total traffic. The nsys pass already runs in every benchmark, so this costs nothing extra.

The catch: nsys samples only the counters in its **metric set**. The stock set for this chip, `gb20b`, has clocks, engine activity and warp occupancy, and **no memory counters** (also noted in [01-platform-gb10.md](01-platform-gb10.md)). nsys does accept a custom set (`--gpu-metrics-set=file:<path>`), so the question was which memory counters it would actually sample on GB10.

## 3. Step 1: which counters can be sampled (synthetic test)

The test workload was 10 copies of a 2 GiB bf16 tensor: **21.47 GB read and 21.47 GB written, 42.94 GB in total**. Each candidate counter was added to a copy of `gb20b.config` and profiled.

| Counter | Meaning | Result |
|---|---|---|
| `lts__t_sectors` | all L2 traffic | ✅ **42.98 GB measured vs 42.94 GB expected (0.1%)** |
| `lts__t_sectors_aperture_sysmem_lookup_miss` | L2 misses to memory (ncu's DRAM figure on this chip) | ❌ silently dropped |
| `lts__d_sectors_fill_sysmem` | L2 fills from memory | ❌ silently dropped |
| `lts__t_sectors_lookup_miss` / `_hit` | L2 misses / hits | ❌ silently dropped |
| `lts__t_sectors_op_read` / `_write` | L2 reads / writes | ❌ silently dropped |
| `lts__d_sectors` | L2 data-stage sectors | ❌ silently dropped |

"Silently dropped" means nsys exits 0 and records only the stock 19 metrics. It gives no error.

**Result:** nsys can sample **all L2 traffic**, accurately, but **not memory (DRAM) bytes**. Those are not the same thing. Data found in L2 never reaches memory, so L2 traffic is always **at least** memory traffic. The next step was to measure how big the gap is on a real model.

Files: [experiments/nsys-gpu-metrics/](../experiments/nsys-gpu-metrics/).

## 4. Step 2: how close L2 is to memory bytes (Qwen3-0.6B)

### Why Qwen

- **Exact memory bytes need ncu tier 1.** On Kimi that takes many hours and still truncates. On Qwen3-0.6B it covers the whole phase in about 30 minutes, without truncation. So Qwen was the only model where one run could produce both numbers: nsys L2 and exact ncu memory bytes.
- **It is a sanity anchor.** Its 1.19 GB of weights tell us roughly what a decode step must read.
- **It is fast and already set up**, and was used for the smoke tests.

### The run

- `runs/20261005T203205Z__Qwen3-0.6B__decode-focused__hf__l2check` (2026-10-06 02:02–02:35)
- Decode-focused workload: 128 → 128 tokens, batch 1
- ncu tier 1 on, giving exact memory bytes. ncu covered **all** kernels: 1,594 prefill and 1,618 decode, not truncated.
- nsys used the L2-extended metric set, through a platform-profile override, with no code change.
- Compared with [experiments/nsys-gpu-metrics/compare_run.py](../experiments/nsys-gpu-metrics/compare_run.py).

### Results

| Per instance | nsys sampled L2 | ncu L2 | ncu memory bytes (exact) |
|---|---|---|---|
| **Decode step** | **1.42 GB** | 1.47 GB | **1.34 GB** |
| **Prefill** (128 tokens) | **4.66 GB** | 4.70 GB | **2.46 GB** |

| Ratio | Decode | Prefill | What it tells us |
|---|---|---|---|
| nsys L2 ÷ ncu L2 | 0.96 | 0.99 | nsys measures L2 correctly on a real model |
| nsys L2 ÷ ncu memory | **1.06** | **1.89** | how far L2 overstates memory traffic |
| ncu L2 ÷ ncu memory | 1.10 | 1.91 | the same gap, measured by ncu alone |

### What it means

1. **nsys's L2 figure is right.** It agrees with ncu's exact L2 count within 1–4%.
2. **For decode, L2 is a good stand-in for memory bytes: 6–10% high.** Decode reads each weight once per token with no reuse, so almost everything that passes through L2 also comes from memory. As a cross-check, the weights are 1.19 GB and ncu measured 1.34 GB.
3. **For prefill, L2 overstates memory bytes by about 1.9×.** Prefill reuses activations across many tokens, and those hits are served from L2.

## 5. What it means for the MoE runs

| Runs | Byte figure | Trust |
|---|---|---|
| Decode sweep (12) and long context (1) | nsys L2 per decode step | Good: a slight overestimate, ~10% on Qwen. The routing differences tested are 3–14×. |
| Short prefill (9) | nsys L2 per prefill | **L2 traffic, not memory bytes.** Use it to compare routing modes against each other, not as absolute GB. |

## 6. How it is built in

| Piece | Where |
|---|---|
| Metric file: stock `gb20b` plus `lts__t_sectors`. nsys requires `alias:` to equal the file name. | `configs/nsys/gb20b_l2.config` |
| Switch, default on (`false` = stock set) | `NsysConfig.sample_l2_traffic` in `nsight_bench/config.py` |
| Picks the extended file when the switch is on and the chip has one | `NsysRunner.gpu_metric_set()` and `L2_METRIC_SETS` in `nsight_bench/runners/nsys_runner.py` |
| Per-phase L2 bytes: the phase's kernels merged into GPU-busy windows (gaps under 1 ms joined), then the samples in each window summed | `NsysReport.sampled_l2_bytes_per_instance()` in `nsight_bench/parsers/nsys_parse.py` |
| `PhaseAnalysis.nsys_l2_bytes`. Phases are built from nsys alone when ncu is skipped. | `nsight_bench/analysis/assemble.py` |
| **Decode physics check without tier 1:** compares against nsys L2 (`DecodeExpectation.measured_source = "nsys_l2"`). The verdict carries the caveat. | `assemble.py`, `analysis/derive.py` |
| Report: "L2 traffic (nsys)" column in "Was the GPU actually busy?"; "Measured (nsys-sampled L2)" in the physics-check table | `report/markdown.py`, `report/html.py` |

With ncu tier 1 on, the physics check still uses ncu's exact DRAM bytes; nsys L2 is only reported alongside.

To turn it off for a profile:

```yaml
nsys:
  sample_l2_traffic: false
```

Checked: the three CPU check suites pass. Re-assembling the Qwen run gives the same numbers as the standalone comparison (1.421 / 4.660 GB). A copy with ncu removed gives a physics-check ratio of 1.16× from nsys L2, against 1.10× from ncu. A run without L2 samples reports "not measured".

## 7. Choosing a profiling depth: no ncu, tier 2 only, or tier 1 + tier 2

With nsys L2 sampling available, a run can be profiled at three depths. The first three stages (calibration, baseline, nsys) are the same in all three; they differ only in the ncu stage.

```bash
nsbench run ... --skip-ncu        # no ncu
nsbench run ... --tiers 2         # ncu tier 2 only (the default in every profile)
nsbench run ... --tiers 1,2       # ncu tier 1 + tier 2
```

### What each runs

| | No ncu | Tier 2 only (default) | Tier 1 + tier 2 |
|---|---|---|---|
| Calibration, baseline, nsys | ✅ | ✅ | ✅ |
| ncu tier 1: ~44 counters on **every** kernel in a phase | – | – | ✅ |
| ncu tier 2: full sections on the **top 5–8** kernels | – | ✅ | ✅ |
| How tier 2 picks its kernels | – | nsys timing | nsys by default; tier 1 if `rank_source: tier1` |

### What each gives

| Result | No ncu | Tier 2 only | Tier 1 + tier 2 |
|---|---|---|---|
| Real timing and throughput (baseline) | ✅ | ✅ | ✅ |
| Memory footprint, KV cache size, routing observations | ✅ | ✅ | ✅ |
| Timeline, GPU-busy %, kernel launch counts (nsys) | ✅ | ✅ | ✅ |
| **Bytes per phase** | L2 traffic only (nsys sampling): close for decode, inflated for prefill | same | **exact DRAM bytes**, plus L1/L2 bytes and hit rates per level |
| Decode physics check | against nsys L2 (labelled as such) | same | against **exact** DRAM bytes |
| Per-kernel bytes | – | sampled kernels only | ✅ every kernel |
| Register spills, FLOPs, arithmetic intensity | – | sampled kernels only | ✅ every kernel |
| **Why kernels are slow:** stall reasons, occupancy, waves per SM | – | ✅ top kernels | ✅ |
| Per-kernel L2 vs DRAM comparison | – | ✅ sampled kernels | ✅ |

### What each costs, on Kimi-Linear-48B

| | No ncu | Tier 2 only | Tier 1 + tier 2 |
|---|---|---|---|
| Extra time per run | 0 | ~25–30 min (two more 50 GB model loads, plus ~60 launches replayed ~34 passes each) | **~13 h per decode phase**; prefill is not feasible (~12,000 launches per decode step and ~190,000 per prefill, ~15 passes each) |
| Coverage | – | first 60 matching launches per phase: mostly early-layer elementwise kernels, rarely the routed experts | stops at the launch cap (5,384), so **truncated**: ~44% of one decode step, ~3% of a prefill |
| Memory risk on the 119 GB unified pool | lowest | high on large phases: ~190 driver allocation failures at batch 32, and **it crashed the DGX on a 16k-token prefill** (2026-10-06) | the same replay mechanism over far more kernels |

On a small dense model, tier 1 is practical. Qwen3-0.6B has ~1,600 launches per phase, ~16 min each, untruncated. That is how nsys L2 was validated against exact DRAM bytes (section 4).

### Which to use

- **Comparing routing modes, batch sizes or prompt lengths on Kimi:** no ncu is enough. Bytes, time, expert counts and launches come from baseline and nsys. Tier 2 adds stall reasons, but its sample mostly misses the expert kernels. For which of the 22 MoE runs had tier 2 and why, see `runs/moe-queue/findings.md`, "ncu coverage".
- **Why a kernel is slow:** tier 2. On Kimi, make it sample later layers first, e.g. per-layer NVTX ranges via `--annotate-layers`.
- **Exact DRAM bytes:** tier 1, practical on small dense models. On Kimi the realistic route would be a reduced tier 1 (about 12 counters, decode only, one step), accepting hours of runtime and the memory risk. Better: fix Kimi's per-expert kernel loop first, so there are far fewer kernels to profile.
- **Never run ncu on Kimi long-context prefill on this machine** (progress.md caveat 12).

## 8. Limits and caveats

1. **L2 traffic is not DRAM traffic.** It is an upper bound: ~6–10% high for decode, ~1.9× for prefill (Qwen).
2. **Validated on one small dense model at batch 1.** On Kimi, especially at batch 32 where activations grow, the decode gap may be wider. No Kimi run has exact DRAM bytes to confirm it. One ncu tier-1 cross-check on a Kimi decode run would, at the cost of hours.
3. **Phase totals, not per-kernel figures.** Sampling gives traffic per phase (every 100 µs), not per kernel. Per-kernel bytes still need ncu.
4. **Phase windows are approximate.** Samples are assigned by time, so a sample spanning a phase boundary is counted in one phase. With a 100 µs period against decode steps of tens to hundreds of ms, this is small.
5. **Only verified on GB10 (`gb20b`).** `L2_METRIC_SETS` maps only that chip. On any other chip the harness falls back to the stock set and reports L2 as "not measured".

## 9. Reproducing

```bash
# Synthetic test: GPU must be idle; run inside tmux
experiments/nsys-gpu-metrics/run_test.sh

# Qwen L2-vs-DRAM comparison: needs ncu tier 1 on, so it takes about 30 min
~/envs/nsbench/bin/nsbench run --model configs/models/qwen3-0.6b.yaml \
    --workload configs/workloads/decode-focused.yaml --profile configs/profiles/standard.yaml \
    --tiers 1 --no-sweep --tag l2check
~/envs/nsbench/bin/python experiments/nsys-gpu-metrics/compare_run.py runs/<that run>
```

`experiments/nsys-gpu-metrics/platform_profile_l2.json` is the override used for the original Qwen check, from before the extended set became the default. It is no longer needed.
