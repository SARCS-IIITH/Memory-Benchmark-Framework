# 10. Kimi-Linear MoE sweep through llama.cpp, with exact DRAM bytes

Status (2026-10-14 09:15): **all 22 workloads done, all ok**: natural routing 5/5
(2026-10-12 23:58 - 10-13 06:05), forced routing 8/8 (10-13 07:03-15:53), batch 9/9 (10-13
18:02 - 10-14 07:44). Energy measured for all 22 (batch runs right after each run, the rest
07:44-08:06). All 22 in one table, with bandwidth and energy
sections: `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/results.md`.

## 1. Why this sweep

The 22 MoE workloads in `configs/workloads/moe/` were first run on 2026-10-06 on the FP8
safetensors checkpoint through Hugging Face transformers (`results/kimi-linear-48b-a3b-fp8__hf-transformers/moe-queue/results.md`,
docs/07). ncu tier 1 was off for all 22: under transformers an eager MoE runs 12,147 kernels per
decode step and 144,031 per 128-token prefill, so tier 1 with kernel replay takes hours per
phase and doubles memory. Without tier 1 those runs have no exact DRAM traffic, only
nsys-sampled L2 traffic (an upper bound, 1.75x the expected bytes on Kimi decode). The GPU was
also idle ~70% of the time, because transformers routes experts on the host.

llama.cpp runs the same model as Q8_0 GGUF with routing on the GPU (1,306 kernels per decode
step, 1,864 per prefill, GPU busy 93-97%). With ncu application replay, tier 1 + 2 take about
48 minutes per workload and memory stays at an unprofiled run's ~57 GiB (progress.md,
"ncu application replay test"). So this sweep re-runs the workloads through llama.cpp with
tier 1 on, to get exact DRAM bytes per decode step and per prefill.

## 2. Setup

| | |
|---|---|
| Model | `configs/models/kimi-linear-48b-q8-gguf.yaml` (Kimi-Linear-48B-A3B, Q8_0, 52.25 GB) |
| Backend | llamacpp (llama-cpp-python, flash attention on, native MLA: latent K cache only) |
| Profile | `configs/profiles/moe.yaml` (tier 2 top 5 kernels ranked from nsys) |
| ncu | tier 1 + 2, application replay; both automatic for llamacpp since 2026-10-12 (`ncu.auto_tier1`, `ncu.replay_mode: auto`) |
| Queue | `scripts/run_kimi_q8_llamacpp_queue.sh` in tmux `kimi-q8-queue` |
| Status log | `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/natural-b1/queue/queue.log` (one line per start / wait / finish) |
| Summary | `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/natural-b1/queue/results.md` (one row per run, rewritten after each start and finish) |
| Per run | `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/natural-b1/queue/<n>-<workload>.log` and `.mem.csv` (5-s /proc/meminfo samples, memory guard at 2 GB for 60 s) |
| **All 22 in one place** | **`results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/`**: `results.md` at the top (one row per workload, all sets, status of the ones not run; `scripts/summarize_kimi_q8_all.py`, refreshed after every run) and one folder per set: `natural-b1/`, `forced-b1/` (moved in when its queue ends), later `batch/`, plus `validation-forced-routing/` and `validation-batch/` |
| Energy | `scripts/measure_energy.py` → `metrics/energy.json` per run (section 2.1); batch runs right after each run (`ENERGY=1`), batch-1 runs in a separate pass |

### 2.1 How energy is measured

The profiling stages cannot give energy: ncu replays kernels at locked clocks and the baseline
stage only reads power once before and once after. Energy is therefore a separate,
**unprofiled** pass per run (`scripts/measure_energy.py RUN_DIR`, method in
`nsight_bench/energy.py`). The steps:

1. **Same run, nothing attached.** Load the run's own `run_config.json` (model, workload,
   prompt ids, routing, batch) and the same backend; no nsys or ncu. Fixed / disjoint runs
   re-run themselves in `~/envs/nsbench-moeroute` (the patched build). Wait until no other
   process holds the GPU (the script's own context excluded).
2. **Warm up.** Prefill + all decode steps, `warmup_iters` times (at least once), untimed.
3. **Counter resolution.** Poll NVML's cumulative energy counter
   (`nvmlDeviceGetTotalEnergyConsumption`, mJ) every ~1 ms for 2 s and record how often and by
   how much it changes (`counter_resolution`; on GB10 every ~100 ms, ~0.84 J steps). This is
   why every window below lasts >= 10 s.
4. **Start a sampler.** A thread reads `power.draw` (`nvmlDeviceGetPowerUsage`) every 20 ms
   for the whole measurement, for the cross-check in step 8.
5. **Idle.** Sleep 1 s, then read the counter across 8 s with the model resident and nothing
   running: idle W = energy / time.
6. **Prefill.** Read the counter, run prefill (with a device sync) repeatedly until the window
   lasts >= 10 s and at least 5 times, read the counter. J per prefill = window energy /
   repetitions; J per prompt token also divides by `prompt x batch`.
7. **Decode.** A decode step cannot be repeated without a prefill before it, and the context
   caps the steps per prefill, so decode is measured in segments: run a prefill *outside* the
   window, then read the counter, run the workload's `generate - 1` decode steps (synced), read
   the counter. Repeat until the summed windows reach >= 10 s (at most 240 s of wall time,
   at least 3 segments). J per step = summed energy / (segments x steps); J per token also
   divides by the batch size.
8. **Cross-checks** (both should be ~1.00):
   - *Counter / sampled*: the same windows integrated from the 20-ms `power.draw` samples
     (trapezoid rule), counter energy / sampled energy.
   - *Seg / sub* (decode only): one continuous counter reading over all segments, prefills
     included, minus segments x the J per prefill from step 6. It is immune to the counter's
     ~100 ms steps at each short segment's edges.
9. **Derived numbers.** Average W = energy / window time; above-idle J = energy - idle W x
   window time (the energy the work adds); pJ per DRAM bit = J per decode step / (the run's
   ncu tier-1 DRAM bytes per step x 8), i.e. all GPU energy, compute included, per bit of DRAM
   traffic, not the energy of the DRAM itself. SM clock, temperature and power are recorded
   before and after.

**Scope and accuracy.** NVML on GB10 covers the GPU rail only ("GPU Memory Power" and "Module
Power" read N/A), so the LPDDR5X, the CPU and the rest of the module are missing: every figure
is a lower bound on system energy. NVIDIA publishes no accuracy for GB10's power telemetry; the
cross-checks so far range 0.87-1.06, so quote absolute values as **+/-10-15%**. Comparisons
between runs are tighter (same sensor, same method, same session).

**Cost.** One model load (~52 GB, 1-2 min) plus 1-2 min of measuring per run; the 16k run
~3-4 min (each of its prefills is 8.9 s and every decode segment needs one).

## 3. Which workloads

Natural routing at batch 1 ran first (5 workloads); the rest needed new code:

| Workload | Status | Note |
|---|---|---|
| moe-decode-natural-b1 | done | |
| moe-prefill-natural-p16, -p64, -p512 | done | |
| moe-longctx-natural-p16384 | done | |
| moe-decode-natural-b8, -b16, -b32 | done | batch > 1 added and GPU-verified on Kimi 2026-10-13 (section 3.2); set `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/batch/`, started 18:02 |
| 8 batch-1 fixed / disjoint workloads | done | validation passed 07:03 (section 3.1); 07:03-15:53, all ok |
| 6 batch > 1 fixed / disjoint workloads | done | same batch set (patched build, 3.1 + 3.2) |

Forced routing is now possible through a patched llama.cpp (section 3.1), and batch > 1
through several sequences in one context (section 3.2).

### 3.1 Forced routing on llama.cpp

`experiments/llamacpp-backend/forced-routing.patch`, built into `~/envs/nsbench-moeroute`.
With `LLAMA_MOE_FORCE_ROUTING=fixed|disjoint` (set by the backend from the workload's `routing`),
llama.cpp takes each MoE layer's expert ids from a graph input instead of the router's top-k,
using the same formula as the HF hook (`nsight_bench/compat/routing.py`): fixed = experts 0..7
for every token; disjoint = token t of pass p gets `(p·T·8 + t·8 + j) mod 256`. Differences
from the HF hook, none of which change the bytes measured:

- Expert weights are the router's own scores for the forced experts, renormalised and scaled
  by 2.446 (HF: uniform at the same total). This keeps the router matmul in the graph, as HF
  keeps its gate running.
- Natural routing fuses the router's top-k into one `topk_moe_cuda` kernel; with forced ids
  that fusion does not apply and ~4 small kernels per layer run instead.
- llama.cpp runs the last layer on the output token only, which gets that token's ids.
- No routing observer: the distinct-expert counts follow from the formula and are checked
  by the DRAM bytes.

Checks before any forced result is trusted: `check_forced_routing_cpu.py` (CPU, 17/17 passed
2026-10-13) and `check_forced_routing_gpu.py` (GPU, run automatically after the natural sweep:
patch off changes nothing; forced kernel sequences repeat exactly, as ncu application replay
needs). Results go to `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/forced-b1/`; validation to
`results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/validation-forced-routing/validation.md`.

### 3.2 Batch > 1 on llama.cpp

`backends/llamacpp.py` runs `batch_size` sequences in one context, mirroring the hf backend's
`[batch, prompt]` tensor: the same per-row prompt ids; the prefill is one `llama_decode` of all
`batch x prompt` tokens (one ubatch, each row's last token returning logits); each decode step is
one `llama_decode` of `batch` tokens, one per sequence. The context is a copy of the bindings'
own parameters with only `n_ctx`, `n_batch`/`n_ubatch` and `n_seq_max` changed; batch 1 keeps
the original path. With forced routing the patch sees `batch x prompt` tokens per prefill pass
and `batch` per decode pass, the same token counts as the hf hook.

Checked on CPU (`check_batch_cpu.py`): batch-3 rows give bit-identical logits and the same
tokens as each row alone. Checked on Kimi on the GPU 2026-10-13
(`results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/validation-batch/batch-validation.md`): batch 4 vs each row alone gives
prefill logits with cosine 0.9995-0.99997 and the same first token in every row (greedy
continuations agree 8/8 in three rows; one forks at step 6, a near-tie between batched and
single-sequence kernels); natural-b8 and disjoint-b8 each show one identical kernel sequence per
phase across two runs, so application replay is safe; b32 (4,096-token prefill) runs, peak
60.6 GiB.

## 4. Results

### 4.1 Natural routing, batch 1 (2026-10-12 23:58 - 2026-10-13 06:05, 5/5 ok)

All four stages ok in every run; tier 1 + 2 with application replay. Full table with L2,
cache and per-run notes: `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/natural-b1/queue/results.md`. DRAM is exact (ncu
tier 1), per decode step or per prefill.

| Workload | Prompt | Prefill ms | Prefill tok/s | Decode ms/step | Decode tok/s | GPU busy P/D | Decode DRAM GB | Expected GB | Ratio | Decode BW of ceiling | Prefill DRAM GB | Run time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-prefill-natural-p16 | 16 | 78.9 | 203 | 18.33 | 54.6 | 94% / 85% | 3.60 | 3.44 | 1.05 | 88% | 15.04 | 61 min |
| moe-prefill-natural-p64 | 64 | 145.1 | 441 | 18.80 | 53.2 | 97% / 83% | 3.59 | 3.44 | 1.05 | 85% | 28.08 | 62 min |
| moe-decode-natural-b1 | 128 | 202.0 | 634 | 17.47 | 57.2 | 97% / 92% | 3.60 | 3.44 | 1.05 | 91% | 36.72 | 70 min |
| moe-prefill-natural-p512 | 512 | 352.8 | 1,451 | 18.81 | 53.2 | 98% / 86% | 3.62 | 3.44 | 1.05 | 85% | 64.59 | 64 min |
| moe-longctx-natural-p16384 | 16,384 | 8,859.7 | 1,849 | 18.24 | 54.8 | 97% / 93% | 3.74 | 3.57 | 1.05 | 90% | 1,110.33 | 107 min |

### 4.2 Forced routing, batch 1 (2026-10-13 07:03-15:53, 8/8 ok)

Validation first (`results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/validation-forced-routing/validation.md`, all passed): the
patched build with routing natural has the unpatched kernel sequences and DRAM within 0.06%;
fixed and disjoint have one kernel sequence per phase, identical across runs (1,991 prefill /
1,436 decode kernels vs natural's 1,864 / 1,306: the unfused top-k).

Full table with L2, cache and memory: `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/forced-b1/queue/results.md`.

| Workload | Prompt | Prefill ms | Decode ms/step | Decode DRAM GB | Ratio | Prefill DRAM GB | Natural prefill GB | Run time |
|---|---|---|---|---|---|---|---|---|
| moe-prefill-fixed-p16 | 16 | 28.9 | 17.80 | 3.60 | 1.05 | 4.33 | 15.04 | 64 min |
| moe-prefill-disjoint-p16 | 16 | 137.7 | 18.26 | 3.60 | 1.05 | 27.02 | 15.04 | 63 min |
| moe-prefill-fixed-p64 | 64 | 38.2 | 17.97 | 3.60 | 1.05 | 6.07 | 28.08 | 64 min |
| moe-prefill-disjoint-p64 | 64 | 269.3 | 18.82 | 3.60 | 1.05 | 53.42 | 28.08 | 66 min |
| moe-prefill-fixed-p512 | 512 | 158.8 | 18.23 | 3.62 | 1.05 | 21.89 | 64.59 | 63 min |
| moe-prefill-disjoint-p512 | 512 | 399.1 | 19.05 | 3.62 | 1.05 | 74.37 | 64.59 | 64 min |
| moe-decode-fixed-b1 | 128 | 52.0 | 16.85 | 3.60 | 1.05 | 8.48 | 36.72 | 70 min |
| moe-decode-disjoint-b1 | 128 | 305.4 | 17.57 | 3.60 | 1.05 | 56.60 | 36.72 | 71 min |

### 4.3 Batch > 1 (2026-10-13 18:02 - 2026-10-14 07:44, 9/9 ok)

Patched build for every run (routing natural uses the unpatched path, section 3.1). Energy was
measured right after each run. Full table: `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/batch/queue/results.md`.
Experts / layer is measured, not assumed: at batch 8, DRAM of the routed-expert kernel
(`mul_mat_vec_q_moe`) / 0.1955 GB per expert (all layers); from batch 16 llama.cpp runs the
experts through `mul_mat_q`, which also does the dense matmuls, so the count is 8 + (DRAM -
fixed's DRAM at the same batch) / 0.1955. Disjoint checks the method: 64.1 / 128.7 / 258 for
exactly 64 / 128 / 256.

| Workload | Prefill ms | Decode ms/step | Decode tok/s | Decode DRAM GB | Expected GB | Ratio | Experts / layer | Decode BW of ceiling | Prefill DRAM GB | Decode J/token | Peak GiB | Run time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-b1 (4.2, for reference) | 52.0 | 16.85 | 59.4 | 3.60 | 3.44 | 1.05 | 8 | 96% | 8.48 | 0.502 | 55.3 | 70 min |
| natural-b1 (4.1) | 202.0 | 17.47 | 57.2 | 3.60 | 3.44 | 1.05 | 8 | 91% | 36.72 | 0.500 | 70.1 | 70 min |
| disjoint-b1 (4.2) | 305.4 | 17.57 | 56.9 | 3.60 | 3.44 | 1.05 | 8 | 91% | 56.60 | 0.531 | 54.9 | 71 min |
| moe-decode-fixed-b8 | 411.5 | 31.18 | 256.5 | 5.12 | 4.05 | 1.26 | 8.1 | 74% | 73.00 | 0.162 | 59.7 | 74 min |
| moe-decode-natural-b8 | 608.1 | 68.67 | 116.5 | 13.51 | 13.72 | 0.98 | 51.0 | 87% | 118.69 | 0.255 | 59.2 | 76 min |
| moe-decode-disjoint-b8 | 620.4 | 81.75 | 97.9 | 16.08 | 15.00 | 1.07 | 64.1 | 88% | 123.99 | 0.281 | 59.2 | 80 min |
| moe-decode-fixed-b16 | 908.4 | 46.64 | 343.1 | 7.07 | 4.76 | 1.49 | 8 | 67% | 156.36 | 0.077 | 59.6 | 84 min |
| moe-decode-natural-b16 | 1,084.4 | 94.52 | 169.3 | 16.61 | 23.13 | 0.72 | 56.8 | 79% | 199.39 | 0.161 | 92.9* | 88 min |
| moe-decode-disjoint-b16 | 1,081.3 | 159.26 | 100.5 | 30.67 | 28.22 | 1.09 | 128.7 | 86% | 203.27 | 0.271 | 60.6 | 97 min |
| moe-decode-fixed-b32 | 1,834.5 | 73.70 | 434.2 | 10.75 | 6.17 | 1.74 | 8 | 65% | 314.47 | 0.063 | 60.0 | 93 min |
| moe-decode-natural-b32 | 2,203.7 | 149.79 | 213.6 | 24.84 | 36.54 | 0.68 | 80.1 | 74% | 358.81 | 0.135 | 62.8 | 102 min |
| moe-decode-disjoint-b32 | 1,916.8 | 305.30 | 104.8 | 59.63 | 54.66 | 1.09 | 258 | 87% | 361.02 | 0.294 | 59.7 | 124 min |

\* natural-b16's 92.9 GiB is not the workload: memory rose above 70 GiB only from 22:47 to
22:56, during one ncu tier-1 decode pass, while every other b16/b32 run (natural-b32 included)
peaked at 59.6-62.8 GiB. Most likely another process on the shared machine; ~28 GB stayed
available and the memory guard did not trigger.

## 5. Findings

1. **Decode reads what the model says it must, at every prompt length.** 3.59-3.62 GB per
   step for prompts of 16-512 tokens, 1.05x the prediction (3.35 GB routed-active Q8 weights +
   cache read + 0.042 GB KDA state write-back). At 16k context it rises to 3.74 GB (prediction
   3.57: the latent MLA cache adds 0.18 GB), still 1.05x. **The 5% is the KDA state, moved
   about four times per step instead of once** (found 2026-10-13 from per-kernel tier-1 bytes):
   `k_get_rows_float_vec` gathers each sequence's state (read + write, 0.084 GB) and
   `gated_delta_net_cuda` reads it and writes the new one (0.086 GB), 0.17 GB against the
   0.042 GB in the prediction. That is ~0.13 of the +0.16 GB; the rest is small activation
   traffic. It scales linearly with batch (finding 11: 1.40 / 2.80 / 5.61 GB at batch 8 / 16 /
   32, 0.175 GB per sequence). Not the Q8_0 block scales, which the GGUF tensor
   sizes already include.
   How the expectation is computed: `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/results.md`, "How Expected GB is
   calculated".
2. **Decode is bandwidth-bound.** 85-91% of the measured 227.7 GB/s DRAM ceiling at 53-57
   tok/s, GPU busy 83-93% in decode. Under transformers the same model ran 7.1 tok/s with the
   GPU ~30% busy (docs/07).
3. **Prefill bytes follow the experts actually used, and both engines see the same router
   skew.** Measured prefill DRAM vs the weights of the distinct experts the HF sweep observed
   per layer on the same prompt ids (routed 50.05 GB x n/256 + 1.79 GB dense/lm_head):
   p16 15.04 vs 14.3 GB (64 experts), p64 28.08 vs 25.6 (122), p128 36.72 vs 31.7 (153).
   Uniform routing would predict 21.7, 45.2 and 51.1 GB, far above the measurements.
   Caveat: the expert counts were observed on the FP8 model under transformers; the Q8 router
   may pick slightly differently.
4. **From 512 tokens, non-weight traffic dominates the growth.** p512 reads 64.6 GB, more than
   all 51.8 GB of weights, so at least ~13 GB is activations / attention / cache (measured
   directly by fixed routing: ~18.5 GB, finding 8); at 16k the prefill moves 1.11 TB (68 MB per
   token, L2 2.85 TB).
5. **Prefill time.** llama.cpp is 8x faster than transformers at 16 tokens (79 vs 660 ms) and
   5.7x at 512 (353 vs 2,000 ms), but the same at 16k (8.86 vs 8.89 s), where the work is
   compute rather than launch overhead.
6. **Memory.** Peak in use 55-56 GiB for p16-p512 (weights + ~3 GiB), 70 GiB for the 16k run
   (larger compute buffers for one 16k-token ubatch). Run 1 also peaked at 70 GiB, but it
   overlapped with the forced-routing build and CPU tests on the same machine.
7. **Forced routing brackets natural routing as designed (all 8 runs).** At batch-1 decode
   every mode touches 8 experts per layer, and all read 3.60-3.62 GB per step. In prefill the
   order is fixed < natural < disjoint at every length:

   | Prefill DRAM GB | 16 tokens | 64 | 128 | 512 |
   |---|---|---|---|---|
   | fixed (8 experts) | 4.33 | 6.07 | 8.48 | 21.89 |
   | natural | 15.04 | 28.08 | 36.72 | 64.59 |
   | disjoint (all 256 from 32 tokens) | 27.02 | 53.42 | 56.60 | 74.37 |

   Disjoint lands on the formula: 16 tokens x 8 = 128 experts -> 50.05 x 128/256 + 1.79 =
   26.8 GB predicted vs 27.02. Prefill time follows the experts touched (128 tokens: 52 / 202 /
   305 ms; 512 tokens: 159 / 353 / 399 ms fixed / natural / disjoint).
8. **Non-weight prefill traffic is ~35-45 MB per prompt token.** Fixed routing reads only its
   3.35 GB of 8-expert weights, so the rest is everything else (activations, attention, KDA,
   MoE intermediates): 0.98 / 2.72 / 5.13 / 18.54 GB at 16 / 64 / 128 / 512 tokens, i.e. 61 /
   42 / 40 / 36 MB per token. Disjoint agrees (74.37 - 51.8 GB of weights = 22.6 GB at 512).
   Natural = weights of the experts used + this: at 512 tokens 64.59 - 18.5 = ~46 GB of
   weights, about 225 of 256 experts per layer. This traffic, not the weights, is what grows
   with prompt length, and the first place to look for prefill savings (tier-2 data per kernel).
9. The 16k run's ncu prefill collection ended with an "SW Power Cap" throttle flag; this
   affects ncu kernel durations only, not bytes or the baseline timings.

10. **Energy (all 22 runs; method in section 2.1).** GPU rail only, a lower bound on system
    energy (LPDDR5X and CPU are not reported on GB10). Idle 9.4-10.6 W with the model resident.
    - *Decode, batch 1:* 0.50 / 0.50 / 0.53 J per token natural / fixed / disjoint at 28-30 W,
      ~17.5 pJ of GPU energy per DRAM bit; at 16k context 0.51 J (27 W). Same bytes, same
      energy: routing does not matter at batch 1, as for the bytes (finding 7).
    - *Not valid: decode energy of the six moe-prefill-\* workloads.* They generate 4 tokens,
      so each decode segment is 3 steps (~55 ms), shorter than the counter's ~100 ms update and
      straight after a prefill whose power has not settled. Their figures rise with prompt
      length (0.41-0.84 J/token) although the decode work is identical, and their cross-checks
      are 0.82-1.68. They are shown as "n/v" in results.md; their prefill figures are valid.
      A fix for future runs: measure decode with a longer generation than the workload's.
    - *Prefill:* per prompt token, fixed / natural / disjoint 0.044 / 0.133 / 0.225 J at 16
      tokens, 0.019 / 0.084 / 0.156 at 64, 0.015 / 0.072 / 0.116 at 128, 0.014 / 0.033 / 0.037
      at 512: the experts touched set the energy, as they set the bytes. 16k tokens: 498 J,
      0.030 J per token, at only 24.7 W (lower than any other prefill; compute-heavy attention /
      KDA kernels at low power, or the "SW Power Cap" ncu flagged in that run, finding 9).
    - *Batch:* finding 11.
    - *Accuracy:* every valid measurement's cross-checks are within 0.84-1.08; quote absolute
      values as +/-10-15%, comparisons tighter.
11. **Batch > 1: throughput follows bytes per token until the batch gets large; then the KDA
    state and a few non-weight kernels limit it.**
    - *Experts actually used per layer* (measured, table 4.3): natural 51 / 57 / 80 at batch 8
      / 16 / 32, against 56 / 102 / 163 for uniform routing. The router concentrates, so
      natural's expected value is too high and its ratio falls (0.98 / 0.72 / 0.68). The HF FP8
      sweep observed 31 / 50.5 / 86: close at b16 and b32, not at b8 (section 6).
    - *KDA state traffic grows with every sequence.* `k_get_rows_float_vec` + `gated_delta_net_cuda`
      move 1.40 / 2.80 / 5.61 GB per step at batch 8 / 16 / 32 (0.175 GB per sequence), against
      0.34 / 0.67 / 1.34 GB in the prediction (finding 1). That is the whole excess of the forced
      modes: fixed reads +1.07 / +2.31 / +4.58 GB over expected. In fixed-b32 the KDA state is
      52% of all decode bytes (5.61 of 10.75 GB), more than all weights read.
    - *Throughput* (tok/s, batch 1 / 8 / 16 / 32): fixed 59 / 257 / 343 / 434, natural 57 / 117
      / 169 / 214, disjoint 57 / 98 / 100 / 105. Disjoint is flat: every row adds 8 new experts,
      so bytes per token stay ~1.9-2.0 GB and it stays at 86-88% of the DRAM ceiling. Natural
      gains as rows share experts (1.69 / 1.04 / 0.78 GB per token). Fixed stops scaling
      linearly and falls to 74 / 67 / 65% of the ceiling.
    - *What limits fixed routing at large batch* (per-kernel times, fixed-b32's 73.7 ms step):
      expert/dense matmuls (`mul_mat_q`, from batch 16) 25.8 ms, `concat_non_cont` 17.9 ms and
      `gated_delta_net_cuda` 12.9 ms. `concat_non_cont` grows with batch (5.9 / 11.9 / 17.9 ms
      at b8 / 16 / 32) while moving almost no DRAM (0.05 GB at b8): latency-bound per-sequence
      work, likely the KDA conv-state concatenation. With the KDA state traffic it is the first
      thing to look at for batched Kimi on llama.cpp; the weights are no longer the limit.
    - *Energy per token* (batch 1 / 8 / 16 / 32): fixed 0.50 / 0.162 / 0.077 / 0.063 J, natural
      0.50 / 0.255 / 0.161 / 0.135 J, disjoint 0.53 / 0.281 / 0.271 / 0.294 J. Energy follows
      bytes per token: disjoint gains nothing from batching beyond batch 8, fixed-b32 costs 8x
      less than batch 1. GPU power stays 26-31 W except fixed-b8 (41 W); pJ per DRAM bit 17.5-24 (fixed-b8 32).
    - *Memory:* peak 59.2-62.8 GiB at every batch (one outlier explained under table 4.3). The
      batch-32 decode adds only ~8 GiB over the 52 GB of weights.

Bandwidth and energy per run: `results/kimi-linear-48b-a3b-q8_0-gguf__llamacpp/results.md`, sections "Bandwidth" and
"Power and energy" (energy is GPU-rail only on GB10, a lower bound on system energy; method and
accuracy are explained there).

## 6. Comparison with the HF FP8 sweep

Not written yet; all 22 runs are now in, so the data is complete. What should and should not carry over (progress.md, "HF MoE
idle-time conclusions re-scoped"): expert counts, router skew and routing-mode ordering are
properties of the model and should agree; launch counts, idle time, tok/s, the L2/DRAM ratio
and cache size are properties of the engine and will not. FP8 and Q8_0 are also different
weights (3.50 vs 3.35 GB routed-active per batch-1 decode step).
