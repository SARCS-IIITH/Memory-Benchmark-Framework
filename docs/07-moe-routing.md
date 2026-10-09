# MoE Routing Workloads

How the harness controls which experts a mixture-of-experts (MoE) model uses, why that matters for memory traffic, and what each of the MoE workloads should show. The numbers are for **Kimi-Linear-48B-A3B (FP8)**, the first MoE model run on this harness.

> **Status (2026-10-06):** **all 22 workloads have run on the GPU; results are in [section 5](#5-measured-results-2026-10-06).** The batch-1 self-check passed (all three routing modes within 0.1%). The CPU test still passes (28/28, `experiments/kimi-linear-compat/check_routing_cpu.py`). Sections 1–4 are the design and the predictions as written before the runs; section 5 is what was measured.

---

## 1. Background: why routing changes memory traffic

Kimi's weights total about 50 GB. Each token uses only a small part of them:

| Part | Size | When it is read |
|---|---|---|
| **Always-read**: attention (KDA + MLA), the shared expert, dense layer 0, routers, norms, output layer (LM head) | ~2.0 GB | Every pass, whatever the routing |
| **Routed experts**: 256 experts in each of 26 MoE layers, ~7 MB each per layer | ~47 GB | Only the experts a pass actually uses |

In each MoE layer, the **router** picks **8 of the 256 experts** for every token. The GPU must read an expert's weights from memory before it can use that expert.

The key fact: **if several tokens in the same pass pick the same expert, its weights are read once and shared by all of them.** So the memory a pass reads depends on **how many different experts the pass touches**, not on how many tokens it holds.

A **pass** is one run through the model:
- **Decode:** one pass makes one new token for each sequence in the batch, so **tokens per pass = batch size**.
- **Prefill:** one pass processes the whole prompt, so **tokens per pass = prompt length**.

**Each token always uses exactly 8 experts per layer, at every batch size.** What grows with batch size or prompt length is the number of *different* experts the whole pass needs. That number is between:
- **8**, when all tokens pick the same experts, and
- **min(256, 8 × tokens per pass)**, when no two tokens share one.

### Nothing carries over between passes

The GPU's L2 cache is **25 MB**. One decode step reads about **3.5 GB** of weights. By the time the next step starts, the previous step's weights are long gone from the cache, so every step reads its experts from memory again, even if they are the same experts as last time.

So:
- **Within one pass**, tokens that share experts save memory. Routing matters here.
- **Across passes**, nothing is saved. Using the same experts again next step costs exactly as much as using new ones.

---

## 2. The three routing modes

Set with the `routing:` key in a workload file.

| Mode | What happens | Different experts per pass | Role |
|---|---|---|---|
| `natural` | The model's own router chooses. Nothing is changed. | Between the two bounds, by chance overlap | The realistic result |
| `fixed` | Every token is forced to experts **0–7** in every layer | Always **8** | **Lower bound**: least memory possible |
| `disjoint` | Token 1 gets experts 0–7, token 2 gets 8–15, and so on, wrapping at 256. The block moves on every pass, so successive steps use new experts. | **min(256, 8 × tokens)** | **Upper bound**: most memory possible |

The two forced modes give the floor and the ceiling. `natural` shows where the real model falls between them.

### What forcing changes, and what it doesn't

- **Changes:** after the model loads, the harness attaches a hook to each of the 26 router modules (`KimiMoEGate`). The router still runs normally, but the hook replaces its chosen experts with the forced ones. It also gives the 8 experts equal weights (1/8 each of the router's usual per-token total), so the numbers flowing through the model stay a sensible size.
- **Unchanged:** the weights on disk, the downloaded model code (`modeling_kimi.py`), attention and the expert computations themselves. Nothing persists after the process exits.
- **Consequence:** tokens are processed by experts the model didn't choose, so **the generated text is meaningless** in `fixed` and `disjoint` runs. The **memory traffic is real**: the forced experts' weights are genuinely read and used. The run manifest carries a `routing_warning` saying so.

### Why forcing and not prompt choice

A prompt such as a single token repeated tends to make tokens route alike. But it can't *guarantee* 8 experts per layer:
- in prefill, the same token at different positions still has different internal state;
- in decode, the model picks the next token itself.

So a prompt gives only an *approximate* lower bound, and you only learn how close it got by measuring. It can't produce the disjoint case at all. Forcing gives exact bounds. A prompt-based variant (real router, repeated-token prompt) could be added later as a realistic "concentrated" case. The observer described below would measure how close to 8 it gets.

### The observer

In **baseline** runs only, the hook also records which experts were used: for each layer and phase, how many different experts each pass touched, and how often each expert was picked. It is not installed in nsys or ncu runs, because its few extra small GPU kernels would show up in the profiles.

The result goes into the run manifest under `backend.routing`:

| Field | Meaning |
|---|---|
| `distinct_experts_per_pass` | Mean, minimum, maximum and per-layer value. **The measured version of the table in section 4.** |
| `expected_distinct_experts_per_pass` | The prediction, for comparison |
| `experts_ever_used_per_layer` | How many experts were used at least once over the whole run |
| `top_k_share_per_layer` | Share of all picks that went to the layer's 8 most popular experts. 8/256 ≈ 0.03 means perfectly even; 1.0 means one fixed set. |

For `natural`, this shows whether the router really spreads tokens evenly, as the formula assumes, or favours some experts.

---

## 3. Expected bytes (the predicted-vs-measured check)

For every decode step the harness **predicts** how many bytes must be read from memory, **measures** the real number with Nsight Compute, and compares the two.

> **Measured bytes now come from nsys by default** (2026-10-06). ncu tier 1 is off: on Kimi a decode step is ~12,000 kernel launches, so tier 1 takes many hours per run. Instead nsys samples all **L2 traffic** (`lts__t_sectors`) during the normal timeline pass, and the check compares that against the expectation. The report labels it "Measured (nsys-sampled L2)".
>
> L2 traffic is an upper bound on DRAM traffic. On Qwen3-0.6B decode it read 1.42 GB against ncu's exact 1.34 GB DRAM (+6%; +10% against ncu's own L2 count). For prefill it read ~1.9× DRAM, because activations are reused in L2. So the decode ratios run slightly high, and prefill GB are L2 traffic, not DRAM. This was validated on a small dense model only, not on Kimi. The method and the validation are in [08-nsys-l2-sampling.md](08-nsys-l2-sampling.md).

```
expected bytes = always-read weights                    (~2.0 GB)
               + routed experts × (different experts ÷ 256)  (~47 GB × fraction)
               + cache read, plus the KDA state written back
```

- A ratio between **0.75× and 1.6×** is reported as consistent. That confirms both the measurement and our model of what the step does.
- A ratio outside that range points to a bug or an effect not accounted for. The report gives the usual causes.

**What changed.** The "different experts" figure used to assume **batch 1 with natural routing**, which is always 8. Now it follows the workload's batch size and routing:

| Mode | Different experts per layer for T tokens per pass |
|---|---|
| `fixed` | 8 |
| `disjoint` | min(256, 8T) |
| `natural` | 256 × (1 − (1 − 8/256)^T) |

The `natural` formula assumes each token picks experts **evenly and at random**. Real routers have favourite experts, so the true number is probably somewhat **lower**. The observer measures it.

At batch 1 with natural routing, the new calculation gives **exactly the same number as before**, so existing results don't move.

The automatic check covers **decode only**. Prefill numbers in section 4 are for comparing by hand.

---

## 4. What each workload should show

All 22 files are in `configs/workloads/moe/`. Prompts are currently random token IDs (`prompt_source: synthetic`). A different sequence per batch row is already in place, but real text is still to do.

### 4.1 Decode sweep: 12 runs, prompt 128, 32 new tokens

Files: `moe-decode-{natural,fixed,disjoint}-b{1,8,16,32}.yaml`

Different experts the step must load, and expected weight bytes **per decode step**:

| Batch | Fixed | Natural | Disjoint |
|---|---|---|---|
| 1 | 8 → **3.5 GB** | 8 → **3.5 GB** | 8 → **3.5 GB** |
| 8 | 8 → **3.5 GB** | ~57 → **12.6 GB** | 64 → **13.8 GB** |
| 16 | 8 → **3.5 GB** | ~102 → **20.8 GB** | 128 → **25.6 GB** |
| 32 | 8 → **3.5 GB** | ~163 → **32.1 GB** | 256 → **49.2 GB** |

**Batch 1: all three modes should match.** One token per step means exactly 8 experts per layer, in any mode:
- `fixed` reuses experts 0–7 every step;
- `disjoint` uses 0–7, then 8–15, then 16–23, …;
- `natural` uses whatever the router picks.

Every step reads 8 experts' weights either way. Reusing the same 8 gives no advantage, because they don't survive in the 25 MB cache until the next step (section 1). This makes batch 1 a **self-check**: if `fixed` comes out clearly lower than `disjoint` here, the hook or the measurement is wrong.

**Larger batches: the modes separate.** At batch 32, one step processes 32 tokens together, each picking 8 experts:
- **Fixed:** all 32 pick experts 0–7. The step loads 8 experts once and all 32 tokens share them. Total stays 3.5 GB, so cost per token falls from 3.5 GB at batch 1 to about **0.11 GB** at batch 32. This is the best batching can ever do.
- **Disjoint:** token 1 takes 0–7, token 2 takes 8–15, … token 32 takes 248–255. Nothing is shared, so the step loads **all 256 experts**, about 49 GB, or about 1.5 GB per token. This is the worst case.
- **Natural:** tokens overlap by chance, about 163 different experts, about 32 GB, or about 1.0 GB per token.

**The main result is where `natural` lands between the two bounds.** Close to `fixed` means the router keeps reusing a few popular experts and batching is very effective. Close to `disjoint` means tokens are spread out and batching saves little on expert traffic.

### 4.2 Short prefill: 9 runs, batch 1, 4 new tokens

Files: `moe-prefill-{natural,fixed,disjoint}-p{16,64,512}.yaml`

In prefill, all prompt tokens go through in **one pass**, so prompt length acts like batch size above.

| Prompt | Fixed | Natural | Disjoint |
|---|---|---|---|
| 16 | 8 → ~3.5 GB | ~102 → ~21 GB | 128 → ~26 GB |
| 64 | 8 → ~3.5 GB | ~221 → ~43 GB | 256 → ~49 GB |
| 512 | 8 → ~3.5 GB | ~256 → ~49 GB | 256 → ~49 GB |

What to look for:
- **Short prompts (16, 64)** are where routing matters. Even a 16-token prompt can pull in a large share of the experts.
- **At 512 tokens, `natural` and `disjoint` should look almost the same:** nearly every expert gets used anyway. Only `fixed` stays small.
- These are **weight** bytes only. Prefill also moves activations, and it is not covered by the automatic check, so compare against this table by hand.

### 4.3 Long context: 1 run, natural routing, batch 1, prompt 16,384, 32 new tokens

File: `moe-longctx-natural-p16384.yaml`

This run is about the **cache**, the stored memory of earlier tokens. Kimi has two kinds of attention layer:

| Layer type | Count | Cache size |
|---|---|---|
| **KDA** (linear attention) | 20 | **Fixed**, about 40 MB in total however long the text |
| **MLA** | 7 | **Grows by about 143 KB per token** as transformers stores it, so about 2.3 GB at 16k tokens |

Expected per decode step: about **3.5 GB of weights + about 2.3 GB of cache**. We expect the MLA cache to become a large share of decode traffic while the KDA part stays flat. That is the hybrid design's advantage, measured.

MLA could store a compressed cache of about 8 KB per token, but the transformers code stores the expanded form, so we measure what transformers actually does. The report shows both sizes side by side.

### 4.4 Time won't follow bytes exactly

Kimi's MoE code runs each *used* expert as separate small GPU jobs, and it stops to sync with the CPU in every MoE layer. More different experts means more jobs, so:
- `disjoint` and large-batch `natural` runs will be **slower than their extra bytes alone explain**;
- this overhead is likely why decode runs at about 10 tokens/s instead of the ~90 the memory bandwidth would allow.

**Use bytes for comparing routing modes.** Time also includes this launch overhead.

---

## 5. Measured results (2026-10-06)

All 22 workloads ran on the DGX Spark on 2026-10-06 (runs 1–21 from `scripts/run_moe_queue.sh`, 03:14–13:44; long context re-run 16:07–16:20 after the first attempt crashed the machine, see 5.6). Every completed run passed every stage.
- Raw per-run results and caveats: `runs/moe-queue/results.md` and `findings.md`.
- Run-by-run log: `progress.md`.

**Read the GB figures with care.** The bytes here are **L2 traffic sampled by nsys**, not DRAM bytes; ncu tier 1, the only exact DRAM source, takes ~13 h per Kimi run ([08-nsys-l2-sampling.md](08-nsys-l2-sampling.md)). On Kimi, L2 traffic runs at about **1.9× the weight bytes a step needs** (5.5); DRAM itself was never measured on Kimi. So compare the GB *between* runs, not as absolute DRAM figures. Expert counts come from the routing observer (section 2) and are exact.

### 5.1 Decode sweep: measured

Per decode step. Experts = distinct experts per layer per step, observed (the section 4.1 prediction in brackets).

| Batch | fixed: L2 / time / tok/s | natural: experts / L2 / time / tok/s | disjoint: experts / L2 / time / tok/s |
|---|---|---|---|
| 1 | 6.32 GB / 141 ms / 7.1 | 8 (8) / 6.32 GB / 141 ms / 7.1 | 8 / 6.33 GB / 137 ms / 7.3 |
| 8 | 8.56 GB / 138 ms / 57.8 | **31 (57)** / 16.77 GB / 364 ms / 22.0 | 64 / 28.16 GB / 658 ms / 12.2 |
| 16 | 11.13 GB / 147 ms / 108.5 | **50 (102)** / 26.15 GB / 550 ms / 29.1 | 128 / 53.16 GB / 1,221 ms / 13.1 |
| 32 | 15.25 GB / 160 ms / 200.2 | **86 (163)** / 42.97 GB / 896 ms / 35.7 | 256 / 102.07 GB / 2,379 ms / 13.4 |

**Batch 1: the self-check passed.** natural / fixed / disjoint read 6.3204 / 6.3225 / 6.3269 GB per step, all within 0.1%. Reusing the same 8 experts every step (`fixed`) saves nothing over 8 new ones (`disjoint`), exactly as section 4.1 predicted: experts do not survive in L2 between steps.

**Larger batches: traffic is linear in distinct experts.**

| | Batch 8 | Batch 16 | Batch 32 |
|---|---|---|---|
| L2 per extra expert, from fixed → disjoint | 0.350 GB | 0.350 GB | 0.350 GB |
| natural's effective experts, read off that line | 31.5 | 50.9 | 87.2 |
| natural's experts, counted by the observer | 31.2 | 50.5 | 86.0 |
| Time per extra expert per step | 9.3 ms | 9.0 ms | 9.0 ms |

Two independent measurements, nsys L2 and the routing observer, agree on how many experts natural routing loads. Predictions made before each run from this line landed within 0.02–1.6%.

**The main result: where `natural` lands.** It sits about **a third of the way from `fixed` to `disjoint`**, closer to the best case than section 4.1 assumed:

| Batch | natural's position between fixed (0) and disjoint (1), by experts | by L2 |
|---|---|---|
| 8 | 0.41 | 0.42 |
| 16 | 0.35 | 0.36 |
| 32 | 0.31 | 0.32 |

The router uses **about half the experts** the uniform-random formula of section 3 predicts, at every batch size. Kimi's router has strongly favoured experts (5.4), so batching shares experts far more than chance would.

**What batching buys, by routing mode** (decode throughput, batch 1 → 32): `fixed` 7.1 → 200 tok/s (**28×**, step time nearly flat at 138–160 ms); `natural` 7.1 → 36 tok/s (**5×**); `disjoint` 7.3 → 13 tok/s (**1.8×**). Under natural routing, step time grows by ~9 ms per extra expert, so most of batching's potential gain is lost to Kimi's per-expert kernel launches (5.5).

### 5.2 Short prefill: measured

Per prefill. The 128-token row comes from the batch-1 decode runs (their prompt is 128 tokens).

| Prompt | fixed: L2 / time | natural: experts (formula) / L2 / time | disjoint: experts / L2 / time |
|---|---|---|---|
| 16 | 8.8 GB / 0.135 s | **65 (102)** / 28.6 GB / 0.66 s | 128 / 50.8 GB / 1.22 s |
| 64 | 14.8 GB / 0.142 s | **122 (222)** / 54.7 GB / 1.19 s | 256 / 101.6 GB / 2.34 s |
| 128 | 25.2 GB / 0.155 s | **153 (252)** / 74.4 GB / 1.50 s | 256 / 109.8 GB / 2.38 s |
| 512 | 85.4 GB / 0.217 s | **204 (256)** / 151.0 GB / 2.00 s | 256 / 163.0 GB / 2.36 s |

- **Short prompts are where routing matters most, as predicted**, but natural routing reaches far fewer experts than section 4.2 expected: 16 tokens reach 65 experts (not 102), and 512 tokens reach 204 (not ~256).
- **Section 4.2 expected natural and disjoint to look almost the same at 512 tokens.** By expert count (204 vs 256) and time (2.00 vs 2.36 s) they are close but not the same.
- **The same 0.35 GB per extra expert holds in prefill** (16 and 64 tokens); at 512 tokens it is 0.335.
- **Prefill time follows experts, not tokens.** `fixed` prefill barely changes with prompt length (0.135 → 0.217 s for 16 → 512 tokens), while natural/disjoint take 5–10× longer at the same length. Natural-p64 (122 experts) takes as long as disjoint-p16 (128 experts).
- **The fixed floor grows with prompt length** (activations): 8.8 / 14.8 / 25.2 / 85.4 GB at 16 / 64 / 128 / 512 tokens, slightly less than linearly.
- **Anomaly at large batched prefills:** at batch 8 and 16 (1,024 and 2,048 tokens), natural prefill read *more* L2 than disjoint despite fewer experts. See 5.4; L2 GB do not rank routing modes reliably for large prefills.

### 5.3 Long context: measured

| | 128-token context (b1 decode runs) | 16,384-token context |
|---|---|---|
| Decode time per step | ~140 ms | **174 ms** |
| Decode L2 per step | ~6.3 GB | **13.35 GB** (+7.0 GB) |
| Cache (measured) | small | **2.40 GB** (section 4.3 predicted ~2.3 GB) |
| KDA recurrent state | 42 MB | **42 MB**: flat, as designed |
| Prefill | – | 8.89 s (1,843 tok/s), GPU busy 92% |

- **The cache prediction held:** 2.40 GB measured against ~2.3 GB predicted for MLA as transformers stores it (expanded). The compressed MLA latent would be 0.13 GB, 18× smaller. The report shows both.
- **The hybrid design works as intended:** only the 7 MLA layers grow with context; the 20 KDA layers' state stays at 42 MB.
- **Cache sizes across all 22 runs** (measured from the live cache tensors at the end of generation):
  - MLA K/V is **143.4 KB per token per sequence** in every run, matching the ~143 KB predicted.
  - KDA recurrent + conv state is **41.9 + 2.0 MB per sequence**, independent of context.
  - Routing doesn't change either.
  - Below **~292 tokens per sequence** the fixed KDA state is the larger part of the cache: ~2/3 of it in the 160-token decode sweep.
  - At 16k tokens MLA K/V is 98% of the cache. All-MLA attention would have needed ~9 GB there instead of 2.4 GB.
  - Full table: `runs/moe-queue/results.md`, "KV cache and recurrent state".
- **But the cache costs ~2.9× its size in traffic per step** (+7.0 GB for a 2.4 GB cache). The cache in use is Kimi's own `KimiDynamicCache`, and its `update()` **does** rebuild each layer's K and V with `torch.cat` on every token (confirmed in `modeling_kimi.py`, 5.7). *Still a hypothesis:* that this explains the ~3×: each step reads the old cache, writes a new copy, then attention reads it, ≈ 3 × 2.4 GB. The traffic split itself was not measured.
- **The MLA layers run as standard multi-head attention**, which is why the cache is the expanded 143 KB/token and not MLA's 8 KB/token. See 5.7.
- **The time cost is modest** (+34 ms/step), because decode is launch-bound (5.5).

### 5.4 Router skew, and the prefill anomaly it causes

The observer's `top_k_share_per_layer` (share of all token → expert assignments that go to each layer's 8 most-used experts) is **~0.44 for natural routing** in batched prefill, against 0.031 for perfectly even routing (8/256). Natural routing piles tokens onto a few favourite experts.

That is why natural uses about half the experts the formula predicts (5.1, 5.2). It also produces an L2 artefact in large prefills:

| Prefill | natural: experts / L2 | disjoint: experts / L2 |
|---|---|---|
| batch 8 × 128 tokens | 221 / **239.1 GB** | 256 / **235.9 GB** |
| batch 16 × 128 tokens | 233 / **411.4 GB** | 256 / **380.8 GB** |

**Mechanism:** a popular expert receiving ~450 tokens is processed in several 128-token tiles, and each tile re-reads the expert's weights through L2. This was confirmed with ncu on non-expert matrix multiplies in the 512-token prefill (⚠️ ncu tier 2 sampled only the first ~4–5 linear layers, so this is the same kernel family, not the experts themselves): 4 token tiles → **7.5× more L2 traffic than DRAM traffic, 84% L2 hits**. The re-reads are almost all hits, so natural routing very likely still reads *less DRAM* than disjoint here, even though it moves more L2. At 512 tokens per pass the skew effect is smaller, and disjoint is back above natural (163 vs 151 GB). Decode is unaffected: at most 32 tokens per step, so every expert fits in one tile.

### 5.5 What limits Kimi: launches, not bandwidth

- **The GPU is idle most of the time.** Busy share in decode: 29–54% in every run. In prefill it's 29–66%, except where few experts process many tokens per pass (fixed routing with ≥ 512 tokens: 84–98%) and the 16k-token prefill (92%).
- **Time follows kernel launches per used expert.** Kimi's `moe_infer` loops over used experts in Python and syncs with the CPU in every MoE layer. Decode steps run 12k launches at 8 experts, 34k at natural-b8, and 238k at disjoint-b32. Each extra expert costs ~9 ms per step regardless of batch.
- **Section 4.4 predicted this:** decode runs at **7.1 tok/s** at batch 1 (predicted "about 10"), far below what memory bandwidth would allow.
- **L2 traffic is ~2× the weights read.** Each extra expert adds 0.35 GB of L2, against ~0.18 GB of weights (47 GB of routed experts ÷ 256). On Qwen3-0.6B decode, L2 was only 6% above DRAM ([08](08-nsys-l2-sampling.md)). This is why the absolute GB here should not be read as DRAM bytes, and why the decode physics-check ratios come out at ~1.2–2.3×.

**Why ~2×? Partly explained (2026-10-06, from existing traces and ncu tier 2 of the b1 decode runs):**
- **No weight dequantization.** The decode step multiplies directly with FP8 weights (CUTLASS FP8 GEMM, ~904 launches per step). There is no kernel converting weights to bf16, so "weights unpacked first" is ruled out.
- ⚠️ *ncu tier 2, first-layer sample.* **The FP8 GEMM itself moves ~1.54× its memory traffic through L2, even at one token:** 45.4 MB of L2 against 29.4 MB filled from memory over 5 sampled launches. This is the same in natural, fixed and disjoint. For comparison, Qwen's whole bf16 decode step was ~1.06–1.10×. Why this kernel does so is not established; likely its tiling or operand loading re-touches lines that are already in L2.
- **Per-token activation quantization** (~900 launches each of abs, max, clamp, convert-to-FP8 per step) runs at ~4× L2/memory, but moves little data at batch 1.
- **Unexplained remainder:** 1.54× covers most but not all of the ~1.9× (0.35 GB of L2 per extra expert against ~0.18 GB of expert weights). Candidates: the per-expert small kernels, the expert-size estimate, and ncu's cold-cache replay (its memory fills are an upper bound).
- **Caveat:** the sampled GEMM launches are from early layers (tier 2 takes the first launches), not necessarily routed experts; same kernel family.

### 5.6 Caveats

1. **Bytes are L2 traffic, not DRAM** (see the top of this section and 5.5). Relative comparisons hold. Absolute GB are ~1.9× the weight bytes needed; how much they overstate actual DRAM traffic on Kimi has not been measured. The FP8 GEMM alone shows 1.54× L2/memory.
2. **No exact-DRAM cross-check exists on Kimi.** The L2-vs-DRAM validation was on Qwen3-0.6B at batch 1.
3. **Prompts are synthetic random token IDs.** Natural routing on real text may spread differently. Real-text prompts are still to do (section 8).
4. **Large prefills:** L2 GB do not rank routing modes reliably (5.4). Use expert counts and time.
5. **Forced routing makes the generated text meaningless.** Only the traffic is real.
6. **Stall data (ncu tier 2) exists for 8 of the 22 runs only**, and its 60-launch sample covers only the first ~4–5 linear layers: it never reached the routed experts. **No routing conclusion in this section depends on ncu**; the ncu-based points (5.4 mechanism, 5.5 GEMM ratio) are marked ⚠️. Which runs and why: `runs/moe-queue/findings.md`, "ncu coverage".
7. **The first long-context attempt, with ncu, crashed the DGX by exhausting memory.** The re-run without ncu is the one reported here. Never run ncu on Kimi long-context prefill on this machine. Why memory grew so far (~100 GB in one process): 5.7.
8. **Batch-32 ncu runs were collected under memory pressure** (~190 driver allocation failures, no data lost). Timing and L2 come from separate processes and are unaffected.


### 5.7 MLA runs as standard attention, and why memory reached ~100 GB

*Added 2026-10-10. Prompted by an `nvidia-smi` screenshot taken during the queue: at **14:20:12 on 2026-10-06**, one `nsbench` Python process (PID **330905**) was using **94,977 MiB (~99.6 GB)** of GPU memory.*

#### MLA layers are expanded and run as standard multi-head attention

Kimi-Linear's 7 MLA layers are designed to cache a small compressed form of each token's keys and values. In the transformers code that ran here (`modeling_kimi.py`, `KimiMLAAttention.forward`), each MLA layer:

1. computes the compressed form, `compressed_kv`: 512 latent values + 64 position (RoPE) values per token;
2. **immediately expands it** with `kv_b_proj` into full keys and values for all 32 heads (`num_key_value_heads = 32`);
3. **stores the expanded keys and values** in the cache (`past_key_values.update(key_states, value_states, …)`);
4. runs ordinary attention over them (`sdpa`, since `flash_attention_2` isn't installed).

So at run time **the MLA layers behave exactly like standard multi-head attention (MHA)**. The compressed form only exists for a moment inside each layer. The 64 position values per token are also stored 32 times, once per head, although they are the same for every head.

| Cached per token, all 7 MLA layers | Bytes | At 16,416 tokens (run 22: 16,384 prompt + 32 generated) |
|---|---|---|
| **Expanded K + V (what ran)**: 32 heads × (192 K + 128 V) × 2 B × 7 | **143,360 B** | **2.353 GB**, measured exactly: `cache_state_bytes.kv = 2,353,397,760` |
| Compressed latent (what MLA is designed to cache): (512 + 64) × 2 B × 7 | 8,064 B | 0.132 GB |

The expanded cache is **17.8× larger** than MLA intends. On top of that, the cache class in use (`KimiDynamicCache`, recorded in the run manifest) grows each layer's K and V with `torch.cat` on **every token**, which builds a new, larger copy of the whole layer cache each step. That is the likely source of the extra decode traffic at long context (5.3).

An inference engine that implements MLA properly caches the compressed form and doesn't copy the cache on every token. **llama.cpp does this for Kimi-Linear** (confirmed from its code, 2026-10-10, not yet from a run): its converter (`conversion/kimi_linear.py`) splits `kv_b_proj` into `k_b_proj` and `v_b_proj`, and its runtime (`src/models/kimi-linear.cpp`, "MLA KV cache enabled" branch) folds the key expansion into the query (`q_nope_absorbed`), caches only the 512 compressed + 64 position values per token as one key/value shared by all 32 query heads, and applies `v_b_proj` after attention. That is 8,064 B/token instead of 143,360 B, ~0.13 GB instead of 2.35 GB at 16k tokens. An older GGUF without the split takes a fallback branch that expands like transformers. To confirm on a run: the load log's "KV buffer size" line.

#### Which run the ~100 GB process was

| Evidence | What it shows |
|---|---|
| `runs/moe-queue/queue.log`: `13:44:21 START 22/22 moe-longctx-natural-p16384 ncu=ncu` | At 14:20 the queue was on run 22, the 16,384-token long-context run, **with ncu on** |
| `runs/20261006T081421Z__…__moe-longctx-natural-p16384__hf__moe/logs/ncu_tier2_prefill.stdout.log`, first line: `==PROF== Connected to process 330905` | **PID 330905 was the process being profiled by ncu tier 2 during the 16k-token prefill** |
| Same log, last line: `Profiling "unrolled_elementwise_kernel": 0%` | ncu never finished profiling even its first kernel |

So the ~100 GB was **run 22's ncu stage**: the stage that later ran the DGX out of memory (caveat 7 in 5.6; progress.md, run 22). The other 21 runs used short contexts, and their processes stayed at ~50–58 GB.

#### Why that process used so much memory

The ~100 GB is four things stacked on top of each other:

| Part | Size | Where the number comes from |
|---|---|---|
| **1. FP8 weights** | **~50 GB** | Measured in every Kimi run: "Model weights resident" 50.01 GB |
| **2. Working memory for a 16,384-token prefill** | **~8.8 GB** above the weights | Same workload without any profiler (re-run `runs/20261006T103757Z__…__moe-longctx-natural-p16384__hf__moe`): PyTorch peak allocated **58.82 GB** |
| …of which the **expanded MLA cache** | 2.35 GB (compressed would be 0.13 GB) | `cache_state_bytes.kv`, above |
| …of which KDA recurrent + conv state | 0.04 GB | `cache_state_bytes`, flat at any context length |
| …the rest: prefill activations for 16k tokens | ~6.4 GB | Hidden states, MoE intermediates and attention inputs for 16,384 tokens at once. In the short runs this is ~0–2 GB. |
| **3. PyTorch holding freed memory for reuse** | **~6.9 GB** | Same re-run: peak *reserved* 65.70 GB vs peak *allocated* 58.82 GB. PyTorch's caching allocator keeps freed blocks instead of returning them, so nvidia-smi still counts them. The nsys allocation timeline agrees: 65.71 GB. |
| **4. ncu's kernel replay** | **~34 GB** | *Inferred, not measured directly:* 99.6 GB (nvidia-smi) − 65.7 GB (the same workload without ncu) |

**Why ncu adds so much.** ncu measures a kernel by running it ~15 times and reading different counters each time. For each run to give the same result, it saves a copy of the memory the kernel may change before the first run, and restores it before each later one. Those copies are made inside the profiled process, so nvidia-smi counts them against it. In a 16k-token prefill the tensors a kernel touches are large, so the saved copies are large too. How much ncu saved was not logged; ~34 GB is the difference between the two measurements above.

**Why it crashed the machine instead of just failing.** GB10 has no separate GPU memory: the CPU and GPU share one ~119 GB pool. ~100 GB in this one process, plus the operating system, the file cache and other users' processes, exhausted it. The kernel's out-of-memory killer then couldn't free anything useful. ncu marks the profiled process as unkillable (`oom_score_adj -1000`), and its memory is GPU allocations rather than ordinary process memory, so the killer only removed a small desktop process. The machine thrashed for ~1.5 h before it went down (progress.md, run 22).

#### So was MLA expansion the reason?

**Only a small part: ~2.2 GB of the ~100 GB.** The memory grew because of, in order of size:

1. **The weights: ~50 GB**, fixed for this model in FP8.
2. **ncu's replay copies: ~34 GB** (inferred), only present while ncu profiles, and large because the prompt was large.
3. **The 16k-token prompt: ~8.8 GB** of prefill working memory, including the 2.35 GB expanded MLA cache (2.2 GB more than the compressed form would need).
4. **PyTorch's held blocks: ~6.9 GB.**

#### What to take from this

- **Never run ncu on a long-context prefill of a large model on this machine**, whichever engine runs it. The replay copies scale with the data each kernel touches. Long-context runs: `--skip-ncu` plus the memory guard in `scripts/rerun_moe_longctx.sh`.
- **The MLA expansion is a property of HF transformers' Kimi code, not of the model.** Reports already show both cache sizes (expanded and compressed). An engine that keeps the compressed form would cut the 16k-token cache from 2.35 GB to 0.13 GB and remove the per-token `torch.cat` copy, but would not change parts 1, 2 (activations) or 4.
- **llama.cpp (see [09-llamacpp-comparison.md](09-llamacpp-comparison.md) section 9.6)** would change parts 2 and 3: a compressed MLA cache (confirmed from its code, above), and memory reserved once at load instead of PyTorch's caching allocator. It would not change ncu's replay copies (part 4), so the rule above still applies.

---

## 6. Files changed

| File | Change |
|---|---|
| `nsight_bench/compat/routing.py` (**new**) | The routing hook. `install_routing(model, mode, observe)` finds every top-k router that returns `(expert indices, expert weights)`, such as Kimi's `KimiMoEGate`, and attaches a forward hook. For `fixed`/`disjoint` it replaces the indices and weights; with `observe` it records what was used. Also `expected_distinct_experts()`, the formulas in section 3. Forced indices are built from shapes and a step counter only, so the GPU never has to wait for the CPU. |
| `nsight_bench/compat/__init__.py` | Exports `install_routing`, `RoutingController`, `ROUTING_MODES` |
| `nsight_bench/config.py` | `WorkloadConfig.routing` (default `natural`, anything else rejected). New `ModelConfig.routed_read_fraction(tokens_per_pass, routing)`. `decode_read_weight_bytes()` now takes `batch_size` and `routing`; the defaults give the old batch-1 answer. |
| `nsight_bench/backends/hf_transformers.py` | `load()` installs the hook, which observes in baseline runs only. `prefill()`/`decode_step()` tell it the current phase. `describe()` writes `routing` and `routing_warning` into the manifest. `teardown()` removes the hook. |
| `nsight_bench/backends/base.py` | New `Backend.profile_mode` ("baseline", "nsys" or "ncu") |
| `nsight_bench/worker.py` | Sets `backend.profile_mode` before loading |
| `nsight_bench/analysis/assemble.py` | The decode check passes the workload's batch size and routing into `decode_read_weight_bytes()`. The expectation's source label records both. |
| `configs/workloads/moe/*.yaml` (**new**, 22 files) | The workloads in section 4 |

Not changed: the downloaded Kimi files, workload code in `nsight_bench/workloads/`, the ncu runner, and the reports.

### Limits

- **Forced routing supports DeepSeek-V3-style routers only:** a class named `…Gate` with `top_k` and `num_experts` that returns `(indices, weights)`. Kimi qualifies. Qwen3-MoE and Mixtral compute top-k inside the MoE block instead, so a forced mode on those **stops with an error** rather than silently running natural routing.
- `natural` on a model with no such router (any dense model, such as Qwen3-0.6B) installs nothing and behaves exactly as before.

---

## 7. Running

All 22, queued, inside tmux (GPU must be idle; the script waits if it is not):

```bash
tmux new-session -d -s moe-queue -c ~/Memory-Benchmark-Framework "./scripts/run_moe_queue.sh; exec bash"
tail -f runs/moe-queue/queue.log                            # one line per run
~/envs/nsbench/bin/python scripts/summarize_moe_queue.py    # -> runs/moe-queue/results.md
```

One run by hand:

```bash
nsbench run --model configs/models/kimi-linear-48b-fp8.yaml \
            --workload configs/workloads/moe/moe-decode-fixed-b8.yaml --profile configs/profiles/moe.yaml
```

Then:
- in the run's `metrics/baseline_result.json`, check `backend.routing.phases.decode.distinct_experts_per_pass` against section 4;
- in the report, check the decode expectation ratio. Without tier 1 it is measured against nsys-sampled L2 traffic; see the note in section 3.

## 8. Still to do

1. ~~CPU test of the hook~~: **done, 28/28 pass.** Run it with `~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_routing_cpu.py`. It builds one Kimi MoE layer from Kimi's own code, shrunk to 16 small experts, and checks that:
   - `natural` is bit-identical;
   - `fixed` gives exactly k experts;
   - `disjoint` gives no overlap and rotates;
   - the install guards and the 22 workload files work;
   - the formulas and Kimi's expected bytes match section 3.
2. ~~**GPU check**~~ **done 2026-10-06 (section 5):**
   - ✅ the three batch-1 decode runs read the same bytes (within 0.1%);
   - ✅ at batch 32, fixed < natural < disjoint (15.3 < 43.0 < 102.1 GB);
   - ⚠️ "each within about 10% of the table" **could not be tested as written**. The table predicts DRAM weight bytes, but the runs measured L2 traffic, which on Kimi is ~2× the weight bytes (5.5). Natural routing also used about half the formula's experts (5.1). The forced modes match their exact expert counts, and the per-expert slope is constant (0.35 GB of L2), so the structure of the prediction holds; the absolute scale is unconfirmed without an exact-DRAM run.
3. ~~**Rank tier-2 kernels from the nsys timeline** instead of tier-1 ncu.~~ **Done 2026-10-06.** Tier 1 is now off by default, so the runs no longer time out.
4. ~~Find a cheap way to measure bytes.~~ **Done 2026-10-06:** nsys samples all L2 traffic (`configs/nsys/gb20b_l2.config`, on by default). DRAM-side counters cannot be sampled on GB10. Still open: one exact-DRAM cross-check on Kimi (ncu tier 1, hours) to confirm the L2-to-DRAM gap at batch 32.
5. ~~**The 22 runs**~~ **done 2026-10-06** (section 5). Runs 1–21 via `scripts/run_moe_queue.sh`; long context re-run without ncu via `scripts/rerun_moe_longctx.sh` after the first attempt crashed the DGX (5.6).
6. *(optional)* Real-text prompts with a different slice per batch row, and the prompt-based repeated-token variant. Now more important: natural routing's expert counts (5.1, 5.2) were measured on random token IDs.
7. **Open questions from the results:**
   - Why Kimi's L2 traffic is ~2× the weight bytes (5.5): needs one exact-DRAM run (ncu tier 1, reduced, decode only).
   - Whether the prefill re-read effect (5.4) also holds on the routed-expert kernels: needs ncu targeted at later layers (`--annotate-layers`).
   - Whether `DynamicCache` copying explains long context's ~2.9× cache traffic (5.3).
