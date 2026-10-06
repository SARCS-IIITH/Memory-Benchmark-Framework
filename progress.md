# Progress log

Running log of setup and model-onboarding work on the DGX Spark (GB10) as user `imc`.
Update this file whenever something is finished or decided, so work can resume after a logout.

_Last updated: 2026-10-06 IST_

---

## ▶ Resume here

| Item | State |
|---|---|
| Env `~/envs/nsbench` | ✅ built, verified |
| `nsbench preflight` | ✅ passes |
| Kimi-Linear FP8 download | ✅ **done** (2026-10-04 ~20:30). 49.96 GB, verified. Config: `configs/models/kimi-linear-48b-fp8.yaml` |
| Step A: harness support for Kimi-Linear | ✅ **done**. CPU checks 21/21 and GPU checks 13/13 pass. |
| Qwen3-0.6B smoke test | ✅ **passed** (2026-10-05 12:14–12:20, GPU idle). See "Smoke test" below. ✅ **Passed again 2026-10-06 00:31–00:37** with `--tiers 1,2`: tier 2 ranked from nsys works on the GPU (60 launches profiled per phase, real stall data). Run `runs/20261005T190132Z__smoke-model__default__hf__smoke`. |
| Step B: correct analysis for hybrid / MLA / MoE | ✅ **done** (2026-10-05). Analysis checks 27/27 pass, step-A checks still 21/21, Qwen report unchanged. See "Step B" below. |
| Step C: MoE workloads | ✅ **ALL 22 MoE RUNS COMPLETE (2026-10-06 16:20).** Runs 1–21 from the queue; run 22 (long context) re-run with `--skip-ncu` after the OOM crash. Results: `runs/moe-queue/results.md` (auto-generated table + findings), per-run log and caveats below. History: run 22 re-run WITHOUT ncu started 2026-10-06 16:07 in tmux `moe-longctx` (`scripts/rerun_moe_longctx.sh`: `--skip-ncu`, plus a memory guard that kills the run if MemAvailable < 2 GB for 60 s). Log `runs/moe-queue/22-moe-longctx-natural-p16384-rerun-skip-ncu.log`; START/DONE/FAIL go to queue.log. `moe-results` auto-updater restarted at the same time. ⚠️ Earlier: **queue stopped 2026-10-06 by a DGX out-of-memory crash during run 22.** Runs 1–21 completed and passed every stage; run 22 (long context) did not (caveat 12). **Do NOT restart `run_moe_queue.sh` as-is** (it would re-run run 22 with ncu); re-run long context with `--skip-ncu`. Earlier: the 22-run queue started 2026-10-06 03:14 in tmux `moe-queue` (`scripts/run_moe_queue.sh`, profile `configs/profiles/moe.yaml`). Live status: `runs/moe-queue/queue.log`. Results table: `runs/moe-queue/results.md` (regenerate with `~/envs/nsbench/bin/python scripts/summarize_moe_queue.py`). See **"MoE queue"** below for the order, the caveats, and how to resume. |
| Kimi-Linear sanity run (real 50 GB model, no profiler) | ✅ **passed** (2026-10-05 ~12:35). Loads as FP8, gives correct and fluent answers. See "Kimi sanity run" below. |
| First profiled Kimi-Linear run | ✅ **Attempt 2 done 2026-10-06 00:41–01:33** (tmux `kimi-quick`, quick profile, tier 2 only ranked from nsys, default workload 512 → 64, batch 1, natural). All stages ok. Run `runs/20261005T191148Z__kimi-linear-48b-fp8__default__hf`. **Results:** prefill 2.0 s (255 tok/s); decode 140.7 ms/token (**7.1 tok/s**). The GPU is **idle 68% of each decode step** and 65% of prefill, so Kimi is bound by launches and host syncs, not memory (consistent with `moe_infer`'s per-layer `.cpu()` sync and per-expert loop). Tier 2 sampled `device_kernel` (the GEMMs) only 4–5 times in 60 launches; elementwise kernels took the rest, as predicted. |
| **Measuring bytes without tier 1** | ✅ **Solved for decode, 2026-10-06. Full write-up: [docs/08-nsys-l2-sampling.md](docs/08-nsys-l2-sampling.md).** nsys samples all L2 traffic (DRAM counters can't be sampled on GB10). Qwen check: decode L2 1.42 GB vs exact DRAM 1.34 GB (**+6–10%**); prefill **~1.9×** DRAM. Built into the harness, on by default. |

### Next steps (in order)

Code first; all of it is CPU-only work and can be done while the GPU is busy:
1. ~~Routing hook~~ **DONE 2026-10-05** (`nsight_bench/compat/routing.py`, `WorkloadConfig.routing`).
2. ~~Routing-aware expected bytes~~ **DONE 2026-10-05.**
   - CPU check: `experiments/kimi-linear-compat/check_routing_cpu.py`, 28/28 pass. The harness and analysis CPU checks still pass.
   - Full write-up: `docs/07-moe-routing.md`.
   - Not yet run on the GPU.
3. ~~**Rank tier-2 kernels from nsys** (option 3) and skip tier-1 ncu.~~ **DONE 2026-10-06.** Tier 1 is off in every profile (`ncu.tiers: [2]`); turn it back on with `--tiers 1,2`. See to-do 3(c).
4. *(nice to have)* **Real-text prompts**, a different slice per batch row, so natural routing is realistic.
5. ~~Stale "routing not implemented" comment in the 22 MoE workload files~~ removed 2026-10-06.

Then the GPU work:

6. ~~First profiled Kimi run~~ **done 2026-10-06** (see table).
7. ~~Custom nsys metric test + Qwen L2-vs-DRAM check~~ **done 2026-10-06** (see table); L2 sampling is built into the harness.
8. ▶ **The 22 MoE runs: RUNNING** (queue started 2026-10-06 03:14; see "MoE queue" below).
9. ~~Write up the routing results~~ **done 2026-10-06: [docs/07-moe-routing.md](docs/07-moe-routing.md) section 5 "Measured results"** (decode sweep, prefill, long context, router skew and the prefill anomaly, launch-bound behaviour, caveats). Sections 1–4 are kept as the pre-run predictions. Open questions are in docs/07 section 8, item 7: the ~2× L2/weight gap, the re-read effect on expert kernels, and `DynamicCache` copying.

Details for 1–4 are under to-do 3 below.

**Before anything uses the GPU**, check it is free:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=utilization.gpu --format=csv
```

**Always run benchmarks inside tmux** (user's standing rule, 2026-10-06), so a laptop disconnect doesn't kill them. Use one named session per job and leave the existing `1`, `2` and `kimi-test` sessions alone:
```bash
tmux new-session -d -s <name> -c ~/Memory-Benchmark-Framework \
  "~/envs/nsbench/bin/nsbench run ... 2>&1 | tee runs/<name>-$(date +%Y%m%d-%H%M).log; exec bash"
tmux attach -t <name>      # watch; detach with Ctrl-b then d
```

---

## MoE routing: CPU check ✅ (2026-10-05)

Run with: `~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_routing_cpu.py`. **28/28 pass.** No GPU or weights needed. It builds one real Kimi MoE layer from Kimi's own code, shrunk to 16 experts with top-4. Full write-up: `docs/07-moe-routing.md`.

| Area | Result |
|---|---|
| Gate discovery | `KimiMoEGate` found; 16 experts, top-4 read correctly |
| Natural | Output bit-identical with the hook installed; removing the hook restores the model; observer counted the router's distinct experts correctly (13 of 16 for 6 tokens) |
| Fixed | Every token goes to experts 0–3; weights equal (0.6115 each), summing to the router's own per-token total; output changes, shape doesn't; observer sees exactly 4 experts |
| Disjoint | No overlap within a pass (4 tokens × 4 = all 16 experts); a larger pass covers every expert; each token's experts distinct; rotates between passes; full layer runs |
| Safety checks | Forced routing on a model without a router stops with an error; unknown `routing` value rejected; all 22 workload files load with the right `routing` |
| Distinct experts (256 experts, top-8) | T=1: 8/8/8 · T=8: 8/57.4/64 · T=16: 8/102.0/128 · T=32: 8/163.3/256 (fixed/natural/disjoint) |
| Kimi expected weight bytes per decode step | Batch 1, natural: **3.50 GB**, unchanged from before. Batch 32: fixed **3.50**, natural **32.12**, disjoint **49.20 GB**; disjoint = every routed expert; fixed doesn't grow with batch. |

Regression: `check_harness_cpu.py` and `check_analysis_cpu.py` still pass.

**Not yet verified, because it needs the GPU:**
- the hook installing during a real `load()`
- the `routing` summary appearing in `baseline_result.json`
- real DRAM bytes: the three batch-1 decode runs should read the same bytes, and batch 32 should order fixed < natural < disjoint

---

## Step B: correct analysis for hybrid / MLA / MoE ✅ (2026-10-05)

**What changed.** Only the analytic ("expected") side. Measurement and profiling code is untouched, and so are the workload files.

- **`nsight_bench/config.py` (`ModelConfig`).** New fields, filled by `discover_model`:
  - **MLA:** `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`.
  - **Hybrid layout:** `linear_attn_layers`, `sliding_attn_layers` + `sliding_window`, `no_cache_layers`, and the linear-state shape (`linear_attn_num_heads/key_head_dim/value_head_dim/conv_kernel/conv_channels/kind/state_dtype_bytes`). Parsed from three config spellings:
    - Kimi's `linear_attn_config` (1-based `kda_layers`)
    - generic `layer_types` (Qwen3.5/3.6 `linear_attention`, Gemma `sliding_attention`)
    - Nemotron-H's `hybrid_override_pattern` (`*` attn, `M` Mamba-2, `E`/`-` MoE/MLP)
  - `full_attn_head_dim` (Gemma 4 `global_head_dim`), `first_k_dense_layers`.
  - **`weight_bytes_by_role`:** exact checkpoint bytes per tensor role (routed_experts / embedding / lm_head / multimodal / other), from the safetensors headers. No weights are loaded.
  - **`quant_bits`:** read from `quantization_config.config_groups[*].weights.num_bits`. `bits_per_weight` prefers it.
- **New / changed methods:**
  - `kv_cache_breakdown()` → `{kv, recurrent_state, conv_state}` by layer kind. MLA is sized as transformers stores it (expanded).
  - `kv_cache_bytes()` is now the sum of that. Identical to the old formula for uniform models.
  - `mla_latent_kv_bytes()`: what an MLA-native engine would cache.
  - `decode_read_weight_bytes()`: weights a batch-1 decode step reads, from tensor roles:
    - routed experts × top_k/E
    - no input-embedding gather, no vision/audio
    - everything else in full
  - `active_weight_bytes()` uses it when roles exist; the old formula remains as the fallback.
  - `describe_attention()` gives one-line attention descriptions for reports and `discover`.
- **`analysis/assemble.py`, the physics check:**
  - MoE weights now come from `decode_read_weight_bytes()`. The old code scaled attention by the expert ratio, which disagreed with `ModelConfig`.
  - It adds the linear-attention **state write-back** (`DecodeExpectation.state_write_bytes`).
  - The footprint gains `cache_state_bytes`, `mla_expanded_kv_bytes` and `mla_latent_kv_bytes`.
  - The misplaced "falls back to on-disk size" tied-embedding warning now fires only when that's actually true.
- **Reports** (`report/markdown.py`, `report/html.py):**
  - The "Shape" line uses `describe_attention()`.
  - The physics table shows the weight source, the MoE resident total, and a "Recurrent state written back" row.
  - The footprint shows the cache breakdown and an "MLA latent equivalent" row.
- **`nsbench discover`** prints the attention layout, decode-read weights (MoE), and the cache split plus MLA latent per length.
- **`docs/03-methodology.md`:** the physics-check description now covers MoE, linear attention and MLA.

**Before → after, Kimi-Linear FP8:**

| | Old | New |
|---|---|---|
| Attention description | "32 heads / 32 KV heads, head_dim 72" | 7 MLA (kv_lora_rank 512, rope 64) + 20 KDA |
| Weights read per batch-1 decode token | 2.7 GB (inconsistent formula) | **3.50 GB**: other 1.28 + lm_head 0.76 + routed 47.17 × 8/256 |
| Cache @ 84 tokens (measured: 11.5 / 40.0 / 1.9 MiB) | wrong shape | **11.48 / 40.00 / 1.88 MiB** ✓ |
| Cache @ 8k tokens | 2,038 MB | **1,218 MB** (HF-expanded) · MLA latent would be 66 MB |
| Expected decode bytes also include | n/a | KDA state write-back, 40 MiB |

**Other local models** (headers only, `/opt/ai-models`):
- All parse without errors.
- Uniform models are unchanged: Qwen3-0.6B, Qwen3-4B, Qwen3-30B-A3B.
- Hybrid KV estimates @ 8k tokens, now much smaller:

| Model | Old | New |
|---|---|---|
| Qwen3.5-4B (24 DeltaNet + 8 attn) | 1,074 MB | 320 MB |
| Qwen3.6-35B (30 DeltaNet + 10 attn) | 671 MB | 233 MB |
| Gemma-4-26B (25 sliding + 5 full) | 2,013 MB | 881 MB |
| Gemma-4-E4B | 705 MB | 272 MB |
| Nemotron-3-Nano (6 attn + 23 Mamba-2 + 23 MoE-only) | 436 MB | 76 MB |

- Only Kimi's state shape has been checked against a measured cache. Gated DeltaNet, Mamba-2 and Gemma's sliding/global widths are documented shapes, flagged as unverified in discovery notes.

**Verified:**
- `experiments/kimi-linear-compat/check_analysis_cpu.py`: **27/27**, CPU only. Covers:
  - formulas vs exact measured caches (reduced Kimi: 1,474,560 B KV, 6 MiB state; real Kimi at 84 tokens)
  - Qwen KV equal to the old formula at 1–32k tokens
  - role sums equal the checkpoint
  - physics check, footprint and both reports on a synthetic Kimi run
- `check_harness_cpu.py` still 21/21.
- Smoke-run report regenerated with the new code (on a copy): only the "Shape" line changed (now shows `head_dim 128`). Every number is identical.
- `configs/models/kimi-linear-48b-fp8.yaml` and `qwen3-0.6b.yaml` re-discovered with the new fields. Kimi keeps `device_map: cuda`, and Qwen's existing values are unchanged.

**Known, not changed (by design or out of scope):**
- Dense models still use the resident size for the weight expectation, not the tensor-role read, so dense results are unchanged.
  - For tied models like Qwen the two are equal.
  - For untied dense models and Gemma-4-E4B (7 GB of per-layer embedding gathers, 1 GB vision), the dense expectation over-counts. Worth revisiting.
- The batch > 1 MoE expectation (more distinct experts read) is not modelled yet. That belongs with the teammate's batch-sweep workload.
- `assemble._artifact` prefers the recorded absolute path when it exists, so re-reporting a *copied* run on the same machine reads the original's files. This is intentional for relocated runs; the check script rewrites paths.

---

## Smoke test (Qwen3-0.6B): PASSED, 2026-10-05

- Run: `runs/20261005T064431Z__smoke-model__default__hf__smoke/` (`report.md`, `report.html`, `manifest.json`)
- Log: `~/nsbench-logs/smoke_20261005_1214.log`
- Started with `setsid nohup`, so it would survive a logout:
  ```bash
  setsid nohup env ENV_DIR=~/envs/nsbench bash scripts/smoke_test.sh > ~/nsbench-logs/smoke_<ts>.log 2>&1 < /dev/null &
  ```
- GPU was checked idle first: no compute processes, 0% utilisation, SM clock at 305 MHz.
- **All 4 stages ok:** calibration, baseline, nsys, ncu. Markdown and HTML reports written.
- **Calibration gate PASS, +0.02%:**
  - 268.4 MB expected, 268.5 MB measured in the standalone calibrate
  - 536.87 MB as 536.98 MB in the run
  - Matches the original author's accuracy.
- **Measured ceilings this time:**
  - dense bf16 GEMM 92.0 TFLOP/s (ridge at 412 FLOP/byte)
  - L2 984 GB/s (4.4× LPDDR5X)
  - These are slightly below the README's ~98 TFLOP/s and ~1035 GB/s. Run-to-run or thermal variation is plausible; worth watching, not a problem.
- Baseline timing (Qwen3-0.6B, 128 prompt → 8 generated, batch 1): prefill 16.3 ms (7,853 tok/s), decode 71.8 tok/s.
- **"PARTIAL DATA" warnings are expected.** The smoke test caps ncu at 120 kernels on purpose. nsys counted 1,594 (prefill) and 1,618 (decode) kernels per instance, so real runs use the default cap.

---

## Kimi sanity run (real model, no profiler): PASSED, 2026-10-05

- Script: `experiments/kimi-linear-compat/sanity_real_model.py`. It refuses to run if the GPU is busy; `--force` overrides.
  - It loads `configs/models/kimi-linear-48b-fp8.yaml` through the real `HFTransformersBackend` and answers 2 chat prompts with the harness's own prefill/decode loop.
  - Log: `~/nsbench-logs/kimi_sanity_20261005_1229.log`. The first attempt, `_1224`, failed on a bug in the script itself, now fixed: Kimi's tokenizer `apply_chat_template` returns a **string** even when asked for tensors.
- **Load:** 256–280 s from disk.
  - **50.0 GB resident**: FP8 48.4 GB, bf16 1.5 GB, fp32 0.1 GB.
  - All **20,257 FP8 layers** on the optimized path.
  - SDPA attention. All compat adapters fired.
  - The `load_warning` "attn_implementation='sdpa' unsupported" is the expected first-try refusal; the retry then succeeds.
- **Output is correct and fluent:**
  - "What is the capital of France…?" → "The capital of France is **Paris**, and the river that flows through it is the **Seine**." It stopped at end-of-turn by itself.
  - The memory-bandwidth explanation is fluent and on-topic, cut only by the 64-token cap.
  - So the FP8 weights and scales load correctly, and the harness decode loop is correct for this model.
- **Decoding state at 84 tokens:**
  - MLA kv 11.5 MiB (= 7 layers × 84 × 20 KB, exactly as predicted)
  - KDA recurrent state 40.0 MiB
  - conv 1.9 MiB
- **Memory:** 46.5 GiB still free (CUDA's count) with the model loaded and generating.
  - CUDA's "free" excludes Linux page cache. The pool showed 113 GB "available" before load, of which ~56 GB was page cache.
- **Speed (one unprofiled run, indicative only):**
  - Prefill: 520 ms for 21 tokens. The first call took 9.2 s, because fla Triton kernels JIT-compile on first use; benchmark warmup absorbs that.
  - **Decode only ~10.5 tok/s.** A pure bandwidth bound would allow ~90 tok/s (~2.7 GB/token at ~240 GB/s).
  - Likely cause: kernel-launch overhead from HF's per-expert MoE loop and per-layer FP8 activation quantization.
  - **Worth investigating in the first profiled run.** It's probably launch- or latency-bound, not bandwidth-bound. Also expect a high ncu kernel count; check `truncated`.

---

## Decisions

- **No git commits (decided 2026-10-05).** All changes stay uncommitted in this working tree on purpose. The teammate works in the same `imc` account and the same directory, so they see everything as-is.
  - Run `git status` to see what's changed vs upstream.
  - Don't `git checkout .`, `git stash` or `git reset`: that would wipe steps A and B.
- **Everything lives under `imc`'s home.** No shared groups and no shared stores.
  - Python env: `/home/imc/envs/nsbench`
  - Model weights: `/home/imc/ai-models`, in a Hugging Face cache layout
- `/opt/ai-models` is a folder another user filled with models. We can read it but not write to it. The Qwen3-0.6B config uses it, which is fine.
- **Model under study:** `nm-testing/Kimi-Linear-48B-A3B-Instruct-FP8-DYNAMIC`. It's 50 GB, chosen over the 98 GB bf16 Base.
- **Ruled out:**
  - **Kimi K2.6:** ~600 GB, doesn't fit.
  - **Mistral Small 4 NVFP4:** Mistral-native format, so it needs a vLLM backend.
  - **Mistral Small 4 FP8:** 121 GB, doesn't fit.
- **The GPU is shared with `sarcs`**, who runs SGLang/vLLM benchmarks. Check it is idle before any GPU job. Coordinating time slots is recommended.
- **Known limitation, accepted for now:** HF's Kimi code caches MLA layers **expanded** (~143 KB/token), not as MLA's compressed latent (~8 KB/token). Reports should state both figures. Measuring true MLA caching would need a vLLM backend.

---

## Download (DONE)

`nm-testing/Kimi-Linear-48B-A3B-Instruct-FP8-DYNAMIC`, registry key `kimi-linear-48b-fp8`.
- **Finished 2026-10-04 ~20:30.**
  - Snapshot: `~/ai-models/hub/models--nm-testing--Kimi-Linear-48B-A3B-Instruct-FP8-DYNAMIC/snapshots/c3a71758d772dc8ff6c207a034a722cb8decff77`
  - Integrity: all 10 shards from the index present (49.96 GB), and every tensor in the index found in its shard header.
  - `check_harness_cpu.py <snapshot>`: 21/21 pass on the real files.
- Started with `setsid nohup`, so it **survives SSH logout and Claude session end**.
- ~5 MB/s average.
- History:
  - The first attempt died with the Claude session at 7.5 GB.
  - On restart, the fetcher deletes stale `.incomplete` files by design, so it started over.

```bash
pgrep -af "nsbench fetch"   # running?
du -sh ~/ai-models          # done at ~50 GB
tail ~/ai-models/fetch.log  # prints the snapshot path when finished

# ONLY if pgrep shows nothing and it is not finished (restarting discards partial files):
cd ~/Memory-Benchmark-Framework && setsid nohup ~/envs/nsbench/bin/nsbench fetch \
    --models kimi-linear-48b-fp8 --store ~/ai-models > ~/ai-models/fetch.log 2>&1 < /dev/null &
```
- Weights land in `~/ai-models/hub/models--nm-testing--Kimi-Linear-48B-A3B-Instruct-FP8-DYNAMIC/snapshots/<hash>/`.
- Never start long jobs as a plain background task of a Claude session. They die with it.

---

## Done

### Environment
- Env `~/envs/nsbench`: torch 2.13.0+cu130, transformers 5.18.0. Self-check passed on GB10 (sm_121, 48 SMs, 128.5 GB unified memory, 25.17 MB L2).
- Kimi extras installed. They're listed in `setup/requirements-kimi.txt`:
  - `fla-core==0.4.0` and `flash-linear-attention==0.4.0`, **pinned** (0.5.x broke `fused_kda_gate`)
  - `compressed-tensors==0.19.0`, `tiktoken==0.14.0`, `einops==0.8.2`
- To rebuild from scratch:
  ```bash
  ENV_DIR=~/envs/nsbench bash setup/create_env.sh
  ~/envs/nsbench/bin/pip install -r setup/requirements-kimi.txt
  ```
- `nsbench preflight` passes. The regenerated `configs/platform_profile.json` matches the original apart from `detected_at` and `python_executable`. 45/45 tier-1 metrics available. ncu counters allowed for `imc`.

### Repo changes (uncommitted on purpose; see Decisions)
- **Per-user paths.** Hard-coded `/home/sarcs/...` and `/home/samarthamp/...` became `${HOME}/envs/nsbench` in:
  - `setup/create_env.sh`, `setup/setup_trtllm_env.sh`
  - `scripts/*.sh`
  - `nsight_bench/backends/trtllm.py`
  - `README.md`, `docs/04-usage.md`
- `configs/models/qwen3-0.6b.yaml` now points at `/opt/ai-models/...`, which `imc` can read.
- `scripts/smoke_test.sh` falls back to `/opt/ai-models` for its model.
- `configs/model-registry.yaml`:
  - added `qwen3-0.6b` (the groups referenced it)
  - added `kimi-linear-48b-fp8`
- **Bug fix:** `nsight_bench/metrics.py` gained `MetricAvailability.probed`. `cli.py:118` used it but it was never defined, so every full `nsbench preflight` crashed at the end.
- `nsight_bench/registry.py`: the fetcher now also downloads `special_tokens_map.json`.

### Step A: harness support for Kimi-Linear ✅ (2026-10-04)
**New: `nsight_bench/compat/transformers_compat.py`.** It bridges remote code written for transformers 4.5x onto 5.x at runtime, without editing the model's files:
- `apply_remote_code_shims()` runs before the tokenizer and model load. It restores three moved or renamed APIs:
  - `OutputRecorder` moved to `transformers.utils.output_capturing`
  - `bytes_to_unicode` moved to `convert_slow_tokenizer`
  - `create_causal_mask(input_embeds=, cache_position=)` became `inputs_embeds=`, and `cache_position` was dropped
- `adapt_model_cache_api(model)` gives a remote cache class the 5.x methods `get_query_offset()`, `get_mask_sizes(int, layer)` and `is_sliding`.
- `model_manages_own_cache(model)` is true when the model's module defines its own `*Cache` class.
- `use_sdpa_if_flash_attn_missing(model)`: Kimi forces `flash_attention_2`, which isn't installed, so this switches every config to `sdpa`. SDPA matched the eager reference: 0.33% logit difference, same argmax.
- `is_fp8_compressed_tensors(config)`: true only if every config group is 8-bit float.

**Changed: `nsight_bench/backends/hf_transformers.py`.**
- `trust_remote_code` models get the shims before load, and the cache and attention adapters after.
- FP8 compressed-tensors checkpoints load with `CompressedTensorsConfig(use_optimized_inference=True)`, so they stay **FP8 at 50 GB**. The default mode unpacks to bf16, ~98 GB.
- **Cache fix:** a model that owns its cache gets `past_key_values=None`. The old forced `DynamicCache()` crashed Kimi with an assertion at `modeling_kimi.py:915`. The cache class actually used is recorded.
- **KV measurement:** `_measure_kv_bytes` now counts KV **plus recurrent and conv state**, deduplicated by storage. The breakdown goes to the manifest as `cache_state_bytes`. Kimi's KDA state is ~40 MiB and was previously missed.
- `describe()` now records:
  - `cache_managed_by_model`
  - `cache_state_bytes`
  - `fp8_optimized_inference`
  - `compat_adapters`
  - `attn_warning`
- Requesting `sdpa` at load makes Kimi's class raise (it declares no sdpa support). The backend's existing retry handles this before any weights load, and the post-load adapter then switches to sdpa.

**Verified:**
- `experiments/kimi-linear-compat/check_harness_cpu.py`: **21/21 pass.** CPU and meta device only, no GPU, no weights. Covers:
  - shims and tokenizer
  - full 48B model on meta (49.12B params)
  - cache and attention adapters
  - FP8 decision (20,257 FP8 layers, 50.0 GB resident)
  - cache handling and measurement on a hybrid cache
  - an ordinary `DynamicCache`, measured the same as before
- `experiments/kimi-linear-compat/check_harness_gpu.py`: **13/13 pass** (run while the GPU was idle, ~2 GB for a few seconds). A reduced Kimi-Linear (3 KDA + 1 MLA layers, random bf16 weights) goes through the **real** `HFTransformersBackend`:
  - discover and load
  - adapters and sdpa
  - fla KDA kernels on GB10
  - prefill + 8 decode steps on `KimiDynamicCache`
  - live cache breakdown, which matches the expected shapes exactly
  - It refuses to run if another process is on the GPU, unless `--force`.
- Regression check: the Qwen3-0.6B path is unchanged. No shims, `DynamicCache`, no attention switch, not FP8.

```bash
cd ~/Memory-Benchmark-Framework
~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_harness_cpu.py
~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_harness_gpu.py   # GPU must be idle
```
`experiments/kimi-linear-compat/kimi-fp8-meta/` holds Kimi's config, code and tokenizer files (no weights, git-ignored). Both scripts default to it. Once downloaded, pass the snapshot path instead.

**Still unverified, because it needs the real weights:**
- FP8 `weight_scale` tensors loading through the real `from_pretrained`
- coherent output text
- real memory headroom

**Found but not needing a fix:** Kimi's `_tied_weights_keys` is a 4.x-style list, which crashes 5.x's `save_pretrained`. Loading never touches it (`tie_word_embeddings=false`), and the harness never saves. The GPU test works around it.

---

## nsys L2 sampling (built into the harness 2026-10-06)

Everything is in **[docs/08-nsys-l2-sampling.md](docs/08-nsys-l2-sampling.md)**: why it was needed, the synthetic counter test, why Qwen was used, the Qwen results, how it is wired in, the switch (`nsys.sample_l2_traffic`), caveats and how to reproduce. In short: decode bytes from nsys L2 run ~6–10% above DRAM; prefill ~1.9×; validated on Qwen3-0.6B only.

## MoE queue (started 2026-10-06 03:14)

- **What:** all 22 `configs/workloads/moe/*.yaml` on Kimi-Linear, one after another, in tmux `moe-queue`. The script is `scripts/run_moe_queue.sh` and the profile `configs/profiles/moe.yaml`: tier 2 only, top 5, ranked from nsys; nsys L2 sampling on; nsys and ncu timeouts 2 h.
- **Order:** decode b1 ×3 (self-check: all three modes must read the same bytes), then b8 ×3, b16 ×3, b32 ×3, prefill p16 ×3, p64 ×3, p512 ×3, then long context.
- **ncu tier 2 on 8 runs:** decode b1 ×3, decode b32 ×3, prefill-natural-p512, longctx (crashed; caveat 12). **The other 14 used `--skip-ncu`:** decode b8 ×3, b16 ×3, prefill p16 ×3, p64 ×3, fixed-p512, disjoint-p512. Tier 1 was off for all 22. **Why:** chosen by me when building the queue, following the earlier "tier-2 on 8 runs" plan, to keep the sweep ~12 h instead of ~18–19 h (each ncu run costs ~25–30 min more). The 8 cover the extremes; b8/b16 are bracketed by b1/b32. **What it loses:** only the stall breakdown. In the 8 runs that have it, stalls barely change with routing (decode memory-wait share: b1 29/30/37%, b32 55/59/54% for natural/fixed/disjoint), and the 60-launch sample mostly misses the expert kernels anyway (caveat 6). Full table and reasoning: `runs/moe-queue/findings.md`, "ncu coverage".
- **Auto-updating results (since 2026-10-06 11:40):** tmux `moe-results` runs `scripts/watch_moe_results.sh`, which regenerates `runs/moe-queue/results.md` within a minute of every queue.log change and exits after QUEUE END. Its own log: `runs/moe-queue/results-watcher.log`. If it's gone, restart with `tmux new-session -d -s moe-results -c ~/Memory-Benchmark-Framework "./scripts/watch_moe_results.sh; exec bash"`.
- **Tracking:** one line per start/finish/failure in `runs/moe-queue/queue.log`; each run's console in `runs/moe-queue/<n>-<name>.log`. `scripts/summarize_moe_queue.py` rebuilds `runs/moe-queue/results.md` (table plus per-run caveats).
- **Resume after a crash or logout:** `tmux attach -t moe-queue`. If the session is gone, check the GPU is free and start it again (`tmux new-session -d -s moe-queue -c ~/Memory-Benchmark-Framework "./scripts/run_moe_queue.sh; exec bash"`). It skips runs already marked DONE. The queue waits if another process holds the GPU (other users are on this machine).
- **Estimated time:** ~12 h (my estimate). Run 1 (with tier 2) took 39 min; the 14 runs without ncu should be shorter; batch 32 and long context are unknowns.

### Caveats to carry into the results (keep this list current)
1. **Bytes are L2 traffic, not DRAM** ([docs/08](docs/08-nsys-l2-sampling.md)). For decode they are an upper bound, 6–10% high on Qwen. The gap may widen at batch 32, where activations grow. For prefill, L2 ran ~1.9× DRAM on Qwen, so prefill GB are **not** DRAM bytes; compare routing modes relatively only.
2. **The validation was on a dense 0.6B model at batch 1**, not on Kimi. No Kimi run has exact DRAM bytes (tier 1 would be ~13 h).
3. **The `natural` expected-experts formula assumes uniform random routing** (confirmed wrong for Kimi: b8 decode observed 31 vs formula 57; 128-token prefill 153 vs 252). Real routers have favourites. The earlier Kimi run (512-token prefill, natural) observed a **mean of 204 distinct experts per layer, range 157–246**, against 256 from the formula. So `natural` "expected" bytes run high, and the observed counts (the Experts/pass columns) are the better yardstick.
4. **Kimi is launch- and sync-bound, not memory-bound.** In the 512→64 run the GPU was idle 68% of each decode step and 65% of prefill. Times will not follow bytes. Use bytes to compare routing modes.
5. **Prompts are synthetic random token IDs** (real-text prompts still to do). Natural routing on random tokens may differ from routing on real text.
6. **Tier-2 sampling is launch-ordered.** On Kimi, elementwise kernels took ~55 of the 60 slots and the GEMMs only 4–5. Stall data on the GEMMs is thin.
7. **Forced routing makes the generated text meaningless.** Only the traffic is real (docs/07).
8. **Timing comes from the unprofiled baseline.** nsys runs perturb timing slightly; ncu durations are never used for performance.
9. **(added after run 1) Kimi's decode L2 is 1.75× the expected bytes, against ~1.1× on Qwen.** So the Qwen-derived "+6–10%" L2-to-DRAM gap probably does **not** carry over to Kimi; part of Kimi's L2 traffic never reaches DRAM. Until an exact-DRAM cross-check exists (ncu tier 1 on one Kimi decode step: hours), treat Kimi decode GB as **L2 traffic**: use the differences between routing modes and batch sizes, not the absolute GB or the ratio against expected. **Update 2026-10-06 17:00:** no weight dequantization happens (decode uses FP8 weights directly in a CUTLASS FP8 GEMM). ncu tier 2 shows that GEMM moving **1.54× its memory traffic through L2** even at batch 1 (45.4 vs 29.4 MB, identical in all three b1 runs), which explains most of the ~1.9×; the rest is unexplained. Details: docs/07 section 5.5.
10. **(added after run 10) Memory pressure at batch 32 with ncu tier 2.** ncu kernel replay on top of the 50 GB model and batch-32 state pushed the unified pool to ~3 GB available, and the GPU driver logged 190 failed allocations (`NV_ERR_NO_MEMORY`, `journalctl -k`). Nothing was killed and the collections completed, but **tier-2 stall data from the b32 runs was collected under memory pressure**. If any b32 tier-2 numbers look odd, this is the first suspect. Timing and nsys L2 come from separate processes (baseline, nsys) and are unaffected.
11. **(added 2026-10-06 12:50) Prefill anomaly: at large token counts L2 GB do not rank routing modes reliably.** natural-b8/b16 prefill read *more* L2 than disjoint (239 vs 236 GB; 411 vs 381 GB) despite fewer experts. Measured: natural routing sends ~44% of assignments to each layer's top 8 experts (`top_k_share_per_layer` 0.43 vs 0.031 even). Hypothesis: popular experts get hundreds of tokens, so their matrix multiplies span several token tiles, and each tile re-reads the weights through L2 (most likely L2 hits, not DRAM). **Run 19's ncu confirmed the mechanism on non-expert GEMMs** (4 token tiles → 7.5× L2/memory, 84% hits); the experts themselves weren't sampled. Full write-up: `runs/moe-queue/findings.md`, which is included in results.md.
12. **(added 2026-10-06 15:55) ncu on a 16k-token prefill crashed the DGX.** Never run ncu (any tier) on Kimi long-context prefill on this machine: kernel replay's save/restore buffers plus the 50 GB model exhausted the unified pool, the profiled process is OOM-unkillable, and the box thrashed for ~1.5 h until it was restarted (15:47). Other users' work was affected. **The queue must not be restarted as-is**: it would re-run run 22 with ncu. Re-run long context with `--skip-ncu` only.
13. **(added 2026-10-06 17:30) ncu tier-2 data is a first-layer sample only; don't use it for routing conclusions.** The 60-launch cap took the first ~4–5 linear layers (5 FP8 GEMMs plus activation quantization) in every run and never reached the routed experts, KDA attention or the LM head. **No routing result depends on ncu.** The main results table uses only baseline (timing, expert counts), nsys (L2 GB, busy %, launches) and model formulas (expected GB); every decode check used nsys L2 (`measured_source = nsys_l2`), even in the 8 ncu runs. ncu-based claims (stall profiles, run 19's 7.5× re-read evidence, the FP8 GEMM's 1.54×) are marked ⚠️ in results.md/findings.md and docs/07. Full source map: `runs/moe-queue/findings.md`, "Where each number comes from".
- *(2026-10-07)* **KV cache results added to reporting:** results.md now has a "Cache MB (MLA K/V + KDA state)" column and a "KV cache and recurrent state" section (MLA 143.4 KB/token/seq, KDA state 41.9 + 2.0 MB/seq, crossover ~292 tokens, latent would be 18× smaller); docs/07 5.3 summarises it. The data was always in each run's report.md "Memory footprint" table and `summary.json` (`footprint.cache_state_bytes`), but hadn't been surfaced.

### Decode sweep summary (12/12 done, 2026-10-06 11:43)

Per decode step. L2 = nsys-sampled L2 traffic (on Kimi ≈ 2× the weight bytes read; caveats 1, 9). Experts = distinct experts per layer per step, observed by the routing hook.

| Batch | fixed: L2 / time / tok/s | natural: L2 / time / tok/s / experts (random formula) | disjoint: L2 / time / tok/s |
|---|---|---|---|
| 1 | 6.32 GB / 141 ms / 7.1 | 6.32 GB / 141 ms / 7.1 / 8 (8) | 6.33 GB / 137 ms / 7.3 |
| 8 | 8.56 GB / 138 ms / 57.8 | 16.77 GB / 364 ms / 22.0 / 31 (57) | 28.16 GB / 658 ms / 12.2 |
| 16 | 11.13 GB / 147 ms / 108.5 | 26.15 GB / 550 ms / 29.1 / 50 (102) | 53.16 GB / 1,221 ms / 13.1 |
| 32 | 15.25 GB / 160 ms / 200.2 | 42.97 GB / 896 ms / 35.7 / 86 (163) | 102.07 GB / 2,379 ms / 13.4 |

**Model that fits all 12 runs:** L2/step ≈ floor(B) + 0.35 GB × (experts − 8), where floor(B) is fixed routing's figure (6.32 + ~0.29–0.32 GB per extra row); time/step ≈ fixed's time + ~9 ms × (experts − 8). Every prediction made before a run landed within 0.02–1.6%.
**Findings:** (1) Kimi's router concentrates hard: natural uses ~half the experts uniform routing predicts at every batch. (2) Bytes and time both scale linearly with distinct experts. (3) Kimi is launch-bound (GPU busy 31–54%): time per step follows kernel launches per used expert, not bandwidth. (4) Batching is nearly free under fixed routing (28× throughput at b32) but gives only 5× under natural routing and ~1.8× under disjoint.

### Per-run log (append as each run finishes)
| # | Run | Result | Key numbers | Caveats / notes |
|---|---|---|---|---|
| 1 | moe-decode-natural-b1 (03:14–03:54, 39 min, with tier 2) `runs/20261005T214435Z__kimi-linear-48b-fp8__moe-decode-natural-b1__hf__moe` | ✅ all stages ok | Decode **141 ms/step** (7.1 tok/s), GPU busy 31%. **Decode L2 6.32 GB/step vs expected 3.61 GB → 1.75×** ("HIGHER than expected"). Experts/pass: decode 8 (exp 8); prefill (128 tok) **153 (exp 252)**. Prefill 1.50 s, L2 74.4 GB (144k launches, GPU idle 70%). | **1.75× is far above Qwen's 1.06–1.16×.** Either Kimi's L2-to-DRAM gap is much larger than Qwen's (likely: FP8 dequant, per-token activation quant, KDA state and ~12k small kernels all add L2 traffic that may not reach DRAM), or decode really reads ~2.7 GB more than modelled. Can't tell which without an exact DRAM run. **Key test: runs 2–3 (fixed-b1, disjoint-b1) should match run 1.** If all three are ~6.3 GB, the excess is a constant per-step overhead and routing comparisons stay valid relative to it. Prefill expert count again well below the uniform formula (153 vs 252; caveat 3). |
| 2 | moe-decode-fixed-b1 (03:54–04:32, 38 min, with tier 2) `runs/20261005T222356Z__kimi-linear-48b-fp8__moe-decode-fixed-b1__hf__moe` | ✅ all stages ok | **Self-check: decode L2 6.3225 GB/step vs natural-b1's 6.3204 GB (0.03% apart)**; decode 141.0 vs 140.5 ms. Experts/pass: decode 8, prefill 8 (forcing works: all 26 gates hooked, exactly 8 in every layer). **Prefill: 0.155 s vs natural's 1.504 s (9.7× faster)**; prefill L2 25.2 vs 74.4 GB; prefill launches 12.6k vs 144k per pass. | Batch-1 self-check passes (2 of 3 so far). Decode excess over expected (1.75×) is the same in both, consistent with a **constant per-step overhead** rather than a routing effect. **Prefill finding:** fewer distinct experts cut prefill time ~10×, because Kimi launches kernels per used expert (launch-bound, caveat 4). Prefill L2 with fixed is 25 GB against ~3.5 GB of expert weights, so prefill L2 is dominated by activations, not weights (caveat 1). Generated text meaningless as expected (3 distinct tokens). |
| 3 | moe-decode-disjoint-b1 (04:32–05:18, 45 min, with tier 2) `runs/20261005T230230Z__kimi-linear-48b-fp8__moe-decode-disjoint-b1__hf__moe` | ✅ all stages ok | Decode L2 **6.3269 GB/step**; 136.7 ms/step. Experts/pass: decode 8 per step, **rotating through all 256 across steps** (ever-used 256 per layer); prefill 256 (exp 256). Prefill 2.38 s, L2 109.8 GB. | **✅ BATCH-1 SELF-CHECK PASSED:** natural / fixed / disjoint decode L2 = 6.3204 / 6.3225 / 6.3269 GB, all within 0.1%. Reusing the same 8 experts (fixed) gives no saving over 8 new ones each step (disjoint), as docs/07 §4.1 predicted: experts don't survive in L2 between steps. The routing hook and the nsys L2 measurement are behaving. **Prefill L2 is linear in experts used:** fixed 8 → 25.2 GB, natural 153 → 74.4, disjoint 256 → 109.8; the fit from fixed/disjoint (0.341 GB per extra expert) predicts natural at 74.6 GB (0.3% off). 0.341 GB per expert across 26 layers is ~1.85× an expert's weight bytes (~0.18 GB), close to the ~1.9× L2/DRAM prefill gap seen on Qwen. Prefill time also tracks experts: 0.155 / 1.50 / 2.38 s. |
| 4 | moe-decode-natural-b8 (05:18–05:35, 16 min, no ncu) `runs/20261005T234811Z__kimi-linear-48b-fp8__moe-decode-natural-b8__hf__moe` | ✅ all stages ok | Decode **364 ms/step = 22.0 tok/s** (3.1× batch-1 throughput), busy 33%, 33.5k launches/step (b1: 12.1k). **Decode L2 16.77 GB/step** vs expected 13.48 (1.24×). **Experts/pass decode: observed 31, formula 57.** Prefill (8×128 tokens): 221 experts (formula 256), 2.30 s, L2 239 GB. | **The router concentrates far more than uniform-random:** 8 tokens per step touch 31 distinct experts per layer, not ~57. So the "expected" bytes here (computed from the formula's 57) overstate the weight read, and the 1.24× ratio isn't comparable to b1's 1.75×. Possible contributor: synthetic random-token prompts, and greedy decode producing similar tokens across rows → more overlap than real text would give (caveat 5). L2 at batch 8 also includes 8× the activation traffic, so the b1 overhead (2.7 GB) isn't simply additive. **Decomposition waits for fixed-b8 (8 experts, the batch-8 floor) and disjoint-b8 (64 experts)**; natural-b8's L2 position between them gives its effective expert count, independent of the formula. |
| 5 | moe-decode-fixed-b8 (05:35–05:46, 11 min, no ncu) `runs/20261006T000501Z__kimi-linear-48b-fp8__moe-decode-fixed-b8__hf__moe` | ✅ all stages ok | Decode **138 ms/step = 57.8 tok/s** (same step time as b1, so 8× the throughput); busy 39%; 12.3k launches/step. **Decode L2 8.56 GB/step** (b1 fixed: 6.32). Experts 8 / 8. Prefill (8×128) 0.40 s, L2 170 GB. | **Batch-8 floor.** Going b1 → b8 with the same 8 experts adds 2.24 GB/step of L2: activations, KV and state for 8 rows. **natural-b8 vs fixed-b8:** +8.21 GB for 31 − 8 = 23 extra experts per layer → **0.357 GB of L2 per extra expert**, close to prefill's 0.341 GB/expert (runs 1–3). One expert across 26 layers is ~0.18 GB of weights, so **Kimi's L2 traffic is ~1.9–2.0× the weight bytes it reads, in decode too.** Qwen decode was ~1.06×. *Hypothesis, later partly checked (see caveat 9):* no dequant step exists; the FP8 GEMM itself shows 1.54× L2/memory. **Time cost of spreading:** natural-b8 takes 364 ms/step against fixed's 138 because Kimi launches kernels per used expert (33.5k vs 12.3k launches). **Prediction for disjoint-b8** (64 experts): ≈ 8.56 + 56 × 0.357 ≈ **28.6 GB/step**. |
| 6 | moe-decode-disjoint-b8 (05:46–06:09, 22 min, no ncu) `runs/20261006T001632Z__kimi-linear-48b-fp8__moe-decode-disjoint-b8__hf__moe` | ✅ all stages ok | Decode **658 ms/step = 12.2 tok/s**, busy 32%, 63.3k launches/step. **Decode L2 28.16 GB/step** (predicted from run 5: 28.6, so 1.6% off). Experts 64 / 256 (exact). Prefill (8×128) 2.56 s, L2 235.9 GB. | **✅ Batch-8 set complete; the linear per-expert model holds.** fixed / natural / disjoint = 8.56 / 16.77 / 28.16 GB/step at 8 / 31 / 64 experts. Slope (fixed→disjoint) **0.350 GB L2 per expert**, which puts natural at **8 + 8.21/0.350 = 31.5 effective experts, matching the 31.2 observed by the routing observer.** So nsys L2 and the observer agree. **Time is linear in experts too:** 138 / 364 / 658 ms, about 9.3 ms per extra expert per step (fixed + 23 × 9.3 = 352 ms vs natural's measured 364). Natural routing at b8 costs 2.6× fixed's step time, from launches per expert, not bytes (GPU busy only 32–39%). ⚠️ **Prefill anomaly:** natural-b8 prefill L2 (239.1 GB, 221 experts) is slightly *above* disjoint-b8 (235.9 GB, 256 experts). Likely activation traffic depends on how unevenly tokens spread over experts (natural is skewed, disjoint perfectly even). Another reason prefill GB are only comparable loosely (caveat 1). |
| 7 | moe-decode-natural-b16 (06:09–06:29, 20 min, no ncu) `runs/20261006T003905Z__kimi-linear-48b-fp8__moe-decode-natural-b16__hf__moe` | ✅ all stages ok | Decode **550 ms/step = 29.1 tok/s**, busy 34%, 51.2k launches/step. **Decode L2 26.15 GB/step.** **Experts/pass decode: observed 50.5, formula 102** (half). Prefill (16×128) 233 experts (formula 256), 2.70 s, L2 411 GB. | Router concentration again: 16 tokens share ~50 experts, not ~102. **Consistency check with the b1/b8 model:** the non-expert floor grows ~0.32 GB/step per extra row (b1 → b8 fixed: +2.24 GB over 7 rows), giving a b16 floor of ≈ 11.1 GB; plus (50.5 − 8) × 0.35 ≈ 14.9 GB of experts → **26.0 GB predicted vs 26.15 measured**. **Predictions:** fixed-b16 ≈ **11.1 GB**/step; disjoint-b16 (128 experts) ≈ 11.1 + 120 × 0.35 ≈ **53 GB**/step. Throughput rises with batch even under natural routing (7.1 → 22.0 → 29.1 tok/s at b1/b8/b16) but far below fixed's linear scaling. |
| 8 | moe-decode-fixed-b16 (06:29–06:40, 11 min, no ncu) `runs/20261006T005907Z__kimi-linear-48b-fp8__moe-decode-fixed-b16__hf__moe` | ✅ all stages ok | Decode **147 ms/step = 108.5 tok/s** (15× batch 1), busy 47%. **Decode L2 11.13 GB/step (predicted ≈ 11.1).** Experts 8 / 8. Prefill (16×128) 0.86 s, L2 338 GB. | Batch-16 floor exactly as predicted from the b1 → b8 per-row growth (~0.32 GB/step per row). With fixed routing, step time barely grows with batch (141 → 138 → 147 ms at b1/b8/b16), so throughput scales almost linearly. The GPU is busier at b16 (47% vs 31–39%) but still mostly idle. Natural-b16's L2 sits at 11.13 + 42.5 × 0.35 = 26.0 vs 26.15 measured. |
| 9 | moe-decode-disjoint-b16 (06:40–07:12, 32 min, no ncu) `runs/20261006T011051Z__kimi-linear-48b-fp8__moe-decode-disjoint-b16__hf__moe` | ✅ all stages ok | Decode **1,221 ms/step = 13.1 tok/s**, busy 32%, 121.5k launches/step. **Decode L2 53.16 GB/step (predicted ≈ 53).** Experts 128 / 256 (exact). Prefill (16×128) 2.78 s, L2 381 GB. | **✅ Batch-16 set complete:** fixed / natural / disjoint = 11.13 / 26.15 / 53.16 GB/step at 8 / 50.5 / 128 experts. Slope 0.350 GB per expert, identical to b8's. Natural = 11.13 + 42.5 × 0.350 = 26.0 (measured 26.15). Time slope ~8.95 ms per expert per step (b8: ~9.3). Prefill again: natural-b16 (411 GB, 233 experts) above disjoint-b16 (381 GB, 256 experts), the same skew effect as b8 (caveat 1). **Next: b32, the heaviest runs** (disjoint-b32 ≈ 256 experts × 32 tokens; expect ~2.4 s/step and ~240k launches/step, so very large nsys traces). Watch these for nsys timeouts or memory problems during trace parsing. |
| 10 | moe-decode-natural-b32 (07:12–08:31, **79 min**, with tier 2) `runs/20261006T014258Z__kimi-linear-48b-fp8__moe-decode-natural-b32__hf__moe` | ✅ all stages ok (tier 2: 60 + 60 launches) | Decode **896 ms/step = 35.7 tok/s**, busy 35%, 84.2k launches/step. **Decode L2 42.97 GB/step.** **Experts/pass decode: observed 86, formula 163.** Prefill (32×128) 241 experts, 3.46 s, L2 752 GB. nsys trace 4.0 GB sqlite, parsed fine. | Model check: fixed-b32 floor (15.25, run 11) + (86 − 8) × 0.35 = **42.6 predicted vs 42.97 measured (1%)**. Router concentration holds at b32 (86 vs 163 random). ⚠️ **Memory pressure during ncu tier 2:** the worker held ~110 GB of the 119 GB unified pool (model + batch-32 state + ncu replay buffers), with ~3 GB available at 08:17. The kernel logged **190 `NVRM ... NV_ERR_NO_MEMORY` messages** in two bursts (07:47–07:57 and 08:11–08:19, i.e. during runs 10's two ncu stages) and more up to 09:11 (run 11). **No process was OOM-killed and both ncu collections completed**, so the failed allocations were absorbed (likely ncu's save/restore buffers retrying). See caveat 10. |
| 11 | moe-decode-fixed-b32 (08:31–09:13, 41 min, with tier 2) `runs/20261006T030159Z__kimi-linear-48b-fp8__moe-decode-fixed-b32__hf__moe` | ✅ all stages ok (tier 2: 60 + 60) | Decode **160 ms/step = 200.2 tok/s** (28× batch 1), busy 54%, 12.3k launches/step. **Decode L2 15.25 GB/step.** Experts 8 / 8. Prefill (32×128) 2.01 s, L2 674 GB. | Batch-32 floor. Predicted from the b1 → b8 per-row slope: 16.2 GB; measured 15.25 (6% lower), so per-row non-expert traffic grows slightly slower at large batch (0.29 GB/row from b1 → b32 vs 0.32 from b1 → b8). Fixed-routing throughput keeps scaling almost linearly (7.1 → 57.8 → 108.5 → 200.2 tok/s). |
| 12 | moe-decode-disjoint-b32 (09:13–11:43, **149 min**, with tier 2) `runs/20261006T034348Z__kimi-linear-48b-fp8__moe-decode-disjoint-b32__hf__moe` | ✅ all stages ok (tier 2: 60 + 60) | Decode **2,379 ms/step = 13.4 tok/s**, busy 32%, **238k launches/step**. **Decode L2 102.07 GB/step.** Experts 256 / 256 (every expert, every step). Prefill (32×128) 3.39 s, L2 742 GB. nsys trace **10.7 GB sqlite**; ranking parse ~20 min (orchestrator peak ~7 GB RSS), report parse again ~20 min. | **✅ DECODE SWEEP COMPLETE (12/12).** Prediction from fixed-b32 + 248 × 0.35 GB = **102.05 vs 102.07 measured.** Time slope 8.95 ms/expert/step, the same as b8/b16. Memory: dipped to 1.1 GB (10:27) and 2.9 GB (11:03) available during ncu, with 16+ `NV_ERR_NO_MEMORY` driver messages; no kills, full collections (caveat 10). Why 149 min: two 20-min trace parses plus heavy ncu replay. **After this run the queue WAITED from 11:43:** another user (`vedant-tejas`, `pipeline/real_mc4_timer.py`, 284 MB) started a GPU job, and the queue resumes when it ends (shared-GPU rule). **The wait lasted 2 min: the queue resumed at 11:45:09** with run 13. |
| 13 | moe-prefill-natural-p16 (11:45–11:54, 9 min, no ncu) `runs/20261006T061509Z__kimi-linear-48b-fp8__moe-prefill-natural-p16__hf__moe` | ✅ all stages ok | **Prefill (16 tokens) 0.660 s** (24 tok/s), GPU busy 29%, 63.6k launches. **Prefill L2 28.56 GB.** **Experts/pass: observed 64.5 (range 40–83 across layers), formula 102.** Decode (4 tokens) 140 ms/step, L2 6.27 GB/step, matching the b1 decode runs. | Even 16 tokens pull in ~65 distinct experts per layer, but the router again concentrates (64.5 vs 102 random). **Predictions** using the prefill slope from runs 1–3 (0.341 GB per expert): non-expert floor at p16 ≈ 28.56 − 56.5 × 0.341 ≈ 9.3 GB → **fixed-p16 ≈ 9.3 GB**, **disjoint-p16 (128 experts) ≈ 50 GB**. Prefill time should follow experts: fixed ≪ natural < disjoint. |
| 14 | moe-prefill-fixed-p16 (11:54–12:04, 9 min, no ncu) `runs/20261006T062450Z__kimi-linear-48b-fp8__moe-prefill-fixed-p16__hf__moe` | ✅ all stages ok | **Prefill (16 tokens) 0.135 s**, 4.9× faster than natural-p16, busy 30%, 12.3k launches (natural: 63.6k). **Prefill L2 8.80 GB** (predicted ≈ 9.3, 5.7% off). Experts 8. Decode 132 ms/step, L2 6.27 GB/step. | Slope from fixed/natural p16: (28.56 − 8.80) / 56.5 = **0.350 GB per expert**, identical to the decode slope; the runs 1–3 prefill estimate was 0.341. Revised **disjoint-p16 prediction: 8.80 + 120 × 0.350 ≈ 50.8 GB**. Prefill time again follows launches per used expert. |
| 15 | moe-prefill-disjoint-p16 (12:04–12:15, 10 min, no ncu) `runs/20261006T063443Z__kimi-linear-48b-fp8__moe-prefill-disjoint-p16__hf__moe` | ✅ all stages ok | **Prefill (16 tokens) 1.218 s**, busy 29%, 121.6k launches. **Prefill L2 50.80 GB (predicted 50.8).** Experts 128 (exact). Decode 139 ms/step, L2 6.27 GB/step. | **✅ p16 set complete:** fixed / natural / disjoint prefill = 8.80 / 28.56 / 50.80 GB at 8 / 64.5 / 128 experts; times 0.135 / 0.660 / 1.218 s. Slope 0.350 GB/expert, the same as decode. **Prefill floor grows with prompt length:** fixed p16 8.80 GB → fixed p128 (run 2) 25.17 GB, ≈ 0.146 GB per extra token. **Predictions for p64:** fixed ≈ 8.80 + 48 × 0.146 ≈ **15.8 GB**; disjoint (256 experts) ≈ 15.8 + 248 × 0.35 ≈ **102.6 GB**; natural = 15.8 + (experts − 8) × 0.35, with experts from the observer. |
| 16 | moe-prefill-natural-p64 (12:15–12:25, 10 min, no ncu) `runs/20261006T064508Z__kimi-linear-48b-fp8__moe-prefill-natural-p64__hf__moe` | ✅ all stages ok | **Prefill (64 tokens) 1.187 s** (54 tok/s), busy 31%, 115.8k launches. **Prefill L2 54.65 GB.** **Experts/pass: observed 121.8 (range 86–158), formula 222.** Decode 139 ms/step, L2 6.29 GB/step. | Prediction from the p16 floor-per-token + 0.35 GB/expert: 15.8 + 113.8 × 0.35 = **55.6 vs 54.65 measured (1.7%)**. Router concentration is strongest here relative to the formula: 64 tokens touch ~122 experts, not ~222. Natural-p64 prefill (1.19 s) takes about as long as disjoint-p16 (1.22 s) because they use similar expert counts (122 vs 128): time follows experts, not tokens. |
| 17 | moe-prefill-fixed-p64 (12:25–12:34, 9 min, no ncu) `runs/20261006T065519Z__kimi-linear-48b-fp8__moe-prefill-fixed-p64__hf__moe` | ✅ all stages ok | **Prefill (64 tokens) 0.142 s**, 8.4× faster than natural-p64, busy 36%. **Prefill L2 14.78 GB** (predicted ≈ 15.8, 6.5% low). Experts 8. Decode 136 ms/step, L2 6.29 GB/step. | The prefill floor grows a little less than linearly with tokens (8.80 / 14.78 / 25.17 GB at p16 / p64 / p128), so the linear per-token estimate overshoots slightly. The slope from fixed/natural p64 is (54.65 − 14.78) / 113.8 = **0.350 GB per expert, the 4th independent pair giving 0.350**. Fixed prefill time barely changes with prompt length (0.135 / 0.142 / 0.155 s at p16 / p64 / p128). Revised **disjoint-p64 prediction: 14.78 + 248 × 0.350 ≈ 101.6 GB**. |
| 18 | moe-prefill-disjoint-p64 (12:34–12:45, 10 min, no ncu) `runs/20261006T070453Z__kimi-linear-48b-fp8__moe-prefill-disjoint-p64__hf__moe` | ✅ all stages ok | **Prefill (64 tokens) 2.338 s**, busy 31%. **Prefill L2 101.64 GB (predicted 101.6).** Experts 256 (all). Decode 136 ms/step, L2 6.29 GB/step. | **✅ p64 set complete:** fixed / natural / disjoint = 14.78 / 54.65 / 101.64 GB at 8 / 122 / 256 experts; times 0.142 / 1.187 / 2.338 s. **Predictions for p512** (no fixed-p512 floor yet, so as differences): natural − fixed ≈ (experts − 8) × 0.35, about 69 GB if natural uses ~204 experts (as in the pre-queue Kimi 512-token run); disjoint − fixed ≈ 248 × 0.35 ≈ 87 GB. **Watch for** the prefill skew effect seen at b8/b16 (natural above disjoint despite fewer experts): 512 tokens put many more tokens per expert, so it may reappear at p512. |
| 19 | moe-prefill-natural-p512 (12:45–13:24, 38 min, with tier 2) `runs/20261006T071520Z__kimi-linear-48b-fp8__moe-prefill-natural-p512__hf__moe` | ✅ all stages ok (tier 2: 60 + 60) | **Prefill (512 tokens) 2.00 s**, busy 35%. **Prefill L2 150.97 GB.** **Experts/pass 203.9 (range 157–246), formula 256**; top-8 share 0.44. Decode 140 ms/step, L2 6.48 GB/step. | Matches the pre-queue Kimi 512-token run (204 experts, 2.0 s), so the result is reproducible. **ncu tier 2 confirms the prefill-anomaly mechanism (caveat 11) on non-expert GEMMs:** at 4 token tiles, matrix multiplies move ~7.5× their memory traffic through L2 (80 MB vs 10.7 MB, 84% L2 hits); at 1 tile, 2.6×. Not yet seen on routed-expert GEMMs (the sample covers early layers). Details in `runs/moe-queue/findings.md`. |
| 20 | moe-prefill-fixed-p512 (13:24–13:33, 9 min, no ncu) `runs/20261006T075409Z__kimi-linear-48b-fp8__moe-prefill-fixed-p512__hf__moe` | ✅ all stages ok | **Prefill (512 tokens) 0.217 s = 2,360 tok/s**, 9.2× faster than natural-p512. **GPU busy 84%**, the first phase in the whole sweep where the GPU is mostly busy. **Prefill L2 85.36 GB.** Experts 8. Decode 136 ms/step, L2 6.48 GB/step. | natural − fixed at p512 = 65.6 GB for 196 extra experts → **0.335 GB/expert** (vs 0.350 elsewhere; slightly lower, plausibly because the skew re-read effect, caveat 11, partly offsets). The fixed floor grows sub-linearly with tokens: 8.8 / 14.8 / 25.2 / 85.4 GB at p16 / p64 / p128 / p512. **With only 8 experts and 512 tokens, Kimi's prefill finally keeps the GPU busy**: launch overhead is amortised over big matrix multiplies. **Prediction for disjoint-p512** (256 experts, ~16 tokens each, one tile): 85.4 + 248 × 0.35 ≈ **172 GB**. That would put it *above* natural (151 GB), unlike the b8/b16 prefill anomaly, because at 512 tokens natural's popular experts get ~225 tokens (2 tiles) rather than ~450. Moderate confidence. |
| 21 | moe-prefill-disjoint-p512 (13:33–13:44, 10 min, no ncu) `runs/20261006T080345Z__kimi-linear-48b-fp8__moe-prefill-disjoint-p512__hf__moe` | ✅ all stages ok | **Prefill (512 tokens) 2.355 s**, busy 36%. **Prefill L2 162.98 GB** (predicted ≈ 172, 5% low). Experts 256. Decode 135 ms/step, L2 6.49 GB/step. | **✅ p512 set complete:** fixed / natural / disjoint = 85.4 / 151.0 / 163.0 GB at 8 / 204 / 256 experts; times 0.217 / 2.00 / 2.36 s. As predicted, disjoint lands *above* natural here (unlike the b8/b16 prefill anomaly, caveat 11): at 512 tokens natural's popular experts get ~225 tokens (2 tiles) instead of ~450. **All 21 completed runs passed every stage.** |
| 22 | moe-longctx-natural-p16384 (started 13:44) `runs/20261006T081421Z__kimi-linear-48b-fp8__moe-longctx-natural-p16384__hf__moe` | ❌ **NOT COMPLETED: the DGX ran out of memory and went down** | **Calibration, baseline and nsys finished** (baseline 13:44–13:50; nsys 13:50–13:56, 677 MB sqlite), so timing and trace data exist on disk. Baseline: ok, **measured KV cache 2.40 GB at 16,384 tokens** (docs/07 predicted ~2.3 GB for MLA as transformers stores it), driver peak 127.7 GB. **No manifest and no report**: the orchestrator writes those only at the end. | **What happened:** ncu tier 2 on the 16k-token prefill began ~13:58 and **never finished profiling even its first kernel** (log stops at `Profiling "unrolled_elementwise_kernel": 0%`, 14:02). ncu's kernel replay saves and restores the memory each kernel touches; with a 16k-token prefill on top of the 50 GB model that exhausted the 119 GB unified pool (my watcher saw 2.6 GB available at 14:02). The system then **thrashed for ~1.5 h**: the kernel OOM killer fired at 15:25 but could only kill a tiny desktop process, because the profiled Python process is marked unkillable (`oom_score_adj -1000`, set by ncu) and its memory is GPU/unified allocations, not ordinary RSS. The previous boot ends at 15:43:43; the machine was back up at **15:47** (probably a manual/forced restart). tmux sessions `moe-queue` and `moe-results` were lost with it. See caveat 12. |
| 22b | moe-longctx-natural-p16384 **RE-RUN with `--skip-ncu`** (16:07–16:20, 12 min) `runs/20261006T103757Z__kimi-linear-48b-fp8__moe-longctx-natural-p16384__hf__moe` | ✅ all stages ok (calibration, baseline, nsys); memory guard never tripped | **Prefill (16,384 tokens) 8.89 s = 1,843 tok/s, GPU busy 92%**, prefill L2 3.59 TB, 245 experts/layer. **Decode 174 ms/step (5.8 tok/s)** vs ~140 ms at 128-token context; busy 45%. **Decode L2 13.35 GB/step** vs ~6.3 GB at short context → **+7.0 GB/step from the 16k context.** Measured KV cache 2.40 GB (expanded MLA as transformers stores it: 2.35 GB; true MLA latent would be 0.13 GB, 18× smaller); KDA state 42 MB, flat. | **Long-context finding:** the extra 7.0 GB/step is ~2.9× the 2.40 GB cache. *Hypothesis (not verified):* transformers' `DynamicCache` concatenates the cache on every update, so each step reads the old cache (2.4 GB) and writes a new copy (2.4 GB), and attention then reads it (2.4 GB): ≈ 7.2 GB. That's the first cause the report's verdict lists ("HIGHER than expected", 2.25×). The time cost is modest (+34 ms/step), since decode stays launch-bound. Only the 7 MLA layers carry this; the 20 KDA layers' state stays at 42 MB whatever the context. **MoE sweep complete: 22/22.** |

## To do (in order)

1. [x] **Smoke test** on Qwen3-0.6B: PASSED 2026-10-05. See "Smoke test" above.
2. [x] **Step B: make the numbers correct for this model.** DONE 2026-10-05. See "Step B" above.
3. [ ] **Step C: MoE workloads.** Taken over from the teammate 2026-10-05 (user confirmed).
   - [x] 22 workload YAMLs in `configs/workloads/moe/` (2026-10-05):
     - decode `moe-decode-{natural,fixed,disjoint}-b{1,8,16,32}`: prompt 128, generate 32
     - prefill `moe-prefill-{…}-p{16,64,512}`: batch 1, generate 4
     - `moe-longctx-natural-p16384`
   - [x] **`routing:` implemented 2026-10-05** (see docs/07-moe-routing.md). Was: `_from_dict` ignores unknown keys, so fixed/disjoint currently run as natural. Needed:
     - a routing hook on Kimi's MoE gate
     - a `WorkloadConfig.routing` field
     - routing-aware `decode_read_weight_bytes()`: fixed = 8 distinct experts, disjoint = min(256, 8T), natural = E·(1−(1−k/E)^T), with T = tokens per pass
   - [ ] *(nice to have)* Real-text prompt file under `configs/prompts/`. Also fix `hf_transformers.py` (~L185), which copies the same text into every batch row.
   - [x] (c) **Rank tier-2 kernels from the nsys timeline; tier 1 off by default. DONE 2026-10-06.** Why: tier 1 at its 5,384-launch cap covers only 2.8% of a Kimi prefill (190,726 launches) and 44% of a decode step (12,147 launches), at ~15 kernels/min.
     - Toggles: `ncu.tiers` (default `[2]`; `[1, 2]` or `--tiers 1,2` = tier 1 back on) and `ncu.rank_source` (`nsys` default, `tier1` = the original ranking, or `--rank-source`). Each ranking source falls back to the other.
     - Code: `NcuRunner.rank_kernels_from_nsys()` and `_choose_ranking()` in `runners/ncu_runner.py`; the orchestrator passes the nsys sqlite path; `NsysReport.kernel_time_by_name(key=)`. The deep dive weights stalls by nsys time when tier 1 is absent (`deep_dive._split_by_instantiation`). `assemble._load_phases` builds tier-2-only phases (`traffic_collected=False`); the reports show "not measured" plus a warning. The decode physics check is skipped.
     - Smoke test now passes `--tiers 1,2`, so it keeps exercising tier 1 and the nsys ranking.
     - Checked (CPU only; the GPU was busy): all 7 tier/rank_source/fallback branches with stubbed ncu; nsys ranking on the smoke and Kimi timelines (Kimi: 158 s to parse the 1.2 GB sqlite); a tier-1-free report renders correctly; the tier-1 report is unchanged.
     - **Not yet checked: a real GPU run with tier 2 ranked from nsys.** Run the smoke test first.
   Original items:
   - [ ] Real-text prompts instead of random token IDs.
   - [ ] Fix `hf_transformers.py` text prompts: every batch row is currently an identical copy, so the rows route to identical experts.
   - [ ] Decode batch sweep (1 to 32).
   - [ ] Short-prompt prefill sweep, and long context.
   - **Ours, once the batch sweep lands:** the physics check assumes batch 1, i.e. top_k experts per layer. For batch B, expected distinct experts per layer is `E·(1−(1−k/E)^B)`, so `decode_read_weight_bytes()` needs a batch argument. Without it, B > 1 runs will read "HIGHER than expected".
4. [x] **After the download:** done 2026-10-04.
   - CPU checks pass on the snapshot (21/21).
   - `nsbench discover` wrote `configs/models/kimi-linear-48b-fp8.yaml`, with `device_map: cuda` set.
   - The discover output shows the step-B bugs concretely:
     - `head_dim 72`, which is wrong for MLA
     - analytic "KV cache @ 8192 = 2038 MB". The real figure is ~1.2 GB as HF stores it (7 MLA layers × ~143 KB/token + ~42 MB KDA state), or ~110 MB with true MLA latent caching.
     - FP8 shown as "compressed-tensors (8 bits/weight)". That happens to be right here, but only by accident.
5. [x] **Sanity run** of the real model: PASSED 2026-10-05 (see above).
6. [ ] **First profiled Kimi-Linear run** (GPU idle). Attempt 1 (2026-10-05 17:19, balanced, standard profile) was stopped during tier-1 ncu. 3(c) has landed (tier 1 off, nsys ranking), so rerun after the smoke test: `nsbench run --model configs/models/kimi-linear-48b-fp8.yaml --profile configs/profiles/quick.yaml`. Expect the run to have no byte totals or physics check; add `--tiers 1,2` only for a run that needs them, and expect it to truncate. Check:
   - why decode is ~10 tok/s, not ~90: launch-bound? (nsys kernel count and gaps; ncu per-kernel)
   - memory headroom
   - the ncu `truncated` flag
   - `cache_state_bytes` in the manifest

---

## Kimi-Linear-48B-A3B (FP8) key facts

| | |
|---|---|
| Repo | `nm-testing/Kimi-Linear-48B-A3B-Instruct-FP8-DYNAMIC`: Neural Magic test org, no model card, accuracy unvalidated |
| Size | 50 GB (10 shards). FP8 per-channel weights, dynamic per-token FP8 activations. lm_head, router, embedding, norms in bf16. |
| Layers | 27: 20 KDA (linear attention) and 7 MLA (1-based layers 4, 8, 12, 16, 20, 24, 27, i.e. 0-based 3, 7, …, 26) |
| MoE | 256 experts, top-8 plus 1 shared, layer 0 dense, ~3B active per token. 49.12B params. |
| Decoding state | KDA: 32×128×128 fp32 per layer, **~40 MiB fixed**. MLA (as HF stores it): **~143 KB/token**. True MLA latent: ~8 KB/token. |
| Code | `trust_remote_code`, written for transformers 4.57. Runs on 5.18 via `nsight_bench/compat`. Needs fla 0.4.0. |
| Memory fit | ~50 GB of 128 GB. Comfortable alone; check what else is on the GPU first. |
