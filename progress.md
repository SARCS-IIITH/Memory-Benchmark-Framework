# Progress log

Running log of setup and model-onboarding work on the DGX Spark (GB10) as user `imc`.
Update this file whenever something is finished or decided, so work can resume after a logout.

_Last updated: 2026-10-05, ~15:35 IST_

---

## ▶ Resume here

| Item | State |
|---|---|
| Env `~/envs/nsbench` | ✅ built, verified |
| `nsbench preflight` | ✅ passes |
| Kimi-Linear FP8 download | ✅ **done** (2026-10-04 ~20:30). 49.96 GB, verified. Config: `configs/models/kimi-linear-48b-fp8.yaml` |
| Step A: harness support for Kimi-Linear | ✅ **done**. CPU checks 21/21 and GPU checks 13/13 pass. |
| Qwen3-0.6B smoke test | ✅ **passed** (2026-10-05 12:14–12:20, GPU idle). See "Smoke test" below. |
| Step B: correct analysis for hybrid / MLA / MoE | ✅ **done** (2026-10-05). Analysis checks 27/27 pass, step-A checks still 21/21, Qwen report unchanged. See "Step B" below. |
| Step C: MoE workloads | 👤 **a teammate on this same `imc` account is doing this.** Don't edit `configs/workloads/` or `nsight_bench/workloads/`. |
| Kimi-Linear sanity run (real 50 GB model, no profiler) | ✅ **passed** (2026-10-05 ~12:35). Loads as FP8, gives correct and fluent answers. See "Kimi sanity run" below. |
| First profiled Kimi-Linear run | ⬜ **ready.** Waiting only on the teammate's workloads (step C), or run now on an existing workload. GPU must be idle. |

**Before anything uses the GPU**, check it is free:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=utilization.gpu --format=csv
```

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

## To do (in order)

1. [x] **Smoke test** on Qwen3-0.6B: PASSED 2026-10-05. See "Smoke test" above.
2. [x] **Step B: make the numbers correct for this model.** DONE 2026-10-05. See "Step B" above.
3. [ ] **Step C: MoE workloads. A TEAMMATE IS DOING THIS** (same `imc` account). Don't edit workload files. Items, for reference:
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
6. [ ] **First profiled Kimi-Linear run** (quick profile, GPU idle). Step B is done; ready. `nsbench run --model configs/models/kimi-linear-48b-fp8.yaml --profile configs/profiles/quick.yaml`. Check:
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
