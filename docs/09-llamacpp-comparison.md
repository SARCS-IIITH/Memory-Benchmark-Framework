# Comparing the testbench with llama.cpp

What `nsight_bench` and llama.cpp each are, how they test, what results each produces, which of those results can be compared, and the plan for a side-by-side Qwen3-0.6B run.

> **Status (2026-10-10):** **phase 1 and the main phase 2 run are done.** On identical BF16 weights, llama.cpp decodes Qwen3-0.6B **2.4× faster** than HF transformers (165.9 against 68.6 tok/s). Run inside the testbench, llama.cpp decodes at 172.3 tok/s, within 4% of its own benchmark, so the testbench adds no overhead; it reaches ~99% of the measured memory bandwidth ceiling and reads 1.23 GB per decode step against 1.34 GB for HF. Results: section 7 (phase 2 in 7.8). Section 2 explains what each engine is. Still to do: a `greedy` token-mode run (7.8.6). **Recommendation (section 9): use llama.cpp as the default engine from now on**, with HF transformers kept for FP8, batch sweeps, forced routing and per-layer detail.

## TL;DR

We checked whether the testbench's results reflect real inference by comparing it with llama.cpp, a proper inference engine. The testbench itself measures correctly, but until now it ran models through Hugging Face transformers, a general-purpose Python library rather than an engine built for speed (section 2). On identical Qwen3-0.6B weights, llama.cpp generated text about **2.4× faster** (≈166–172 vs ≈68 tokens/s). Plugged into the testbench, llama.cpp matched its own benchmark within 4%. So the measuring setup doesn't slow anything down; the slowness came from transformers (7.8.1). It had two causes. For each Qwen3-0.6B token, transformers sends about 1,600 tiny GPU jobs (kernel launches), against 369 for llama.cpp, so the GPU sat idle ~30% of the time, against 5% for llama.cpp. These counts are model-specific (Kimi-Linear on transformers was ~12,000 per token). The general reason: PyTorch runs every small operation in the Python model code as its own kernel, while llama.cpp merges chains of small operations into single kernels and handles all of an MoE layer's chosen experts in one kernel, so the gap grows for MoE models. Its kernels were also slower, with half their time spent on small steps that move almost no data (7.8.2, 7.8.3). The amount of memory read per token was nearly the same (1.34 GB vs 1.23 GB, against ~1.22 GB the model must read; 7.8.4). So the testbench's **byte measurements describe the model**, but its **timings describe transformers**. With llama.cpp, the model runs at ~99% of the machine's memory bandwidth limit, so it's limited by memory, not by overhead as the transformers runs suggested (7.8.5). The file format (GGUF vs safetensors) doesn't affect speed; it's just what each engine reads. GGUF can't store FP8, though, so Kimi would have to run at a near-equivalent 8-bit format, Q8_0 (2.5, 6.4).

Going forward, we recommend **llama.cpp as the default engine**, keeping transformers for what only it can do: FP8 models, batch sizes above 1, forced MoE routing and per-layer detail (section 9). For Kimi-Linear, the llama.cpp build used here supports the model, and it should need far fewer GPU jobs per step (an estimate, not yet run), which would make detailed ncu tier 1 profiling practical again (9.6). Separately, in the transformers Kimi runs, the MLA layers were expanded and ran as ordinary attention, storing ~18× more cache than MLA is designed to. That wasn't the main reason one profiled process reached ~100 GB, though. That was ncu working on the 16k-token long-context run: ~50 GB of weights, ~16 GB from the long prompt and PyTorch's held memory (only ~2 GB of it from the expanded MLA cache; llama.cpp would cache MLA compressed instead, 9.6), and roughly 34 GB of ncu's own copies for replaying kernels, which ran the machine out of memory. Never run ncu on long-context runs of large models on this machine, whichever engine is used ([07-moe-routing.md](07-moe-routing.md) section 5.7).

---

## 1. Why this comparison

Every result so far comes from one engine: Hugging Face transformers, driven by the harness's own prefill/decode loop. Several findings may belong to that engine rather than to the model or to GB10:

- Kimi-Linear's GPU is idle 68% of each decode step (launches and host syncs).
- Kimi's MLA layers are cached expanded, ~143 KB/token instead of ~8 KB/token ([07-moe-routing.md](07-moe-routing.md)).
- Long context adds ~7 GB of traffic per decode step, possibly from `DynamicCache` copying ([07-moe-routing.md](07-moe-routing.md) section 5.3).

Running the same model on an optimised engine shows how much of what we measure is transformers. llama.cpp is the engine chosen for this check, because the review raised GGUF/llama.cpp specifically. It also builds from source in the home directory without root or docker, which rules out TensorRT-LLM on this host (see `nsight_bench/backends/trtllm.py`). vLLM would be a valid second comparison later; it reads the existing safetensors and is the route to native MLA caching.

## 2. What each one is

### 2.1 Was the testbench using an inference engine?

It depends on the definition.

- **Loose sense: software that runs a model to produce output.** Yes. HF transformers + PyTorch ran Qwen and Kimi and produced real tokens. That is inference, which is why this doc calls it "the testbench's engine".
- **Strict sense, the usual industry meaning: software built specifically to run models fast in production** (vLLM, SGLang, TensorRT-LLM, llama.cpp). By that definition, **no**. HF transformers is a **model library**, a reference implementation. Its job is to define hundreds of models clearly and correctly in Python for research, fine-tuning and checking results. Speed comes second.

The testbench also doesn't use transformers' own generation function (`generate()`). It runs the prefill and decode loop itself and calls the model's `forward()` once per token, so each generated token gets its own NVTX range for the profilers (module docstring of `hf_transformers.py`). That gives clean attribution, but costs speed. So what ran before this comparison was:

```
the testbench's own loop  →  HF transformers model code  →  PyTorch (eager)  →  GPU
```

| Layer | What it is |
|---|---|
| `nsight_bench` | Loads the model, runs a hand-written prefill/decode loop with NVTX ranges around each phase, and drives nsys and ncu around it. A **profiling harness**, not an engine. |
| HF transformers (`nsight_bench/backends/hf_transformers.py`) | Runs the model's maths. A reference implementation: eager PyTorch, no CUDA graphs, one `forward()` call per token. |
| The model's own code | For Kimi, its `trust_remote_code` file, including the per-expert loop and per-layer `.cpu()` sync. |

| | Before (HF transformers) | llama.cpp |
|---|---|---|
| Runs inference? | Yes | Yes |
| Built as an inference engine? | No: a general model library on a general framework | Yes |
| Optimised for speed? | No: eager PyTorch, ~1,600 launches/token, GPU 71% busy (Qwen decode) | Yes: fused kernels, CUDA graphs, GPU 93% busy (phase 2 smoke run) |
| Typical use | Research, fine-tuning, reference results | Running models for real |

The testbench was therefore measuring a correct but unoptimised way of running the model. Its byte counts are still right, because every engine must read the same weights. Its timings show what HF transformers achieves, not what the model can do on GB10 with an inference engine (section 7.6).

> **Terminology.** "Transformers" in this doc means **Hugging Face's `transformers` Python library**. It does not mean the transformer *architecture*. Qwen, Llama and Kimi are all transformer models, and both engines here run a transformer model.

### 2.2 How HF transformers + PyTorch runs a token

1. **The model is written in Python.** Hugging Face's `modeling_qwen3.py` defines each part (attention, feed-forward, normalisation) as Python classes.
2. **Each line of that Python calls a PyTorch operation:** a matrix multiply, an add, a normalisation, a rotation (RoPE), attention (`sdpa`).
3. **PyTorch runs operations one at a time ("eager mode").** For each one it picks a GPU kernel (cuBLAS for matrix multiplies, its own kernels for the small steps) and sends it to the GPU straight away.
4. **The result is ~1,600 separate GPU launches per Qwen token**, each sent from Python through PyTorch. Each send costs a few microseconds of CPU time, and the GPU sits idle when the CPU can't keep up: 29% of each decode step.
5. **Memory is managed as it goes.** The KV cache (`DynamicCache`) grows by building new, larger tensors as tokens are added.

PyTorch is a general-purpose framework, built for research and training as well as inference. Faster modes exist (`torch.compile`, CUDA graphs), but the testbench doesn't use them.

### 2.3 How llama.cpp runs a token

1. **The model is written in C++.** llama.cpp has its own description of each architecture, Qwen3 included, from which it builds a **compute graph**: the full list of operations for a step, worked out in advance.
2. **ggml, llama.cpp's own tensor library, runs that graph.** Its CUDA backend uses kernels written for this exact job. For example, `mul_mat_vec_f` multiplies a BF16 weight matrix by a single token's vector, which is exactly the shape of a decode step (it tops the phase 2 smoke run's decode kernel list).
3. **It merges operations ("fusion")** where possible, for example a normalisation and the multiply after it. That gives ~369 kernels per Qwen decode step instead of ~1,600.
4. **It uses CUDA graphs.** The whole sequence of kernels is recorded once and then replayed with a single command per token. The CPU barely takes part, so the GPU is rarely left waiting: 93% busy in the phase 2 smoke run.
5. **Memory is allocated once, up front.** Weights, the KV cache (sized for the whole context) and working space are all reserved at load time. Nothing is allocated while generating.
6. **No Python is involved.**

| | HF transformers + PyTorch | llama.cpp |
|---|---|---|
| Model written in | Python | C++ |
| Runs operations with | PyTorch, general-purpose | ggml, built for inference |
| GPU kernels | General (cuBLAS + PyTorch's own) | Hand-written for these exact shapes, plus fusion |
| Launches per Qwen decode step | ~1,600, each sent separately | ~369, sent together as one CUDA graph |
| CPU involvement per token | High (Python + PyTorch per operation) | Very low |
| Memory | Allocated as needed; cache grows | All reserved at load time |
| Built for | Flexibility, research, training | Fast inference |

llama.cpp also ships several tools. Phase 1 used `llama-bench`, which measures speed.

### 2.4 What runs llama.cpp in each phase

**Phase 1** used llama.cpp's own benchmarking program. The testbench wasn't involved, so the only result is tokens per second:

```
llama-bench  →  llama.cpp  →  GPU
   └ times itself with a stopwatch; no profiler attached
```

**Phase 2** runs the same llama.cpp engine **inside the testbench**. The testbench loads llama.cpp as a library, the way it loads HF transformers, drives it token by token, puts its NVTX ranges around prefill and each decode step, and runs its usual stages with nsys and ncu:

```
testbench (Python): calibration, unprofiled timing, nsys pass, ncu pass, analysis, report
   └─ llamacpp backend (nsight_bench/backends/llamacpp.py)
        └─ llama-cpp-python 0.3.36: a thin Python wrapper (ctypes)
             └─ libllama.so: the llama.cpp C library, compiled with CUDA for GB10 (sm_121a)
                  └─ ggml + its CUDA kernels → GPU
```

- **`llama-cpp-python` doesn't run the model itself.** It only lets Python call functions in the compiled llama.cpp library.
- **The library is the same llama.cpp as phase 1:** the same commit (`0c1e570`, verified file for file, section 6.4), compiled for the same GPU target.
- **The backend makes one call per step:** `llama_decode` once for the whole prompt, then once per token. Python is involved once per token, not ~1,600 times.
- **PyTorch is still loaded in the process, but doesn't run the model.** The testbench uses it for its own tools: the NVTX markers that tell nsys and ncu which part is prefill and which is a decode step, the profiler start/stop signals, memory sampling, and generating the prompt token IDs. The HF tokenizer is loaded only to read the vocabulary size, so the prompt IDs are identical to the HF run.

### 2.5 safetensors and GGUF

Both are file formats that store a model's weights. They differ in what the file holds and which software expects it.

**safetensors (Hugging Face's format)**

- **It stores tensors only.** A small header lists each tensor's name, data type, shape and position in the file; the raw numbers follow.
- **Everything else is in separate files** next to it: `config.json` (the architecture), the tokenizer files (vocabulary, merge rules), generation settings. A Hugging Face model is a **folder**, not one file.
- **Tensor names follow Hugging Face's Python code**, for example `model.layers.0.self_attn.q_proj.weight`.
- **Number types are standard:** FP32, FP16, BF16, FP8, INT8. Quantization schemes, such as Kimi's FP8 with scale factors, are stored as extra tensors, with `config.json` saying how to interpret them.
- **"Safe"** means loading it can't run hidden code, unlike PyTorch's older pickle-based `.bin` files.
- **Read by** HF transformers, vLLM, SGLang, TensorRT-LLM and others: the general standard.

**GGUF (llama.cpp's format)**

- **It is one self-contained file**: weights plus the architecture settings, the full tokenizer vocabulary and merge rules, and the chat template, all in a key-value header.
- **Tensor names follow llama.cpp's convention**, for example `blk.0.attn_q.weight`.
- **It has llama.cpp's own number types built in:** F32, F16, BF16, plus block-quantized types (Q8_0, Q4_K, IQ2…) designed together with llama.cpp's kernels. It has **no FP8 type** (section 6.4).
- **It is laid out for fast loading**, so llama.cpp can map the file straight into memory.
- **Read by** llama.cpp and the tools built on it (Ollama, LM Studio). HF transformers can read some GGUF files, but turns the weights back into PyTorch tensors.

| | safetensors | GGUF |
|---|---|---|
| What it holds | Weights only | Weights + architecture + tokenizer + chat template |
| A model is | A folder of several files | One file |
| Tensor naming | Hugging Face convention | llama.cpp convention |
| Quantization | Standard types; scheme described in `config.json` | llama.cpp's own block types, built into the file |
| FP8 | Yes | No |
| Mainly read by | HF transformers, vLLM, SGLang, TensorRT-LLM | llama.cpp, Ollama, LM Studio |

**Here**, the conversion copied the same BF16 numbers into the GGUF and added Qwen's settings and tokenizer from the folder. The only difference is that it kept both copies of the embedding matrix (section 6.4).

**The file format doesn't make anything faster.** The speed difference comes from the engine (2.2 and 2.3). GGUF matters only because llama.cpp needs it. A comparison against vLLM would read the original safetensors files directly.

## 3. Side by side

| | Testbench | llama.cpp (`llama-bench`) |
|---|---|---|
| Purpose | Explain where bytes go in the memory hierarchy, and why | Measure tokens per second |
| Who runs the model | HF transformers (+ the model's Python code) | llama.cpp's own C++/CUDA implementation of each architecture |
| Model format | `.safetensors`, the original checkpoint at original precision | `.gguf`, a converted file; usually quantized (Q4_K_M, Q8_0…), or F16/BF16 |
| Execution | Eager PyTorch, many small kernel launches, host involved every step | Fused kernels, CUDA graphs on decode, little host work |
| How a run works | Four collections of one workload: calibration gate, unprofiled timing baseline, nsys, ncu | Each test repeated (default 5) and averaged |
| Profilers | Central; nsys and ncu produce most of the results | None; it times itself with wall clock |
| Self-validation | Yes: DRAM-byte derivation checked against a known kernel (+0.02%); decode traffic compared with prediction | No; reports mean ± stddev |
| Output | `report.md`, `report.html`, `manifest.json`, `metrics/`, `raw/` per run | One row per test: tokens/s ± stddev (markdown, CSV, JSON or SQL) |
| Time per run | Qwen ~6–30 min; Kimi ~50 min without ncu tier 1 | Seconds to minutes |

## 4. What each one tests and produces

### The testbench

Per phase (prefill, one decode step), from one run:

- **Timing**, from the unprofiled baseline: prefill time and tok/s, decode ms/token and tok/s.
- **GPU busy %**, from nsys: the share of each phase spent running kernels, versus gaps from launch latency or host syncs. Plus kernel launches per phase.
- **Memory traffic**: L2 bytes from nsys sampling (on by default, [08-nsys-l2-sampling.md](08-nsys-l2-sampling.md)); exact DRAM bytes and per-level bytes and hit rates from ncu tier 1, when it is on.
- **Bytes per generated token**, and amplification between levels.
- **Footprint**: weights and KV cache / recurrent state, measured and compared with prediction.
- **Roofline position and bandwidth utilisation**, against ceilings measured on this machine (~242 GB/s DRAM, ~1035 GB/s L2, ~98 TFLOP/s bf16).
- **Top kernels and their stall reasons**, from ncu tier 2.
- **Physics check**: does a decode step read about the weights plus the KV cache, as it must.

### llama.cpp (`llama-bench`)

- **pp** (prompt processing): prefill tok/s, e.g. `pp128`.
- **tg** (text generation): decode tok/s, e.g. `tg128`.
- Optionally `-d N` (run the test after N tokens are already in context) and `-pg P,G` (prompt then generate, combined).
- Model size and parameter count, backend, GPU layers, build commit.

Nothing about memory traffic, cache behaviour or kernels. Other llama.cpp tools cover output quality (`llama-perplexity`) and serving (`llama-server`), not the memory hierarchy.

### The difference in testing style

- **The testbench is diagnostic.** It slows the model down on purpose to look inside (NVTX ranges, ncu replaying kernels), so it keeps timing in a separate unprofiled run and never reports profiled durations as performance.
- **`llama-bench` is a speed test.** It treats the engine as a black box and reports throughput. It cannot say *why* a number is what it is.

## 5. What the comparison can and cannot tell us

There are two kinds of "accuracy" here, and the comparison answers only one.

1. **Are the testbench's measurements correct?** A different engine cannot answer this, because it does different work: different kernels, cache layout and launch pattern. The numbers will differ even if both are measured perfectly. Measurement correctness is checked against independent sources for the *same* run, which is already done: the calibration gate, nsys L2 against ncu exact bytes ([08-nsys-l2-sampling.md](08-nsys-l2-sampling.md)), and measured against predicted KV cache.
2. **Are the testbench's runs representative of real inference?** This is what the comparison answers: how far transformers is from an optimised engine, and which findings come from the engine rather than the model or GB10.

To make (2) meaningful, profile llama.cpp with the **same nsys setup**. Then GPU busy % and L2 bytes per decode step are measured the same way on both engines, not compared against llama.cpp's self-reported speed alone.

## 6. Plan: Qwen3-0.6B on both

### Why Qwen3-0.6B first

- llama.cpp supports Qwen3 well. Support for Kimi-Linear (KDA + MLA + MoE) has not been checked.
- We already have a full testbench run on it, including exact DRAM bytes from ncu tier 1 ([08-nsys-l2-sampling.md](08-nsys-l2-sampling.md) section 4).
- It is small and fast, so both sides run in minutes.

### Matched settings

| Setting | Testbench | llama.cpp |
|---|---|---|
| Weights | `/opt/ai-models/.../Qwen3-0.6B` safetensors | **The same safetensors**, converted with `convert_hf_to_gguf.py --outtype bf16` |
| Precision | bf16 | bf16 (no quantization, so the weights are identical) |
| Prompt / generated tokens | 128 / 128 (`decode-focused` workload) | prefill `-p 128 -n 0`; decode `-p 0 -n 128` |
| Decode context | Decodes after the 128-token prompt | `-d 128`, so decode also starts at 128 tokens of context. Plain `tg128` starts from an empty context. |
| Batch | 1 | 1 (single sequence) |
| Repetitions | `repeat: 5`, `warmup_iters: 3` | `-r 5` (llama-bench also runs a warmup) |
| All layers on GPU | yes | `-ngl 99` |
| Prompt content | synthetic, seed 1234 | llama-bench's own tokens; content does not matter for a dense model |

Optionally repeat at the `balanced` shape (512 → 64) for a second point.

### Two phases

The work is split in two, done in order:

- **Phase 1: standalone `llama-bench`.** llama.cpp's own speed, with no harness involved.
- **Phase 2: a llama.cpp backend inside the testbench** (`--backend llamacpp`, through `llama-cpp-python`). Both engines then go through the same calibration, baseline, nsys and ncu stages and get the same report. This replaces profiling `llama-bench` by hand and splitting phases by kernel timing. The phase 1 `llama-bench` number is the check on the backend: if the backend's decode tok/s is close to it, the harness adds no overhead to llama.cpp.

**Both phases must use the same llama.cpp version.** `llama-cpp-python` bundles its own llama.cpp, so phase 1 builds exactly the commit that the chosen `llama-cpp-python` release bundles. Otherwise the phase 2 check would partly compare two llama.cpp versions.

### Phase 1 steps

1. **Check the GPU is idle** (`nvidia-smi`, see progress.md). Run every step that uses the GPU inside a named tmux session.
2. **Pin the version.** `llama-cpp-python` 0.3.36 (2026-09-30) bundles llama.cpp commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
3. **Build llama.cpp** at that commit under `~/llama.cpp` with CUDA (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`).
4. **Convert the model** to BF16 GGUF in a separate venv (`~/envs/llamacpp`), so the `nsbench` env is not touched. Output to `~/ai-models/gguf/Qwen3-0.6B-bf16.gguf`.
5. **Sanity check**: one short `llama-cli` prompt to confirm the output is fluent.
6. **Testbench runs**, fresh, the same day, on `decode-focused`: one with `--skip-ncu` and one with `--tiers 1,2` (exact DRAM bytes and the decode physics check).
7. **llama-bench speed run**, no profiler: `scripts/run_llamabench_qwen.sh`. Two invocations, because `-d` applies to every test in one: prefill `-p 128 -n 0 -d 0` (from an empty context, like the testbench) and decode `-p 0 -n 128 -d 128` (after 128 cached tokens). Saves JSON, a markdown table and a metadata file (commit, GGUF size, GPU clocks) under `runs/llamacpp-qwen/`.
8. **Fill in section 7** and update progress.md.

### 6.4 Phase 1 setup log (2026-10-09)

- **Network.** The DGX's own wired connection (`10.1.73.x`) downloads from GitHub at ~60–95 KB/s and PyPI is slow or times out. A full `git clone` of llama.cpp and a single-commit `git fetch` both timed out. What worked is a resumable download of the commit's source tarball (`~/downloads/fetch_phase1.sh`, run in tmux). `~/llama.cpp` therefore has no `.git`; the commit is recorded in `~/llama.cpp/PINNED_COMMIT`.
- **Build.** cmake 3.28, CUDA 13.0.88, GCC 13.3. cmake turned `121` into **`121a`** itself (the GB10-specific target), so no extra flag was needed. `-DLLAMA_CURL=OFF`. Built without errors: `llama-bench`, `llama-cli`, `llama-completion`. The only error in the log is a failed download of web-UI files, which nothing here uses.
- **Defaults worth knowing.** `llama-bench` already puts every layer on the GPU (`-ngl` default `-1`) and uses flash attention `auto`. The testbench's Qwen config uses `sdpa`.
- **Same llama.cpp in both phases: verified.** The `llama-cpp-python` 0.3.36 source (downloaded for phase 2, not installed) bundles `vendor/llama.cpp`. It is file-for-file identical to the `~/llama.cpp` built here: 0 differing files across `src`, `include`, `ggml/src` and `ggml/include`.
- **Conversion env.** `~/envs/llamacpp`: CPU-only torch 2.14.1, transformers 4.57.6, numpy, sentencepiece, protobuf. `gguf` is not installed: its build needs poetry. That doesn't matter, because `convert_hf_to_gguf.py` loads `gguf` from `~/llama.cpp/gguf-py` itself.
- **Conversion.** `convert_hf_to_gguf.py <Qwen snapshot> --outtype bf16 --outfile ~/ai-models/gguf/Qwen3-0.6B-bf16.gguf`: 5 s on the CPU, 1,509,347,168 bytes, 311 tensors, all BF16 except the F32 norm weights. **The GGUF holds two copies of the embedding matrix** (`token_embd.weight` and `output.weight`, 311 MB each), because the checkpoint stores `lm_head.weight` even though it declares tied embeddings. So llama.cpp holds ~1.50 GB resident against the testbench's 1.19 GB. Bytes per decode step are not affected: a step reads all of the output matrix and one row of the input embedding, the same as the testbench reading the tied matrix once.
- **No FP8 in GGUF.** The converter's `--outtype` choices are `f32, f16, bf16, q8_0, tq1_0, tq2_0`. An FP8 checkpoint is turned back into full precision during conversion. This doesn't affect Qwen, which is BF16. For Kimi the closest stand-in is Q8_0 (~8.5 bits/weight against FP8's ~8): almost the same bytes per decode step, but a different number format.

## 7. Results

> **A note on the word "transformers".** In this section, "transformers" always means **Hugging Face's `transformers` Python library**: the software the testbench uses to run the model. It does not mean the transformer *architecture*. Qwen is a transformer model on both sides. llama.cpp runs that same model with its own C++ code and GPU kernels, and doesn't use the Hugging Face library, PyTorch or Python at run time. The library was used once, by `convert_hf_to_gguf.py`, to read Qwen's config and tokenizer while writing the GGUF file.

### 7.1 What was compared

The same model (Qwen3-0.6B), the same weights at the same precision (BF16), on the same machine (GB10), doing the same job: read a 128-token prompt, then generate 128 tokens one at a time, batch 1. The only difference is the software running the model:

- **Testbench**: the harness, with HF transformers running the model.
- **llama.cpp**: its own engine, timed by its built-in `llama-bench` tool.

Any difference in speed therefore comes from the software, not from the model or the hardware.

Generating text has two phases:

- **Prefill**: the model reads the whole prompt in one go. All 128 tokens are processed together, so each weight read from memory is used 128 times.
- **Decode**: the model produces one new token at a time. For every token it must read **all of its weights** (~1.2 GB for Qwen) from memory again, to get just one token out. That's why decode is usually the slow part, and why memory traffic matters most there.

### 7.2 The runs

All on 2026-10-09 except the first:

| Name | Run | Notes |
|---|---|---|
| Earlier testbench | `runs/20261005T203205Z__Qwen3-0.6B__decode-focused__hf__l2check` | 2026-10-06, ncu tier 1 + 2. Kept for reference. |
| Testbench, no ncu | `runs/20261009T164538Z__Qwen3-0.6B__decode-focused__hf__llamacpp-cmp` | 22:15 |
| Testbench, tier 1 + 2 | `runs/20261009T171859Z__Qwen3-0.6B__decode-focused__hf__llamacpp-cmp-t1` | 22:48–23:25. Adds exact memory bytes from ncu. |
| llama.cpp | `runs/llamacpp-qwen/llamabench-20261009-2334-*` | 23:34, 10 repetitions. An earlier 5-repetition run (`…-2333-*`) agrees within 2%. |
| llama.cpp inside the testbench (phase 2) | `runs/20261009T184419Z__Qwen3-0.6B__decode-focused__llamacpp__llamacpp-bench-t1` | 2026-10-10 00:14–00:28, `--backend llamacpp`, ncu tier 1 + 2, token mode `bench`. Same GGUF, same llama.cpp commit as `llama-bench`. |

All timings are **medians**. The testbench reports the median of 5 timed repeats after 3 discarded warm-up rounds. For llama.cpp, the median is taken from the per-repetition samples in the JSON, not `llama-bench`'s printed average (why: 7.7).

### 7.3 The results

| Metric | HF, earlier | HF, no ncu | HF, tier 1 + 2 | llama.cpp, `llama-bench` (phase 1) | llama.cpp in the testbench, tier 1 + 2 (phase 2) |
|---|---|---|---|---|---|
| Prefill, 128 tokens: time | 16.1 ms | 16.2 ms | 17.2 ms | **9.97 ms** | **10.91 ms** |
| Prefill tok/s | 7,948 | 7,892 | 7,432 | **12,835** | **11,735** |
| Decode ms/token | 14.4 ms | 14.6 ms | 15.0 ms | **6.03 ms** | **5.80 ms** |
| Decode tok/s | 69.3 | 68.6 | 66.8 | **165.9** | **172.3** |
| Kernel launches, prefill | 1,594 | 1,594 | 1,594 | – | **670** |
| Kernel launches, per decode step | 1,618 | 1,618 | 1,618 | – | **369** |
| GPU busy, prefill | 66% | 66% | 66% | – | **85%** |
| GPU busy, decode | 72% | 71% | 71% | – | **95%** |
| GPU kernel time per decode step (nsys) | 9.80 ms | 9.75 ms | 9.75 ms | – | **5.87 ms** |
| L2 bytes, prefill (nsys) | 4.66 GB | 4.66 GB | 4.66 GB | – | **4.51 GB** |
| L2 bytes per decode step (nsys) | 1.42 GB | 1.42 GB | 1.42 GB | – | **1.33 GB** |
| Exact DRAM bytes, prefill (ncu) | 2.46 GB | not run | 2.46 GB | – | **2.25 GB** |
| Exact DRAM bytes per decode step (ncu) | 1.34 GB | not run | 1.34 GB | – | **1.23 GB** |
| Decode physics check | 1.10× (ncu DRAM) | 1.16× (nsys L2) | 1.10× (ncu DRAM) | – | **1.01×** (report prints 0.81×; see 7.8.4) |
| Calibration: measured memory ceiling | – | – | 228.0 GB/s | – | 214.5 GB/s |

In short, comparing the no-ncu testbench run with llama.cpp:

| | Testbench (HF transformers) | llama.cpp | How much faster llama.cpp is |
|---|---|---|---|
| Prefill (128-token prompt) | 16.2 ms | 9.97 ms | **1.6×** |
| Decode (time per new token) | 14.6 ms | 6.03 ms | **2.4×** |
| Decode (tokens per second) | 68.6 | 165.9 | **2.4×** |

**llama.cpp generates text about 2.4 times faster, on identical weights.**

### 7.4 Why the testbench is slower: two separate reasons

nsys splits one testbench decode step (13.7 ms in the profiled trace) into:

- **9.75 ms when the GPU is doing work** (running kernels), and
- **~4 ms when the GPU is idle**, waiting for the CPU to send it the next piece of work.

**Reason 1: idle gaps.** HF transformers runs the model as ~1,600 separate small GPU jobs (kernel launches) per token, each sent from Python on the CPU. Sending each one takes a little time, and the GPU sits idle in between. llama.cpp avoids most of this: it combines operations into fewer, bigger kernels and uses **CUDA graphs**, which send a whole pre-recorded sequence of GPU work in one go.

**Reason 2: the GPU work itself is slower.** If idle gaps were the only problem, removing them would leave the testbench at 9.75 ms per token. But llama.cpp's *entire* step, gaps included, takes only 6.03 ms. So even counting only the time the GPU is working, HF transformers is slower. Its kernels are general-purpose PyTorch operations; llama.cpp's are hand-written for exactly this job.

**The testbench loses time in both places:** about 4 ms waiting, plus about 3.7 ms of slower work. Phase 2 will show which kernels account for the second part.

### 7.5 How close each one gets to the hardware limit

A decode step must read about **1.22 GB** from memory: 1.19 GB of weights plus a 29 MB KV cache (the testbench's predicted figure). This machine's memory delivers at most about **228 GB/s**, as measured by the calibration at the start of the tier 1 + 2 run. So the fastest a decode step could possibly be is:

> 1.22 GB ÷ 228 GB/s ≈ **5.4 ms per token**

| | Time per token | Effective speed reading memory | Share of the hardware limit |
|---|---|---|---|
| Physical limit | ~5.4 ms | 228 GB/s | 100% |
| llama.cpp | 6.03 ms | ~202 GB/s | **~89%** |
| Testbench | 14.6 ms | ~92 GB/s (1.34 GB measured ÷ 14.6 ms) | **~40%** |

**llama.cpp runs almost as fast as this hardware allows.** Its speed is set mostly by how fast memory can deliver the weights. The testbench uses less than half of what the memory can give, because the GPU spends so much time idle or in slower kernels.

*Assumption at the time:* llama.cpp reads the same ~1.22 GB per token. **Phase 2 measured it: 1.23 GB per decode step** (ncu tier 1, exact). The assumption held; see 7.8.4.

### 7.6 What this means for the testbench

The testbench measures two kinds of things, and this comparison says something different about each.

**1. Bytes moved through memory: trustworthy, and they describe the model.**
- The tier 1 + 2 run measured **1.34 GB** per decode step, against **1.22 GB** predicted from the model's size: within 10% (the physics check, 1.10×).
- It's identical to the 2026-10-06 run.
- Every engine has to read those weights, so these numbers are facts about the model, not about HF transformers.

**2. Timings: real, but they describe HF transformers, not the model.**
- 68.6 tok/s is true for HF transformers on GB10. It isn't "how fast Qwen runs on GB10"; llama.cpp shows the same model can do 165.9.
- The same goes for conclusions drawn from timing. The testbench report says Qwen decode is limited by dispatch (the GPU waiting on launches), not by memory. That's true for HF transformers. With llama.cpp the same model is limited almost entirely by memory.

**Use the testbench's byte numbers as facts about the model, and its timing numbers as facts about HF transformers.** This applies to the Kimi-Linear findings too, for example the 68% GPU idle time per decode step ([07-moe-routing.md](07-moe-routing.md) section 5.5).

#### Why bytes describe the model and time describes the engine (general)

**Bytes are set by the model.** To produce one token, the GPU must read every weight the token passes through, plus the cache of earlier tokens. How many bytes that is depends on the model: its size, its precision (BF16 = 2 bytes per weight, FP8 or Q8_0 ≈ 1), how many experts are active (MoE), and the context length. Every engine has to read those bytes, so they come out nearly the same whichever engine runs the model. On Qwen: 1.34 GB (HF) vs 1.23 GB (llama.cpp), against ~1.22 GB the model needs. Engines add small extras on top, such as writing and re-reading intermediate results. The exception is when the engine changes **what is stored**: caching MLA expanded instead of compressed ([07-moe-routing.md](07-moe-routing.md) section 5.7), copying the whole cache every token (`torch.cat`), or running a different precision (FP8 vs Q8_0). So bytes are *mostly* about the model; check for those exceptions.

**Time is set by the engine.** How long those bytes take depends on how efficiently the engine keeps the GPU reading them: how many kernels it launches, how long the GPU waits for the CPU between them, and how well each kernel streams data. Same bytes, very different times: 14.6 ms per Qwen token on HF, 5.8–6.0 ms on llama.cpp.

**Memory-bound vs overhead-bound.** Memory can only deliver so much per second (~215–228 GB/s measured on this machine), which sets a floor on the time per token:

> fastest possible time ≈ bytes needed ÷ memory bandwidth

For Qwen: 1.22 GB ÷ ~215–228 GB/s ≈ 5.4–5.7 ms.

- **Memory-bound:** the engine runs close to that floor. The GPU spends its time streaming weights as fast as memory allows, and only faster memory or fewer bytes (a smaller model, lower precision) would make it quicker. llama.cpp: 5.8–6.0 ms, ~99% of the measured limit (7.8.5).
- **Overhead-bound:** the engine runs well above the floor because the GPU keeps waiting on the CPU and on tiny kernels. Faster memory wouldn't help, because the time goes elsewhere. HF transformers: 14.6 ms, ~40% of the limit.

The same model got opposite diagnoses depending on the engine. "What limits this model" only describes the model when the engine runs near the floor; otherwise it describes the engine.

**Rule of thumb for future runs:** compare measured time with **bytes ÷ bandwidth**. If they're close, the model is memory-bound and the timing reflects the model and hardware. If time is much higher, look at the engine (GPU idle %, launch counts) before drawing conclusions about the model. As a general pattern, single-sequence decode with a good engine is usually memory-bound. Prefill and large batches reuse each weight many times, so they tend to move towards being limited by compute instead.

### 7.7 Measurement notes

**Why the median and not `llama-bench`'s average.** In both `llama-bench` runs the first prefill attempt took ~17.5 ms and every later one 9.6–10.3 ms, despite llama-bench's own warm-up run. The GPU clock read 305 MHz just before the run, so the first attempt probably ran while the GPU was still speeding up. That one slow attempt drags the printed average down (11,658 ± 2,384 and 12,321 ± 1,793 tok/s); the median ignores it. The testbench also reports medians, after 3 discarded warm-up rounds, so medians on both sides keep the comparison fair. Decode had no such problem: every attempt was 165–169 tok/s.

**The testbench's own run-to-run variation.** The two testbench runs today differ by 3–6% in speed (68.6 vs 66.8 tok/s decode; 16.2 vs 17.2 ms prefill), while their bytes, launches and busy % are identical. Each timing comes from its own unprofiled run, and the tier 1 + 2 run's timing stage didn't overlap any other work started here. So treat about ±5% as normal noise on this shared machine. The gaps to llama.cpp (60% and 140%) are far bigger, so the conclusions stand.

**The comparison is fair on token handling.** One possible unfairness: if one side copied each new token back to the CPU and the other didn't, the copying side would pay for a wait on every token. Neither does in the timed part. `llama-bench` doesn't pick tokens at all, and the testbench keeps generated tokens on the GPU until timing ends (`GenerationState.generated_tokens` in `nsight_bench/backends/base.py`).

**Sanity check of llama.cpp's output.** `llama-completion`, greedy decoding, prompt "The capital of France is" → "Paris. The capital of Italy is Rome. …": fluent and correct. Its log confirmed the device `CUDA0 (NVIDIA GB10)`, **all 29/29 layers on the GPU**, flash attention on, and CUDA graphs on and reused.

**One unexplained detail.** The same log says llama.cpp used `0.00 MiB` of GPU memory for the model (`CUDA0 model buffer size`) and for the KV cache. It clearly ran on the GPU (above). The likely explanation is that llama.cpp reports memory this way on a machine where CPU and GPU share one pool, but that's not confirmed. It doesn't affect the speed results; check it in phase 2 before trusting any llama.cpp memory figure.

### 7.8 Phase 2: llama.cpp inside the testbench

Phase 1 answered **how much faster** llama.cpp is. Phase 2 runs the same llama.cpp engine inside the testbench (section 2.4), so it gets the same full report as HF transformers and shows **why**.

Run: `runs/20261009T184419Z__Qwen3-0.6B__decode-focused__llamacpp__llamacpp-bench-t1`, all four stages ok, no warnings. Same workload (128 → 128, batch 1, 5 repeats after 3 warm-ups) and same ncu depth (tier 1 + 2) as the HF tier 1 + 2 run. Token mode `bench`, which copies `llama-bench`'s loop: predetermined tokens, no read-back of the output, and a wait for the GPU after every token (7.8.6).

#### 7.8.1 The check: does the testbench slow llama.cpp down?

| | `llama-bench` on its own (phase 1) | llama.cpp inside the testbench (phase 2) | Difference |
|---|---|---|---|
| Decode | 6.03 ms/token, 165.9 tok/s | 5.80 ms/token, 172.3 tok/s | testbench **3.9% faster** |
| Prefill, 128 tokens | 9.97 ms, 12,835 tok/s | 10.91 ms, 11,735 tok/s | testbench **9% slower** |

**Decode passes the check.** Inside the testbench, llama.cpp decodes at the same speed as `llama-bench`; it's even 4% faster, which is within the run-to-run noise seen today (7.7). The harness's per-token loop, markers and timing **add no measurable overhead** to decode. So the testbench's llama.cpp decode numbers can be trusted, and the 2.4× gap to HF transformers is the engine, not the measuring setup.

**Prefill is 9% slower inside the testbench.** The report shows the GPU idle for 15% of the prefill phase (1.73 ms). *Likely cause, not verified:* the timed prefill includes host-side setup the backend does in Python before the GPU work starts: clearing the cache's bookkeeping and building the 128-token array for the C library. `llama-bench` does its equivalent in C++. The gap is ~1 ms on one call, so it barely affects any conclusion, but prefill timings from this backend should be read as slightly pessimistic.

#### 7.8.2 Reason 1 confirmed: far fewer launches, far less idle time

| Per decode step | HF transformers | llama.cpp | |
|---|---|---|---|
| GPU kernel launches | 1,618 | **369** | 4.4× fewer |
| Different kernel types | 13 | **8** | |
| GPU busy | 71% | **95%** | |
| Wall time per step (nsys trace) | 13.7 ms | **6.16 ms** | |

| Per prefill (128 tokens) | HF transformers | llama.cpp |
|---|---|---|
| GPU kernel launches | 1,594 | **670** |
| GPU busy | 66% | **85%** |

llama.cpp sends 4.4× fewer GPU jobs per token, and replays them as a CUDA graph, so the GPU is busy 95% of each decode step. With HF transformers ~4 ms of every 13.7 ms step is idle; with llama.cpp it's ~0.3 ms.

#### 7.8.3 Reason 2 confirmed: the kernels themselves are faster

GPU time per decode step, counting only when the GPU is working (nsys): **9.75 ms** for HF transformers against **5.87 ms** for llama.cpp. Where the time goes (ncu tier 1, share of each engine's own kernel time; ncu durations are measured under profiling, so only the shares are compared):

**llama.cpp**, 8 kernel types:

| Kernel | Launches/step | Share of time | Share of DRAM bytes | What it does |
|---|---|---|---|---|
| `mul_mat_vec_f` | 169 | **71.0%** | **97.0%** | Weight matrix × one token's vector: reads the weights |
| `rms_norm_mul_rope_f32` | 56 | 8.7% | 0.2% | Normalisation, scaling and RoPE **fused into one kernel** |
| `rms_norm_f32` | 57 | 7.8% | 0.1% | Normalisation |
| `flash_attn_ext_vec` | 28 | 5.5% | 2.5% | Attention over the KV cache |
| `flash_attn_combine_results` | 28 | 3.3% | 0.1% | Combines attention partial results |
| `k_set_rows` | 28 | 3.3% | 0.0% | Writes the new token into the KV cache |

**HF transformers**, 13 kernel types (top 6):

| Kernel | Launches/step | Share of time | Share of DRAM bytes | What it does |
|---|---|---|---|---|
| `vectorized_elementwise_kernel` | 684 | **28.9%** | 1.2% | Small element-by-element steps (adds, multiplies, activations…) |
| `cublasGemvParamsEx` | 141 | 26.4% | 63.2% | Weight matrix × vector (cuBLAS): reads weights |
| `elementwise_kernel` | 337 | 15.2% | 1.0% | More element-by-element steps |
| `cutlass_80_wmma_tensorop_bf16…` | 56 | 9.1% | 26.4% | Weight matrix multiply (CUTLASS): reads weights |
| `unrolled_elementwise_kernel` | 114 | 5.9% | 0.4% | More element-by-element steps |
| `reduce_kernel` | 114 | 5.4% | 0.7% | Reductions (e.g. inside normalisation) |

**The difference in one sentence:** llama.cpp spends **71%** of its GPU time in the kernel that actually reads the weights; HF transformers spends only **~35%** there, and **~50%** in over 1,100 tiny element-by-element kernels that move almost no data (~3% of bytes) but each cost launch and scheduling time. llama.cpp fuses those small steps (for example `rms_norm_mul_rope_f32`), so they nearly disappear.

#### 7.8.4 Memory bytes: the assumption from phase 1 holds

| Per decode step | HF transformers | llama.cpp |
|---|---|---|
| Exact DRAM bytes (ncu tier 1) | 1.34 GB | **1.23 GB** |
| Predicted minimum (weights read per step + KV cache) | 1.22 GB | 1.22 GB |
| Ratio, measured ÷ predicted | 1.10× | **1.01×** |
| L2 traffic (nsys) | 1.42 GB | 1.33 GB |
| L2 hit rate | 9.1% | 8.1% |

- **llama.cpp reads almost exactly the minimum a decode step needs:** 1.23 GB against 1.22 GB predicted. Phase 1's assumption (7.5) was right.
- **HF transformers reads ~9% more than llama.cpp** (1.34 vs 1.23 GB). *Likely sources, not verified:* its many small kernels writing and re-reading intermediate results, and `DynamicCache` handling. It's a real but modest cost; the big difference between the engines is time, not bytes.
- **Prefill:** llama.cpp 2.25 GB of DRAM traffic against 2.46 GB for HF.

**Why the report prints 0.81× instead of 1.01×.** The report's "expected" figure takes the weights the backend says are resident, 1.50 GB. For llama.cpp that includes **both** embedding copies (section 6.4). But a decode step reads only one row of the input embedding table, which llama.cpp keeps on the CPU side anyway (`CPU_Mapped model buffer`, 296.75 MiB). The right expectation is the same 1.22 GB as for HF, which gives **1.01×**. The measured bytes are correct; only the expectation in this report is too high. *To fix later:* have the llamacpp backend report the weights a step reads, not everything resident.

#### 7.8.5 How close to the hardware limit

| | Decode bytes ÷ time | Share of the ceiling this run measured | Share of 228.0 GB/s (HF run's calibration) |
|---|---|---|---|
| llama.cpp | 1.23 GB ÷ 5.80 ms ≈ **212.7 GB/s** | **99%** of 214.5 GB/s | 93% |
| HF transformers | 1.34 GB ÷ 15.0 ms ≈ **89.5 GB/s** | 39% of 228.0 GB/s | 39% |

**llama.cpp decode runs at the memory limit of this machine.** Decode is limited by memory bandwidth, as the physics says it should be. HF transformers reaches less than 40%, because of idle time and slow small kernels.

The two calibrations measured different ceilings (214.5 and 228.0 GB/s, 6% apart) on different runs. The GPU is shared and the driver reported 31.9 GB of the unified pool in use by all processes during the llama.cpp run, so the achievable bandwidth varies a little from run to run.

#### 7.8.6 Notes

- **Token mode.** This run used `bench` mode, copying `llama-bench`: a predetermined token each step, the output never read, and `llama_synchronize` after every token. `llama-bench`'s own generation loop (`test_gen` in `llama-bench.cpp`) waits for the GPU after every token, so llama.cpp reached 166–172 tok/s *with* a full wait per token. The HF backend never waits per token. A `greedy` mode (read the output, pick the most likely token, as real generation does) is built in (`NSBENCH_LLAMACPP_TOKEN_MODE=greedy`) and not yet run.
- **Footprint figures.** PyTorch's memory statistics show 0 B, as expected: llama.cpp allocates its own memory, which PyTorch can't see. Weights resident 1.50 GB (both embedding copies). The KV cache isn't measured directly, so the report uses the predicted 29.36 MB; llama.cpp's own log shows 28.00 MiB (29.36 MB) for the same context.
- **The "0.00 MiB" puzzle from phase 1 is resolved.** Those lines came from a dry-run sizing pass that `llama-completion` makes before loading anything ("fitting params to device memory"). The real load allocates 1,137 MiB of weights on the GPU, 296.75 MiB for the input embedding table on the CPU side, and a 28.00 MiB KV cache.
- **Not covered:** batch sizes above 1 (the backend runs one sequence), other models, and Kimi-Linear (llama.cpp support not checked; no FP8 in GGUF, section 6.4).

#### 7.8.7 What this means

1. **The testbench measures correctly.** Run on a fast engine, it reports fast numbers that match that engine's own benchmark (decode within 4%). The slowness in the HF runs belongs to HF transformers, not to the measuring setup.
2. **Both of phase 1's explanations are now measured, not inferred:** HF transformers loses time to idle gaps (71% vs 95% busy; 1,618 vs 369 launches) *and* to slower kernel work (9.75 vs 5.87 ms of GPU time per step, half of it in tiny element-by-element kernels).
3. **The byte figures carry over between engines.** The model needs ~1.22 GB per decode step. llama.cpp reads 1.23 GB, HF transformers 1.34 GB. Memory-traffic findings from the testbench describe the model, within ~10%.
4. **Qwen3-0.6B decode on GB10 is memory-bound** when run by a real inference engine, at ~99% of the measured bandwidth ceiling. The HF-based finding that it is "dispatch-bound" describes HF transformers only.

## 8. Caveats

- **Different kernels.** Even at bf16 the two engines compute the same model differently, so small differences in bytes are expected.
- **llama-bench prompt content** is not the testbench's synthetic prompt. For a dense model this does not change the work done.
- **CUDA graphs and nsys.** Without `--cuda-graph-trace=node`, nsys shows each graph as one launch and the busy % is not comparable.
- **nsys L2 is an upper bound on DRAM** (+6% for Qwen decode, ~1.9× for prefill; [08-nsys-l2-sampling.md](08-nsys-l2-sampling.md)). The same applies on both sides.
- **One model, one shape.** A result on Qwen3-0.6B does not carry over to Kimi-Linear automatically. The llama.cpp build used here does include Kimi-Linear (`src/models/kimi-linear.cpp`, `conversion/kimi_linear.py`), but converting and running it has not been tried (section 9.6).

## 9. Recommendation: llama.cpp as the default engine from now on

### 9.1 The decision

For profiling a model's inference behaviour on GB10, **use the testbench with `--backend llamacpp` by default.** Keep `--backend hf` (HF transformers) for the specific cases in 9.3, and label those timings as HF transformers timings.

Nothing about *how* we measure changes. It's the same testbench with the same stages: calibration, unprofiled timing, nsys, ncu tier 1 and tier 2. Only the engine that runs the model changes (section 2.4).

### 9.2 Why

**1. It measures inference the way it really runs.** HF transformers is a model library, not an inference engine (section 2.1). On identical weights it decodes Qwen3-0.6B at 68.6 tok/s against llama.cpp's 172.3, and spends 29% of each decode step idle (section 7.8). Timings and "what limits this model" conclusions from HF describe HF transformers, not the model on GB10. llama.cpp's describe what an inference engine achieves.

**2. It shows the hardware's real limit.** With llama.cpp, Qwen3-0.6B decode ran at ~99% of the measured memory bandwidth ceiling (7.8.5). The model is memory-bound, as the physics says a decode step should be. With HF the same model looked "dispatch-bound", because Python overhead and ~1,100 tiny kernels per token hid the memory behaviour. A memory benchmark should show the memory limit, not the overhead of an unoptimised engine.

**3. The testbench has been checked against llama.cpp's own benchmark.** Inside the testbench, llama.cpp decoded at 172.3 tok/s against 165.9 from `llama-bench` on its own: within 4% (7.8.1). So the testbench adds no measurable overhead, and its llama.cpp numbers can be trusted.

**4. Its bytes match what the model must read.** ncu tier 1 measured 1.23 GB per decode step against 1.22 GB predicted: 1.01× (7.8.4). HF read 1.34 GB (1.10×). The bytes measured with llama.cpp are about as close to the model's true minimum as we can get.

**5. Profiling is much cheaper, so tier 1 becomes usable on large models.** ncu tier 1 profiles every kernel, so its cost grows with the number of kernels:

| Kernels per step | HF transformers | llama.cpp |
|---|---|---|
| Qwen3-0.6B decode step (measured) | 1,618 | **369** |
| Qwen3-0.6B prefill, 128 tokens (measured) | 1,594 | **670** |
| Kimi-Linear decode step | ~12,000 (measured), so ~13 h of tier 1 | ~500–1,000 (*estimate*, 9.6) |
| Kimi-Linear prefill, 512 tokens | ~190,000 (measured), so tier 1 not feasible | ~1,000–2,000 (*estimate*, 9.6) |

On Kimi the HF kernel count is why tier 1 had to be switched off, leaving only nsys L2 estimates and no exact memory bytes ([08-nsys-l2-sampling.md](08-nsys-l2-sampling.md)). llama.cpp merges each MoE layer's experts into one kernel per projection and runs Kimi's linear attention as one dedicated kernel, so tier 1 should become practical again.

**6. It's the engine the review asked about.** The review raised GGUF/llama.cpp (section 1), and it installs in the home directory without root or docker.

### 9.3 When to use HF transformers instead

The llama.cpp backend, as built, can't do the following:

| Need | HF transformers | llama.cpp backend |
|---|---|---|
| **FP8 checkpoints** (e.g. Kimi-Linear FP8) | ✅ Runs the real FP8 | ❌ GGUF has no FP8; Q8_0 is the closest stand-in (section 6.4) |
| **Batch size above 1** (e.g. the Kimi b8/b16/b32 sweeps) | ✅ | ❌ One sequence only. llama.cpp itself can do more; the backend doesn't yet. |
| **Forced MoE routing** (fixed / disjoint, [07-moe-routing.md](07-moe-routing.md)) | ✅ | ❌ Natural routing only |
| **Per-layer markers** (`--annotate-layers`) | ✅ | ❌ |
| **Measured KV cache and memory footprint** | ✅ From the live tensors | ⚠️ Predicted values; PyTorch can't see llama.cpp's memory |
| **Architectures llama.cpp doesn't support** | ✅ Almost anything, incl. `trust_remote_code` | ❌ Must be implemented in llama.cpp |

When HF is used for one of these, its **byte** figures remain useful (within ~10% of llama.cpp's on Qwen). Its **timings** should be reported as "HF transformers", not as the model's speed on GB10.

### 9.4 How to work from now on

1. **New model: run llama.cpp first.** Timing, busy %, launches, exact bytes, top kernels and the main conclusions come from this run.
2. **For the first few new models, also run HF once** and compare bytes. Qwen3-0.6B agreed within ~10%; it's worth confirming this holds for MoE and hybrid models before relying on llama.cpp alone.
3. **Use HF for the cases in 9.3**, labelled as such.
4. **Every reported number names the engine that produced it.** Bytes are mostly about the model; timings are about the engine.
5. **For each new model, check the backend against `llama-bench` once** (`scripts/run_llamabench_qwen.sh` is the template), as in 7.8.1. Decode within ~5–10% means the testbench's numbers for that model can be trusted.

### 9.5 What to fix before relying on it fully

- **Decode physics check prints the wrong expectation** (0.81× instead of 1.01×). The backend reports all resident weights (1.50 GB), including the input embedding table that llama.cpp keeps on the CPU side and that a step reads only one row of (7.8.4). Fix in `nsight_bench/backends/llamacpp.py`: report the weights a step reads.
- **Prefill reads ~9% slow** inside the testbench (7.8.1), probably from the backend's Python setup work inside the timed prefill. Not verified; worth tightening.
- **`greedy` token mode not yet run** (7.8.6). That run measures what reading each token back to the CPU costs, which real generation pays.
- **Validated on one model only** (Qwen3-0.6B, dense). The next model, ideally an MoE such as Qwen3-30B-A3B (on disk at `/opt/ai-models`), should repeat the 7.8.1 check.

### 9.6 What it would mean for Kimi-Linear

*Not run; based on reading llama.cpp's Kimi-Linear implementation and on the Qwen measurements.*

- **Support exists** in the build used here: `src/models/kimi-linear.cpp` (runtime) and `conversion/kimi_linear.py` (converter). The converter handles this checkpoint's format (compressed-tensors, FP8, one group, per-channel scales) and can write the FP8 weights as Q8_0 (`--fp8-as-q8`, ~51 GB).
- **Far fewer kernels.** MoE layers go through `build_moe_ffn`, which handles all 8 chosen experts in one kernel per projection, with no per-layer copy to the CPU. KDA linear attention is one dedicated kernel (`ggml_kda_scan`). Estimate: ~500–1,000 launches per decode step (HF: ~12,000) and ~1,000–2,000 per 512-token prefill (HF: ~190,000), so ncu tier 1 might take roughly 30 min–1 h per decode step and 1–2 h per prefill, if each kernel costs about as much to profile as before. Merged MoE kernels touch more memory each, so that cost may be higher.
- **Faster runs overall.** If decode becomes memory-bound as on Qwen, ~3.7 GB per step at ~210 GB/s gives a ceiling of ~18 ms per token (~55 tok/s), against ~141 ms (~7 tok/s) on HF.
- **What can be redone:** only the 5 natural-routing, batch-1 runs from the 22 (decode b1, prefill p16/p64/p512, long context). Forced routing and batch sweeps stay HF-only for now.
- **MLA runs as real MLA, not expanded.** *Confirmed from the code, not yet from a run.* HF transformers expands Kimi's MLA and runs it as standard multi-head attention, caching 143,360 B per token ([07-moe-routing.md](07-moe-routing.md) section 5.7). llama.cpp's converter splits `kv_b_proj` into `k_b_proj` and `v_b_proj` ("MLA with the absorption optimization"). Its runtime then folds the key expansion into the query, caches only the compressed form (512 + 64 values per token, one key/value shared by all 32 query heads) and applies `v_b_proj` after attention. Cache: **8,064 B per token, ~0.13 GB at 16k tokens instead of 2.35 GB**, written in place instead of rebuilt with `torch.cat` every token. This only holds for a GGUF made by this converter; an older GGUF without the split falls back to the expanded path. Check the load log's "KV buffer size" on the first run.
- **Precision differs:** Q8_0 instead of FP8. Almost the same bytes per step (~6% more), different arithmetic.
- **The long-context run keeps its memory risk.** llama.cpp processes a 16k-token prompt in 512-token chunks, so ncu would see ~32× more prefill kernels. Run it with tier 1 off and the memory guard on, as the re-run in [07-moe-routing.md](07-moe-routing.md) did.
- **Steps to confirm:** convert (CPU, tmux), a short test prompt for correct output, then an nsys-only smoke run to count kernels before committing hours of ncu.
