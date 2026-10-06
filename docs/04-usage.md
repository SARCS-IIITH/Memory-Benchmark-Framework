# Usage

All commands assume the harness environment:

```bash
source ~/envs/nsbench/bin/activate
cd ~/Memory-Benchmark-Framework
# or call it directly, without activating:
~/envs/nsbench/bin/nsbench --help
```

## First time on a machine

```bash
bash setup/create_env.sh      # build ~/envs/nsbench (ENV_DIR= to change)
nsbench preflight             # probe GPU, tools, permissions, metric availability
bash scripts/smoke_test.sh    # end-to-end check on a small model
```

`preflight` writes `configs/platform_profile.json` and is a prerequisite for everything else.
It runs real ncu collections to determine which metrics actually work on the attached GPU --
`ncu --query-metrics` is not a reliable oracle in either direction. **Re-run it whenever the
metric registry changes**; a stale profile silently drops the new metrics from every
collection, and the harness warns when it detects one.

## Benchmarking a checkpoint

```bash
# 1. Turn a checkpoint directory into a config
nsbench discover /path/to/Qwen3.5-4B --name Qwen3.5-4B

# 2. Run it
nsbench run --model configs/models/qwen3.5-4b.yaml \
            --workload configs/workloads/decode-focused.yaml
```

`discover` reads `config.json`, `model.safetensors.index.json` and `hf_quant_config.json` to
work out the architecture, shape and quantization scheme. It handles single-file and sharded
layouts, and reads the transformer shape from a nested `text_config` when the checkpoint is
multimodal. Nothing is hard-coded to a model.

To skip the config file for a one-off:

```bash
nsbench run --model-path /path/to/checkpoint --name my-model
```

`run` executes four collections in order -- calibration, unprofiled baseline, Nsight Systems,
Nsight Compute -- then writes the reports. Each runs in its own subprocess, so a model that
OOMs takes down one stage rather than the run.

## Comparing runs

```bash
nsbench compare runs/*__sweep --out runs/_comparison
```

Or benchmark several checkpoints and compare them in one go:

```bash
./scripts/run_sweep.sh /path/to/model-a /path/to/model-b /path/to/model-c
```

To see how one model behaves as the prompt grows and the KV cache with it:

```bash
./scripts/run_bench.sh /path/to/checkpoint
```

## Choosing a profile

| Profile | ncu tiers | What it is for |
|---|---|---|
| `configs/profiles/quick.yaml` | 2 | Iterating on setup; checking a checkpoint loads and profiles. Deep dive on the top 5 kernels. |
| `configs/profiles/standard.yaml` | 2 | The default. Section deep-dive on the heaviest 8 kernels, ranked from the nsys timeline. |
| `configs/profiles/deep.yaml` | 2, 3 | Adds source-level attribution. Slow, and only useful with SASS line info. |

Tier 1 (the per-level byte totals, hit rates and decode physics check) is **off in every
profile**, because on an eager MoE it takes hours per phase. Turn it back on per run with
`--tiers 1,2`, or in a profile with `tiers: [1, 2]`. Tier 2's kernel ranking comes from the
nsys timeline; `--rank-source tier1` (or `rank_source: tier1`) ranks from tier 1's export
instead, which needs tier 1 on.

## Choosing a workload

| Workload | Shape | What it exercises |
|---|---|---|
| `decode-focused` | 128 -> 128 | Memory-bound decode. The default for memory work. |
| `prefill-focused` | 4096 -> 8 | Compute-bound prefill. |
| `balanced` | 512 -> 64 | A realistic chat-shaped request. |
| `long-context` | 16384 -> 32 | KV cache pressure rather than weight pressure. |
| `layer-attribution` | 512 -> 16 | Adds per-transformer-block NVTX ranges. |

Override any field from the command line:

```bash
nsbench run --model configs/models/x.yaml \
            --prompt-tokens 2048 --generate-tokens 64 --batch-size 4 \
            --repeat 5 --attn eager --tag eager-vs-sdpa
```

## Cost, and the knob that controls it

Nsight Compute replays every kernel, so tier 1, when it is turned on, costs roughly **one
minute per 100 kernels in scope**. An eager transformers decode step runs far more kernels than the layer count
suggests -- about 55-60 launches per transformer block -- so a 28-layer model is around 1600
launches, or ~16 minutes per phase.

```bash
nsbench run --model ... --max-kernels 8000     # raise the cap for a large model
nsbench run --model ... --skip-ncu             # timeline and timing only, minutes not hours
nsbench run --model ... --tiers 1,2            # add tier 1 back: byte totals + physics check
nsbench run --model ... --tiers 1 --top-n 4    # tier 1 only, fewer deep dives
nsbench run --model ... --rank-source tier1    # rank tier 2 from tier 1 (needs --tiers 1,2)
```

`--max-kernels` is a runaway guard, not a sampling knob: below the real kernel count it
**truncates** rather than samples, and the harness flags the phase as partial when that
happens. Raise it rather than accepting truncated totals.

## Reading the output

```
runs/<timestamp>__<model>__<workload>__<backend>[__<tag>]/
  manifest.json     full provenance: host, driver, tool versions, checkpoint digests,
                    verbatim command lines, calibration result, clock/thermal state
  run_config.json   the exact config the workers were given
  report.md         the readable report
  report.html       the same, with charts, self-contained
  metrics/          hierarchy.csv, kernels_*.csv, alloc_events.csv, summary.json
  raw/              *.nsys-rep, *.sqlite, *.ncu-rep, *.csv
  logs/             every subprocess's stdout and stderr
```

Start with `report.md` or `report.html`. Its first section is whether the numbers can be
trusted at all -- the calibration gate and any tripped sentinels -- because if that failed,
everything below it is suspect.

For the interactive views:

```bash
nsys-ui  runs/<id>/raw/timeline.nsys-rep
ncu-ui   runs/<id>/raw/ncu_tier2_decode_step.ncu-rep   # ncu_tier1_* too, if tier 1 was on
```

To re-render reports after changing the report code, without re-profiling:

```bash
nsbench report runs/<id>
```

The allocator history pickle in `metrics/baseline_result.alloc.pickle` opens at
<https://pytorch.org/memory_viz> for a per-allocation timeline with Python stacks.

## Calibrating on its own

```bash
nsbench calibrate --megabytes 256
```

Verifies the DRAM derivation against a known byte count and measures the machine's ceilings.
Worth running on its own after a driver update, or whenever a run's numbers look wrong.

## The TensorRT-LLM backend

Not implemented. `nsbench run --backend trtllm` resolves and fails with an explanation rather
than a stack trace. The reasoning, the constraints found on this machine, and implementation
notes are in `nsight_bench/backends/trtllm.py`; the environment it would need is provisioned
by `setup/setup_trtllm_env.sh`, which is deliberately not run by default.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ERR_NVGPUCTRPERM` | GPU counters restricted to admins. Set `NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/`, `update-initramfs -u`, reboot. |
| "No kernels were profiled" | The NVTX filter matched nothing. Check the worker ran in `--mode ncu`. Note push/pop ranges need a trailing `/` in the filter -- the harness adds it. |
| Blank columns in the report | Stale `platform_profile.json`. Re-run `nsbench preflight`. |
| "PARTIAL DATA" warning | ncu hit its launch cap. Raise `--max-kernels`. |
| ncu timeout | Lower `--top-n`, or raise `ncu.timeout_s` in the profile config. |
| Calibration gate fails | Something changed about how memory is routed. Do not trust the DRAM figures; investigate before benchmarking. |
| nsys warns about CPU sampling | Expected here -- `perf_event_paranoid=4`. GPU tracing is unaffected. |

## Bringing your own model

There is no inference server to set up. `nsbench` does not talk to vLLM, Ollama, llama.cpp or
LM Studio. The `hf` backend (`nsight_bench/backends/hf_transformers.py`) **is** the inference
engine: it loads the weights into its own process with `AutoModelForCausalLM.from_pretrained()`,
moves them to the GPU, and calls the forward pass itself, once for the prompt and once per
generated token. All a model has to be is a folder that HuggingFace `transformers` can load.

That folder is a *checkpoint directory*. A typical one looks like this:

```
Qwen3-4B/
  config.json                        architecture and shape -- required; discover reads it
  generation_config.json             default sampling settings (not used; decoding is greedy)
  model-00001-of-00003.safetensors   the weights, split into shards ...
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  model.safetensors.index.json       ... and the map of which tensor is in which shard
  tokenizer.json                     tokenizer files
  tokenizer_config.json
  modeling_*.py, configuration_*.py  only present for custom architectures
```

A small model may have a single `model.safetensors` and no index. Either layout works.

| Requirement | How to check | If it fails |
|---|---|---|
| A **directory**, not a file | `ls <path>` shows `config.json` | Point at the folder that holds the weights |
| `config.json` present | `ls <path>/config.json` | It is not an HF checkpoint; see [A fine-tuned or custom model](#a-fine-tuned-or-custom-model) |
| Weights in `.safetensors` | `ls <path>/*.safetensors` | `.bin`/`.pt`: re-save as safetensors. GGUF: not supported, so download the original HF repo instead |
| A causal (text-generation) LM | `"architectures"` in `config.json` ends in `ForCausalLM` or `ForConditionalGeneration` | Encoders, embedding models and diffusion models cannot be benchmarked |
| Tokenizer files present | `tokenizer.json` or `tokenizer.model` beside the weights | Copy them from the base model, or set `tokenizer_path:` in the model YAML |
| Fits in memory | bf16 ≈ 2 GB per billion parameters, plus the KV cache `discover` prints | The 128 GB pool is shared with the OS; keep weights + KV under ~100 GB |
| Downloaded **completely** | Shard count matches `model-XXXXX-of-N` | Runs are offline (`HF_HUB_OFFLINE=1`), so nothing is fetched at run time; re-run the download |

### From the HuggingFace Hub

Before downloading, open the model's page on huggingface.co and check the *Files* tab:

- `config.json` and `*.safetensors` files are there. Repos that only hold `.gguf` files are
  conversions for llama.cpp; find the original repo they link to.
- The model card says *Text Generation*, and the total size of the safetensors files fits in memory.
- If the page asks you to accept a licence (Llama, Gemma and others are *gated*), accept it there
  first, then log in once from the terminal with a token from huggingface.co → Settings → Access Tokens:

```bash
hf auth login               # older huggingface_hub: huggingface-cli login
```

**Route A -- the shared store (preferred).** Add a block to
[`configs/model-registry.yaml`](../configs/model-registry.yaml) and let the harness fetch it into
`/opt/ai-models`, where everyone else can reuse it (see [05-suite.md](05-suite.md)):

```yaml
  my-model:
    repo: Org/Model-Name
    label: Model Name
    family: Org
    class: dense          # or moe
    params: 4.0e9
```

```bash
nsbench fetch  --models my-model
nsbench models --models my-model          # wait until it says "ready", not "fetching"
```

The store uses the HF cache layout, so the checkpoint directory is the **snapshot** folder, not
the `models--...` folder above it:

```bash
ls /opt/ai-models/hub/models--Org--Model-Name/snapshots/
# -> c1899de289a04d12100db370d81485cdf75e47ca
nsbench discover /opt/ai-models/hub/models--Org--Model-Name/snapshots/c1899de2.../ --name my-model
```

**Route B -- a one-off download to a plain folder:**

```bash
hf download Org/Model-Name --local-dir ~/models/Model-Name \
    --include "*.safetensors" "*.json" "tokenizer*" "*.model" "*.py"
nsbench discover ~/models/Model-Name --name my-model
```

The `--include` filter skips GGUF, ONNX and `.bin` duplicates that many repos carry and the
harness never reads.

### A fine-tuned or custom model

Whatever produced the model, the goal is the same folder shape as above. Which case you are in
depends on what the training run saved:

**Full fine-tune** (every weight was trained; the output folder has `config.json` and weights).
If it was saved with `save_pretrained`, it is usable as-is. If the weights came out as
`pytorch_model*.bin`, re-save them:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("path/to/finetune", dtype="bfloat16")
model.save_pretrained("path/to/finetune-st", safe_serialization=True)
AutoTokenizer.from_pretrained("path/to/finetune").save_pretrained("path/to/finetune-st")
```

**LoRA / PEFT adapter** (the folder has `adapter_config.json` and `adapter_model.safetensors`,
and no `config.json`). That is only a small diff on top of a base model, so `discover` rejects it.
Merge the adapter into the base model and benchmark the merged result:

```python
from peft import PeftModel                     # pip install peft, into the harness env
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("path/to/base-model", dtype="bfloat16")
merged = PeftModel.from_pretrained(base, "path/to/adapter").merge_and_unload()
merged.save_pretrained("path/to/merged", safe_serialization=True)
AutoTokenizer.from_pretrained("path/to/base-model").save_pretrained("path/to/merged")
```

A merged model has the same shape and size as its base, so its memory traffic should match the
base model's. Benchmark both if you want to confirm that.

**Custom architecture code** (your own `modeling_*.py`). Put the `.py` files inside the
checkpoint folder and register them under `auto_map` in `config.json`, the same way models on the
Hub with custom code do. `discover` sees the `.py` files and turns on `trust_remote_code` by itself.
The model's `forward` must then honour the contract the backend drives:

- Prefill: `model(input_ids=..., attention_mask=..., past_key_values=DynamicCache(), use_cache=True)`.
- Each decode step: `model(input_ids=<one token per sequence>, past_key_values=<cache from the last call>, use_cache=True)`.
- Return an object with `.logits` of shape `[batch, seq, vocab]` and `.past_key_values`.
- Optional: accept `logits_to_keep` (or the older `num_logits_to_keep`). Without it, prefill runs
  the LM head over every prompt position, and the manifest records
  `prefill_computed_full_vocab_logits: true`.

If the model only works through `model.generate()`, or keeps its cache somewhere other than
`past_key_values`, it does not fit the contract. In that case either adapt `forward` or write a
new backend against `nsight_bench/backends/base.py` (see
[06-architecture-onboarding.md](06-architecture-onboarding.md)).

**Quantized checkpoints** (AWQ, GPTQ, bitsandbytes, FP8, NVFP4): see the next section.

### Quantized checkpoints

The harness does not quantize anything itself. It loads whatever the checkpoint already contains
through `transformers`, so whether a quantized model works comes down to whether `transformers`
can load that checkpoint in this env:

- **Pre-quantized checkpoints only.** There is no on-the-fly quantization: the backend never
  passes `load_in_4bit` or a `BitsAndBytesConfig`. For a 4-bit version of a model, download a
  4-bit checkpoint rather than the bf16 one.
- **The scheme's loader library is not installed.** `setup/requirements.txt` carries no AutoAWQ,
  GPTQModel, bitsandbytes, compressed-tensors or ModelOpt. Install the one the model card names.
- **Untested on this platform.** The machine is aarch64 with an sm_121 GPU, and many quantization
  libraries ship kernels built only for x86 or older GPUs. Every model in `/opt/ai-models` is
  bf16, so no quantized path has been exercised yet. Run the compatibility check below before
  profiling.

When `discover` recognises the scheme, the backend skips forcing a dtype at load
(`hf_transformers.py`), so `transformers` honours the checkpoint's own `quantization_config`.

`discover` reads the scheme from metadata, never from the repository name
(see [02-metric-reference.md](02-metric-reference.md)):

| Scheme in metadata | Reported as | Bits/weight assumed |
|---|---|---|
| none (bf16 / fp16) | `none` | 16 |
| fp32 | `none` | 32 |
| AWQ, GPTQ | `awq`, `gptq` | 4 |
| NVFP4, FP4 (anything containing `fp4`, e.g. MXFP4) | `nvfp4`, `fp4` | 4 |
| FP8 (anything containing `fp8`) | `fp8` | 8 |
| bitsandbytes | `bitsandbytes` | 8, even for 4-bit NF4 |
| ModelOpt, compressed-tensors | `modelopt`, `compressed-tensors` | 8 |
| ModelOpt `MIXED_PRECISION` | `mixed` | indeterminate |

The bits column matters less than it looks. Every byte figure that counts (expected decode
traffic, bytes per token) uses the exact on-disk size, which already includes scales and
zero-points. The assumed bits affect only the estimated parameter count and, for MoE models, the
size of the always-active part of the expected traffic.

Two traps:

- **An unrecognised scheme is treated as unquantized.** Schemes not in the table (torchao, HQQ,
  EETQ, Quark, AQLM, VPTQ, ...) come out of `discover` as `quantization none`. The backend then
  forces `dtype: bfloat16` at load, which can fail or silently dequantize the weights. If
  `discover` says `none` for a model you know is quantized, do not trust the run.
- **A declared KV-cache quantization is detected but not applied by the harness.** The KV cache
  is runtime state, not part of the weights. A checkpoint can only *declare* a cache dtype
  (`kv_cache_quant_algo`, `kv_cache_scheme`), and serving engines such as vLLM, SGLang and
  TensorRT-LLM honour that declaration. The `hf` backend instead hands the model a plain
  `DynamicCache`, which stores whatever the attention code passes to `cache.update()`.
  - Usually that is bf16, whatever the checkpoint declares.
  - It is FP8 only if the model's own modelling code quantizes K and V before caching them.

  Either way, the physics check uses the *measured* cache size (the real tensors' element sizes),
  so the numbers stay correct. Check which case you got by comparing the measured
  `kv_cache_bytes` in the run against the KV size `discover` predicted: about 2x the prediction
  means the cache ran in bf16. In that case do not describe the run as "FP8 KV cache".
- **The cache layout follows the `transformers` implementation, not production serving.** For
  multi-head latent attention models (DeepSeek-V3, Kimi K2), serving engines cache the
  compressed latent vector. The HF modelling code decompresses it and caches full per-head K and
  V, which is many times larger per token: **18× on Kimi-Linear-48B** (measured 143.4 KB per
  token per sequence across its 7 MLA layers, against an 8.06 KB latent; docs/07 section 5.3).
  The exact factor depends on head count and dimensions. Decode KV traffic measured here
  describes the HF reference execution.

What a correct quantized run looks like:

- Fewer weight bytes per decode step, and lower bytes per generated token, than the bf16 version.
- A run warning: `Model is quantized (...)`.

If measured decode traffic is well above expected, one cause the report lists is an unfused
dequantization pass, where weights are expanded in memory before the matrix multiply. That is a
real cost of the scheme, not a measurement error.

### Check compatibility before profiling

This snippet makes the same calls the `hf` backend makes, in the same order. If it prints
sensible text, `nsbench run` will be able to load and drive the model. Run it with the harness
env's Python:

```python
# check_model.py  --  python check_model.py /path/to/checkpoint
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, trust_remote_code=True,   # drop dtype= for quantized models
    attn_implementation="sdpa",
).to("cuda").eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
with torch.inference_mode():
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                past_key_values=DynamicCache(), use_cache=True)          # prefill
    cache, nxt, generated = out.past_key_values, out.logits[:, -1:].argmax(-1), []
    for _ in range(16):                                                  # decode steps
        generated.append(nxt)
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)

print(repr(tok.decode(torch.cat(generated, dim=1)[0])))
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

How to read the result:

- An exception on load usually means a missing file, a missing quantization library, or custom
  code that needs `auto_map`.
- An exception in the loop means the forward-pass contract above is not met.
- Gibberish output means wrong weights or the wrong tokenizer. The memory numbers would still be
  measured, but they would describe a broken model.

### A first run, end to end

```bash
# 0. Once per session
source ~/envs/nsbench/bin/activate
cd ~/Memory-Benchmark-Framework

# 1. Once per machine (and again after a driver or metric-registry change)
nsbench preflight

# 2. Get the checkpoint (see above), then turn it into a config
nsbench discover /path/to/checkpoint --name my-model
```

Read what `discover` prints before going further:

- `layers` and `hidden / heads` are non-zero.
- `quantization` is what you expect.
- `weights on disk` matches the size on the model page.
- The `KV cache @...` lines show how much memory long prompts will add.
- Every `note:` line is worth reading: it flags MoE, tied embeddings, custom code and missing weights.

The YAML it writes to `configs/models/` is safe to edit, for example `attn_implementation`,
`dtype` or `tokenizer_path`.

```bash
# 3. Cheapest real run: no ncu, no calibration sweep. Minutes, not hours.
nsbench run --model configs/models/my-model.yaml \
            --workload configs/workloads/decode-focused.yaml \
            --profile configs/profiles/quick.yaml --skip-ncu --no-sweep

# 4. Open runs/<id>/report.html. All stages "ok", and the calibration gate passed?
#    Then add ncu (quick profile: tier-2 deep dive on the top 5 kernels):
nsbench run --model configs/models/my-model.yaml \
            --workload configs/workloads/decode-focused.yaml \
            --profile configs/profiles/quick.yaml

# 5. The real measurement. --tiers 1,2 adds the byte totals and physics check; on a
#    dense model that is about a minute per 100 kernels, on an eager MoE many hours.
nsbench run --model configs/models/my-model.yaml \
            --workload configs/workloads/decode-focused.yaml \
            --profile configs/profiles/standard.yaml --tiers 1,2 --tag full

# 6. Optional: the same model across prompt lengths, or against other models
./scripts/run_bench.sh /path/to/checkpoint my-model
nsbench compare runs/*__full --out runs/_comparison
```

Steps 3 and 4 exist so that a mistake costs minutes. A load failure, out-of-memory error or bad
tokenizer shows up in step 3. A launch-count or NVTX problem shows up in step 4. Only then is the
hour-long step 5 worth paying for.

When reading the report:

- Start with the trust section (calibration gate and sentinels). If that failed, nothing below it
  holds. [01-platform-gb10.md](01-platform-gb10.md) explains why the gate exists.
- Then read the decode physics check, which compares measured traffic against predicted traffic.
  It only exists when tier 1 ran; otherwise the report says so and the byte figures read "not
  measured".
- Then read the hierarchy. [02-metric-reference.md](02-metric-reference.md) says what each number
  is, and [03-methodology.md](03-methodology.md) says what it does and does not mean.

### Common first-timer mistakes

| Symptom | Cause and fix |
|---|---|
| `Not a model directory` / `No config.json` | Pointed at a file, or at `models--Org--Name/` instead of `models--Org--Name/snapshots/<hash>/`. |
| `discover` notes `NO WEIGHTS FOUND` | Weights are `.bin`/`.pt`/GGUF, or the download stopped early. Re-save as safetensors, or re-run the download. |
| `discover` fails on a fine-tune folder with `adapter_config.json` | It is a LoRA adapter, not a model. Merge it (see above). |
| 401 / `GatedRepoError` while downloading | Licence not accepted on the model page, or not logged in (`hf auth login`). |
| `OfflineModeIsEnabled` / "couldn't connect" at run time | Runs are offline on purpose. Something is missing locally; finish the download. |
| CUDA out of memory in the baseline stage | Model + KV cache + activations exceed the pool. Lower `--prompt-tokens` / `--batch-size`, or use a smaller or quantized checkpoint. |
| `ImportError` naming `autoawq`, `gptqmodel`, `bitsandbytes`, ... | Quantized checkpoint whose loader is not installed. Install it into the harness env. |
| Manifest shows `load_warning` about `attn_implementation` | The architecture does not support the requested kernel, so it fell back to the default. Results are valid, but label them accordingly. |
| Manifest shows `prefill_computed_full_vocab_logits: true` | The model's `forward` has no `logits_to_keep`, so prefill traffic includes a full-vocabulary LM head. Expected for some custom models; note it when comparing. |
| Run takes hours | ncu replays every kernel. Use `--skip-ncu` or the `quick` profile while iterating; see [Cost](#cost-and-the-knob-that-controls-it). |
