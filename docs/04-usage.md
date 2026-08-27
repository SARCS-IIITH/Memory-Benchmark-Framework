# Usage

All commands assume the harness environment:

```bash
source /home/sarcs/envs/samarthamp/bin/activate
cd /home/sarcs/samarthamp
# or call it directly, without activating:
/home/sarcs/envs/samarthamp/bin/nsbench --help
```

## First time on a machine

```bash
bash setup/create_env.sh      # build envs/samarthamp
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
| `configs/profiles/quick.yaml` | 1 | Iterating on setup; checking a checkpoint loads and profiles. |
| `configs/profiles/standard.yaml` | 1, 2 | The default. Full memory metrics plus section deep-dive on the heaviest kernels. |
| `configs/profiles/deep.yaml` | 1, 2, 3 | Adds source-level attribution. Slow, and only useful with SASS line info. |

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

Nsight Compute replays every kernel, so tier 1 costs roughly **one minute per 100 kernels in
scope**. An eager transformers decode step runs far more kernels than the layer count
suggests -- about 55-60 launches per transformer block -- so a 28-layer model is around 1600
launches, or ~16 minutes per phase.

```bash
nsbench run --model ... --max-kernels 8000     # raise the cap for a large model
nsbench run --model ... --skip-ncu             # timeline and timing only, minutes not hours
nsbench run --model ... --tiers 1 --top-n 4    # tier 1 only, fewer deep dives
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
ncu-ui   runs/<id>/raw/ncu_tier1_decode_step.ncu-rep
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
