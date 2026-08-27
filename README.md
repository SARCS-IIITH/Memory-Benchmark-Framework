# nsight_bench

Memory-hierarchy benchmarking for AI models on **NVIDIA DGX Spark (GB10)**, built on Nsight
Systems and Nsight Compute.

Point it at a `.safetensors` checkpoint and it measures how many bytes move at every rung of
the hierarchy -- registers, shared memory, L1, L2, unified LPDDR5X -- separately for prefill
and decode, then writes a report that says what the numbers mean and whether they can be
trusted.

```bash
bash setup/create_env.sh
source /home/sarcs/envs/samarthamp/bin/activate

nsbench preflight
nsbench discover /path/to/checkpoint --name my-model
nsbench run --model configs/models/my-model.yaml
```

## The problem this solves

Every NVIDIA profiling guide reads memory traffic from `dram__bytes.sum`. **On GB10 that
counter does not exist** -- there is no `dram__*` metric at all, and no `ctc__*` NVLink-C2C
counter either. The GPU has no private VRAM: it shares one coherent LPDDR5X pool with the
Grace CPU.

What does exist is the L2 aperture breakdown, and on this part everything past L2 is tagged
`sysmem`. So:

> **DRAM bytes = `lts__t_sectors_aperture_sysmem_lookup_miss.sum` x 32 B**

That substitution is the foundation of every memory number here, so the harness does not take
it on faith. Before every benchmark it profiles a streaming kernel of known size and checks
the derivation reproduces it. On this machine, 536.9 MB of known traffic measured as
537.0 MB -- an error of **+0.02%** -- with the device and peer aperture sentinels both at
zero. The gate result is stamped into every run and every report.

## What it measures

- **The full hierarchy per phase.** Bytes and hit rates at each level, prefill against
  decode, plus the amplification between rungs -- how much of what the SM asked for survived
  to reach memory.
- **Bytes per generated token**, normalised so models of different sizes and quantization
  schemes compare directly.
- **Measured ceilings.** LPDDR5X bandwidth, L2 bandwidth and dense bf16 GEMM throughput, all
  measured on the machine rather than taken from a datasheet, so utilisation percentages and
  roofline positions mean something. On this box: **~242 GB/s** from LPDDR5X, **~1035 GB/s**
  from L2 (a 4.3x prize for keeping a working set on chip), and **~98 TFLOP/s** dense bf16 --
  which puts the roofline's ridge at ~405 FLOP/byte, while a decode step runs at about 12.
- **Honest timing**, from a separate unprofiled run. Nsight Compute replays every kernel, so
  profiled durations are not performance and are never reported as such.
- **Footprint from three independent sources**, because NVML reports nothing on this part.
- **A physics check.** A decode step must read every weight and the whole KV cache to produce
  one token, so its traffic is predictable. Measured against predicted is the strongest
  available test that the scoping and the derivation are both right.

## Layout

```
setup/          environment provisioning (create_env.sh, setup_trtllm_env.sh)
configs/        platform profile, model configs, workload and profiler presets
nsight_bench/
  metrics.py            the metric registry -- single source of truth
  platform.py           hardware/tool/permission/metric-availability probing
  calibration.py        the byte-accounting gate and the ceiling measurements
  config.py             configs and checkpoint discovery
  orchestrator.py       runs the four collections for one benchmark
  worker.py             the process the profilers wrap
  backends/             hf_transformers (reference), trtllm (scaffolded)
  workloads/            text-generation, multimodal (interface only)
  instrumentation/      NVTX, memory tracking, capture-range control
  runners/              baseline, nsys, ncu (tiered), calibration
  parsers/              ncu CSV and nsys SQLite -> tidy records
  analysis/             hierarchy roll-up, cross-checks, cross-run comparison
  report/               markdown, self-contained HTML with inline SVG charts
scripts/        smoke_test.sh, run_bench.sh, run_sweep.sh
docs/           platform findings, metric reference, methodology, usage
runs/           output, one directory per run
```

## Documentation

| Document | What it covers |
|---|---|
| [docs/01-platform-gb10.md](docs/01-platform-gb10.md) | What GB10 is, the missing counters, the calibration experiment, and every platform trap found |
| [docs/02-metric-reference.md](docs/02-metric-reference.md) | Every metric collected, what it means, and how the reported quantities are derived |
| [docs/03-methodology.md](docs/03-methodology.md) | How to read the numbers, and what the harness deliberately does not claim |
| [docs/04-usage.md](docs/04-usage.md) | Commands, cost, output layout, troubleshooting |

## Cost

Nsight Compute replays every kernel, so tier 1 runs at roughly **one minute per 100 kernels in
scope**. An eager transformers decode step is around 55-60 kernel launches per transformer
block, so a 28-layer model is ~1600 launches, or ~16 minutes per phase. `--skip-ncu` gives
the timeline and honest timing in minutes instead.

## Status

- **HuggingFace transformers backend**: complete, and the reference implementation. The decode
  loop is written by hand rather than delegated to `generate()`, because that is the only way
  to put an NVTX range around exactly one generated token -- which is what makes selective
  Nsight Compute profiling tractable.
- **TensorRT-LLM backend**: scaffolded, not implemented. aarch64 wheels are release-candidate
  only, sm_121 support is unverified, and the NGC container route needs docker access this
  user does not have. Reasoning and implementation notes are in
  `nsight_bench/backends/trtllm.py`.
- **Multimodal workload**: interface, phase structure, NVTX scoping and memory tracking are in
  place and reusable; the per-checkpoint processor input construction is not written.
