# The GB10 platform, and why this harness is shaped the way it is

Everything here was measured on this machine, not read from a specification. Where a number
came from an experiment, the experiment is described so it can be repeated.

## What the hardware is

| Property | Value | How it was determined |
|---|---|---|
| GPU | NVIDIA GB10 | `nvidia-smi` |
| Internal chip name | **GB20B** | `ncu --query-metrics` header, `nsys profile --gpu-metrics-devices=help` |
| Compute capability | **12.1** (sm_121) | `torch.cuda.get_device_properties` |
| SMs | 48 | ditto |
| L2 cache | **25.17 MB** | ditto |
| Shared memory | 100 KB per SM, 48 KB per block by default | ditto |
| Registers | 65536 per SM | ditto |
| Memory | **128.5 GB unified LPDDR5X**, `is_integrated = 1` | ditto |
| Host | 20 ARM cores (10x Cortex-X925 + 10x Cortex-A725), 1 NUMA node, 128 GB | `lscpu`, `numactl --hardware` |
| Toolchain | nsys 2025.3.2, ncu 2025.3.1, CUDA 13.0, glibc 2.39 | `--version` |

The chip name matters more than it looks. The marketing name is "GB10" but Nsight keys its
GPU metric sets on **GB20B**, and passing the wrong set is a hard error. `nsbench preflight`
resolves the chip name and picks the right set (`gb20b`) automatically.

## The central problem: there are no DRAM counters

Every NVIDIA profiling guide reads memory traffic from `dram__bytes.sum`. **On GB20B that
metric does not exist.** Neither does any other `dram__*` counter.

Enumerating the metric namespace shows why. GB20B exposes 5292 metrics across these prefixes:

```
lts__ (2016)   smsp__ (507)   l1tex__ (461)   sm__ (365)   gcc__ (34)
tpc__ (31)     gr__ (18)      gpu__ (15)      idc__ (12)   gpc__ (8)
fe__ (6)       sys__ (4)
```

No `dram__`. No `fbpa__`. And no `ctc__` either -- the NVLink-C2C counters that a Grace-Hopper
part exposes are absent, and Nsight Compute's `C2CLink` section is gated to CC_90 and CC_100
in `C2CLink.section`, so it never applies to a CC_121 part.

That is not an oversight. There is no separate device memory on this part to count: the GPU
shares one coherent LPDDR5X pool with the Grace CPU.

## What replaces it, and the experiment that established it

L2 (LTS) tags every access with the **aperture** it targets -- `device` (private VRAM),
`sysmem` (host-coherent memory), or `peer` (another GPU). On an integrated part everything
past L2 is sysmem.

The experiment: allocate two 256 MiB fp32 arrays and run `torch.mul(a, 2.0, out=b)`. That
kernel reads 256 MiB and writes 256 MiB, so exactly 512 MiB = 536.9 MB must cross the memory
interface. Profiling it:

```
lts__t_sectors_aperture_sysmem_lookup_miss.sum  =  16,780,246 sectors
16,780,246 x 32 B                              =  537.0 MB     <- matches
lts__t_sectors_aperture_device_lookup_miss.sum =           0
lts__t_sectors_aperture_peer_lookup_miss.sum   =           0
```

An error of **+0.02%**. So:

> **DRAM bytes = `lts__t_sectors_aperture_sysmem_lookup_miss.sum` x 32 B**

The `device` and `peer` counters are carried through every collection as **sentinels**. They
must read zero. If a driver update ever starts routing traffic through the device aperture,
the derivation silently halves -- so the harness checks rather than assumes, and every report
states the result. `nsbench calibrate` re-runs this experiment before every benchmark.

## The memory system, measured

A working-set sweep of the same streaming kernel, run without a profiler attached:

| Working set (both arrays) | Bandwidth |
|---|---|
| 2 MiB | 500.3 GB/s |
| 8 MiB | 1,020.2 GB/s |
| 16 MiB | 1,034.7 GB/s |
| 32 MiB | 338.6 GB/s |
| 48 MiB | 270.5 GB/s |
| 64 MiB | 250.5 GB/s |
| 96 MiB | 234.8 GB/s |
| 128 MiB | 232.6 GB/s |
| 256 MiB | 242.1 GB/s |
| 512 MiB | 238.5 GB/s |
| 1,024 MiB | 238.9 GB/s |
| 2,048 MiB | 239.9 GB/s |

The knee falls at a 32 MiB working set against 24 MiB of L2 -- exactly where
capacity says it should. Three ceilings come out of this and are used throughout:

- **LPDDR5X: 242 GB/s.** The roofline's memory bound. Achievable, not a datasheet figure.
- **L2: 1,035 GB/s, 4.3x LPDDR5X.** This multiple is what an L2 hit is worth on this
  part, and the reason L2 hit rate is the headline metric for decode.
- **Compute: 97.9 TFLOP/s** dense bf16 GEMM, putting the roofline's ridge point at
  405 FLOP/byte.

The working-set column counts *both* arrays the streaming kernel touches. Reporting only
the source array would place the knee at half its true value and make L2 look as though it
stopped helping well before capacity. These figures vary a few percent run to run with
clock and thermal state, which is why the harness re-measures them rather than hard-coding
them.

A decode step measured on this machine sits at about 12 FLOP/byte -- some thirty times below
the ridge. That is the structural reason decode is slow here, and no amount of arithmetic
tuning changes it.

## Other platform consequences

**NVML reports no GPU memory.** `nvidia-smi --query-gpu=memory.used,memory.total` returns
`[N/A]` on this part, because there is no discrete pool to report. Footprint is instead
reconstructed from three sources with different blind spots -- the torch allocator,
`cudaMemGetInfo`, and `/proc/meminfo` -- and the harness cross-checks them. See
[03-methodology.md](03-methodology.md).

**nsys GPU metrics carry no bandwidth.** The `gb20b` metric set
(`/opt/nvidia/nsight-systems/*/target-linux-sbsa-armv8/GpuMetrics/gb20b.config`) contains
clocks, engine activity, warp occupancy and tensor activity -- and nothing else. All traffic
figures come from Nsight Compute. nsys contributes the timeline, the allocation events and
the clock record.

One trap: nsys names the metric `GPC Clock Frequency [MHz]` but stores raw cycles per second
in `GPU_METRICS`, applying the 1e-6 multiplier only at display time. Read at face value, a
1.43 GHz clock becomes 1.4 billion MHz. The parser rescales by magnitude.

**Allocation events carry no timestamps.** Every row in
`CUDA_GPU_MEMORY_USAGE_EVENTS` has `start = 0` on this nsys build, while kernel timestamps in
`CUPTI_ACTIVITY_KIND_KERNEL` are populated normally. Sizes and ordering are correct, so the
cumulative live-allocation series is still meaningful -- but it is plotted against event
sequence rather than a clock, and the report says so. Plotting it against the collapsed time
axis would put every point at t=0, which reads as a bug rather than as missing data.

**nsys CPU sampling is unavailable.** `perf_event_paranoid = 4` and sudo needs a password, so
`perf_event_open` is refused. GPU tracing is unaffected; the harness detects this and passes
`--sample=none --cpuctxsw=none` rather than letting nsys retry and warn per thread.

**GPU counters are readable without root.** `/etc/modprobe.d/nvidia-profiling.conf` sets
`NVreg_RestrictProfilingToAdminUsers=0`. Verified by actually collecting a metric, not by
reading the config.

**Clock locking does not work through nvidia-smi.** `--query-supported-clocks` returns `N/A`,
so `ncu --clock-control base` may be a no-op. Rather than trusting the lock, the harness
records the *measured* GPC clock from the nsys pass and flags a run whose clock varied by
more than 15%.

**NVTX filter syntax bites.** `ncu --nvtx-include "name"` matches only **start/end** ranges.
`torch.cuda.nvtx.range_push` creates **push/pop** ranges, which need a trailing slash:
`"name/"`. Without it ncu exits 0, writes no report, and prints only "No kernels were
profiled" -- which reads like the workload never ran. `instrumentation/nvtx.py::ncu_filter`
appends the slash.

**Docker needs sudo**, and this user is not in the `docker` group, so the NGC container route
-- NVIDIA's sanctioned path for TensorRT-LLM on DGX Spark -- is unavailable here.

## Metrics that do not exist, and what to use instead

`nsbench preflight` records these so their absence is explained rather than rediscovered:

| Requested | Why it is missing | Use instead |
|---|---|---|
| `dram__bytes.sum` and all `dram__*` | No DRAM counters on GB20B | `lts__t_sectors_aperture_sysmem_lookup_miss.sum` x 32 B |
| `ctc__rx/tx_bytes_data_user.sum` | No NVLink-C2C counters exposed | nothing equivalent; C2C traffic is not observable here |
| `derived__local_spilling_requests` | Section-internal; fails as a standalone `--metrics` entry | `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st.sum`, or the tier-2 section |
| `sass__inst_executed_register_spilling_*` | Needs `SourceCounters` and SASS line info | tier 3, on binaries built with `-lineinfo` |

One more trap worth stating: `ncu --query-metrics` is not a reliable oracle in either
direction. `launch__*` metrics collect perfectly but never appear in the listing, while
`derived__*` metrics appear in it and then fail. `nsbench preflight` therefore determines
availability by running a real one-kernel collection and eliminating whatever ncu rejects.
