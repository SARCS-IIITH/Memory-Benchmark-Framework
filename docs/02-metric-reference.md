# Metric reference

Every counter the harness collects, what it means, and how the reported quantities are
derived from it. The registry itself lives in `nsight_bench/metrics.py` and is the single
source of truth -- the ncu command line, the parsers and the report tables are all generated
from it, so this document and the code cannot drift apart in what is collected.

All 45 tier-1 metrics below were confirmed to collect on GB20B by running a real one-kernel
collection.

## Two rules that govern every number here

**Rates are re-derived, never averaged.** `lts__t_sector_hit_rate.pct` is a per-kernel
percentage. Averaging it across kernels weights a 200-sector elementwise kernel the same as a
GEMM moving hundreds of megabytes. Aggregate hit rates are computed from the raw counts:

```
hit rate = sum(lookup_hit) / sum(lookup_hit + lookup_miss)
```

which is why `*_lookup_hit`/`*_lookup_miss` are collected alongside the `.pct` metrics.

**A missing counter is not a zero.** Absent metrics propagate as "not measured" all the way
into the report. Reading them as 0.0 would have the report state that a decode step moved no
data through memory, which is worse than saying nothing.

## Unit normalisation

`ncu --csv --page raw` emits a **wide** table with a units row, and it autoscales magnitudes.
`gpu__time_duration.sum` arrives in **milliseconds** despite the metric description implying
nanoseconds; byte metrics arrive as Kbyte/Mbyte/Gbyte depending on size. The parser reads the
units row and normalises everything to ns / byte / sector. Ignoring that row produces values
wrong by factors of a million, in a direction that still looks plausible.

Values also carry locale thousands separators (`"8,400,907"` is one number) and unavailable
metrics arrive as empty strings.

## Tier 1 -- the triage set

Collected over every kernel inside the scoped NVTX range. Roughly 15 replay passes per
kernel; budget about a minute per 100 kernels.

### Registers

| Metric | Unit | Meaning |
|---|---|---|
| `launch__registers_per_thread` | reg/thread | Registers allocated per thread. Past 255 the compiler must spill. |
| `launch__occupancy_limit_registers` | block | Blocks per SM permitted by register pressure alone. |
| `launch__occupancy_limit_shared_mem` | block | Blocks per SM permitted by shared-memory usage alone. |

Registers have no traffic counter -- they are a capacity resource, not a cache. The report
shows pressure here and the resulting *spill* traffic under Local.

### Local memory (register spills)

| Metric | Unit | Meaning |
|---|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` | sector | Spill reloads. |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum` | sector | Spill stores. |

Local memory is backed by the same physical LPDDR5X as global memory. A spill is not a
register-file problem, it is a bandwidth problem: real memory traffic that produces no work.

`bytes = (ld + st) x 32`

### Shared memory

| Metric | Unit | Meaning |
|---|---|---|
| `launch__shared_mem_per_block_static` | byte/block | Statically declared shared memory. |
| `launch__shared_mem_per_block_dynamic` | byte/block | `extern __shared__` allocation. |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | wavefront | Shared-memory wavefronts. |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | wavefront | Load wavefronts. |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | wavefront | Store wavefronts. |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | conflict | Bank conflicts. Each serialises an otherwise parallel access. |
| `smsp__inst_executed_op_shared_ld.sum` | inst | Shared load instructions. |
| `smsp__inst_executed_op_shared_st.sum` | inst | Shared store instructions. |

`bytes ~= wavefronts x 128` (32 banks x 4 B). There is **no shared-memory sector counter**,
so this is the only route to bytes at this level, and it is an estimate. Bank conflicts
inflate the wavefront count, so a conflict-heavy kernel reports more shared bytes than its
data needs -- which is exactly the cost worth seeing. The report says so wherever the number
appears.

### L1 / TEX

| Metric | Unit | Meaning |
|---|---|---|
| `l1tex__t_requests.sum` | request | Requests arriving at L1TEX from the SM. |
| `l1tex__t_sectors.sum` | sector | Sectors moved through L1TEX. |
| `l1tex__t_bytes.sum` | byte | Bytes, reported directly by hardware. |
| `l1tex__t_sector_hit_rate.pct` | % | Per-kernel hit rate. |
| `l1tex__t_sectors_lookup_hit.sum` | sector | Hits, for the aggregate rate. |
| `l1tex__t_sectors_lookup_miss.sum` | sector | Misses, which proceed to L2. |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | sector | Global-load sectors. |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | sector | Global-store sectors. |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | sector/request | **Coalescing quality.** |

The coalescing ratio deserves attention. A warp of 32 threads reading contiguous 4-byte
values touches 128 B = **4 sectors**, so 4.0 is perfect. Values well above that mean the warp
is scattering across cache lines and fetching far more than it uses -- a common cause of a
kernel reading several times the bytes its data actually occupies.

### L2 (LTS)

| Metric | Unit | Meaning |
|---|---|---|
| `lts__t_requests.sum` | request | Requests arriving at L2. |
| `lts__t_sectors.sum` | sector | Sectors moved through L2. |
| `lts__t_sector_hit_rate.pct` | % | Per-kernel hit rate. |
| `lts__t_sectors_lookup_hit.sum` | sector | Hits, for the aggregate rate. |
| `lts__t_sectors_lookup_miss.sum` | sector | Misses. |
| `lts__t_sectors_op_read.sum` | sector | Read sectors. |
| `lts__t_sectors_op_write.sum` | sector | Write sectors. |

With 25 MB of L2 and an L2 hit worth 4.9x the LPDDR5X rate on this part, the L2 hit rate is
the single most informative number for whether a model's working set is being captured.

### Past L2 -- the unified LPDDR5X pool

**This block replaces the `dram__*` counters that GB20B does not have.** See
[01-platform-gb10.md](01-platform-gb10.md) for the calibration.

| Metric | Unit | Meaning |
|---|---|---|
| `lts__t_sectors_aperture_sysmem_lookup_miss.sum` | sector | **The DRAM-bytes substitute.** Misses to sysmem, which is where memory lives here. |
| `lts__t_sectors_aperture_sysmem_lookup_hit.sum` | sector | Traffic L2 *saved* from reaching memory. |
| `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum` | sector | Read traffic to memory. |
| `lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum` | sector | Write traffic to memory. |
| `lts__d_sectors_fill_sysmem.sum` | sector | L2 fill sectors sourced from sysmem -- an independent cross-check. |
| `lts__d_sectors.sum` | sector | Total L2 data-stage sectors. |

**Sentinels -- must read zero:**

| Metric | Why it must be zero |
|---|---|
| `lts__t_sectors_aperture_device_lookup_miss.sum` | GB10 has no private VRAM. |
| `lts__t_sectors_aperture_peer_lookup_miss.sum` | Single-GPU system. |

If either goes non-zero, the sysmem-only derivation is incomplete and every DRAM figure
understates reality. The report says so loudly rather than quietly reporting a wrong number.

### Time, throughput and work

| Metric | Unit | Meaning |
|---|---|---|
| `gpu__time_duration.sum` | ns (arrives in ms) | Kernel duration. |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | % | SM pipeline throughput. |
| `sm__memory_throughput.avg.pct_of_peak_sustained_elapsed` | % | SM-side memory pipeline throughput. |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | % | Closest single "how saturated is memory" number available without DRAM counters. |
| `sm__pipe_tensor_cycles_active.sum` | cycle | Tensor-core active cycles. |
| `sm__ops_path_tensor_src_bf16_dst_fp32.sum` | op | **Tensor-core work** -- the dominant path for bf16 transformer GEMMs. |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | inst | Non-tensor fp32 FMA (2 FLOPs each). |
| `smsp__sass_thread_inst_executed_op_hfma_pred_on.sum` | inst | Non-tensor fp16 FMA (2 FLOPs each). |

FLOPs are measured rather than inferred from parameter counts, which is what makes the
roofline's x-coordinate real.

## Tier 2 -- the deep dive

Whole Nsight Compute sections on the top-N kernels by measured time. Sections rather than a
metric list, because the interesting derived quantities (spill requests, roofline points,
memory-chart edges) are section-internal and cannot be requested individually:

`SpeedOfLight`, `MemoryWorkloadAnalysis`, `MemoryWorkloadAnalysis_Tables`,
`MemoryWorkloadAnalysis_Chart`, `ComputeWorkloadAnalysis`, `LaunchStats`, `Occupancy`,
`SchedulerStats`, `WarpStateStats`, `InstructionStats`, `WorkloadDistribution`.

Tier 2 picks its kernels from **the nsys timeline** by default (`ncu.rank_source: nsys`):
per-kernel GPU time inside the phase's NVTX range, summed over every instance of the phase.
The two tools demangle differently -- nsys reports a full templated signature where ncu reports
a base function name -- so nsys names are first reduced to the same base identifier ncu's
`--kernel-name` filter matches (`base_identifier()`), and the ranking is by that base name.
`rank_source: tier1` restores the original behaviour of ranking from tier 1's own export. Each
source falls back to the other when it has nothing for a phase.

**Tier 1 is off by default** (`ncu.tiers: [2]`). Over every kernel in scope it needs about 15
replay passes per launch. On an eager MoE that is hours per phase: Kimi-Linear runs ~12,000
launches per decode step and ~190,000 per prefill, so it also ends truncated at the launch cap.
Without tier 1 the run has no per-level byte totals, hit rates or decode physics check; the
report marks those "not measured" and says why. Turn it back on with `tiers: [1, 2]` or
`--tiers 1,2`.

Note: `MemoryWorkloadAnalysis` requests `dram__bytes.sum.per_second`, which does not exist
here, so those rows come back `n/a`. The parser tolerates this rather than failing.

## Tier 3 -- source attribution (opt-in)

Adds `SourceCounters` with `--import-source yes`. Useful only when the binary carries SASS
line information, which stock PyTorch and cuBLAS kernels generally do not. Best-effort by
design.

## Quantization detection, and two traps in it

`nsbench discover` reads the scheme from checkpoint metadata, never from the repository name.
Two things caught this out on real checkpoints here and are worth knowing:

- **`quant_algo` is the scheme; `quant_method` is the toolkit.** A checkpoint can declare both
  (`quant_method: modelopt`, `quant_algo: MIXED_PRECISION`). The algorithm determines
  precision; the toolkit says only who produced the file. Preferring the method would label a
  mixed-precision checkpoint as uniformly 8-bit.
- **KV-cache quantization is declared separately from weight quantization**, via
  `kv_cache_quant_algo`. One checkpoint here pairs mixed-precision weights with an **FP8 KV
  cache** — which halves the cache's bytes. The analytic KV size uses the declared element
  size, so the expected-versus-measured decode check does not disagree for a reason that has
  nothing to do with the measurement.

For a mixed-precision checkpoint there is no single bits-per-weight. Every byte figure that
matters uses the **exact on-disk size** instead, so the reported traffic is unaffected; only
the derived parameter count is an estimate.

## Derived quantities

| Quantity | Derivation |
|---|---|
| Bytes at any sector-counted level | `sectors x 32` |
| Shared bytes | `wavefronts x 128` (estimate) |
| **DRAM bytes** | `lts__t_sectors_aperture_sysmem_lookup_miss.sum x 32` |
| Aggregate hit rate | `sum(hits) / sum(hits + misses)` |
| SM -> L2 amplification | `l2_bytes / l1_bytes` |
| **L2 -> DRAM amplification** | `dram_bytes / l2_bytes` |
| Achieved bandwidth | `dram_bytes / sum(kernel durations)` |
| Arithmetic intensity | `total_flops / dram_bytes` |
| Bytes per token | `level_bytes / batch_size` for one profiled decode step |
| **x weights** | `dram_bytes_per_token / model_weight_bytes` |

The last one is the most transferable number the harness produces. It is dimensionless, so a
0.6B bf16 checkpoint and a 30B 4-bit one are held to the same standard. Theory says a decode
step reads the model once per token, so 1.0 is expected; well below means real cache reuse,
well above means something is moving data it need not.

Bandwidth uses summed **kernel** time, not wall clock: an ncu collection's wall clock
includes replay and profiler overhead and would understate bandwidth by an order of
magnitude.
