# Methodology: How the system works, and how to read the numbers

This document is about interpretation -- what each figure does and does not support, and the
decisions behind the measurement design. [02-metric-reference.md](02-metric-reference.md)
covers what is collected; this covers what it means.

## Prefill and decode are measured separately, always

They are the same weights running through the same kernels, and their memory behaviour
differs by an order of magnitude.

**Prefill** processes the whole prompt in one pass. Each weight is read once and amortised
across hundreds of tokens, so arithmetic intensity is high and the phase sits near the
compute roof.

**Decode** produces one token per forward pass. Every weight in the model, plus the whole KV
cache, is read to emit a single token. Arithmetic intensity is near its floor, so decode
*cannot* be compute-bound.

Any end-to-end "tokens per second" number averages these together and hides exactly the
effect being studied. Every table in every report keeps them apart.

The practical consequence: **on this machine a decode step cannot be made fast by better
arithmetic.** The roofline's ridge point is where memory-bound turns into compute-bound, and
a decode step sits far to the left of it.

But "not compute-bound" does not imply "bandwidth-bound", and the harness no longer asserts
that it does. A step whose arithmetic intensity is at the floor can still be limited by
something that is neither: an eager-mode decode step issues well over a thousand kernels per
token, and if the host cannot queue them faster than the device drains them the phase is
bounded by dispatch. That shows up as GPU idle time inside the phase, which no per-kernel
metric can see, so the report measures it directly -- see *Busy time* below. Read that column
before choosing a lever. If the GPU is idle a third of the step, reducing bytes moved will
not help; cutting launch count or fusing kernels will.

## Profiled durations are not performance

Nsight Compute serialises kernel launches and replays each kernel many times to collect a
full metric set. A decode step measured under ncu can be an order of magnitude slower than
the same step running normally. Nsight Systems is far lighter but still installs CUPTI
callbacks.

So the harness runs a **separate unprofiled baseline** whose only job is honest timing. It is
the only source of latency and throughput in any report, and it is labelled as such
everywhere. Quoting a profiled duration as performance is a common and quietly wrong thing to
do; keeping the baseline as its own collection makes the distinction structural rather than a
footnote.

Each collection also runs in its **own subprocess**. Reusing one process would let the first
collection's CUPTI state and allocator condition leak into the next one's numbers.

## Why only one decode step is profiled

A decode step's kernel mix does not change from token to token -- the same layers run in the
same order, with only the KV cache length differing. So profiling one step yields one
instance of every unique kernel, at a fraction of the cost of profiling all of them.

The step chosen is the **last** one, where the KV cache is deepest. That is the worst case
for memory traffic and the point where the cache genuinely competes with the weights for
bandwidth; an early step would flatter the model.

This is why the workload annotates NVTX differently per collector. In nsys mode every decode
step carries the `nsbench.decode_step` name so the timeline is complete. In ncu mode exactly
one does, and the rest are renamed. Comparing raw kernel counts between the two tools would
therefore show a mismatch on every healthy run -- so the cross-check normalises the nsys
count per range instance before comparing.

## Truncation is not sampling

`ncu --launch-count N` **stops** after N launches. It does not sample. A cap below the real
kernel count leaves totals that are a prefix of the phase's traffic, not a scaled-down
version of it, so everything reads low.

This matters more than it sounds: an eager transformers decode step runs far more kernels
than the layer count suggests -- roughly 55-60 launches per transformer block once
normalisation, rotary embedding and elementwise ops are counted, so a 28-layer model is about
1600 launches per step. The default cap is 4000. When a collection hits its cap the harness
flags the phase as partial, skips the expected-versus-measured check (which would otherwise
measure the truncation rather than the model), and says so in both reports.

## Cache state during measurement

<mark>ncu runs with `--cache-control all`, which flushes L2 before each replay pass.</mark> That makes
every kernel's measurement independent of whatever ran before it -- without it, a kernel's
numbers would depend on its position in the launch sequence and would not be reproducible.

The cost is that **reported L2 hit rates are a lower bound**. A real, un-profiled decode loop
may carry data in L2 from one step to the next, and with 25 MB of L2 on this part that
carry-over is not negligible. <mark>If the warm behaviour is what you need, set
`ncu.cache_control: none` in the profile config</mark> -- accepting that each kernel's numbers then
depend on its predecessor, which is the trade being made.

The same flush makes **DRAM byte counts an upper bound**, and that consequence is easy to
miss because it points the other way. Traffic L2 would have absorbed in a warm run is counted
as reaching LPDDR5X here. So any rate computed as *these bytes over un-profiled wall time*
reads high. The report computes exactly that, for both phases, and labels it a bound rather
than a measurement -- because on a healthy run the prefill figure lands above the machine's
own measured ceiling, which is impossible and is the cleanest evidence available that the
numerator is inflated. A bound that can be caught overstating itself is worth more than one
quoted only where it happens to look reasonable.

## Busy time

A phase's wall time is not all device time. The report divides each phase's NVTX range into
the union of kernel intervals inside it and the gaps between them, taken from the Nsight
Systems timeline.

This is the only figure in the harness that can distinguish "the memory system is saturated"
from "the GPU is waiting". Every bandwidth, utilisation and intensity number divides by
kernel time, so all of them describe the busy fraction and none of them notices the idle
one. A decode step that is 65% busy has a third of its latency in launch overhead and host
synchronisation, and quoting a bandwidth utilisation for it without that context points at
the wrong optimisation entirely.

Two things the harness does to keep that fraction honest rather than merely reported:

- The decode loop never touches the host. Sampled tokens stay device tensors and are
  materialised once, after the timer closes. Converting each token to a Python int inside the
  loop forces a device-to-host copy, which is a full synchronisation, and turns a measurement
  of the model into a measurement of round-trip latency.
- Prefill limits the LM head to the final position. Running it over every prompt position
  materialises a `[batch, prompt, vocab]` logits tensor whose other rows are discarded --
  on a large-vocabulary model the single heaviest kernel in the phase, and work no inference
  stack performs. Where the model accepts no such keyword, the run records that prefill
  included it rather than letting the number stand unqualified.

The sections above describe measurement choices that hold across the whole harness. The rest
of this document walks through the four collection stages that apply them, in the order they
actually run.

## Pipeline order and rationale

Every benchmark run executes four stages in a fixed sequence, each as its own isolated
subprocess: **Calibration &rarr; Baseline &rarr; Nsight Systems &rarr; Nsight Compute**
(`CalibrationRunner` &rarr; `BaselineRunner` &rarr; `NsysRunner` &rarr; `NcuRunner`, sequenced by
`nsight_bench/orchestrator.py`). Running each stage in its own process means a crash or a
perturbed CUDA context in one collection cannot leak into the next one's numbers -- every
stage starts clean.

The order is not arbitrary:

- **Calibration first**, Runs a sweep to determine the roofline plot. Verifies the byte-accounting
  derivation, which all the following memory traffic data is derived using (Because GB10 lacks DRAM counters)
- **Baseline second**, before any profiler has touched the GPU. This is the only point in the
  whole run where completely unperturbed timing can be captured -- once Nsight Systems or Nsight
  Compute attach, every duration measured afterward is inflated by instrumentation to some degree.
- **Nsight Systems third**, because it is comparatively cheap (light CUPTI callbacks, no kernel
  replay) and produces a full timeline -- kernel ranking, gaps, busy time -- that the final stage
  can build on.
- **Nsight Compute last**, because it is by far the most expensive collection (it serialises and
  replays every kernel to gather a full metric set), and running it last means it benefits from
  everything already known: which metrics are available on this platform, and whether
  calibration's gate passed.

## Stage 1 -- Calibration (`CalibrationRunner`)

Every DRAM figure this harness reports rests on one substitution: that
`lts__t_sectors_aperture_sysmem_lookup_miss x 32 B` is the traffic reaching LPDDR5X. That
claim is load-bearing, so it is re-verified against a known byte count before every
benchmark, and the pass/fail result is stamped into the manifest and both reports.

**Why this stage exists at all**: GB10 is a unified-memory part and exposes none of the
`dram__*` counters a normal discrete-GPU harness would read for "bytes moved to/from DRAM" --
that metric family simply does not exist on this hardware. So the harness substitutes a
different, available counter instead: an L2-miss counter ncu does expose
(`lts__t_sectors_aperture_sysmem_lookup_miss`), scaled by the sector size, standing in for
DRAM traffic to the unified LPDDR5X pool. That substitution is unverified by construction --
nothing about the counter's name guarantees it measures what it is being assumed to measure --
so before it is trusted for anything else, it has to be checked against a quantity that is
already known. Calibration is that check. It does not measure the model; it proves the
derivation the rest of the harness depends on.

Calibration runs first, before anything else touches the GPU, and produces two deliberately
different collections.

### Byte-accounting gate (profiled with ncu)

Done to verify whether the L2 Cache-miss based estimation of DRAM traffic is accurate or not. 

A streaming kernel -- a 256 MiB elementwise multiply, reading one array and writing another --
is run under Nsight Compute, wrapped in an NVTX range so the profiler scopes exactly to the
kernel of interest and ignores setup allocations. The number of bytes it touches is known
exactly by construction: `expected_bytes = 2 x 4 x elements` (256 MiB read plus 256 MiB
written, 4 bytes per fp32 element). ncu exports the kernel's sector counters, and the
*measured* figure is read the same way every other DRAM byte count in this harness is:
`measured_bytes = sysmem-aperture-miss-sectors x 32 B`.

The gate passes only if **both** of the following hold:

- **Relative error within &plusmn;5%.** Real traffic includes a small amount of instruction fetch
  and page-table-walk overhead that the analytic `2 x 4 x elements` figure does not model, so an
  exact match is not expected -- but anything beyond this tolerance indicates a broken
  derivation, not noise.
- **Device and peer aperture "sentinel" counters read zero.** Alongside the sysmem-aperture
  counter, ncu also reports traffic through the *device* aperture (this GPU's own local memory)
  and the *peer* aperture (another GPU's memory over NVLink or similar). On this platform, none
  of this kernel's traffic should ever route through either -- it should all go through the
  sysmem aperture. These two counters are called sentinels precisely because they are not part
  of what the gate is trying to measure: they are a tripwire. If a future driver or architecture
  change starts routing some of that same traffic through the device or peer aperture instead,
  the sysmem-aperture count would quietly *undercount* total traffic -- and every "bytes moved"
  figure downstream would be silently low -- without necessarily pushing the relative error
  outside &plusmn;5%, since the measured total can still look plausible in isolation. The sentinel
  check catches a failure mode the tolerance check alone would miss: it fails loudly the moment
  traffic starts going somewhere the derivation does not expect, instead of letting every later
  DRAM figure be silently wrong by an unknown factor.

### Bandwidth and compute sweep - Produces the Roofline Plot (unprofiled -- no ncu attached)

The second collection measures achievable bandwidth and compute throughput, and deliberately
runs with **no profiler attached at all**. Nsight Compute's kernel-replay mode times a
serialised re-execution with L2 forcibly flushed before each pass -- that is not achievable
bandwidth, it is the cost of making individual kernel measurements reproducible (see *Cache
state during measurement* above). Measuring a real, sustainable rate means getting the
profiler out of the way entirely.

The same streaming kernel from the byte-accounting gate is run across a sweep of working-set
sizes -- `1, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024` MiB -- chosen to bracket the
platform's ~25.17 MB L2 tightly (16 and 24 MiB sit inside it, 32 and 48 sit just outside, so
the transition lands between sample points rather than being inferred across a wide gap).
Each working-set size is warmed up three times first, to absorb the first-touch page faults a
unified-memory part pays on initial access -- those are substantial here and would otherwise
dominate a small working set's timing. The kernel then runs 20 iterations, timed via plain
wall clock (`time.perf_counter()` bracketing a `cuda.synchronize()`-bounded region), and
bandwidth is `bytes touched / elapsed seconds`. A dense bf16 8192&times;8192 GEMM, comfortably
compute-bound by construction, is timed the same way to establish the achievable compute
ceiling.

**Both ceilings are measured** rather than taken from a datasheet, for the same reason: a
datasheet number assumes clock and sparsity conditions no real kernel meets, so plotting
against it would make every kernel look far from the roof regardless of how well it is
actually doing.

The sweep's working-set points are then split into two regimes, and conflating them is the
easy mistake this classification exists to prevent:

- **L2-resident** (working set &le; 0.75&times; the L2 size) &rarr; `peak_l2_bandwidth_gbps`.
- **Streaming / DRAM-bound** (working set &ge; 4&times; the L2 size, leaving no doubt the data
  cannot be cache-resident) &rarr; `peak_dram_bandwidth_gbps`, the memory-bound roofline ceiling.

On this part, L2-resident bandwidth runs roughly five times the LPDDR5X rate. Taking the
sweep's overall maximum as "peak bandwidth" would put the roofline's memory ceiling far above
anything a real model can reach, and every kernel would look artificially efficient by
comparison -- which is exactly why the two regimes are kept separate rather than collapsed
into one number.

> **Small callout: the sweep does not estimate L2 size.** `l2_cache_bytes` comes from the
> platform profile -- a known, probed hardware spec -- not from the sweep. What the sweep
> *does* derive empirically is the **knee**: the working-set size with the single largest
> bandwidth drop between adjacent sample points. That knee is compared against the
> already-known L2 size purely as a consistency check (does the measured falloff line up with
> where L2 capacity should stop helping?), not as how L2 capacity is discovered in the first
> place.

From the two measured ceilings, the calibration also derives the **ridge point** --
`peak_compute_gflops / peak_dram_bandwidth_gbps`, the arithmetic intensity at which the
roofline turns from memory-bound to compute-bound. Below that intensity, a kernel cannot be
anything but memory-bound no matter how well it is written. LLM decode sits far below it,
which is the structural reason decode cannot be made fast by better arithmetic (see *Prefill
and decode are measured separately, always* above).

### Output: `calibration.json`

The stage's result is written to `calibration.json`, containing: `passed`, `byte_accounting`
(the gate's full result, including relative error and sentinel readings), `sweep` (every
working-set point measured), `l2_cache_bytes`, `knee_mib`, `peak_l2_bandwidth_gbps`,
`peak_dram_bandwidth_gbps`, `peak_compute_gflops`, `compute_peak_detail`, and a set of
human-readable `notes` summarising the gate outcome and the derived ceilings. The
byte-accounting gate's raw artifacts (`calibration_stream.ncu-rep`, its CSV export) and the
sweep's raw payload (`calibration_sweep.json`) are kept alongside it.

If the gate fails, the orchestrator does not abort the run -- the remaining stages still
execute -- but every DRAM-derived figure in that run is flagged as unverified in the manifest.
And regardless of pass/fail, every "% of peak bandwidth" figure anywhere in either report is
computed against this stage's `peak_dram_bandwidth_gbps`: this is the roofline denominator for
the entire benchmark.

## Stage 2 -- Baseline (`BaselineRunner`)

This is the only stage in the harness whose latency and throughput numbers are meaningful,
because it runs the actual model workload with **no profiler attached at all** (see *Profiled
durations are not performance* above). It runs second, immediately after calibration and
before any profiler has touched the GPU -- the only point in the run where timing is
completely unperturbed.

### Timing

Each measured phase is timed with a GPU-synchronised wall clock: `cuda.synchronize()` is
called immediately before starting the clock and immediately after stopping it, so the
duration reflects actual kernel execution rather than how fast the host could submit work.
Without that synchronisation, decode in particular -- which submits kernels far faster than it
computes them -- would look implausibly fast.

Per repeat (the workload runs `cfg.repeat` times), two phases are timed:

- **prefill** -- one forward pass over the whole prompt.
- **decode** -- the entire generation loop, timed as a single block covering every requested
  token, not per individual step.

Before any of this, warmup iterations run and are discarded (the last one at the full
generation length, so the KV cache reaches its measured depth before timing starts) --
absorbing cuBLAS autotuning, lazy module initialisation, allocator growth, and first-touch
page faults that would otherwise make the first measured iteration an outlier.

**`decode_step` -- a derived average, not a per-step measurement.** The harness also reports a
per-token decode figure, `decode_step`, but it is worth being precise about how it is
produced: it is the decode block's total measured seconds divided by the number of generated
tokens. That is an average under an assumption of uniform per-token cost, and the assumption
does not strictly hold in practice -- the KV cache grows with position, so later decode steps
read more cache than earlier ones, and a long decode loop on a thermally-managed box is not
guaranteed to run at a constant clock throughout. `decode_step` should be read as a
*representative* average cost per token, not a claim that every token in the loop cost exactly
the same. The median-with-IQR statistics described later (see *Statistics*) surface
instability *across repeats* of the same measurement, which is a real and useful signal, but
they do not surface step-to-step variance *within* a single decode loop -- that variance is
simply averaged away by this derivation, and nothing in the current pipeline exposes it.

### <mark>Memory footprint: three views</mark>

<mark>(looking into this could be important when studying agents, because cpu vs gpu usage becomes important)</mark>

**Start with the discrete-GPU case, since it is the more familiar one.** On a discrete GPU,
`cudaMemGetInfo`'s total/free figures are scoped to that GPU's own VRAM. So the gap between
the CUDA driver's peak reading and the torch allocator's reserved peak is a meaningful,
directly computable quantity: this process's non-torch CUDA memory -- the CUDA context and
library workspaces that torch's own allocator cannot see. That is the usual reason a harness
would bother tracking more than one view of memory at all.

**GB10 breaks that intuition.** NVML returns nothing here -- there is no discrete VRAM to
report, because the GPU shares one coherent 128 GB LPDDR5X pool with the Grace CPU. Worse, on
this part <mark>`total - free` from the driver blends in every host process and the page cache
alongside whatever CUDA actually took</mark>, so a 1.6 GB process routinely sits inside a 25 GB
system-wide reading. The driver-vs-allocator subtraction that is meaningful on a discrete GPU
is therefore *not* scoped to this process at all here -- attributing that gap to "cuBLAS
workspace" would be wrong by more than an order of magnitude while sounding specific. This is
why a third, independent view is needed that a discrete-GPU harness would not require.

So footprint is reconstructed from three sources, each blind to something different:

1. **The torch caching allocator** (`torch.cuda.memory_stats`) is exact for tensors torch
   allocated, and blind to cuBLAS workspaces, the CUDA context, and anything a fused kernel
   allocates itself. It also reports *cached* memory -- held but not currently in use.
2. **`cudaMemGetInfo`** (the CUDA driver) sees everything CUDA has taken, context and library
   workspaces included -- but as above, "total" is the whole 128 GB unified pool, so "free"
   moves whenever *any* host process allocates, not just this one.
3. **`/proc/meminfo`** (the host kernel) is the only view showing GPU and CPU allocations
   competing for one shared budget -- which is the thing that actually bites on a
   unified-memory box.

**Phase-boundary sampling.** `PhaseMemoryTracker` takes a before/after snapshot across each
phase (`load`, `prefill`, `decode`), resetting and re-reading torch's peak counters at each
boundary so every phase gets its own high-water mark. That reset is not free, though: torch's
own "peak since last reset" would otherwise silently under-report the run's true peak, since
the actual maximum usually occurs during prefill, several resets before the process ends. The
harness works around this by folding each phase's outgoing peak into a run-level maximum
before clearing torch's counters, so per-phase resolution does not cost the run-level figure.

**20 Hz continuous sampling.** In addition to the phase-boundary sampling, a background thread
takes the same three-view snapshot on a fixed clock: every 50 ms, for the entire worker
subprocess's lifetime -- starting before weights are even loaded and continuing until the
process's final cleanup.

Why it exists: a before/after pair per phase only shows the *net* change, so it is blind to
anything that spiked and came back down within that phase. Continuous sampling is what makes
transient peaks visible, for example:

- a burst during weight loading,
- workspace allocation ramping up and back down within a single phase,
- KV-cache growth mid-decode.

Where the data goes:

- Every sample is merged with the phase-boundary snapshots and written out in full as
  `baseline_result.json`'s `memory_timeline`.
- Downstream, only a single reduction of that timeline is actually used -- its peak value --
  which flows into the analysis stage and from there into both reports' memory table and the
  torch-vs-driver agreement check.

As things stand today, the full time series is written to disk but not read again anywhere
else in the codebase. The 20 Hz stream's current job is solely to make that one peak figure
safe against transient spikes; it does not yet drive any plotted memory-over-time view, though
the data is already there if that is ever built.


**Why deltas, rather than absolute snapshots, are the unit recorded.** A delta isolates what
one phase actually *did* to memory, independent of whatever steady-state footprint the process
already happened to be sitting at when the phase began. That makes deltas comparable across
phases and across runs regardless of starting conditions, in a way raw absolute snapshots are
not.

**`agreement_ratio`.** For a given phase, this is:

```
agreement_ratio = cuda_driver_delta / torch_allocator_delta
```

How to read the value:

- **&asymp; 1.0 (high agreement)** -- the torch allocator is successfully tracking almost all of
  the memory being consumed during this phase. Likely entirely from KV cache or other tensor growth between intervals during inference.
- **&gt; 1.0 (low agreement)** -- the driver is allocating significantly more memory than torch
  requested. This indicates that operations outside of the torch allocator's view are consuming
  memory -- such as the initialisation of the CUDA context, or scratchpad memory pools
  ("workspaces") spun up by highly optimised libraries like cuBLAS or cuDNN.

**Contextualizing with `host_delta`.** On a unified-memory architecture, a ratio above 1.0 is
not automatically real workspace or context overhead -- it is worth checking against
`host_delta` before trusting it. If the driver (`cuda_delta`) reports a spike that does not
line up with either the application (`torch_delta`) or the physical system (`host_delta`),
what is most likely being captured is background OS activity or page-cache churn on the shared
pool, not actual model-workload consumption.

This is the same shared-pool caveat raised above, applied concretely: on a discrete GPU, a
ratio above 1.0 can be read fairly directly as isolated non-torch overhead, in bytes. On this
part it cannot be read that directly, because the driver side of the subtraction is not scoped
to this process -- which is exactly why `agreement_ratio` should always be read alongside
`host_delta` here, rather than trusted on its own.

**The honest per-process headline figure on this part** is therefore not derived from the
driver/allocator gap at all -- it is the host-visible delta (`/proc/meminfo`'s available
memory, before versus after) measured specifically across weight loading, since that delta is
scoped to this process's own activity during that window. For contrast, the nsys allocation
timeline reads lower still, because it only counts allocations made inside its capture range --
and weights are loaded before tracing starts, by design.

### Output: `baseline_result.json`

The worker subprocess writes its full result here: `mode`/`ok`, `load_memory` (the weight-load
phase's memory delta), a `result` block (`timings` for every measured phase and repeat,
`memory_deltas`, `kv_cache_bytes`, `generated_token_ids` and their distinct count, and
human-readable `notes`), `backend` and `workload` descriptions, the `memory_timeline` and its
derived `memory_peak_cuda_used`, `allocator_stats` (including the run-level peak-allocated and
peak-reserved figures), and the path to the dumped `allocator_history` pickle. A separate,
much smaller record lives in the run manifest instead -- the command's exit status, a pointer
to this file's path, and the GPU throttle/thermal state recorded around the whole subprocess --
but the measurements themselves live only in `baseline_result.json`.

## Stage 3 -- Nsight Systems (`NsysRunner`)

Baseline already established honest wall-clock time and honest footprint. What it structurally
cannot do is explain *why* a phase took as long as it did -- its stopwatch can't distinguish "the
GPU was working the whole time" from "the GPU was idle, waiting on the host." Nsys is the next
layer on top of baseline specifically to close that one gap: it re-runs the same kind of real
workload, lightly instrumented (CUPTI callbacks only, no kernel replay), and produces a timeline
detailed enough to say how that already-measured duration was actually spent.

### What it measures

- **A full kernel timeline, with every decode step tagged** -- unlike Stage 4 below, where only
  one representative step is ever profiled, nsys annotates all of them, so it can report per-step
  kernel counts and durations across the whole run.
- **The busy/idle split** described in *Busy time* above -- nsys is where that figure actually
  comes from. It's computed from the exported SQLite database by attributing each kernel to the
  NVTX range that was open when the *launching* CUDA runtime call fired, not when the kernel
  itself later executed on the device -- decode kernels routinely outlive the range that queued
  them, so using the kernel's own timestamp would misattribute work across phase boundaries.
  Busy time itself is the *union* of kernel intervals inside a phase, not their sum, so
  overlapping streams aren't double-counted.
- **Memory-adjacent signals** -- worth being precise about what math these actually support,
  because none of it is traffic:
  - `CUDA_GPU_MEMORY_USAGE_EVENTS` is a raw log of allocate/free events, each carrying a
    timestamp and a size. Running a cumulative sum over that log in timestamp order reconstructs
    "bytes currently held" as a function of time -- simple running-total arithmetic, nothing
    more -- and the maximum of that curve is the allocation-timeline peak compared against
    baseline's own peak below. This is GB10's substitute for a VRAM graph, since NVML/nvidia-smi
    report nothing on this part (see *Memory footprint: three views* above).
  - Unified-memory page faults are *counted* per phase -- each fault marks one page being
    migrated between the CPU and the GPU. A fault count is a proxy for migration churn, not a
    byte total.
  - Sampled GPU clocks and engine-activity figures are periodic point-in-time readings, not
    derived quantities.
  - **None of the above is traffic.** All three answer "how much was held, and how did that
    change over time," never "how many bytes moved." That boundary is deliberate, not an
    oversight -- bytes-moved is exactly what nsys's metric set has no rows for, and it's Stage 4's
    job alone.

### What it cannot measure

No memory bandwidth at all -- the metric set nsys uses on this platform has no bytes-moved rows.
That's the reason Stage 4 exists.

### How it runs and what it produces

`nsys profile` wraps the measured loop in a `cudaProfilerApi` capture range, so weight loading is
skipped from the trace. It traces `cuda,nvtx,cublas,cudnn,osrt` -- cublas and cudnn are included
so GEMMs attribute cleanly to a recognisable name instead of showing up as mangled cutlass kernel
names -- and adds GPU-metric sampling only where the chip's metric set is actually known, since a
wrong metric set is a hard error rather than a degraded collection. The run produces
`timeline.nsys-rep`, which is exported to `timeline.sqlite` and queried directly with SQL rather
than through `nsys stats`'s fixed report menu, because the join this harness needs -- attributing
a kernel to its launching call's correlation ID -- isn't one of the canned reports.

### Where it feeds downstream

- Its kernel ranking feeds `NcuRunner` tier 2's kernel-name filter (see Stage 4 below).
- Its per-phase kernel count, normalised per decode-step instance (since nsys tags every step and
  ncu tags exactly one), is the mechanism *Why only one decode step is profiled* above already
  points to as "the cross-check that normalises the nsys count." Concretely: that normalised
  count is compared against ncu's own profiled kernel count, flagged `PARTIAL` if ncu's
  collection was truncated, or `SCOPE MISMATCH` if the two fall outside a 0.7-1.4x band of each
  other.
- Its allocation-timeline peak is cross-checked against baseline's own peak -- against the torch
  allocator's figure on unified memory, or the driver's figure on a discrete GPU, for the same
  reason given in *Memory footprint: three views* above. Routine gaps are explained rather than
  flagged as anomalies: nsys only sees allocations made inside its capture range, so weights
  loaded before tracing starts are invisible to it, by design.

## Stage 4 -- Nsight Compute (`NcuRunner`)

Even with nsys's busy/idle picture in hand, nothing so far has said *what moved*. Ncu is the only
stage reading the memory-hierarchy counters calibration validated -- and it's expensive, because
Nsight Compute serialises and replays every profiled kernel to collect a full metric set (see
*Profiled durations are not performance* above). Keeping that tractable rests on two mechanisms:
NVTX scoping to a single representative step, already covered in *Why only one decode step is
profiled* above, and **tiering**, which is new to this document.

### Tiering

- **Tier 1** applies a curated metric list -- 45 entries in the current registry, more than a
  rough "about 30" estimate would suggest -- to *every* kernel inside the scoped NVTX range.
  Every entry was individually confirmed to actually collect on this GPU by running a real
  one-kernel probe, not by trusting `ncu --query-metrics` (some metrics that look collectable
  from the query listing fail as standalone entries; others collect fine but never appear in that
  listing at all). By category:
  - **Register and occupancy limits** -- registers allocated per thread, and how many blocks per
    SM register pressure or shared-memory usage alone permit. This is the ceiling on parallelism
    a kernel is subject to, before any stall data is even collected.
  - **Local memory (register spills)** -- sectors read from and written to local memory. Spills
    are backed by the same physical DRAM as global memory, so they're expensive in exactly the
    way a naive register-pressure reading wouldn't suggest.
  - **Shared memory** -- static/dynamic allocation size, bank conflicts, load/store instruction
    counts, and wavefronts. Wavefronts are the only path to shared-memory *bytes* here, because
    there is no shared-memory sector counter: one wavefront moves up to 128 B across the 32
    banks, and bank conflicts inflate the wavefront count above the ideal.
  - **L1/TEX** -- requests, sectors, hit rate (plus raw hit/miss sector counts, so an aggregate
    hit rate across many kernels can be computed as a true sum-of-hits over sum-of-lookups rather
    than an average that weights a trivial kernel the same as a large one), and sectors-per-request
    for load coalescing.
  - **L2** -- the same request/sector/hit-rate pattern as L1, described in the preamble as "the
    single most informative number for whether a model's working set is being captured," given 25
    MB of L2 on this part.
  - **The DRAM substitute block** -- the same sysmem-aperture family from calibration
    (`lts__t_sectors_aperture_sysmem_lookup_miss` and its read/write/hit variants), now measured
    per kernel instead of only on the calibration kernel.
  - **Sentinels** -- the same device/peer aperture zero-checks from calibration, run per kernel.
  - **Time and throughput** -- kernel duration, SM pipeline throughput, SM-side memory throughput,
    and a compute-memory pipeline throughput figure described as "the closest single number to
    'how saturated is the memory system', given no DRAM counters are available."
  - **Work counters, for arithmetic intensity** -- bf16 tensor-core operations (the dominant path
    for transformer GEMMs) plus non-tensor fp32/fp16 FMA instruction counts. This is what lets a
    kernel be placed on the roofline against calibration's measured ceilings, since arithmetic
    intensity needs a real FLOP count rather than one inferred from parameter counts.
- **Tier 2 and above** deep-dive only the top-N kernels tier 1 showed actually matter, ranked by
  tier 1's own measured time from tier 1's CSV export -- not nsys's timeline, because nsys and ncu
  demangle kernel names differently and would disagree on granularity. Rather than an individual
  metric list, tier 2 requests whole Nsight Compute *sections*, because some of the quantities
  that matter here (register-spill request counts, for instance) are section-internal and fail as
  standalone `--metrics` entries.

  Eleven sections are requested at tier 2 (`SpeedOfLight`, the three `MemoryWorkloadAnalysis`
  variants, `ComputeWorkloadAnalysis`, `LaunchStats`, `Occupancy`, `SchedulerStats`,
  `WarpStateStats`, `InstructionStats`, `WorkloadDistribution`), but as things stand today the
  analysis layer only reads fields from four of them: `WarpStateStats` (the stall-reason
  vocabulary below), `Occupancy` (achieved occupancy), `LaunchStats` (waves per SM), and
  `MemoryWorkloadAnalysis` (register-spill request counts -- the one field that specifically
  justifies collecting whole sections instead of a flat metric list in the first place). The
  other six sections are present in the exported CSV but not read anywhere in the analysis or
  report code today.

  Prefill and decode are collected as **separate ncu subprocesses** -- a full second model load
  each -- so a metric is always unambiguously attributable to one phase, never blended.

### Launch-count cap, extending *Truncation is not sampling*

Tier 1's `--launch-count` cap (default 4000, auto-scalable from layer count) is the same
truncation guard already covered in the preamble. Tier 2's cap is lower and different in *kind*:
it's sized as the larger of 48 or twelve times the number of top-N kernels being deep-dived, and
it is **deliberate sampling, not truncation** -- tier-2 metrics are read per kernel and never
summed into a phase total the way tier 1's byte counts are, so sampling a subset of launches costs
nothing that truncating tier 1 would.

### The stall-reason vocabulary, and how it becomes a verdict

Tier 2's `WarpStateStats` section reports, for every profiled kernel, the share of stall cycles
attributable to each of 15 named reasons a warp wasn't able to issue an instruction:

- **Memory (long scoreboard)** -- waiting on a global or local memory load. This is the
  memory-bound signature.
- **Shared / L1 (short scoreboard)** -- waiting on shared memory or L1.
- **MIO throttle** -- the memory I/O pipe is saturated.
- **Local/global throttle** -- the local/global instruction queue is full.
- **Math pipe throttle** -- the arithmetic pipes are saturated. This is the compute-bound
  signature.
- **Barrier** -- waiting at a `__syncthreads()`.
- **Memory barrier** -- waiting on a memory fence.
- **Not selected** -- eligible to issue, but another warp was chosen instead -- a sign of healthy
  parallelism, not a problem.
- **Instruction fetch** -- waiting on instruction fetch, often a symptom of a large unrolled loop.
- **Fixed latency** -- a fixed-latency arithmetic dependency.
- **Drain** -- draining outstanding memory operations at kernel exit.
- **Dispatch** -- the dispatcher couldn't issue this cycle.
- **Branch** -- waiting on branch resolution.
- **Other** -- no further elaboration available.
- **Issued** -- the warp issued an instruction this cycle; excluded from the stall totals above.

**Correcting for sampling bias.** Tier 2 only profiles a capped, launch-ordered sample of
kernels, which over-represents whichever kernels happened to launch earliest or most often. To
avoid reporting a distribution skewed by that sampling order, stall shares are time-weighted using
tier 1's *complete* per-kernel duration -- tier 1 ran over every kernel in scope, cheaply --
rather than tier 2's own biased sample count. The result is a phase-wide stall-reason distribution
that reflects where the phase's time actually went, not which kernels happened to get
deep-profiled.

**The verdict.** From that weighted distribution, the harness states the dominant stall reason and
its share; if the memory-wait share (long scoreboard) is 40% or more, it declares the phase
"genuinely latency-bound on memory rather than on arithmetic." If mean achieved occupancy across
the profiled kernels is under 30%, it appends a separate note that there are too few warps in
flight to hide the latency that does occur -- more parallelism would help more than fewer bytes.
Worth flagging: this is asymmetric. Memory-bound and occupancy-limited each get a dedicated
interpretive branch, but a compute-bound phase -- one dominated by math-pipe throttle -- only ever
surfaces as the generic "the dominant stall reason is Math pipe throttle" sentence, with no
equivalent special-cased claim.

A "sample coverage" figure -- what fraction of the phase's total GPU time the tier-2 sample
actually covered, and which significant kernels it missed -- is computed as part of this analysis
but is not rendered in either report today.

### Output: `.ncu-rep` per scope and tier, exported CSV

Every tier exports the same way (`ncu --import ... --csv --page raw`, the only page that
round-trips cleanly), so tier 2's section-based collection still lands in the same wide
per-kernel CSV shape as tier 1 -- there's no separate parsing path to maintain.

- **The decode physics check.** Measured ncu DRAM bytes for the decode step are compared against
  an analytic prediction: resident weight bytes plus KV-cache bytes, both **taken from baseline's
  own measurements**, not re-derived here. This check cannot run without baseline's numbers -- it
  validates ncu's measurement against baseline's, rather than ncu validating itself.
- **Bandwidth at real latency.** ncu's DRAM bytes divided by *baseline's* wall time -- never ncu's
  own replay-inflated time -- checked against calibration's LPDDR5X ceiling as an upper bound.
  Exceeding that ceiling confirms the numerator reads high, for the reasons given in *Cache state
  during measurement* above, rather than signalling a bug.
- **The L2-effectiveness headline.** Decode's measured L2 hit rate is compared against
  calibration's measured L2-vs-DRAM speedup ratio, flagging when a low hit rate means decode is
  reading essentially everything from main memory.

All of this surfaces to a reader in the report's "What limits each phase" section -- a verdict
banner per phase, the weighted stall-reason table, and a per-kernel table of occupancy, waves per
SM, and dominant stall reason for the kernels that matter most.

## Stitching it together: how `assemble.py` builds one report from four stages

No single stage's numbers are independently meaningful. Most of the headline figures in either
report require combining two or three stages' outputs, and `assemble.py` is the one place in the
codebase where that combination happens. Each stage supplies exactly one thing nothing else can:

- **Calibration** -- trust in the byte-counting substitution, plus the measured roofline
  ceilings (`peak_dram_bandwidth_gbps`, `peak_l2_bandwidth_gbps`, `peak_compute_gflops`).
- **Baseline** -- the only honest wall-clock time and footprint, plus the two quantities
  (resident weight bytes, KV-cache bytes) that define what a *correct* decode step's traffic
  should look like.
- **Nsys** -- the structural busy/idle diagnosis and kernel ranking, cheaply, over the whole run.
- **Ncu** -- the actual bytes moved, and the stall-reason verdicts for why the busiest kernels
  behave the way they do.

A few worked examples show how these combine -- each one is fully explained in its owning
stage's section above; this is only the map of who supplies what:

- **Achieved bandwidth** is ncu's phase byte count divided by baseline's real wall time for that
  phase -- two stages, neither sufficient alone (see *Bandwidth at real latency* in Stage 4).
- **% of peak bandwidth** takes that achieved figure and divides it by calibration's
  `peak_dram_bandwidth_gbps` -- three stages now.
- **Reading that percentage correctly** needs a fourth ingredient: nsys's busy/idle split (see
  *Busy time* above and Stage 3). The same percentage means "genuinely memory-bound" in a phase
  that's mostly busy, or "launch-bound, bandwidth is irrelevant" in a phase that's mostly idle --
  and the stage that decides which reading applies measures neither bytes nor bandwidth.
- **The decode physics check** judges ncu's measured bytes against an expectation built entirely
  from baseline's numbers -- resident weight bytes plus KV-cache bytes (see Stage 4's Output
  subsection). This check cannot run without baseline: it validates ncu against baseline, not
  ncu against itself.
- **Footprint agreement** cross-checks baseline's own peak (torch or driver, depending on
  unified vs. discrete memory) against nsys's independently-collected allocation-timeline peak,
  with routine gaps -- weights loaded before nsys's capture range opens -- explained rather than
  flagged (see Stage 3's *Where it feeds downstream*).
- **The nsys/ncu kernel-count cross-check** reconciles two independently-collected views of the
  same kernels -- nsys's per-instance-normalised count against ncu's profiled count -- as an
  internal consistency check rather than a headline figure in its own right.
- **The L2-effectiveness headline** judges ncu's measured L2 hit rate against calibration's
  measured L2-vs-DRAM speedup ratio.

Pull any one stage out and whole classes of these figures either stop being computable -- no
baseline time means no bandwidth denominator at all -- or stop being trustworthy -- no
calibration gate means no verified ceiling to call a percentage "of peak." That is the actual
reason the pipeline has four stages feeding one assembly step, rather than four independent
reports.

## Statistics

The baseline reports a **median with interquartile range** over its repeats. Not a mean:
repeats on a thermally-managed box are not identically distributed, and one iteration
running away during a clock excursion would drag a mean with it. The IQR is what tells you
whether a headline number is stable.

Quartiles are linearly interpolated, so a spread is reported from three repeats rather than
only from four. Nearest-rank quartiles need four samples before Q1 and Q3 land on different
elements, and printing "-" on every headline number of a three-repeat run reads as *not
measured* when the truth is *measured, over three points*. The sample count is printed
alongside so the difference between those two claims stays visible.

Warmup iterations are discarded. They absorb cuBLAS autotuning, lazy module initialisation,
allocator growth, and the first-touch page faults a unified-memory part pays on initial
access. Without them the first measured iteration is an outlier by a wide margin.

Generation is **greedy and deterministic** by default. Sampling would add run-to-run variance
that appears as noise in the memory numbers while telling us nothing about memory.

## Comparing runs

The comparison matrix reports **per-token and per-phase rates**, never raw totals, so runs
with different generation lengths and batch sizes are directly comparable.

The most transferable column is **memory read per token as a multiple of the model's own
weight bytes**. It is dimensionless, so a 0.6B bf16 checkpoint and a 30B 4-bit one are held
to the same standard. Theory says a decode step reads the model once per token:

- **~1.0** -- as expected.
- **well below 1.0** -- real cache reuse, plausible on this part given 25 MB of L2, or a
  scoping problem.
- **well above 1.0** -- activation and workspace traffic, an unfused dequantization pass, or
  a KV cache larger than the analytic estimate.

Runs whose calibration failed, whose collections were truncated, or whose workload shapes
differ are flagged rather than silently tabulated.

## Things this harness deliberately does not claim

- **It does not measure C2C link traffic.** GB10 exposes no `ctc__*` counters and the
  `C2CLink` section is gated to other architectures. Traffic between the Grace CPU and the
  GPU over NVLink-C2C is not observable here.
- **Shared-memory bytes are an estimate**, from wavefronts x 128 B. There is no
  shared-memory sector counter. Bank conflicts inflate the figure.
- **Register "traffic" is not a thing.** Registers are a capacity resource. What is reported
  is pressure, and the spill traffic that results when pressure exceeds capacity.
- **Tier 3 source attribution is best-effort.** It needs SASS line info that stock PyTorch
  and cuBLAS kernels do not carry.
- **Timing under a profiler is not performance**, however tempting the number.
