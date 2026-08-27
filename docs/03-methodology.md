# Methodology: how to read these numbers

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

ncu runs with `--cache-control all`, which flushes L2 before each replay pass. That makes
every kernel's measurement independent of whatever ran before it -- without it, a kernel's
numbers would depend on its position in the launch sequence and would not be reproducible.

The cost is that **reported L2 hit rates are a lower bound**. A real, un-profiled decode loop
may carry data in L2 from one step to the next, and with 25 MB of L2 on this part that
carry-over is not negligible. If the warm behaviour is what you need, set
`ncu.cache_control: none` in the profile config -- accepting that each kernel's numbers then
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

## Calibration before measurement

Every DRAM figure rests on one substitution: that
`lts__t_sectors_aperture_sysmem_lookup_miss x 32 B` is the traffic reaching LPDDR5X. That
claim is load-bearing, so it is re-verified against a known byte count before every
benchmark, and the pass/fail result is stamped into the manifest and both reports.

The gate also checks the device and peer aperture sentinels. If a future driver starts
routing traffic through the device aperture, the derivation silently halves -- the gate turns
that into a loud failure.

The same calibration measures the machine's ceilings (LPDDR5X bandwidth, L2 bandwidth, dense
bf16 GEMM throughput). Measured ceilings are what make a utilisation percentage or a roofline
position meaningful; against a datasheet number every kernel looks far from the roof
regardless of how well it is doing.

## Footprint on a unified-memory part

NVML returns nothing on GB10 -- there is no discrete VRAM to report -- so footprint comes
from three sources, each blind to something different:

1. **The torch caching allocator** is exact for tensors torch allocated, and blind to cuBLAS
   workspaces, the CUDA context, and anything a fused kernel allocates itself. It also
   reports *cached* memory, held but not in use.
2. **`cudaMemGetInfo`** sees everything CUDA has taken, context and library workspaces
   included. On this part "total" is the whole 128 GB unified pool, so "free" also moves when
   *host* processes allocate.
3. **`/proc/meminfo`** is the only view showing GPU and CPU allocations competing for one
   budget -- which is the thing that actually bites on a unified-memory box.

Disagreement between them is signal, but only where the two views measure the same thing.
On a discrete GPU the gap between the driver's peak and the allocator's reserved peak is this
process's non-torch CUDA memory -- the context and the library workspaces -- and quantifying
it is worth doing. **On this part it is not that**, and the report says so instead of
computing it. `total - free` over a shared pool counts every host process and the page cache
alongside anything CUDA took, so a 1.6 GB process routinely sits inside a 25 GB system-wide
reading. Attributing that difference to cuBLAS workspaces would be wrong by more than an
order of magnitude while sounding specific.

The honest per-process figure on a unified part is the host-visible delta measured across
weight loading, which is scoped to this process's own activity. The nsys allocation timeline
reads lower than either, because it only counts allocations made inside the capture range --
weights are loaded before tracing starts, by design.

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
