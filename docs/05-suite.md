# Running the model suite

Benchmarking one model is `nsbench run`. Benchmarking seven and comparing them is
`nsbench suite`, which adds three things that only matter at that scale: a shared weight
store, resumability, and a comparison document built from whatever succeeded.

## The shared store

Weights live in `/opt/ai-models`, not in a per-user cache. These checkpoints run to 70 GB
each; a second person benchmarking the same model should not download it again, and the
machine should not carry three copies because three people tried it.

The store is a standard HuggingFace cache layout, so anything else on the box picks it up by
exporting one variable:

```bash
export HF_HOME=/opt/ai-models
```

Creating it needs root once:

```bash
sudo install -d -o $(id -un) -g $(id -gn) -m 2775 /opt/ai-models
```

Mode `2775` gives you and your group write access and everyone else read and traverse, so
teammates reuse the cache without being able to corrupt it. The setgid bit makes new files
inherit the group. If teammates need to *add* models, create a shared group instead and put
everyone in it.

## Selecting models

The catalogue is [`configs/model-registry.yaml`](../configs/model-registry.yaml). Adding a
model is adding a block to it; nothing else in the harness changes.

```bash
nsbench models                       # catalogue + what the store already holds
nsbench fetch  --models @all         # download (resumable, safe to re-run)
nsbench suite  --models @all         # benchmark everything and compare
```

Selection accepts keys, comma-separated lists, and `@group` references:

| Selection | Meaning |
|---|---|
| `qwen3-4b` | one model |
| `qwen3-4b,gemma4-26b` | two, in that order |
| `@moe` | the four mixture-of-experts models |
| `@dense` | the dense models |
| `@all` | the full seven-model comparison |
| `@smoke` | Qwen3 0.6B only, for checking the pipeline |

Groups expand in place, so `@dense,qwen3-30b` means what it looks like, and duplicates are
dropped while order is kept.

## What the suite does

1. **Fetches** anything missing, resuming at file granularity (see [below](#downloads-stalls-and-what-resume-actually-means)).
2. **Benchmarks each model in turn, smallest first.** A configuration mistake then surfaces
   in the ten-minute model rather than after two hours of the largest one.
3. **Records every outcome to a ledger** (`runs/_suite_<tag>/ledger.json`) as soon as that
   model finishes.
4. **Builds the comparison** from whatever succeeded.

While one model is being profiled, the next ones download in the background. Fetching and
profiling use almost disjoint resources, and on this machine they take comparable time, so
overlapping them removes roughly a third of the wall clock. Pass `--no-prefetch` if you want
the timing pass free of even that much I/O contention.

## Failure and resume

A failure on one model does not cost the others. The suite logs it, continues, and marks the
comparison as incomplete rather than quietly omitting the model.

```bash
nsbench suite --models @all --tag full     # ... fails on model 6 of 7
nsbench suite --models @all --tag full     # resumes: skips the five that worked
```

Resume is the default. `--no-resume` re-runs everything.

The ledger is the source of truth, not the run directories, so a run that produced artefacts
but no usable data is correctly treated as failed.

## Downloads, stalls, and what resume actually means

Two things about fetching 284 GB are worth knowing before you start one, because both are
surprising and both have bitten this harness.

**`snapshot_download` can hang forever.** Against a rate-limited endpoint the remote closes
the connection, the sockets sit in `CLOSE-WAIT`, a worker thread parks in a futex, and the
parent blocks joining it. `HF_HUB_DOWNLOAD_TIMEOUT` does not rescue this: it governs a single
request, not a pool of threads one of which will never return. So `fetch` runs the download in
a **subprocess under a watchdog** that samples bytes on disk and kills the process after 15
minutes of genuinely zero progress, then retries with backoff. Killing it is the only thing
that reliably ends the hang.

**Resume is whole-file, not byte-offset.** huggingface_hub 1.x streams each file into a
temporary name carrying a fresh `uuid4` and opens it with `"wb"`
([PR #4228](https://github.com/huggingface/huggingface_hub/pull/4228) — a shared
`<etag>.incomplete` corrupts the cache on filesystems where `flock` silently succeeds for
every caller, as it does on Lustre, GPFS and some NFS mounts). Nothing can adopt that partial
afterwards. Concretely:

- Files already linked into `blobs/` are **kept**, always. Re-running `fetch` skips them.
- The one file *in flight* when a transfer dies is **lost**, however complete it was.

That is why the stall threshold is 15 minutes rather than something twitchy. A kill 95% of the
way through a 10 GB shard costs 10 GB — about 20 minutes on this link — so killing early to
save four costs more than it saves. It also means a large completed file's hash-and-link pause,
which writes nothing for a while, must not be mistaken for a stall.

Interrupted transfers leave full-size `.incomplete` carcasses behind, because the downloader's
cleanup never runs when the process is killed. `fetch` sweeps them between attempts, judging
them by mtime rather than size — immediately after a kill the abandoned file is the *larger*
one, so size gets it exactly backwards. `nsbench models` applies the same test, which is why a
partial that nothing is writing stops counting toward the percentage.

## Reading the comparison

The document leads with **memory traffic per generated token**, not tokens per second.
Throughput conflates the model, the kernels and the machine; bytes read per token isolates
what this harness measures.

Two columns carry the dense-versus-sparse story, and they have to be read together:

- **x stored** — traffic against the whole checkpoint. A top-8-of-128 model reads about a
  tenth of what it stores. That is not cache reuse; it never touched the other 120 experts.
- **x active** — traffic against the weights a token actually routes through. This is the
  column that makes dense and sparse models comparable at all: ~1.0 is the expected value for
  *every* row regardless of architecture, because every decode step reads its own active set
  once.

A sparse model that reads well under 1.0 **x active** is either genuinely reusing hot experts
across steps, or routing more narrowly than top-k implies.

**GPU busy** should be read before any bandwidth column. Every rate in the document divides
by kernel time, so on a row that is two-thirds busy those rates describe the kernels and not
the step. Deep models and MoE routing both push it down; when it is low the model is bounded
by launch dispatch and moving fewer bytes will not help it.

**Capacity versus throughput** is the section that answers the edge question. On a 128 GB
unified part capacity is the binding constraint, and a sparse model spends a lot of it to buy
bandwidth. `tok/s per GB resident` says whether that trade paid off on this machine.

## Cost

Sized from the seven-model default on a DGX Spark:

| | |
|---|---|
| Download | ~284 GB, ~10 h at 8 MB/s |
| Profiling | ~1–2.5 h per model, ~10–18 h total at `standard` |
| Overlapped | ~20 h wall clock |
| Disk | ~284 GB store + ~2 GB of run artefacts per model |

Throughput on this machine has been measured anywhere between 0.14 and 8.5 MB/s, and the low
end was not HuggingFace — a Cloudflare control transfer was equally slow at the same moment.
Check the link before blaming the hub or concluding a transfer is wedged; `nsbench models`
distinguishes the two, since a slow transfer still advances its percentage and a dead one
does not.

`--profile configs/profiles/quick.yaml` is roughly 3–5x faster and still produces the
traffic and bandwidth comparison; it drops the tier-2 deep dive, so the "why is it slow"
section is empty.

## Launch-count sizing

`ncu --launch-count` stops collection rather than sampling it, so a cap below the real kernel
count leaves totals that are a prefix of the phase and costs that run its physics check.

The suite sizes the cap from model depth — roughly 60 launches per transformer block, doubled
for MoE layers where routing and per-expert GEMMs add their own. A 52-layer MoE gets about
10,000 where a 28-layer dense model gets the 4,000 default.

Passing `--max-kernels` overrides this exactly, including downward. That is deliberate: the
smoke test sets a cap far below the real count to check the plumbing in minutes, and an
automatic mode that overrode an explicit instruction would make the fast path slow again.
