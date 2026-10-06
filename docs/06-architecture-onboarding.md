# Architecture Onboarding — Memory-Benchmark-Framework (`nsight-bench`)

## 0. TL;DR

A pure-Python 3.12 package (`nsight_bench`, console script `nsbench`, ~13.8k LOC, no CUDA/C++/CMake anywhere) that profiles LLM inference (via HuggingFace `transformers`) on NVIDIA GPUs by shelling out to `nsys`/`ncu` CLI tools, parsing their exported SQLite/CSV output, and rendering self-contained Markdown/HTML reports. It does **not** contain hand-written CUDA kernels — the "workload" is real model inference; Nsight profiles whatever kernels PyTorch/transformers launch underneath.

---

## 1. High-Level Module Architecture

```
nsight_bench/                 core package
  cli.py                      argparse entry point (nsbench)
  orchestrator.py             sequences calibration→baseline→nsys→ncu subprocesses
  worker.py                   the actual re-launched-per-mode worker process
  suite.py                    multi-model sweep runner with resumable ledger
  config.py                   ModelConfig/WorkloadConfig/ProfileConfig/RunConfig dataclasses
  platform.py                 GPU/driver/counter-availability detection (gates everything)
  registry.py                 shared HF-cache-style model store + fetch()
  calibration.py              validates DRAM-byte derivation against known byte count
  metrics.py                  canonical metric registry, tiers, units, hierarchy Level enum
  stats.py                    stats helpers (median/IQR etc.)
  gpu_state.py                NVML/nvidia-smi throttle & thermal bracketing

  backends/                   model-execution backends (the "GPU workload" driver)
    base.py                   Backend ABC, register decorator, GenerationState, measured_phase()
    hf_transformers.py        hand-written greedy decode loop over HF transformers
    trtllm.py                 documented no-op stub (blocked on aarch64 wheels)

  workloads/                  benchmark "test" definitions
    base.py                   Workload ABC, WorkloadResult, ProfileMode, registry
    text_generation.py        TextGenerationWorkload — prefill/decode split (the central axis)
    multimodal.py             MultimodalWorkload(TextGenerationWorkload) — partial/extension template

  instrumentation/
    nvtx.py                   Phase enum, nvtx_range/phase/mark — the NVTX contract both profilers key off
    memory.py                 MemorySnapshot/MemorySampler/PhaseMemoryTracker (3 independent memory sources)
    markers.py                start_capture/stop_capture (cudaProfilerStart/Stop) for nsys capture-range

  runners/                    Nsight CLI wrappers (one subprocess launcher per collection stage)
    base.py                   shared runner scaffolding
    baseline.py               untainted wall-clock pass (the only "trustworthy" timing)
    nsys_runner.py            NsysRunner — builds `nsys profile` command lines
    ncu_runner.py             NcuRunner/NcuCollection — two-tier `ncu` invocation strategy
    calibration_runner.py     runs calibration.py's known-byte-count check as a subprocess and runs a sweep to establish DRAM/L2/Compute ceilings

  parsers/                    Nsight output → Python dataclasses
    nsys_parse.py             parse_sqlite() — direct SQL over nsys-exported SQLite (CUPTI/NVTX joins)
    ncu_parse.py               parse_csv() — parses `ncu --csv --page raw`, handles units row/unavailable counters

  analysis/                   derived metrics, cross-checks, comparisons
    assemble.py               assemble() — joins calibration+baseline+nsys+ncu into RunAnalysis, cross-checks
    derive.py                 summarize()/HierarchySummary — raw counters → bytes/bandwidth/hit-rate/AI
    deep_dive.py              tier-2 per-kernel stall-reason analysis
    compare.py                cross-run comparison matrices

  report/                     exporters
    html.py / markdown.py / compare_report.py   self-contained report renderers
    charts.py                 inline SVG chart primitives (hbar_panel, line_chart, roofline_chart)
    format_utils.py           shared number/byte formatting

configs/                      YAML/JSON inputs (NOT code)
  model-registry.yaml         model catalogue consumed by registry.py
  models/                     per-model config files
  profiles/{quick,standard,deep}.yaml    ncu tier depth / kernel caps
  workloads/{balanced,prefill-focused,decode-focused,long-context,layer-attribution}.yaml
  platform_profile.json       cached `nsbench preflight` output

scripts/                      outer-loop bash wrappers around the installed `nsbench` binary
  run_bench.sh, run_sweep.sh, smoke_test.sh, fetch_status.sh

setup/                        environment bootstrap (two separate venvs — trtllm pins its own torch)
  create_env.sh, setup_trtllm_env.sh, requirements.txt

docs/                         platform notes, metric reference, methodology, usage, suite design
download_models.py            standalone HF-download helper (root-level, alongside registry.py's fetch())
pyproject.toml                setuptools build, console_scripts → nsight_bench.cli:main
```

**Categorization against the requested buckets:**
- **Core Benchmarking Orchestration** → `orchestrator.py`, `worker.py`, `suite.py`, `cli.py`, `config.py`, `platform.py`
- **Nsight CLI/API Wrappers** → `runners/nsys_runner.py`, `runners/ncu_runner.py`, `parsers/nsys_parse.py`, `parsers/ncu_parse.py`, `instrumentation/nvtx.py`, `instrumentation/markers.py`
- **Target GPU Workloads** → `backends/` (execution engine) + `workloads/` (test definitions) — note this is real transformer inference, not synthetic CUDA kernels
- **Data Exporters/Parsers/Reports** → `analysis/*`, `report/*`, `metrics.py`, `stats.py`
- **Config/Sweep** → `configs/*.yaml`, `config.py` dataclasses
- **Scripts** → `scripts/*.sh`, `setup/*.sh`
- **Docs** → `docs/*.md`

**Language/build story:** 100% Python, one setuptools package, no compiled artifacts. The "kernels under test" come from PyTorch/transformers, not from this repo. Interconnection: `cli.py → orchestrator.py → subprocess(python -m nsight_bench.worker) → subprocess(nsys/ncu CLI) → parsers → analysis → report`. Bash scripts are an outer loop over the installed `nsbench` binary across configs.

---

## 2. Benchmark Run Lifecycle & Flow

1. **CLI entry**: `cli.py:main()` → `build_parser()` (subcommands `preflight/discover/calibrate/run/report/compare/models/fetch/suite`). `nsbench run` → `cmd_run()` builds a `RunConfig` via `_build_run_config()`.
2. <mark>**Orchestration**: </mark>`orchestrator.py:BenchmarkOrchestrator.run()` runs **four sequential subprocess stages**, deliberately isolated so one profiler's CUDA-context perturbation never leaks into the next: `CalibrationRunner → BaselineRunner → NsysRunner → NcuRunner`. Results fold into a `RunRecord` manifest (`manifest.json`).
3. **Worker/workload selection**: each stage shells out to `python -m nsight_bench.worker`. `worker.py:main()` loads `RunConfig`, resolves the backend (`backends/base.py:get_backend()`), builds the workload (`workloads/base.py:build_workload()`). Mode (`baseline|nsys|ncu`) drives `ProfileMode`-based NVTX scoping.
4. **Kernel execution**: `TextGenerationWorkload.run()` → `prepare_inputs` → `_warmup()` (discards N iters) → opens `markers.start_capture()/stop_capture()` (nsys-only; **this is the `cudaProfilerStart`/`cudaProfilerStop` hook**, `instrumentation/markers.py`) → `_run_once()` → `backend.prefill()` / `backend.decode_step()`.
5. **Profiling capture** (where Nsight is actually invoked):
   - **nsys**: `runners/nsys_runner.py:NsysRunner.build_command()` builds `nsys profile --force-overwrite=true -o <stem> --trace=<cfg.trace> --cuda-memory-usage=... --gpu-metrics-set=<chip-mapped> --capture-range=cudaProfilerApi ...` — tracing starts only at the workload's `cudaProfilerStart` call post-warmup. Output exported via `nsys export --type sqlite`, parsed by `parsers/nsys_parse.py:parse_sqlite()` (raw SQL joining CUPTI kernel-activity correlation IDs to NVTX ranges).
   - **ncu**: `runners/ncu_runner.py`. Two-tier strategy: tier 1 = ~45-metric sweep over every kernel scoped via `--nvtx-include <range>/`, optional and off by default (`ncu.tiers`); tier 2 = deep-dive top-N kernels via `--kernel-name regex:...` plus `--section` args, ranked from the nsys timeline (`rank_kernels_from_nsys()`) or from the tier-1 CSV (`rank_kernels()`) per `ncu.rank_source`. Output exported with `ncu --import ... --csv --page raw`, parsed by `parsers/ncu_parse.py:parse_csv()`.
   - **NVTX contract**: `instrumentation/nvtx.py`'s `Phase` enum defines the range names (`nsbench.prefill`, `nsbench.decode_step`, `nsbench.layer.<i>`) both profilers scope against — in NCU mode only the *last* decode step gets the scoped name so ncu profiles exactly one step.
6. **Report generation**: `cli.py:_emit_reports()` → `analysis/assemble.py:assemble()` merges manifest + baseline/nsys/ncu artifacts into `RunAnalysis` → `report/markdown.py` / `report/html.py` render self-contained output.

**Device synchronization (timing-contamination guard)** — `backends/base.py:_PhaseTimer` (used by `Backend.measured_phase()`):
```python
def __enter__(self):
    self.backend.synchronize()        # drain queue BEFORE starting the clock
    self.timing.start = time.perf_counter()
def __exit__(self, *_exc):
    self.backend.synchronize()        # drain queue BEFORE stopping the clock
    self.timing.end = time.perf_counter()
```
`Backend.synchronize()` calls `torch.cuda.synchronize()`. Used around prefill (`text_generation.py:154`) and decode (`:168`); warmup also ends with an explicit sync before entering the measured region. Token-id materialization (a D2H copy/implicit sync) is deliberately placed *after* the timer closes so it doesn't inflate the measured window.

**Timing-contamination caveats to know:**
- Only the **baseline** pass's wall-clock numbers are trustworthy — ncu "serialises kernel launches and replays each one several times" (documented in `runners/baseline.py`'s own docstring). nsys/ncu-mode timings must never be reported as real performance.
- Under ncu, memory sampling is disabled (`--memory-sample-hz 0`) because a sampled footprint under kernel replay "would describe the replay, not the run."
- Nothing in `PhaseTiming` itself flags which pass produced it — separation into distinct `baseline`/`nsys`/`ncu` manifest sections is the only safeguard against misreading a profiled-run timing as real.

---

## 3. Internal API Contracts & Interfaces

**`nsight_bench/workloads/base.py`** — the workload interface:
```python
class Workload(ABC):
    kind: str = "base"
    def __init__(self, config: WorkloadConfig) -> None: ...
    @abstractmethod
    def run(self, backend: Backend, mode: ProfileMode = ProfileMode.BASELINE,
             sampler: MemorySampler | None = None) -> WorkloadResult: ...
    def describe(self) -> dict: ...   # optional override
```
- `WorkloadResult` (dataclass): `mode`, `timings: list[PhaseTiming]`, `memory_deltas: list[PhaseMemoryDelta]`, `prompt_tokens`, `generated_tokens`, `batch_size`, `kv_cache_bytes`, `profiled_step_context_len`, `layers_annotated`, `generated_token_ids`, `notes`; helpers `phase()`, `phases()`, `median_seconds()`, `iqr_seconds()`, `to_dict()`.
- `ProfileMode` enum: `BASELINE`, `NSYS`, `NCU` — `run()` must branch on this for NVTX-range placement.
- Registry primitives (same file): module-level `_REGISTRY: dict[str, type[Workload]]`, `@register` class decorator, `get_workload(kind)`, `build_workload(config)`, `available_workloads()`.

**Concrete examples**: `workloads/text_generation.py:TextGenerationWorkload` (`kind="text-generation"`, the full reference implementation with warmup/capture/timing/memory/KV-cache tracking) and `workloads/multimodal.py:MultimodalWorkload(TextGenerationWorkload)` (`kind="multimodal"`, subclasses to reuse phase/NVTX/memory plumbing, falls back to parent `run()` when no media requested, raises `NotImplementedError` at documented extension points `_build_multimodal_inputs`/`_encode_media` otherwise).

<mark>**Boilerplate required to add a new workload:**</mark>
1. Subclass `Workload`, set a unique `kind: str` (used as registry key and YAML `kind:` field).
2. Decorate with `@register`.
3. Implement `run()`, returning a populated `WorkloadResult`.
4. Honor `ProfileMode` inside `run()` — place NVTX ranges (`nvtx_range`/`phase`/`mark`) so NCU sees exactly one scoped range and NSYS gets a `markers.start_capture()/stop_capture()` window; discard warmup before opening the capture range.
5. Track memory per phase via `PhaseMemoryTracker(name, sampler)`, append to `result.memory_deltas`.
6. Time phases via `backend.measured_phase(name, tokens=...)`, append to `result.timings`.
7. Optionally override `describe()`.
8. Add new knobs to the shared `WorkloadConfig` dataclass (`config.py`) rather than a bespoke config class — the loader ignores unknown keys.
9. **Critical, easy to miss**: import the new module for its side effect in `worker.py` (currently `from .workloads import multimodal, text_generation  # noqa: F401`). There is **no directory scan or entry_points discovery** — skipping this import means `build_workload()` raises `KeyError: Unknown workload '<kind>'`.

**Config/sweep schema** — `WorkloadConfig` (`config.py`): `name`, `kind`, `prompt_tokens`, `generate_tokens`, `batch_size`, `warmup_iters`, `repeat`, `prompt_source`, `do_sample`, `temperature`, `seed`, `annotate_layers`, plus multimodal placeholders. A "sweep" is not a cartesian-parameter dataclass — it's one YAML file per sweep point (e.g. `configs/workloads/balanced.yaml`, `decode-focused.yaml`, `long-context.yaml`, `prefill-focused.yaml`, `layer-attribution.yaml`), driven by `scripts/run_sweep.sh` calling the CLI once per file. `suite.py:SuiteRunner` sweeps across **models** for one fixed `WorkloadConfig`, not across workload parameters. `RunConfig` composes `ModelConfig + WorkloadConfig + ProfileConfig` and computes a `fingerprint()` (sha256) for dedup/comparison.

**Validation/cross-check hooks a new workload should respect:**
- `analysis/assemble.py:_cross_check()` compares nsys kernel counts per phase against ncu's scoped kernel count and checks expected-vs-measured decode DRAM traffic against `ModelConfig.active_weight_bytes()`/`kv_cache_bytes()`.
- `metrics.py` enforces a hierarchy vocabulary (`Level` enum: `register, local, shared, l1tex, l2, dram, host`) and the rule that missing counters propagate as `None`, never `0.0` (a workload/backend must not silently invent zero traffic) — enforced in `derive.py`'s `LevelSummary.verdict()`.

---

## 4. The Extension Playbook

### (a) Adding a new benchmark workload
1. Create `nsight_bench/workloads/<name>.py`, subclass `Workload` (or `TextGenerationWorkload` if it's a variant of generation), set `kind`, decorate `@register`.
2. Implement `run()` following the pattern in `text_generation.py` — warmup/sync → NVTX-scoped capture region → `backend.measured_phase()` timing → `PhaseMemoryTracker` for memory → return `WorkloadResult`.
3. Add the import line to `worker.py`'s `from .workloads import ...` block (mandatory — this is the entire discovery mechanism).
4. Extend `WorkloadConfig` in `config.py` if new knobs are needed (unknown-key-tolerant loader).
5. Add one or more YAML files under `configs/workloads/` with `kind: <name>` and the new fields.
6. If the workload needs new derived metrics or cross-checks, extend `analysis/derive.py`/`assemble.py` — respect the `Level` hierarchy vocabulary in `metrics.py` and the "`None` not `0.0`" rule for missing counters.
7. Run via `nsbench run --workload configs/workloads/<name>.yaml ...`; verify via `nsbench report`.

### (b) Tracking a new hardware metric from Nsight
1. Add the metric name to the appropriate tier list in `metrics.py` (`tier1_metrics_arg` for the per-kernel metric list, collected only when tier 1 is on, or a new `sections_for_tier(tier)` entry for a deeper ncu `--section`).
2. If it's an nsys GPU-metric-set metric, check `runners/nsys_runner.py`'s chip→`--gpu-metrics-set` mapping — GB10 (this platform) exposes no `dram__*` counters, so verify availability via `platform.py:PlatformProfile.detect()` first (`nsbench preflight`).
3. Update `parsers/ncu_parse.py:parse_csv()` (or `parsers/nsys_parse.py:parse_sqlite()` for trace-based metrics) if the new metric needs special unit handling (thousands separators, "unavailable counter" vs real zero, units row).
4. Feed the parsed value into `analysis/derive.py:summarize()` under the correct `Level`, and into `HierarchySummary`/`LevelSummary` roll-up logic.
5. Surface it in `report/charts.py` / `report/markdown.py` / `report/html.py` if it should appear in generated reports.
6. Consider whether `calibration.py` needs a new sanity check (it currently validates DRAM-byte derivation against a known streaming-array byte count) if the new metric feeds a derived quantity people will trust.

---

## 5. Architectural Friction Points

- **Discovery is manual, not automatic**: workload registration requires editing `worker.py`'s import list by hand. No `__init__.py` auto-import, no entry_points, no directory scan. Easy to forget, and produces a runtime `KeyError` rather than an import-time or static error. If you add many workloads, consider (later, not now) a package-scan or explicit `__all__`-driven registration, but this is out of scope for a read-only review.
- **Sweeps are file-multiplication, not parameterization**: parameter sweeps are done via one YAML file per sweep point plus a bash loop (`scripts/run_sweep.sh`), rather than a declarative cartesian-sweep spec. `SuiteRunner` only sweeps across models for one fixed `WorkloadConfig`. Adding a true parameter-grid sweep (e.g. sweep `batch_size` × `generate_tokens`) means hand-authoring N YAML files or writing your own wrapper script — there's no first-class grid abstraction to build on.
- **`trtllm.py` backend is a stub**: documented as blocked on missing aarch64 wheels and loss of per-token NVTX scoping under CUDA graphs. Any future backend work needs to solve the "NVTX ranges vs. CUDA graph capture" tension the stub's docstring flags — this is a real, not cosmetic, technical blocker.
- **Timing trustworthiness is convention, not type-enforced**: `PhaseTiming` objects from baseline/nsys/ncu passes are structurally identical; only their placement in separate manifest sections (`baseline` vs `nsys` vs `ncu`) signals which is real-world-accurate. Nothing stops new code from accidentally reading nsys/ncu-mode timings as performance numbers — a `trusted: bool` field or distinct return type would be safer if this framework grows more consumers of `WorkloadResult`.
- **Platform-specific counter gaps are hardcoded knowledge**: GB10 exposing no `dram__*` counters (forcing DRAM bytes to be derived from `sysmem_lookup_miss_sectors * 32`) is captured in `metrics.py`/`platform.py`/docs, but a new GPU target would need someone to manually discover and hardcode its equivalent quirks — there's no automated counter-capability probing beyond what `nsbench preflight` already checks.
- **Tier 2's kernel list comes from a ranking**: by default the nsys timeline (`rank_kernels_from_nsys()`, needs the nsys pass), otherwise tier 1's export (`rank_kernels()`); each falls back to the other. A new workload with very different kernel-count/shape characteristics (e.g. many more distinct kernels) could produce regexes that are too broad or exceed command-line length limits — worth watching if you add workloads with very different kernel profiles than transformer decode/prefill.
- **Separate venvs for trtllm vs. main environment** (`setup/create_env.sh` vs `setup_trtllm_env.sh`) reflect a real dependency conflict (trtllm pins its own torch) rather than an oversight — anticipate this friction if you ever want a single unified environment.
