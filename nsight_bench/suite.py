"""Run the benchmark across several models and compare them.

A suite is not just a loop over ``nsbench run``. Three things separate them, and each one is
a lesson from what goes wrong at fifteen hours and six models in.

**It must survive a failure without losing the models that already worked.** A 70 GB
checkpoint that OOMs, a driver hiccup, a Ctrl-C at hour nine -- any of these can end a run,
and none of them should cost the five models already profiled. Every model's outcome is
appended to a ledger on disk as soon as it finishes, and ``--resume`` reads that ledger and
skips what is already done. The comparison is rebuilt from whatever succeeded, so a partial
suite still produces a document.

**Order is chosen, not incidental.** Models run smallest first. A configuration mistake --
a bad profile, a missing permission, a wrong path -- surfaces in the ten-minute model rather
than after two hours of the largest one. It also means the ledger has useful content early.

**The comparison is the deliverable.** Individual run reports are inputs to it. The suite
always regenerates the cross-run document at the end, even when some models failed, and
records which ones are missing so the document is never quietly incomplete.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig, WorkloadConfig, discover_model
from .orchestrator import BenchmarkOrchestrator
from .platform import PlatformProfile
from .registry import Registry, RegistryEntry, fetch, snapshot_path, store_env


def _flushing_log(*args) -> None:
    """Print with an immediate flush, and a UTC timestamp on section lines.

    The timestamp is what makes a finished log forensically useful: "model 4 took nine
    hours" is only visible if each line carries when it happened.
    """
    import sys as _sys

    text = " ".join(str(a) for a in args)
    if text.startswith("["):
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        text = f"{stamp}  {text}"
    print(text, flush=True, file=_sys.stdout)


@dataclass
class ModelOutcome:
    """What happened to one model in the suite."""

    key: str
    label: str
    repo: str
    status: str = "pending"          # pending | ok | failed | skipped
    run_id: str = ""
    run_dir: str = ""
    seconds: float = 0.0
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuiteLedger:
    """Append-only record of a suite's progress, written after every model.

    Held on disk rather than in memory because its whole purpose is to outlive the process
    that created it.
    """

    path: Path
    suite_id: str = ""
    started_at: str = ""
    profile: str = ""
    workload: str = ""
    outcomes: dict[str, ModelOutcome] = field(default_factory=dict)

    @classmethod
    def load_or_create(cls, path: Path, suite_id: str, profile: str, workload: str) -> SuiteLedger:
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                ledger = cls(
                    path=path,
                    suite_id=raw.get("suite_id", suite_id),
                    started_at=raw.get("started_at", ""),
                    profile=raw.get("profile", profile),
                    workload=raw.get("workload", workload),
                )
                for key, data in (raw.get("outcomes") or {}).items():
                    ledger.outcomes[key] = ModelOutcome(**data)
                return ledger
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        return cls(
            path=path, suite_id=suite_id,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            profile=profile, workload=workload,
        )

    def record(self, outcome: ModelOutcome) -> None:
        self.outcomes[outcome.key] = outcome
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "suite_id": self.suite_id,
            "started_at": self.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": self.profile,
            "workload": self.workload,
            "outcomes": {k: v.to_dict() for k, v in self.outcomes.items()},
        }, indent=2) + "\n", encoding="utf-8")

    def completed(self) -> set[str]:
        return {k for k, o in self.outcomes.items() if o.status == "ok"}

    def successful_dirs(self) -> list[str]:
        return [o.run_dir for o in self.outcomes.values() if o.status == "ok" and o.run_dir]


class SuiteRunner:
    """Executes a multi-model benchmark suite."""

    def __init__(
        self,
        registry: Registry,
        entries: list[RegistryEntry],
        profile: PlatformProfile,
        workload: WorkloadConfig,
        profile_config_path: str | Path | None = None,
        output_root: str | Path = "runs",
        python_executable: str | None = None,
        tag: str = "",
        prefetch: bool = True,
        max_kernels: int | None = None,
        log=None,
    ) -> None:
        self.registry = registry
        self.profile = profile
        self.workload = workload
        self.profile_config_path = profile_config_path
        self.output_root = Path(output_root)
        self.python = python_executable
        self.tag = tag or "suite"
        self.prefetch = prefetch
        self.max_kernels = max_kernels
        # Flushed on every line. A suite runs for many hours, almost always redirected to a
        # file rather than a terminal -- and Python block-buffers a non-tty stream, so the
        # default `print` leaves the log empty for hours at a time. Progress you cannot see
        # is indistinguishable from a hang, which is the one thing an operator watching an
        # overnight job must be able to rule out.
        self.log = log if log is not None else _flushing_log
        # Smallest first: a misconfiguration should surface in minutes, not after the
        # largest checkpoint has been loaded and profiled.
        self.entries = sorted(entries, key=lambda e: e.params or 0)
        self._prefetch_thread = None
        self._prefetch_stop = None

    # ---- prefetching ------------------------------------------------------------------

    def _start_prefetch(self, pending: list[RegistryEntry]) -> None:
        """Download upcoming models in the background while the current one is profiled.

        Fetching and profiling are both long and use almost disjoint resources -- one is
        network and page cache, the other is GPU -- so running them in series wastes roughly
        the shorter of the two. On this machine that is not a rounding error: at the observed
        ~5 MB/s the seven checkpoints take about as long to fetch as to profile, so
        overlapping them removes something like a third of the total wall clock.

        The contention this introduces is small enough to accept and worth naming. A 5 MB/s
        transfer writing into page cache is four to five orders of magnitude below the
        240 GB/s the memory system sustains, so it cannot meaningfully move a bandwidth
        measurement. What it can touch is the unprofiled baseline, which is wall-clock timing
        on a shared box -- so anyone who wants the timing figures pristine should pass
        ``--no-prefetch`` and accept the longer run.
        """
        import threading

        if not self.prefetch or not pending:
            return

        self._prefetch_stop = threading.Event()
        stop = self._prefetch_stop

        def worker() -> None:
            for entry in pending:
                if stop.is_set():
                    return
                if snapshot_path(entry, self.registry.store) is not None:
                    continue
                try:
                    fetch(entry, self.registry.store, log=lambda *_: None)
                except Exception:                                # noqa: BLE001
                    # A prefetch failure is not fatal: the model's own turn will retry the
                    # download in the foreground, where the error is reported properly.
                    continue

        self._prefetch_thread = threading.Thread(
            target=worker, name="nsbench-prefetch", daemon=True
        )
        self._prefetch_thread.start()
        self.log(f"    prefetching {len(pending)} upcoming model(s) in the background")

    def _stop_prefetch(self) -> None:
        if self._prefetch_stop is not None:
            self._prefetch_stop.set()

    # ---- fetching ---------------------------------------------------------------------

    def ensure_weights(self, entry: RegistryEntry) -> Path:
        """Return the local snapshot, downloading it if the store does not have it."""
        path = snapshot_path(entry, self.registry.store)
        if path is not None:
            return path
        fetch(entry, self.registry.store, log=self.log)
        path = snapshot_path(entry, self.registry.store)
        if path is None:
            raise FileNotFoundError(
                f"{entry.repo} downloaded but no usable snapshot found in "
                f"{self.registry.store}. The transfer may have been interrupted; "
                f"re-run 'nsbench fetch --models {entry.key}'."
            )
        return path

    # ---- one model --------------------------------------------------------------------

    def run_one(self, entry: RegistryEntry) -> ModelOutcome:
        outcome = ModelOutcome(
            key=entry.key, label=entry.display, repo=entry.repo,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        started = time.time()
        try:
            path = self.ensure_weights(entry)

            model = discover_model(path, name=entry.display)
            model.dtype = entry.dtype
            model.attn_implementation = entry.attn_implementation
            if not model.benchmarkable:
                raise RuntimeError(
                    f"no usable weights found in {path} -- the snapshot has config files "
                    "but no .safetensors shards"
                )

            run_config = RunConfig(
                model=model,
                workload=self.workload,
                backend="hf",
                output_root=str(self.output_root),
                tag=self.tag,
            )
            if self.profile_config_path:
                from .config import ProfileConfig
                run_config.profile = ProfileConfig.load(self.profile_config_path)
            if self.max_kernels:
                run_config.profile.ncu.max_kernels = self.max_kernels
                run_config.profile.ncu.auto_launch_cap = False

            self.log(f"    {entry.display}: {model.num_layers} layers, "
                     f"{model.weight_bytes_on_disk / 1e9:,.1f} GB"
                     + (f", MoE {model.num_experts}x top-{model.num_experts_per_token}"
                        if model.is_moe else ", dense"))

            record = BenchmarkOrchestrator(
                run_config, self.profile,
                python_executable=self.python, output_root=self.output_root,
            ).run()

            outcome.run_id = record.run_id
            outcome.run_dir = record.root
            outcome.warnings = list(record.warnings)
            outcome.status = "ok" if record.stages_ok.get("ncu") or record.stages_ok.get(
                "baseline") else "failed"
            if outcome.status == "failed":
                outcome.error = "no collection stage produced usable data"

            # Reports are written per model so a failure later in the suite still leaves
            # every finished model individually readable.
            self._write_reports(Path(record.root))

        except KeyboardInterrupt:
            outcome.status = "failed"
            outcome.error = "interrupted"
            raise
        except Exception as exc:                                 # noqa: BLE001
            outcome.status = "failed"
            outcome.error = f"{type(exc).__name__}: {exc}"[:600]
        finally:
            outcome.seconds = time.time() - started
            outcome.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return outcome

    def _write_reports(self, run_dir: Path) -> None:
        from .analysis.assemble import assemble, write_metric_tables
        from .report.html import write_html_report
        from .report.markdown import write_markdown_report

        try:
            analysis = assemble(run_dir)
            write_metric_tables(analysis)
            write_markdown_report(analysis)
            write_html_report(analysis)
        except Exception as exc:                                 # noqa: BLE001
            self.log(f"    report generation failed for {run_dir.name}: {exc}")

    # ---- the suite --------------------------------------------------------------------

    def run(self, resume: bool = True, comparison_dir: str | Path | None = None) -> SuiteLedger:
        suite_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}__{self.tag}"
        ledger_path = self.output_root / f"_suite_{self.tag}" / "ledger.json"
        ledger = SuiteLedger.load_or_create(
            ledger_path, suite_id,
            profile=str(self.profile_config_path or "default"),
            workload=self.workload.name,
        )

        for key, value in store_env(self.registry.store).items():
            os.environ[key] = value

        done = ledger.completed() if resume else set()
        total = len(self.entries)

        self.log(f"\nSuite '{self.tag}': {total} models, ledger at {ledger_path}")
        if done:
            self.log(f"Resuming -- {len(done)} already complete: {', '.join(sorted(done))}")

        # Everything after the first model still to run can be fetched while that first one
        # is being profiled.
        todo = [e for e in self.entries if e.key not in done]
        self._start_prefetch(todo[1:])

        try:
            self._run_models(ledger, done, total)
        finally:
            self._stop_prefetch()

        self._compare(ledger, comparison_dir)
        return ledger

    def _run_models(self, ledger: SuiteLedger, done: set[str], total: int) -> None:
        for index, entry in enumerate(self.entries, 1):
            if entry.key in done:
                self.log(f"\n[{index}/{total}] {entry.display} -- already done, skipping")
                continue

            self.log(f"\n[{index}/{total}] {entry.display}  ({entry.repo})")
            outcome = self.run_one(entry)
            ledger.record(outcome)

            mins = outcome.seconds / 60
            if outcome.status == "ok":
                self.log(f"    done in {mins:,.1f} min -> {outcome.run_dir}")
            else:
                self.log(f"    FAILED after {mins:,.1f} min: {outcome.error}")
                self.log("    continuing with the next model; "
                         "re-run with --resume to retry this one")

    def _compare(self, ledger: SuiteLedger, comparison_dir: str | Path | None) -> None:
        """Build the cross-run document from whatever succeeded."""
        from .analysis.compare import build_comparison
        from .report.compare_report import write_comparison_reports

        dirs = ledger.successful_dirs()
        if not dirs:
            self.log("\nNo model completed successfully; no comparison to build.")
            return

        out_dir = Path(comparison_dir or self.output_root / f"_comparison_{self.tag}")
        comparison = build_comparison(dirs)

        missing = [o.label for o in ledger.outcomes.values() if o.status != "ok"]
        if missing:
            comparison.warnings.append(
                "This comparison is incomplete: "
                + ", ".join(missing)
                + " did not produce usable data, so the table below is missing them. "
                  "Re-run with --resume to retry."
            )

        md, html, csv = write_comparison_reports(comparison, out_dir)
        self.log(f"\nComparison across {len(dirs)} models:")
        self.log(f"  {md}")
        self.log(f"  {html}")
        self.log(f"  {csv}")
