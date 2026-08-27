"""Nsight Compute collection: the per-kernel memory hierarchy.

This is where the memory numbers actually come from. Nsight Systems can say a kernel took
1.4 ms; only Nsight Compute can say it moved 268 MB past L2 and hit in L2 five percent of
the time.

The cost problem, and how it is handled
---------------------------------------
Nsight Compute replays every profiled kernel several times -- once per pass needed to cover
the requested counters. A full section set is thousands of metrics and dozens of passes. Run
that over a whole generation loop and the collection takes hours to days.

Two things keep it tractable, and both are load-bearing:

1. **NVTX scoping.** Collection is filtered to one prefill and one decode step. A decode
   step's kernel mix does not change from token to token, so one step yields one instance of
   every unique kernel -- full coverage at 1/N the cost.
2. **Tiering.** Tier 1 collects a curated ~30-metric memory list over every kernel in scope,
   which is cheap. Tier 2 then spends the expensive full-section budget on only the handful
   of kernels that tier 1 showed actually matter.

Tier 2 selects its kernels from *tier 1's own export*, reduced to the base identifiers
ncu's ``--kernel-name`` filter actually matches on. Two naming mismatches make this fiddly and
both bite silently, since ncu reports either as the generic "No kernels were profiled":

* nsys and ncu demangle differently, so a regex built from the Nsight Systems timeline will
  not match what ncu matches against.
* ncu's own CSV export and its own filter disagree: the export carries the full demangled
  signature while, under ``--kernel-name-base function``, the filter wants the bare
  identifier. :func:`base_kernel_name` bridges that.

Prefill and decode are collected separately, in their own processes. That costs a second
model load, but it is the only way to be certain a metric belongs to the phase it is filed
under, and phase attribution is the primary axis of every report here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import metrics as M
from ..config import RunConfig
from ..gpu_state import GpuStateRecorder
from ..instrumentation.nvtx import ncu_filter
from ..parsers.ncu_parse import base_identifier, parse_csv
from ..platform import PlatformProfile
from .base import CommandResult, RunPaths, run_command

#: NVTX ranges collection is scoped to, and the phase each maps to in the reports.
SCOPES: tuple[tuple[str, str], ...] = (
    ("prefill", "nsbench.prefill"),
    ("decode_step", "nsbench.decode_step"),
)


@dataclass
class NcuCollection:
    """One ncu invocation's artefacts."""

    scope: str
    tier: int
    report_path: Path | None = None
    csv_path: Path | None = None
    command: CommandResult | None = None
    kernel_count: int = 0
    launch_cap: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.csv_path and self.csv_path.exists() and not self.error)

    @property
    def truncated(self) -> bool:
        """Whether ncu stopped at the launch cap rather than at the end of the range.

        Hitting the cap exactly is the signature. It matters because ``--launch-count`` does
        not sample -- it stops -- so a truncated collection's totals are a prefix of the
        phase's traffic, not a scaled-down version of it, and summing them understates
        everything.
        """
        return bool(self.launch_cap and self.kernel_count >= self.launch_cap)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "tier": self.tier,
            "report_path": str(self.report_path) if self.report_path else None,
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "kernel_count": self.kernel_count,
            "launch_cap": self.launch_cap,
            "truncated": self.truncated,
            "ok": self.ok,
            "error": self.error or None,
            "command": self.command.to_dict() if self.command else None,
        }


class NcuRunner:
    """Runs the tiered Nsight Compute collections for one benchmark."""

    name = "ncu"

    def __init__(
        self,
        run_config: RunConfig,
        paths: RunPaths,
        profile: PlatformProfile,
        python_executable: str | None = None,
    ) -> None:
        self.run_config = run_config
        self.paths = paths
        self.profile = profile
        self.python = python_executable or sys.executable
        self.repo_root = Path(__file__).resolve().parents[2]

    # ---- command construction ---------------------------------------------------------

    def launch_cap(self) -> int:
        """Tier-1 ``--launch-count``, scaled to what this model's decode step will emit.

        The configured cap is a runaway guard, not a sampling knob: ncu *stops* at it, so a
        cap below the real kernel count leaves totals that are a prefix of the phase rather
        than a scaled-down version, and the harness has to discard the physics check for that
        run. A single fixed number cannot serve both a 28-layer dense model and a 52-layer
        mixture of experts.

        The estimate comes from the one thing that reliably predicts launch count: depth. An
        eager transformers block issues roughly 55-60 launches once normalisation, rotary
        embedding and elementwise ops are counted -- measured at 1,618 launches over 28
        layers on this harness. MoE layers issue more, because routing, gather and per-expert
        GEMMs are extra launches per block, so they carry a multiplier.

        The configured value is treated as a floor, never a ceiling: raising
        ``ncu.max_kernels`` still works, but leaving it at the default no longer silently
        truncates a deep model.
        """
        cfg = self.run_config.profile.ncu
        model = self.run_config.model
        layers = model.num_layers
        if not cfg.auto_launch_cap or not layers:
            return cfg.max_kernels

        per_layer = 60.0
        if model.is_moe:
            # Routing, permutation and per-expert GEMMs roughly double a block's launches.
            per_layer *= 2.0
        # 1.6x headroom over the estimate, plus the non-layer kernels (embedding, final norm,
        # LM head, sampling) that do not scale with depth.
        estimate = int(layers * per_layer * 1.6) + 200
        return max(cfg.max_kernels, estimate)

    def _common_flags(self, nvtx_range: str, report_stem: Path) -> list[str]:
        cfg = self.run_config.profile.ncu
        return [
            "ncu",
            "--target-processes", "all",
            "--nvtx",
            # ncu_filter appends the trailing '/' that marks this as a push/pop range.
            # Without it the filter matches only start/end ranges and silently selects
            # nothing -- see instrumentation/nvtx.py.
            "--nvtx-include", ncu_filter(nvtx_range),
            # Kernels are identified by demangled base function name in both tiers, so the
            # names tier 1 reports can be fed straight back into tier 2's filter.
            "--kernel-name-base", "function",
            "--replay-mode", cfg.replay_mode,
            "--cache-control", cfg.cache_control,
            "--clock-control", cfg.clock_control,
            "--launch-count", str(self.launch_cap()),
            "--print-summary", "none",
            "--force-overwrite",
            "-o", str(report_stem),
        ]

    def _worker_argv(self, config_path: Path, result_path: Path) -> list[str]:
        return [
            self.python, "-m", "nsight_bench.worker",
            "--config", str(config_path),
            "--mode", "ncu",
            "--out", str(result_path),
            # Memory sampling is disabled under ncu: the profiler serialises and replays
            # kernels, so a sampled footprint timeline would describe the replay, not the run.
            "--memory-sample-hz", "0",
        ]

    def build_tier1_command(
        self, config_path: Path, result_path: Path, nvtx_range: str, report_stem: Path
    ) -> list[str]:
        available = set(self.profile.metric_available) or None
        metrics_arg = M.tier1_metrics_arg(available)
        return (
            self._common_flags(nvtx_range, report_stem)
            + ["--metrics", metrics_arg]
            + self._worker_argv(config_path, result_path)
        )

    def tier_n_launch_cap(self, tier: int) -> int:
        """How many launches tier 2+ should profile.

        Deliberately far below the tier-1 cap. ncu's filter matches base identifiers, which
        are coarser than the ranking that produced them -- several template instantiations
        share one name -- so a list of six ranked kernels can select hundreds of launches. A
        full-section collection over hundreds of launches costs as much as tier 1 did and
        adds nothing: instantiations of the same kernel behave alike, so a handful of
        instances of each is representative.

        Unlike the tier-1 cap this is sampling, not truncation, because tier-2 metrics are
        read per kernel and never summed into the phase totals.
        """
        cfg = self.run_config.profile.ncu
        return max(48, cfg.top_n_kernels * 12)

    def build_tier_n_command(
        self,
        config_path: Path,
        result_path: Path,
        nvtx_range: str,
        report_stem: Path,
        tier: int,
        kernel_names: list[str],
    ) -> list[str]:
        argv = self._common_flags(nvtx_range, report_stem)
        cap = str(self.tier_n_launch_cap(tier))
        argv[argv.index("--launch-count") + 1] = cap
        for section in M.sections_for_tier(tier):
            argv += ["--section", section]
        if tier >= 3:
            argv += ["--import-source", "yes"]
        if kernel_names:
            argv += ["--kernel-name", f"regex:{_kernel_regex(kernel_names)}"]
        return argv + self._worker_argv(config_path, result_path)

    # ---- execution --------------------------------------------------------------------

    def run(self, config_path: Path) -> dict:
        cfg = self.run_config.profile.ncu
        collections: list[NcuCollection] = []
        gpu_windows: dict[str, dict] = {}

        for scope, nvtx_range in SCOPES:
            with GpuStateRecorder() as gpu:
                tier1 = self._collect_tier1(config_path, scope, nvtx_range)
            collections.append(tier1)
            gpu_windows[scope] = gpu.window.to_dict() if gpu.window else {}

            if not tier1.ok:
                continue

            top_kernels = self.rank_kernels(tier1.csv_path, cfg.top_n_kernels)
            for tier in sorted(t for t in cfg.tiers if t >= 2):
                if not top_kernels:
                    collections.append(NcuCollection(
                        scope=scope, tier=tier,
                        error="tier 1 produced no kernels to rank; nothing to deep-dive",
                    ))
                    continue
                collections.append(
                    self._collect_tier_n(config_path, scope, nvtx_range, tier, top_kernels)
                )

        truncated = [c for c in collections if c.truncated]
        warnings = [
            f"ncu {c.scope} tier {c.tier} stopped at the {c.launch_cap}-launch cap. "
            "--launch-count halts collection rather than sampling, so this phase's totals "
            "cover only the first {cap} kernels and understate its real traffic. Raise "
            "ncu.max_kernels (--max-kernels) and re-run.".replace("{cap}", str(c.launch_cap))
            for c in truncated
        ]

        return {
            "runner": self.name,
            "scopes": [s for s, _ in SCOPES],
            "tiers": list(cfg.tiers),
            "top_n_kernels": cfg.top_n_kernels,
            "max_kernels": cfg.max_kernels,
            "launch_cap_used": self.launch_cap(),
            "clock_control": cfg.clock_control,
            "cache_control": cfg.cache_control,
            "collections": [c.to_dict() for c in collections],
            "truncated": bool(truncated),
            "warnings": warnings,
            "gpu_state": gpu_windows,
        }

    def _collect_tier1(self, config_path: Path, scope: str, nvtx_range: str) -> NcuCollection:
        cfg = self.run_config.profile.ncu
        stem = self.paths.raw / f"ncu_tier1_{scope}"
        result_path = self.paths.metrics / f"ncu_tier1_{scope}_worker.json"
        argv = self.build_tier1_command(config_path, result_path, nvtx_range, stem)

        command = run_command(
            argv, log_dir=self.paths.logs, log_name=f"ncu_tier1_{scope}",
            timeout=cfg.timeout_s, cwd=self.repo_root,
        )
        collection = NcuCollection(
            scope=scope, tier=1, command=command, launch_cap=self.launch_cap()
        )
        report_path = stem.with_suffix(".ncu-rep")

        if not report_path.exists():
            collection.error = _diagnose(command, nvtx_range)
            return collection

        collection.report_path = report_path
        collection.csv_path = self.export_csv(report_path, self.paths.metrics /
                                              f"ncu_tier1_{scope}.csv")
        if collection.csv_path:
            collection.kernel_count = _count_kernels(collection.csv_path)
        else:
            collection.error = "ncu report written but CSV export failed"
        return collection

    def _collect_tier_n(
        self, config_path: Path, scope: str, nvtx_range: str, tier: int,
        kernel_names: list[str],
    ) -> NcuCollection:
        cfg = self.run_config.profile.ncu
        stem = self.paths.raw / f"ncu_tier{tier}_{scope}"
        result_path = self.paths.metrics / f"ncu_tier{tier}_{scope}_worker.json"
        argv = self.build_tier_n_command(
            config_path, result_path, nvtx_range, stem, tier, kernel_names
        )

        command = run_command(
            argv, log_dir=self.paths.logs, log_name=f"ncu_tier{tier}_{scope}",
            timeout=cfg.timeout_s, cwd=self.repo_root,
        )
        # launch_cap left at 0: tier 2's cap is deliberate sampling, not truncation, and its
        # metrics are read per kernel rather than summed into phase totals.
        collection = NcuCollection(scope=scope, tier=tier, command=command)
        report_path = stem.with_suffix(".ncu-rep")

        if not report_path.exists():
            collection.error = _diagnose(command, nvtx_range)
            return collection

        collection.report_path = report_path
        collection.csv_path = self.export_csv(report_path, self.paths.metrics /
                                              f"ncu_tier{tier}_{scope}.csv")
        if collection.csv_path:
            collection.kernel_count = _count_kernels(collection.csv_path)
        return collection

    # ---- export & ranking -------------------------------------------------------------

    def export_csv(self, report_path: Path, csv_path: Path) -> Path | None:
        """Export a .ncu-rep to raw CSV.

        ``--page raw`` gives one row per (kernel, metric) with the full metric name intact,
        rather than the human-formatted section tables where values carry unit suffixes and
        thousands separators. It is the only page that round-trips cleanly into analysis.
        """
        command = run_command(
            ["ncu", "--import", str(report_path), "--csv", "--page", "raw"],
            log_dir=self.paths.logs,
            log_name=f"ncu_export_{report_path.stem}",
            timeout=1800,
        )
        stdout_path = Path(command.stdout_path)
        if not command.ok or not stdout_path.exists() or stdout_path.stat().st_size == 0:
            return None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(stdout_path.read_text(errors="replace"))
        return csv_path

    def rank_kernels(self, csv_path: Path | None, top_n: int) -> list[str]:
        """Rank kernels by measured GPU time, returning the top N names.

        Deep-diving anything else would spend the expensive full-section budget on kernels
        that contribute nothing to the phase's runtime.

        Ranking uses tier 1's own export, so the names handed to tier 2's ``--kernel-name``
        filter are exactly the strings ncu itself produced. Ranking from the Nsight Systems
        timeline instead would give demangled full template signatures, which do not match
        what ncu matches against.
        """
        if not csv_path or not csv_path.exists():
            return []

        report = parse_csv(csv_path)
        durations: dict[str, float] = {}
        for kernel in report.kernels:
            if not kernel.name:
                continue
            durations[kernel.name] = durations.get(kernel.name, 0.0) + kernel.duration_ns

        ranked = sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in ranked[:top_n]]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


#: The base identifier ncu's --kernel-name filter matches on. Shared with the parser so the
#: filter and the report's kernel labels can never disagree about what a kernel is called.
base_kernel_name = base_identifier


def _kernel_regex(names: list[str]) -> str:
    """Build an anchored alternation regex over the base identifiers of these kernels.

    Anchored so a short name cannot select every longer kernel containing it as a substring,
    and escaped because kernel identifiers can carry regex metacharacters.

    Base-name matching is coarser than the ranking that produced the list: several template
    instantiations share one identifier, so tier 2 may profile more launches than tier 1
    ranked. That is a superset, not a mismatch -- and it is the finest granularity ncu's
    filter offers.
    """
    bases = {base_kernel_name(n) for n in names if n}
    escaped = sorted(re.escape(b) for b in bases if b)
    return "^(" + "|".join(escaped) + ")$"


def _count_kernels(csv_path: Path) -> int:
    """Number of distinct kernel launches in an export."""
    return len(parse_csv(csv_path).kernels)


def _diagnose(command: CommandResult, nvtx_range: str) -> str:
    """Turn an ncu failure into something actionable.

    ncu's own errors are terse and its most common failure here -- an NVTX filter that
    matched nothing -- reports as a bare warning with a zero exit code, which is easy to
    mistake for success.
    """
    tail = (command.stderr_tail or "") + (command.stdout_tail or "")

    if "ERR_NVGPUCTRPERM" in tail:
        return (
            "GPU performance counters are restricted to admin users. Enable them with:\n"
            "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | "
            "sudo tee /etc/modprobe.d/nvidia-profiling.conf\n"
            "  sudo update-initramfs -u && sudo reboot"
        )
    if "No kernels were profiled" in tail:
        detail = (
            f"The NVTX filter '{ncu_filter(nvtx_range)}' matched no kernels. Either the "
            "workload never reached that phase, or the range name changed. Check that the "
            "worker ran in --mode ncu, which is what emits exactly one scoped decode step."
        )
        if "match only start/end ranges" in tail:
            detail += (
                "\n  ncu also reported that the expression matched only start/end ranges -- "
                "the trailing '/' that marks a push/pop range is missing from the filter."
            )
        return detail
    if "Failed to find metric" in tail:
        missing = re.findall(r"Failed to find metric \S*?([a-z0-9_]+__[a-zA-Z0-9_.]+)", tail)
        return (
            "ncu rejected metrics unavailable on this GPU: "
            + ", ".join(sorted(set(missing)))
            + ". Re-run 'nsbench preflight' to refresh the metric availability probe."
        )
    if command.returncode == 124:
        return (
            f"ncu timed out. Full-section replay is slow; lower ncu.top_n_kernels or raise "
            f"ncu.timeout_s.\n{tail[-500:]}"
        )
    return f"ncu failed (exit {command.returncode}).\n{tail[-800:]}"
