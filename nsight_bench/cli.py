"""``nsbench`` -- the command-line entry point.

Commands follow the order you would actually use them in::

    nsbench preflight                      # what can this machine measure?
    nsbench discover /path/to/model        # turn a checkpoint into a config
    nsbench calibrate                      # is the DRAM derivation sound right now?
    nsbench run --model configs/models/x.yaml
    nsbench report runs/<id>
    nsbench compare runs/*/

``preflight`` is not optional ceremony. It probes which Nsight Compute metrics genuinely
collect on the attached GPU, and the answer differs enough between architectures -- GB10
exposes no ``dram__*`` counters at all -- that running without it means discovering the
problem hours into a profiling session instead of seconds into setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import platform as platform_mod
from .config import ModelConfig, ProfileConfig, RunConfig, WorkloadConfig, discover_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_PATH = REPO_ROOT / "configs" / "platform_profile.json"
DEFAULT_RUNS = REPO_ROOT / "runs"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if sys.stdout.isatty() else text


def _load_profile(path: str | None, allow_missing: bool = False) -> platform_mod.PlatformProfile:
    profile_path = Path(path or DEFAULT_PROFILE_PATH)
    if profile_path.exists():
        profile = platform_mod.PlatformProfile.load(profile_path)
        unprobed = profile.unprobed_metrics()
        if unprobed:
            # Warn rather than fail: the run still produces useful data, just with blank
            # columns where the unprobed metrics would have been.
            print(
                f"WARNING: {profile_path.name} predates {len(unprobed)} metric(s) now in the "
                f"registry ({', '.join(unprobed[:3])}"
                + (", ..." if len(unprobed) > 3 else "")
                + "). They will be missing from this run. Run 'nsbench preflight' to refresh.",
                file=sys.stderr,
            )
        return profile
    if allow_missing:
        return platform_mod.detect(probe_metric_availability=False)
    raise SystemExit(
        f"No platform profile at {profile_path}.\n"
        "Run 'nsbench preflight' first -- it records which metrics this GPU can actually "
        "collect, which the profiling commands depend on."
    )


# --------------------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> int:
    out_path = Path(args.out or DEFAULT_PROFILE_PATH)
    print(_bold("Probing platform capabilities (this runs real ncu collections)..."))

    # --quick skips the metric probe, which is slow. Carrying forward a previous probe's
    # result matters: an empty availability list is not "nothing works", it is "not checked",
    # and overwriting a good profile with an empty one would leave every later run reporting
    # blank metric columns for no visible reason.
    previous = (
        platform_mod.PlatformProfile.load(out_path) if (args.quick and out_path.exists())
        else None
    )

    profile = platform_mod.detect(
        probe_metric_availability=not args.quick,
        python_executable=sys.executable,
        workdir=out_path.parent,
    )
    if previous is not None and previous.metric_available:
        profile.metric_available = previous.metric_available
        profile.metric_missing = previous.metric_missing
        profile.notes.append(
            f"metric availability carried forward from the previous probe "
            f"({previous.detected_at}); re-run without --quick to refresh it"
        )
    profile.save(out_path)

    print()
    print(profile.describe())
    if profile.notes:
        print()
        print(_bold("Notes:"))
        for note in profile.notes:
            print(f"  - {note}")
    if profile.metric_missing:
        print()
        print(_bold("Unavailable metrics:"))
        for name, reason in sorted(profile.metric_missing.items()):
            print(f"  {name}\n      {reason}")

    print()
    print(f"Saved -> {out_path}")

    if not profile.permissions.ncu_counters_allowed:
        print()
        print(_bold("BLOCKING: ") + profile.permissions.ncu_permission_detail)
        return 1
    if not profile.availability.ok:
        print()
        print(_bold("BLOCKING: required metrics are unavailable: ")
              + ", ".join(profile.availability.missing_required()))
        return 1
    if not profile.availability.probed:
        print()
        print("Metric availability was not probed. Runs will request the full registry and "
              "let ncu reject what it cannot collect.")
        print("Run 'nsbench preflight' without --quick for a definitive answer.")
    return 0


# --------------------------------------------------------------------------------------
# discover
# --------------------------------------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> int:
    model = discover_model(
        args.path,
        name=args.name,
        dtype=args.dtype,
        attn_implementation=args.attn,
    )

    print(_bold(f"{model.name}"))
    print(f"  path            {model.path}")
    print(f"  architecture    {model.architecture} ({model.model_type})")
    print(f"  layers          {model.num_layers}")
    print(f"  hidden          {model.hidden_size}")
    print(f"  attention       {model.describe_attention()}")
    print(f"  quantization    {model.quantization or 'none'} "
          f"({model.bits_per_weight} bits/weight)")
    print(f"  weights on disk {model.weight_bytes_on_disk / 1e9:.2f} GB "
          f"across {len(model.safetensors_files)} file(s)")
    print(f"  est. parameters {model.param_count / 1e9:.2f} B")
    if model.is_moe and model.weight_bytes_by_role:
        read, _ = model.decode_read_weight_bytes()
        print(f"  decode reads    {read / 1e9:.2f} GB of weights per batch-1 token "
              "(routed-active, from the checkpoint's tensors)")
    for length in (512, 2048, 8192):
        parts = model.kv_cache_breakdown(length)
        if not parts:
            continue
        line = f"  KV cache @{length:>5} {sum(parts.values()) / 1e6:8.1f} MB"
        if len(parts) > 1:
            line += "  (" + ", ".join(f"{k} {v / 1e6:,.1f}" for k, v in parts.items()) + ")"
        latent = model.mla_latent_kv_bytes(length)
        if latent is not None:
            line += f"  | MLA latent would be {latent / 1e6:,.1f} MB"
        print(line)
    for note in model.notes:
        print(f"  note: {note}")

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "configs" / "models" / f"{_slugify(model.name)}.yaml"
    )
    written = model.save(out_path)
    print()
    print(f"Saved -> {written}")
    return 0


def _slugify(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower() or "model"


# --------------------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .runners.base import RunPaths
    from .runners.calibration_runner import CalibrationRunner

    profile = _load_profile(args.platform_profile)
    out_dir = Path(args.out or (DEFAULT_RUNS / "_calibration"))
    paths = RunPaths(out_dir).create()

    print(_bold("Calibrating the memory derivation against a known byte count..."))
    result = CalibrationRunner(
        paths, profile, python_executable=sys.executable, megabytes=args.megabytes
    ).run(include_sweep=not args.no_sweep)

    print()
    print(_bold("Byte-accounting gate: ") + result.byte_accounting.summary())

    if result.sweep:
        l2_mib = result.l2_cache_bytes / (1024 * 1024)
        print()
        print(_bold(f"Working-set sweep (L2 = {l2_mib:.1f} MiB):"))
        for point in result.sweep:
            marker = "in L2 " if point.working_set_mib <= l2_mib * 0.75 else (
                "stream" if point.working_set_mib >= l2_mib * 4 else "  ~   "
            )
            print(f"  {point.working_set_mib:>8.0f} MiB  {marker}  "
                  f"{point.bandwidth_gbps:>8.1f} GB/s")

    print()
    for note in result.notes:
        print(f"  - {note}")

    from .runners.base import write_json

    written = write_json(paths.metrics / "calibration.json", result.to_dict())
    print()
    print(f"Saved -> {written}")
    return 0 if result.passed else 1


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    if args.model:
        model = ModelConfig.load(args.model)
    elif args.model_path:
        model = discover_model(args.model_path, name=args.name)
    else:
        raise SystemExit("Provide either --model <config.yaml> or --model-path <checkpoint dir>")

    if args.dtype:
        model.dtype = args.dtype
    if args.attn:
        model.attn_implementation = args.attn

    workload = WorkloadConfig.load(args.workload) if args.workload else WorkloadConfig()
    workload.name = args.workload_name or workload.name
    if args.prompt_tokens:
        workload.prompt_tokens = args.prompt_tokens
    if args.generate_tokens:
        workload.generate_tokens = args.generate_tokens
    if args.batch_size:
        workload.batch_size = args.batch_size
    if args.repeat:
        workload.repeat = args.repeat
    if args.warmup is not None:
        workload.warmup_iters = args.warmup
    if args.annotate_layers:
        workload.annotate_layers = True

    profile_config = ProfileConfig.load(args.profile) if args.profile else ProfileConfig()
    if args.top_n:
        profile_config.ncu.top_n_kernels = args.top_n
    if args.max_kernels:
        profile_config.ncu.max_kernels = args.max_kernels
        # Explicit beats automatic: honour the number the caller gave, including one
        # deliberately below the real kernel count for a fast plumbing check.
        profile_config.ncu.auto_launch_cap = False
    if args.tiers:
        profile_config.ncu.tiers = tuple(int(t) for t in args.tiers.split(","))

    return RunConfig(
        model=model,
        workload=workload,
        profile=profile_config,
        backend=args.backend,
        output_root=str(args.output or DEFAULT_RUNS),
        tag=args.tag or "",
    )


def cmd_run(args: argparse.Namespace) -> int:
    from .orchestrator import BenchmarkOrchestrator

    profile = _load_profile(args.platform_profile)
    run_config = _build_run_config(args)

    print(_bold(f"Benchmarking {run_config.model.name}"))
    print(f"  backend    {run_config.backend}")
    print(f"  workload   {run_config.workload.prompt_tokens} prompt tokens -> "
          f"{run_config.workload.generate_tokens} generated, "
          f"batch {run_config.workload.batch_size}")
    print(f"  ncu tiers  {list(run_config.profile.ncu.tiers)}  "
          f"(top {run_config.profile.ncu.top_n_kernels} kernels deep-dived)")
    print()

    record = BenchmarkOrchestrator(
        run_config, profile, python_executable=sys.executable, output_root=run_config.output_root
    ).run(
        skip_calibration=args.skip_calibration,
        skip_baseline=args.skip_baseline,
        skip_nsys=args.skip_nsys,
        skip_ncu=args.skip_ncu,
        calibration_sweep=not args.no_sweep,
    )

    print()
    print(_bold("Stages:"))
    for stage, ok in record.stages_ok.items():
        print(f"  {'ok  ' if ok else 'FAIL'}  {stage}")
    if record.warnings:
        print()
        print(_bold("Warnings:"))
        for warning in record.warnings:
            print(f"  - {warning}")

    print()
    print(f"Run -> {record.root}")

    if not args.no_report:
        return _emit_reports(Path(record.root), open_after=False)
    return 0 if record.ok else 1


# --------------------------------------------------------------------------------------
# report / compare
# --------------------------------------------------------------------------------------


def _emit_reports(run_dir: Path, open_after: bool = False) -> int:
    from .analysis.assemble import assemble, write_metric_tables
    from .report.html import write_html_report
    from .report.markdown import write_markdown_report

    analysis = assemble(run_dir)
    tables = write_metric_tables(analysis)
    md_path = write_markdown_report(analysis)
    html_path = write_html_report(analysis)

    print()
    print(_bold("Reports:"))
    print(f"  markdown  {md_path}")
    print(f"  html      {html_path}")
    print(f"  tables    {len(tables)} files in {run_dir / 'metrics'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(f"{run_dir} does not look like a run directory (no manifest.json)")
    return _emit_reports(run_dir)


def cmd_compare(args: argparse.Namespace) -> int:
    from .analysis.compare import build_comparison
    from .report.compare_report import write_comparison_reports

    run_dirs = [Path(d) for d in args.run_dirs]
    valid = [d for d in run_dirs if (d / "manifest.json").exists()]
    if not valid:
        raise SystemExit("No valid run directories given (each needs a manifest.json)")
    if len(valid) < len(run_dirs):
        print(f"Skipping {len(run_dirs) - len(valid)} path(s) without a manifest.json")

    comparison = build_comparison(valid)
    out_dir = Path(args.out or (DEFAULT_RUNS / "_comparison"))
    md_path, html_path, csv_path = write_comparison_reports(comparison, out_dir)

    print(_bold(f"Compared {len(comparison.runs)} runs"))
    print(f"  markdown  {md_path}")
    print(f"  html      {html_path}")
    print(f"  csv       {csv_path}")
    return 0


def _load_registry(args: argparse.Namespace):
    from .registry import load_registry

    return load_registry(getattr(args, "registry", None), getattr(args, "store", None))


def cmd_models(args: argparse.Namespace) -> int:
    """List the catalogue and what the shared store already holds."""
    from .registry import store_free_bytes, store_status

    registry = _load_registry(args)
    entries = registry.resolve(args.models) if args.models else list(registry.entries.values())
    rows = store_status(registry, entries)

    print(_bold(f"Model store: {registry.store}"))
    print(f"  free space {store_free_bytes(registry.store) / 1e12:,.2f} TB\n")
    print(f"  {'key':<16}{'class':<7}{'status':<12}{'on disk':>10}{'':>7}  model")
    print("  " + "-" * 80)
    have = need = 0
    for entry, row in zip(entries, rows):
        size = f"{row['bytes'] / 1e9:,.1f} GB" if row["bytes"] else "-"
        if row["state"] == "ready":
            have += row["bytes"]
            status, progress = "ready", ""
        elif row["state"] == "fetching":
            have += row["bytes"]
            need += max(0, (row["expected_bytes"] or 0) - row["bytes"])
            status = "fetching"
            progress = f"{row['pct']:.0f}%" if row["pct"] is not None else ""
        else:
            need += entry.params * 2 if entry.params else 0
            status, progress = "absent", ""
        print(f"  {entry.key:<16}{entry.model_class:<7}{status:<12}{size:>10}{progress:>7}"
              f"  {entry.display}")
    print("  " + "-" * 80)
    print(f"  {have / 1e9:,.1f} GB on disk"
          + (f", ~{need / 1e9:,.0f} GB still to fetch" if need else ", nothing to fetch"))

    fetching = [r for r in rows if r["state"] == "fetching"]
    if fetching:
        print(f"\n  {len(fetching)} transfer(s) in progress -- "
              "'fetching' counts partial blobs, so a model is not usable until 'ready'.")

    print(f"\n  groups: " + ", ".join(f"@{g}" for g in sorted(registry.groups)))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download selected models into the shared store."""
    from .registry import fetch, snapshot_path, store_free_bytes

    registry = _load_registry(args)
    entries = registry.resolve(args.models)

    pending = [e for e in entries if snapshot_path(e, registry.store) is None]
    if not pending:
        print(f"All {len(entries)} selected model(s) already in {registry.store}.")
        return 0

    estimate = sum(e.params * 2 for e in pending if e.params)
    free = store_free_bytes(registry.store)
    print(_bold(f"Fetching {len(pending)} model(s) into {registry.store}"))
    print(f"  roughly {estimate / 1e9:,.0f} GB to download, {free / 1e12:,.2f} TB free\n")
    if estimate and free < estimate * 1.15:
        raise SystemExit(
            f"Not enough space: need ~{estimate / 1e9:,.0f} GB plus headroom, "
            f"{free / 1e9:,.0f} GB free."
        )

    store = Path(registry.store)
    if not store.exists():
        raise SystemExit(
            f"The model store {store} does not exist.\n"
            f"Create it once, with permissions that let other users reuse it:\n"
            f"  sudo install -d -o $(id -un) -g $(id -gn) -m 2775 {store}"
        )

    failed = []
    for index, entry in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {entry.display}")
        try:
            path = fetch(entry, registry.store)
            size = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))
            print(f"    {size / 1e9:,.1f} GB -> {path}")
        except Exception as exc:                                 # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failed.append(entry.key)

    if failed:
        print(f"\n{len(failed)} model(s) failed: {', '.join(failed)}")
        print("Re-run the same command to resume; completed files are not re-downloaded.")
        return 1
    print(f"\nAll {len(pending)} model(s) fetched.")
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    """Benchmark several models end to end and build the comparison."""
    from .suite import SuiteRunner

    registry = _load_registry(args)
    entries = registry.resolve(args.models)
    profile = _load_profile(args.platform_profile)

    workload = (
        WorkloadConfig.load(args.workload) if args.workload else WorkloadConfig()
    )
    for attr, value in (
        ("prompt_tokens", args.prompt_tokens), ("generate_tokens", args.generate_tokens),
        ("batch_size", args.batch_size), ("repeat", args.repeat),
        ("warmup_iters", args.warmup),
    ):
        if value is not None:
            setattr(workload, attr, value)

    runner = SuiteRunner(
        registry=registry,
        entries=entries,
        profile=profile,
        workload=workload,
        profile_config_path=args.profile,
        output_root=args.output_root or DEFAULT_RUNS,
        python_executable=sys.executable,
        tag=args.tag,
        prefetch=not args.no_prefetch,
        max_kernels=args.max_kernels,
    )

    print(_bold(f"Benchmark suite: {len(entries)} model(s)"))
    for entry in runner.entries:
        print(f"  {entry.key:<16}{entry.model_class:<6}{entry.display}")

    ledger = runner.run(resume=not args.no_resume, comparison_dir=args.comparison_out)

    ok = [o for o in ledger.outcomes.values() if o.status == "ok"]
    bad = [o for o in ledger.outcomes.values() if o.status != "ok"]
    print()
    print(_bold(f"Suite complete: {len(ok)} succeeded, {len(bad)} failed"))
    for outcome in bad:
        print(f"  FAILED {outcome.label}: {outcome.error}")
    return 0 if not bad else 1


# --------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------


def _add_registry_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--registry", help="Model catalogue YAML (default configs/model-registry.yaml)")
    p.add_argument("--store", help="Shared model store (default /opt/ai-models)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsbench",
        description="Memory-hierarchy benchmarking for AI models on NVIDIA DGX Spark (GB10), "
                    "built on Nsight Systems and Nsight Compute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- preflight --
    p = sub.add_parser("preflight", help="Probe GPU, tools, permissions and metric availability")
    p.add_argument("--out", help=f"Where to write the profile (default {DEFAULT_PROFILE_PATH})")
    p.add_argument("--quick", action="store_true",
                   help="Skip the metric availability probe (faster, less useful)")
    p.set_defaults(func=cmd_preflight)

    # -- discover --
    p = sub.add_parser("discover", help="Read a checkpoint directory and write a model config")
    p.add_argument("path", help="HuggingFace-format checkpoint directory")
    p.add_argument("--name", help="Override the model name")
    p.add_argument("--dtype", default=None, help="bfloat16 | float16 | float32")
    p.add_argument("--attn", default=None, help="sdpa | eager | flash_attention_2")
    p.add_argument("--out", help="Output config path")
    p.set_defaults(func=cmd_discover)

    # -- calibrate --
    p = sub.add_parser("calibrate",
                       help="Verify the DRAM derivation and measure the bandwidth ceiling")
    p.add_argument("--megabytes", type=int, default=256, help="Streaming array size")
    p.add_argument("--no-sweep", action="store_true", help="Skip the working-set sweep")
    p.add_argument("--out", help="Output directory")
    p.add_argument("--platform-profile", help="Path to platform_profile.json")
    p.set_defaults(func=cmd_calibrate)

    # -- run --
    p = sub.add_parser("run", help="Run a full benchmark: calibrate, time, trace, profile")
    p.add_argument("--model", help="Model config YAML (from 'nsbench discover')")
    p.add_argument("--model-path", help="Checkpoint directory, discovered on the fly")
    p.add_argument("--name", help="Model name when using --model-path")
    p.add_argument("--backend", default="hf", help="hf | trtllm")
    p.add_argument("--workload", help="Workload config YAML")
    p.add_argument("--workload-name", help="Name for this workload in the run id")
    p.add_argument("--prompt-tokens", type=int)
    p.add_argument("--generate-tokens", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--repeat", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--annotate-layers", action="store_true",
                   help="Add per-transformer-block NVTX ranges")
    p.add_argument("--dtype")
    p.add_argument("--attn")
    p.add_argument("--profile", help="Profiler config YAML")
    p.add_argument("--tiers", help="Comma-separated ncu tiers, e.g. '1,2'")
    p.add_argument("--top-n", type=int, help="Kernels to deep-dive in tier 2")
    p.add_argument("--max-kernels", type=int,
                   help="ncu --launch-count cap. Below the phase's real kernel count this "
                        "truncates rather than samples; the run flags it when that happens")
    p.add_argument("--output", help=f"Run output root (default {DEFAULT_RUNS})")
    p.add_argument("--tag", help="Extra label for the run directory")
    p.add_argument("--platform-profile")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-nsys", action="store_true")
    p.add_argument("--skip-ncu", action="store_true")
    p.add_argument("--no-sweep", action="store_true", help="Skip the calibration sweep")
    p.add_argument("--no-report", action="store_true", help="Do not render reports afterwards")
    p.set_defaults(func=cmd_run)

    # -- report --
    p = sub.add_parser("report", help="(Re)generate reports for an existing run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    # -- compare --
    # -- models --
    p = sub.add_parser("models", help="List the model catalogue and what the store holds")
    _add_registry_args(p)
    p.add_argument("--models", help="Selection to show (default: everything)")
    p.set_defaults(func=cmd_models)

    # -- fetch --
    p = sub.add_parser("fetch", help="Download models into the shared store")
    _add_registry_args(p)
    p.add_argument("--models", required=True,
                   help="Comma-separated keys, or @group (e.g. @all, @moe, qwen3-4b)")
    p.set_defaults(func=cmd_fetch)

    # -- suite --
    p = sub.add_parser(
        "suite",
        help="Benchmark several models and build the comparison document",
        description="Fetches anything missing, benchmarks each model in turn (smallest "
                    "first), and writes a cross-model comparison. Resumable: a failure on "
                    "one model does not cost the others.",
    )
    _add_registry_args(p)
    p.add_argument("--models", required=True,
                   help="Comma-separated keys, or @group (e.g. @all, @moe, qwen3-4b)")
    p.add_argument("--profile", help="Profiler config YAML (default: standard)")
    p.add_argument("--platform-profile", help="Platform profile JSON from 'nsbench preflight'")
    p.add_argument("--workload", help="Workload config YAML")
    p.add_argument("--prompt-tokens", type=int)
    p.add_argument("--generate-tokens", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--repeat", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--max-kernels", type=int,
                   help="Exact ncu --launch-count. Omit to size it from model depth, which "
                        "is what keeps a deep MoE from silently truncating.")
    p.add_argument("--tag", default="suite", help="Names the run group and the ledger")
    p.add_argument("--output-root", help=f"Where runs are written (default {DEFAULT_RUNS})")
    p.add_argument("--comparison-out", help="Where the comparison document goes")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run models the ledger already records as complete")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Do not download upcoming models while profiling the current one. "
                        "Slower overall, but removes all I/O contention from the timing pass.")
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("compare", help="Build a cross-run comparison matrix")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", help="Output directory")
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
