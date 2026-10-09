"""In-process benchmark entry point -- the program the profilers actually wrap.

Every collection mode launches this same module as a subprocess::

    python -m nsight_bench.worker --config run.json --mode nsys --out result.json

Running the workload in a fresh process for each collection is what keeps the modes
comparable. Nsight Compute serialises and replays kernels, Nsight Systems installs CUPTI
callbacks, and both leave the CUDA context in a state that is not representative of a normal
run. Reusing one process across modes would let the first collection's overhead leak into
the next one's numbers.

It also means the harness process itself never imports torch, so an OOM or a segfault in a
model under test takes down the worker and not the whole sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nsight_bench.worker",
        description="Run one benchmark workload in this process (launched under a profiler).",
    )
    parser.add_argument("--config", required=True, help="RunConfig JSON/YAML path")
    parser.add_argument(
        "--mode", default="baseline", choices=["baseline", "nsys", "ncu"],
        help="Collector this process is running under; controls NVTX scoping",
    )
    parser.add_argument("--out", required=True, help="Where to write the result JSON")
    parser.add_argument(
        "--memory-sample-hz", type=float, default=20.0,
        help="Background memory sampling rate; 0 disables",
    )
    parser.add_argument(
        "--allocator-history", action="store_true",
        help="Record the torch allocator history and dump it next to the result",
    )
    args = parser.parse_args(argv)

    # Imported here rather than at module scope so --help works without torch present.
    from .backends import hf_transformers, llamacpp, trtllm  # noqa: F401  (registers backends)
    from .backends.base import get_backend
    from .config import RunConfig
    from .instrumentation import memory as mem
    from .workloads import multimodal, text_generation  # noqa: F401  (registers workloads)
    from .workloads.base import ProfileMode, build_workload

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_config = RunConfig.load(args.config)
    mode = ProfileMode(args.mode)

    payload: dict = {"mode": mode.value, "ok": False}
    sampler: "mem.MemorySampler | None" = None
    backend = None

    try:
        if args.memory_sample_hz > 0:
            sampler = mem.MemorySampler(interval_s=1.0 / args.memory_sample_hz).start()
        if args.allocator_history:
            mem.start_allocator_history()

        backend_cls = get_backend(run_config.backend)
        backend = backend_cls(run_config.model, run_config.workload)
        backend.profile_mode = mode.value

        # Weight loading is a phase in its own right: on a unified-memory part it is the
        # largest single memory event in the run, and its cost is pure host-to-device
        # traffic that never appears in a decode step's counters.
        with mem.PhaseMemoryTracker("load", sampler) as load_mem:
            backend.load()
            backend.synchronize()
        payload["load_memory"] = load_mem.delta.to_dict() if load_mem.delta else None

        workload = build_workload(run_config.workload)
        result = workload.run(backend, mode=mode, sampler=sampler)

        payload["ok"] = True
        payload["result"] = result.to_dict()
        payload["backend"] = backend.describe()
        payload["workload"] = workload.describe()

    except Exception as exc:                                     # noqa: BLE001
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if sampler is not None:
            timeline = sampler.stop()
            payload["memory_timeline"] = timeline.to_rows()
            payload["memory_peak_cuda_used"] = timeline.peak("cuda_used")
        payload["allocator_stats"] = mem.allocator_stats()
        if args.allocator_history:
            dumped = mem.dump_allocator_history(out_path.with_suffix(".alloc.pickle"))
            payload["allocator_history"] = str(dumped) if dumped else None
        if backend is not None:
            try:
                backend.teardown()
            except Exception:                                    # noqa: BLE001
                pass

    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    if not payload["ok"]:
        error = payload.get("error", {})
        print(
            f"worker failed: {error.get('type')}: {error.get('message')}",
            file=sys.stderr,
        )
        return 1

    print(f"worker ok: {args.mode} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
