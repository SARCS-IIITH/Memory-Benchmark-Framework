"""Check the llamacpp backend (nsight_bench/backends/llamacpp.py) without the GPU.

Covers what can go wrong before a run starts: registration, the GGUF model config, the token
mode switch, input validation, and -- the point of the comparison -- that the prompt token ids
are identical to the hf backend's. Nothing here loads weights or touches the GPU.

    cd ~/Memory-Benchmark-Framework
    ~/envs/nsbench/bin/python experiments/llamacpp-backend/check_llamacpp_cpu.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # prove nothing here needs the GPU

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import numpy as np
    from transformers import AutoTokenizer

    from nsight_bench.backends import hf_transformers, llamacpp  # noqa: F401  (registers)
    from nsight_bench.backends.base import available_backends, get_backend
    from nsight_bench.config import ModelConfig, WorkloadConfig

    hf_cfg = ModelConfig.load(REPO / "configs/models/qwen3-0.6b.yaml")
    gguf_cfg = ModelConfig.load(REPO / "configs/models/qwen3-0.6b-gguf.yaml")
    workload = WorkloadConfig.load(REPO / "configs/workloads/decode-focused.yaml")

    print("Registration")
    check("llamacpp is registered", "llamacpp" in available_backends(), str(available_backends()))
    check("get_backend('llamacpp') resolves", get_backend("llamacpp") is llamacpp.LlamaCppBackend)
    check("hf still registered", get_backend("hf") is hf_transformers.HFTransformersBackend)

    print("GGUF model config")
    check("path is the GGUF file", gguf_cfg.path.endswith(".gguf") and Path(gguf_cfg.path).is_file(),
          gguf_cfg.path)
    check("tokenizer_path is the HF snapshot", gguf_cfg.tokenizer_path == hf_cfg.path)
    differ = sorted(k for k in vars(hf_cfg) if getattr(hf_cfg, k) != getattr(gguf_cfg, k))
    expected = sorted(["path", "tokenizer_path", "weight_bytes_on_disk", "safetensors_files", "notes"])
    check("only the intended fields differ from qwen3-0.6b.yaml", differ == expected, str(differ))
    check("same predicted decode bytes", hf_cfg.decode_read_weight_bytes() == gguf_cfg.decode_read_weight_bytes())
    check("GGUF size recorded", gguf_cfg.weight_bytes_on_disk == Path(gguf_cfg.path).stat().st_size)

    print("Token mode switch")
    for mode in ("bench", "greedy"):
        os.environ["NSBENCH_LLAMACPP_TOKEN_MODE"] = mode
        check(f"mode '{mode}' accepted", llamacpp.LlamaCppBackend(gguf_cfg, workload).token_mode == mode)
    os.environ["NSBENCH_LLAMACPP_TOKEN_MODE"] = "sampled"
    try:
        llamacpp.LlamaCppBackend(gguf_cfg, workload)
        check("unknown mode rejected", False)
    except ValueError:
        check("unknown mode rejected", True)
    os.environ.pop("NSBENCH_LLAMACPP_TOKEN_MODE")
    check("default mode is bench", llamacpp.LlamaCppBackend(gguf_cfg, workload).token_mode == "bench")

    print("Input validation (raised before any weights load)")
    wide = WorkloadConfig.load(REPO / "configs/workloads/decode-focused.yaml")
    # batch > 1 is supported since 2026-10-13 (check_batch_cpu.py); only a nonsense size is refused.
    wide.batch_size = 0
    try:
        llamacpp.LlamaCppBackend(gguf_cfg, wide).load()
        check("batch_size < 1 rejected", False)
    except ValueError:
        check("batch_size < 1 rejected", True)
    try:
        llamacpp.LlamaCppBackend(hf_cfg, workload).load()
        check("non-GGUF path rejected", False)
    except ValueError:
        check("non-GGUF path rejected", True)

    print("Prompt token ids match the hf backend")
    hf = hf_transformers.HFTransformersBackend(hf_cfg, workload)
    hf.device = "cpu"
    hf.tokenizer = AutoTokenizer.from_pretrained(hf_cfg.path)
    hf_ids = hf.prepare_inputs()["input_ids"][0].numpy()
    lc_ids = llamacpp.LlamaCppBackend(gguf_cfg, workload).prepare_inputs()
    check("same length", len(hf_ids) == len(lc_ids) == workload.prompt_tokens, f"{len(hf_ids)} / {len(lc_ids)}")
    check("identical ids", bool(np.array_equal(hf_ids, lc_ids)), f"first 5: {lc_ids[:5].tolist()}")
    check("int32, contiguous (what llama_decode takes)",
          lc_ids.dtype == np.int32 and lc_ids.flags["C_CONTIGUOUS"])

    print("ncu replay mode (auto: application for llamacpp, kernel otherwise)")
    import tempfile

    from nsight_bench.config import NcuConfig, ProfileConfig, RunConfig
    from nsight_bench.platform import PlatformProfile
    from nsight_bench.runners.base import RunPaths
    from nsight_bench.runners.ncu_runner import NcuRunner

    def replay_flag(backend: str, cfg: ModelConfig, setting: str | None = None) -> str:
        profile = ProfileConfig.load(REPO / "configs/profiles/moe.yaml")
        if setting:
            profile.ncu.replay_mode = setting
        run = RunConfig(model=cfg, workload=workload, profile=profile, backend=backend)
        with tempfile.TemporaryDirectory() as tmp:
            runner = NcuRunner(run, RunPaths(Path(tmp)), PlatformProfile())
            argv = runner.build_tier1_command(
                Path(tmp) / "c.json", Path(tmp) / "r.json", "nsbench.prefill", Path(tmp) / "s"
            )
        return argv[argv.index("--replay-mode") + 1]

    check("profiles default to auto", all(
        ProfileConfig.load(p).ncu.replay_mode == "auto"
        for p in sorted((REPO / "configs/profiles").glob("*.yaml"))))
    check("llamacpp + auto -> application", replay_flag("llamacpp", gguf_cfg) == "application")
    check("hf + auto -> kernel (unchanged)", replay_flag("hf", hf_cfg) == "kernel")
    check("explicit kernel honoured on llamacpp", replay_flag("llamacpp", gguf_cfg, "kernel") == "kernel")
    check("explicit application honoured on hf", replay_flag("hf", hf_cfg, "application") == "application")
    try:
        NcuConfig(replay_mode="range")
        check("unsupported replay mode rejected", False)
    except ValueError:
        check("unsupported replay mode rejected", True)

    print("ncu tier 1 (auto: on for llamacpp, profile's tiers otherwise)")
    from nsight_bench.cli import _build_run_config, build_parser

    def run_tiers(backend: str, model: str, *extra: str) -> tuple[int, ...]:
        args = build_parser().parse_args([
            "run", "--model", f"configs/models/{model}", "--backend", backend,
            "--workload", "configs/workloads/moe/moe-decode-natural-b1.yaml",
            "--profile", "configs/profiles/moe.yaml", *extra])
        run = _build_run_config(args)
        with tempfile.TemporaryDirectory() as tmp:
            return tuple(NcuRunner(run, RunPaths(Path(tmp)), PlatformProfile()).tiers)

    check("llamacpp + moe.yaml -> tiers 1,2", run_tiers("llamacpp", "kimi-linear-48b-q8-gguf.yaml") == (1, 2))
    check("hf + moe.yaml -> tier 2 only (unchanged)", run_tiers("hf", "kimi-linear-48b-q8-gguf.yaml") == (2,))
    check("explicit --tiers 2 honoured on llamacpp",
          run_tiers("llamacpp", "kimi-linear-48b-q8-gguf.yaml", "--tiers", "2") == (2,))
    check("explicit --tiers 1,2 honoured on hf",
          run_tiers("hf", "kimi-linear-48b-q8-gguf.yaml", "--tiers", "1,2") == (1, 2))
    check("auto_tier1: false in a profile turns it off",
          NcuConfig(tiers=(2,), auto_tier1=False).resolved_tiers("llamacpp") == (2,))

    print("Cache sizing follows the engine (llama.cpp: latent MLA)")
    from nsight_bench.analysis.assemble import RunAnalysis
    from nsight_bench.config import cache_engine_for_backend

    kimi_q8 = ModelConfig.load(REPO / "configs/models/kimi-linear-48b-q8-gguf.yaml")
    # llama.cpp's own allocation log for this GGUF (CPU context, 2026-10-12):
    #   256 cells:    K (f16) 1.97 MiB, V 0.00 MiB, 7 layers; R (f32) 2.81 MiB, S (f32) 40.00 MiB
    #   16,640 cells: K (f16) 127.97 MiB
    lc256 = kimi_q8.kv_cache_breakdown(256, engine="llama.cpp")
    check("llama.cpp MLA cache = latent, K only (256 cells: 2,064,384 B = 1.97 MiB)",
          lc256.get("kv") == 2_064_384, str(lc256.get("kv")))
    check("llama.cpp MLA cache at 16,640 cells = 127.97 MiB",
          kimi_q8.kv_cache_breakdown(16_640, engine="llama.cpp").get("kv") == 134_184_960)
    check("llama.cpp KDA state S = 40.00 MiB", lc256.get("recurrent_state") == 41_943_040)
    check("llama.cpp KDA conv state R = 2.81 MiB (kernel-1, f32)",
          lc256.get("conv_state") == 2_949_120, str(lc256.get("conv_state")))
    hf160 = kimi_q8.kv_cache_breakdown(160)
    check("transformers layout unchanged (expanded MLA 22.94 MB at 160 tokens)",
          hf160 == {"kv": 22_937_600, "recurrent_state": 41_943_040, "conv_state": 1_966_080},
          str(hf160))
    check("kv_cache_bytes passes the engine through",
          kimi_q8.kv_cache_bytes(160, engine="llama.cpp") == 46_182_400)
    check("backend -> engine: llamacpp -> llama.cpp, hf -> transformers",
          cache_engine_for_backend("llamacpp") == "llama.cpp"
          and cache_engine_for_backend("hf") == "transformers")
    check("RunAnalysis picks the engine from the run's backend",
          RunAnalysis(manifest={"config": {"backend": "llamacpp"}}).cache_engine == "llama.cpp"
          and RunAnalysis(manifest={"config": {"backend": "hf"}}).cache_engine == "transformers"
          and RunAnalysis(manifest={}).cache_engine == "transformers")
    try:
        kimi_q8.kv_cache_breakdown(160, engine="vllm")
        check("unknown cache engine rejected", False)
    except ValueError:
        check("unknown cache engine rejected", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: " + "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
