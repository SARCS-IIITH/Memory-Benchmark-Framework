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
    wide.batch_size = 4
    try:
        llamacpp.LlamaCppBackend(gguf_cfg, wide).load()
        check("batch_size > 1 rejected", False)
    except ValueError:
        check("batch_size > 1 rejected", True)
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

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: " + "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
