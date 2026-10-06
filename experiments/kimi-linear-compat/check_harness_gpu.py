"""End-to-end GPU check of the harness's Kimi-Linear support -- no download needed.

Builds a reduced Kimi-Linear (4 layers: 3 KDA + 1 MLA, 16 experts, full 163k vocabulary so
the real tokenizer fits) with random bf16 weights, saves it as a checkpoint with the model's
own remote code, and drives it through the real ``HFTransformersBackend``: load, synthetic
prompt, prefill, decode. That covers what the CPU check cannot -- fla's KDA Triton kernels
on this GPU, the model-managed cache across steps, the attention switch at run time, and
the cache measurement on live state. About 2 GB of GPU memory for a few seconds.

FP8 loading is *not* covered here (the weights are random bf16); it needs the real checkpoint.

    cd ~/Memory-Benchmark-Framework
    ~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_harness_gpu.py [KIMI_DIR] [--force]

Refuses to run while another process is using the GPU, unless --force: this machine is
shared, and even a small job perturbs someone else's benchmark.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
KIMI_DIR = Path(ARGS[0] if ARGS else HERE / "kimi-fp8-meta")
FORCE = "--force" in sys.argv

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def gpu_busy() -> list[str]:
    """Other compute processes on the GPU, as 'name (MiB)' strings."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = os.getpid()
    busy = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) != me:
            busy.append(f"{parts[1]} ({parts[2]} MiB)")
    return busy


def build_tiny_checkpoint(out_dir: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from nsight_bench import compat

    compat.apply_remote_code_shims()
    cfg = AutoConfig.from_pretrained(KIMI_DIR, trust_remote_code=True)
    cfg.num_hidden_layers = 4
    cfg.linear_attn_config = {**cfg.linear_attn_config,
                              "kda_layers": [1, 2, 3], "full_attn_layers": [4]}
    cfg.num_experts = 16
    del cfg.quantization_config            # random bf16 weights; FP8 needs the real checkpoint
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    # Kimi declares _tied_weights_keys as a 4.x-style list; 5.x's save path expects a dict and
    # crashes on it. Only saving is affected -- on load, 5.x returns no tied weights because the
    # config says tie_word_embeddings=false -- and the harness never saves, so this stays a
    # test-only workaround. Kimi ties nothing, so an empty mapping is exact.
    model._tied_weights_keys = {}
    model.save_pretrained(out_dir)
    for pattern in ("*.py", "tokenizer_config.json", "special_tokens_map.json",
                    "tiktoken.model", "chat_template.jinja", "generation_config.json"):
        for src in KIMI_DIR.glob(pattern):
            if not (out_dir / src.name).exists():
                shutil.copy2(src, out_dir / src.name)
    del model


def main() -> int:
    busy = gpu_busy()
    if busy and not FORCE:
        print("GPU in use by: " + ", ".join(busy))
        print("Not running -- re-run when the GPU is free, or pass --force.")
        return 2

    import torch

    from nsight_bench.backends.hf_transformers import HFTransformersBackend
    from nsight_bench.config import ModelConfig, WorkloadConfig, discover_model

    with tempfile.TemporaryDirectory(prefix="tiny-kimi-") as tmp:
        ckpt = Path(tmp)
        print(f"Building a reduced Kimi-Linear checkpoint in {ckpt} ...")
        build_tiny_checkpoint(ckpt)

        print("\n1. Discovery")
        model_cfg: ModelConfig = discover_model(ckpt, name="tiny-kimi")
        check("trust_remote_code enabled automatically", model_cfg.trust_remote_code)
        check("MoE detected", model_cfg.is_moe,
              f"{model_cfg.num_experts} experts, top-{model_cfg.num_experts_per_token}")

        workload = WorkloadConfig(prompt_tokens=64, generate_tokens=8, warmup_iters=0, repeat=1)
        backend = HFTransformersBackend(model_cfg, workload)

        print("\n2. Load through the harness")
        backend.load()
        info = backend.describe()
        check("loaded on the GPU", next(backend.model.parameters()).is_cuda)
        check("cache API adapted", info.get("compat_adapters", {}).get("cache_classes_adapted")
              == ["KimiDynamicCache"])
        check("model owns its cache", info["cache_managed_by_model"])
        check("attention running on sdpa", info["actual_attn_implementation"] == "sdpa",
              info.get("attn_warning", ""))

        print("\n3. Prefill + decode (KDA kernels on this GPU)")
        inputs = backend.prepare_inputs()
        state = backend.prefill(inputs)
        for _ in range(workload.generate_tokens):
            state = backend.decode_step(state)
        torch.cuda.synchronize()
        tokens = state.token_ids()
        check("prefill + 8 decode steps ran", len(tokens) == workload.generate_tokens + 1,
              f"tokens {tokens}")
        check("cache is the model's own", type(state.cache).__name__ == "KimiDynamicCache")

        print("\n4. Cache measurement on live state")
        total = backend._measure_kv_bytes(state.cache)
        breakdown = backend._cache_breakdown
        context = workload.prompt_tokens + workload.generate_tokens
        want_kv = 1 * 32 * context * (192 + 128) * 2          # one MLA layer, expanded K and V
        want_rec = 3 * 32 * 128 * 128 * 4                      # three KDA layers, fp32 state
        check("MLA KV bytes match the expanded-cache shape", breakdown.get("kv") == want_kv,
              f"{breakdown.get('kv'):,} B for {context} tokens")
        check("KDA recurrent state counted", breakdown.get("recurrent_state") == want_rec,
              f"{breakdown.get('recurrent_state', 0) / 2**20:.1f} MiB")
        check("conv state counted", breakdown.get("conv_state", 0) > 0)
        check("total = sum of parts", total == sum(breakdown.values()), f"{total:,} B")

        backend.teardown()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: " + "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1]))
    raise SystemExit(main())
