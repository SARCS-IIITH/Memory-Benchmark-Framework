"""Check the harness's Kimi-Linear support without a GPU and without the weights.

Everything here runs on the CPU or the meta device, so it is safe while someone else is
benchmarking the GPU. It exercises the real harness code paths (nsight_bench.compat and the
HF backend), not a copy of them.

    cd ~/Memory-Benchmark-Framework
    ~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_harness_cpu.py [KIMI_DIR]

KIMI_DIR is any directory holding the checkpoint's config.json, modeling_kimi.py,
configuration_kimi.py, tokenization_kimi.py, tokenizer files and tiktoken.model -- the
downloaded snapshot works, as does the weight-free copy next to this script (the default).
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # prove nothing here needs the GPU

HERE = Path(__file__).resolve().parent
KIMI_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "kimi-fp8-meta")

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from nsight_bench import compat
    from nsight_bench.backends.hf_transformers import HFTransformersBackend
    from nsight_bench.config import ModelConfig, WorkloadConfig

    print(f"Kimi files: {KIMI_DIR}")
    raw_config = json.loads((KIMI_DIR / "config.json").read_text())

    print("\n1. Import shims")
    shims = compat.apply_remote_code_shims()
    check("shims applied", len(shims) >= 3, ", ".join(s.split(".")[-1] for s in shims))
    check("shims are idempotent", compat.apply_remote_code_shims() == shims)

    print("\n2. Tokenizer and config (remote code)")
    tok = AutoTokenizer.from_pretrained(KIMI_DIR, trust_remote_code=True)
    text = "The memory hierarchy of the GB10"
    check("tokenizer round-trips text", tok.decode(tok(text)["input_ids"]) == text,
          type(tok).__name__)
    cfg = AutoConfig.from_pretrained(KIMI_DIR, trust_remote_code=True)
    check("config loads", cfg.num_hidden_layers == 27, type(cfg).__name__)

    print("\n3. Full 48B model on the meta device")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    params = sum(p.numel() for p in model.parameters())
    check("model builds", 48e9 < params < 50e9, f"{params / 1e9:.2f}B params")

    print("\n4. Post-load adapters")
    adapted = compat.adapt_model_cache_api(model)
    check("cache class adapted to 5.x API", adapted == ["KimiDynamicCache"], str(adapted))
    check("model recognised as owning its cache", compat.model_manages_own_cache(model))
    replaced = compat.use_sdpa_if_flash_attn_missing(model)
    impls = {getattr(m.config, "_attn_implementation", None)
             for m in model.modules() if getattr(m, "config", None) is not None}
    check("flash_attention_2 replaced by sdpa everywhere",
          replaced == "flash_attention_2" and impls == {"sdpa"}, f"implementations now {impls}")

    print("\n5. FP8 loading decision")
    check("checkpoint detected as FP8 compressed-tensors", compat.is_fp8_compressed_tensors(raw_config))
    check("a bf16 config is not", not compat.is_fp8_compressed_tensors({"model_type": "qwen3"}))
    int4 = {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {
        "g": {"weights": {"type": "int", "num_bits": 4}}}}}
    check("an INT4 compressed-tensors config is not", not compat.is_fp8_compressed_tensors(int4))

    from transformers.quantizers.auto import AutoHfQuantizer
    from transformers.utils.quantization_config import CompressedTensorsConfig

    qc = CompressedTensorsConfig.from_dict({**raw_config["quantization_config"],
                                            "use_optimized_inference": True})
    quantizer = AutoHfQuantizer.from_config(qc, pre_quantized=True)
    quantizer.use_fp8_kernel = True       # validate_environment() would need the GPU to say so
    # A fresh config: step 4 switched the shared one to sdpa, which Kimi's class rejects at
    # construction (covered in step 7).
    fresh = AutoConfig.from_pretrained(KIMI_DIR, trust_remote_code=True)
    with torch.device("meta"):
        qmodel = AutoModelForCausalLM.from_config(fresh, trust_remote_code=True, dtype=torch.bfloat16)
    quantizer._process_model_before_weight_loading(qmodel)
    kinds = collections.Counter(type(m).__name__ for m in qmodel.modules()
                                if isinstance(getattr(m, "weight", None), torch.Tensor))
    fp8 = kinds.get("CompressedTensorsFP8Linear", 0)
    resident = sum(p.numel() * p.element_size() for p in qmodel.parameters())
    check("every quantized Linear stays FP8", fp8 == 20257, f"{fp8} FP8 layers")
    check("resident weights ~50 GB, not ~98 GB", 45e9 < resident < 55e9, f"{resident / 1e9:.1f} GB")

    print("\n6. HF backend cache handling (CPU tensors)")
    model_cfg = ModelConfig(name="kimi", path=str(KIMI_DIR), trust_remote_code=True)
    backend = HFTransformersBackend(model_cfg, WorkloadConfig())

    backend._model_owns_cache = True
    check("owning model gets past_key_values=None", backend._new_cache() is None)
    backend._model_owns_cache = False
    check("ordinary model still gets a DynamicCache",
          isinstance(backend._new_cache(), DynamicCache))

    kimi_cache_cls = sys.modules[type(model).__module__].KimiDynamicCache
    cache = kimi_cache_cls(cfg)
    seq, heads = 100, 32
    expected = collections.Counter()
    for i, kind in enumerate(cache.layer_types):
        if kind == "full_attention":
            k = torch.zeros(1, heads, seq, 192, dtype=torch.bfloat16)
            v = torch.zeros(1, heads, seq, 128, dtype=torch.bfloat16)
            cache.update(k, v, i)
            expected["kv"] += k.numel() * 2 + v.numel() * 2
        else:
            cache.recurrent_states[i] = torch.zeros(1, heads, 128, 128, dtype=torch.float32)
            cache.conv_states[i] = tuple(torch.zeros(1, heads * 128, 3) for _ in range(3))
            expected["recurrent_state"] += heads * 128 * 128 * 4
            expected["conv_state"] += 3 * heads * 128 * 3 * 4
    total = backend._measure_kv_bytes(cache)
    check("hybrid cache: KV of the 7 MLA layers counted",
          backend._cache_breakdown.get("kv") == expected["kv"], f"{expected['kv']:,} B")
    check("hybrid cache: KDA recurrent state counted (was missed before)",
          backend._cache_breakdown.get("recurrent_state") == expected["recurrent_state"],
          f"{expected['recurrent_state'] / 2**20:.0f} MiB over 20 layers")
    check("hybrid cache: conv state counted",
          backend._cache_breakdown.get("conv_state") == expected["conv_state"])
    check("total is the sum, nothing double-counted", total == sum(expected.values()),
          f"{total:,} B")

    dyn = DynamicCache()
    for i in range(4):
        dyn.update(torch.zeros(1, 8, seq, 128, dtype=torch.bfloat16),
                   torch.zeros(1, 8, seq, 128, dtype=torch.bfloat16), i)
    want = 4 * 2 * 8 * seq * 128 * 2
    got = backend._measure_kv_bytes(dyn)
    check("ordinary DynamicCache measured exactly as before", got == want and
          set(backend._cache_breakdown) == {"kv"}, f"{got:,} B")

    print("\n7. Requested attention implementation at load")
    # The backend requests attn_implementation="sdpa" by default. Kimi's class declares no
    # sdpa support, so transformers refuses at construction -- before any weight loads -- and
    # the backend's retry drops the request. The model then forces flash_attention_2 and the
    # post-load adapter (step 4) switches it to sdpa. This checks the error is one the retry
    # recognises.
    try:
        with torch.device("meta"):
            AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(KIMI_DIR, trust_remote_code=True),
                trust_remote_code=True, attn_implementation="sdpa",
            )
        check("requesting sdpa at construction", True, "accepted outright")
    except (TypeError, ValueError) as exc:
        check("requesting sdpa raises an error the backend's retry catches",
              "attention" in str(exc).lower(), type(exc).__name__)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: " + "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1]))
    raise SystemExit(main())
