"""Load the real Kimi-Linear FP8 checkpoint through the harness and check it talks sense.

Not a benchmark -- no profiler is attached and the timings are indicative only. It answers
the questions the weight-free checks could not:

* do the FP8 weights and their scales load through the real ``from_pretrained`` path?
* does the model produce coherent text (``nm-testing`` checkpoints carry no model card)?
* what is the real resident footprint, and how much of the 128 GB pool is left?
* what does the live decoding state (KV + KDA recurrent state) look like?

It drives the same ``HFTransformersBackend`` prefill/decode loop the profiled runs use, so
coherent output here also means the harness's hand-written decode loop is correct for this
model. Refuses to start while another process is on the GPU, unless --force.

    cd ~/Memory-Benchmark-Framework
    ~/envs/nsbench/bin/python experiments/kimi-linear-compat/sanity_real_model.py [--force]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_YAML = REPO / "configs" / "models" / "kimi-linear-48b-fp8.yaml"
FORCE = "--force" in sys.argv
NEW_TOKENS = 64

PROMPTS = [
    "Explain in two sentences why LLM decoding is limited by memory bandwidth.",
    "What is the capital of France, and what river flows through it?",
]


def gib(n: float) -> str:
    return f"{n / 2**30:.1f} GiB"


def main() -> int:
    sys.path.insert(0, str(REPO))
    from check_harness_gpu import gpu_busy  # noqa: E402  (same directory)

    busy = gpu_busy()
    if busy and not FORCE:
        print("GPU in use by: " + ", ".join(busy) + " -- not running (pass --force to override).")
        return 2

    import torch

    from nsight_bench.backends.hf_transformers import HFTransformersBackend
    from nsight_bench.config import ModelConfig, WorkloadConfig

    cfg = ModelConfig.load(MODEL_YAML)
    print(f"Model:  {cfg.name}\n  path  {cfg.path}\n  device_map={cfg.device_map}  "
          f"quantization={cfg.quantization}  trust_remote_code={cfg.trust_remote_code}")

    free0, total = torch.cuda.mem_get_info()
    print(f"\nMemory before load: {gib(free0)} free of {gib(total)}")

    backend = HFTransformersBackend(cfg, WorkloadConfig(seed=1234))
    t0 = time.perf_counter()
    backend.load()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    free1, _ = torch.cuda.mem_get_info()
    info = backend.describe()

    dtypes: dict[str, int] = {}
    for p in backend.model.parameters():
        key = str(p.dtype).replace("torch.", "")
        dtypes[key] = dtypes.get(key, 0) + p.numel() * p.element_size()
    fp8_layers = sum(1 for m in backend.model.modules()
                     if type(m).__name__ == "CompressedTensorsFP8Linear")
    print(f"\nLoaded in {load_s:.0f} s")
    print(f"  resident parameters  {info.get('parameter_bytes_resident', 0) / 1e9:.1f} GB  "
          f"by dtype: " + ", ".join(f"{k} {v / 1e9:.1f} GB" for k, v in sorted(dtypes.items())))
    print(f"  FP8 linear layers    {fp8_layers}  (fp8_optimized_inference={info['fp8_optimized_inference']})")
    print(f"  attention            {info['actual_attn_implementation']}  -- {info.get('attn_warning', '')}")
    print(f"  compat adapters      {info.get('compat_adapters')}")
    print(f"  load_warning         {info.get('load_warning')}")
    print(f"  memory after load    {gib(free1)} free  (load took {gib(free0 - free1)})")

    tok = backend.tokenizer
    results = []
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        # Kimi's tiktoken tokenizer returns the rendered template as a string even when asked
        # for tensors, so render to text and encode it explicitly. add_special_tokens=False:
        # the template already carries <|im_user|>/<|im_assistant|> markers.
        text_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok(text_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        inputs = {"input_ids": ids.to("cuda"), "attention_mask": torch.ones_like(ids).to("cuda")}

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = backend.prefill(inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eos = set(filter(None, [tok.eos_token_id] + list(getattr(tok, "all_special_ids", []) or [])))
        for _ in range(NEW_TOKENS - 1):
            state = backend.decode_step(state)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        out = state.token_ids()
        if any(t in eos for t in out):                   # cut at the first end-of-turn token
            out = out[: next(i for i, t in enumerate(out) if t in eos)]
        text = tok.decode(out, skip_special_tokens=True).strip()
        backend._measure_kv_bytes(state.cache)
        results.append((prompt, text, len(set(out)), len(out)))
        print(f"\n--- prompt ({ids.shape[1]} tokens): {prompt}")
        print(f"--- answer: {text}")
        print(f"    prefill {1e3 * (t1 - t0):.0f} ms | decode {(NEW_TOKENS - 1) / (t2 - t1):.1f} tok/s "
              f"(indicative, unprofiled, one run) | distinct tokens {len(set(out))}/{len(out)}")
        print(f"    decoding state at {state.position} tokens: "
              + ", ".join(f"{k} {v / 2**20:.1f} MiB" for k, v in backend._cache_breakdown.items()))
        del state

    peak_free, _ = torch.cuda.mem_get_info()
    print(f"\nMemory at end: {gib(peak_free)} free")
    backend.teardown()

    degenerate = [p for p, t, distinct, n in results if n and distinct < max(5, n // 6)]
    if degenerate:
        print("\nWARNING: output looks degenerate (very few distinct tokens) for: " + "; ".join(degenerate))
        return 1
    print("\nDone. Read the answers above: they should be fluent and on-topic.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
