"""GPU check that batch > 1 in the llamacpp backend is correct on Kimi-Linear itself.

The CPU check (check_batch_cpu.py) uses a tiny qwen3moe; Kimi adds a recurrent (KDA) state and
a conv window per sequence, which only the real model exercises. If sequences shared or mixed
that state, a row's logits in a batch would be far from the same row run alone. So: greedy
prefill + a few decode steps at batch 4, then each row alone at batch 1, and compare.

Batched and single-sequence runs use different kernels (e.g. mul_mat_q vs mul_mat_vec_q), so
exact equality is not expected; the thresholds allow float noise, not a different sequence:
per row, prefill logits cosine >= 0.999 and the same top-1 token, and the greedy continuation
reported (it may legitimately fork on a near-tie after the first token).

    ~/envs/nsbench/bin/python experiments/llamacpp-backend/check_batch_gpu.py [--out FILE.md]

Needs an idle GPU and ~60 GB free (one 52 GB model load at a time).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if "nsight_bench" not in sys.modules:
    sys.path.insert(0, str(REPO))

B, T, G = 4, 128, 8
lines: list[str] = []
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    line = f"- [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else "")
    print(line, flush=True)
    lines.append(line)
    if not ok:
        failures.append(name)


def note(text: str) -> None:
    print(text, flush=True)
    lines.append(text)


def make(batch: int):
    from nsight_bench.backends.llamacpp import LlamaCppBackend
    from nsight_bench.config import ModelConfig, WorkloadConfig

    os.environ["NSBENCH_LLAMACPP_TOKEN_MODE"] = "greedy"
    cfg = ModelConfig.load(REPO / "configs/models/kimi-linear-48b-q8-gguf.yaml")
    wl = WorkloadConfig.load(REPO / "configs/workloads/moe/moe-decode-natural-b1.yaml")
    wl = dataclasses.replace(wl, batch_size=batch, prompt_tokens=T, generate_tokens=G)
    be = LlamaCppBackend(cfg, wl)
    be.load()
    return be


def logits(be, index: int):
    import numpy as np

    import llama_cpp

    ptr = llama_cpp.llama_get_logits_ith(be._ctx, index)
    return np.ctypeslib.as_array(ptr, shape=(be._n_vocab,)).astype(np.float64).copy()


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    out = ap.parse_args().out

    note(f"Kimi-Linear Q8 GGUF, batch {B} vs each row alone, prompt {T}, greedy, {G} tokens\n")
    be = make(B)
    inputs = be.prepare_inputs()
    check("batch prompt is [batch, prompt]", inputs.shape == (B, T), str(inputs.shape))
    state = be.prefill(inputs)
    batch_prefill = [logits(be, (s + 1) * T - 1) for s in range(B)]
    batch_tokens = [state.last_token_ids[:, 0].tolist()]
    for _ in range(G - 1):
        state = be.decode_step(state)
        batch_tokens.append(state.last_token_ids[:, 0].tolist())
    be.teardown()
    del be

    single = make(1)
    for s in range(B):
        st = single.prefill(inputs[s])
        ref = logits(single, -1)
        got = batch_prefill[s]
        cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
        rel = float(np.abs(ref - got).max() / (np.abs(ref).max() or 1))
        toks = [int(st.last_token_ids[0, 0])]
        for _ in range(G - 1):
            st = single.decode_step(st)
            toks.append(int(st.last_token_ids[0, 0]))
        in_batch = [step[s] for step in batch_tokens]
        agree = next((i for i, (a, b) in enumerate(zip(toks, in_batch)) if a != b), len(toks))
        check(f"row {s}: prefill logits match (cosine >= 0.999)", cos >= 0.999,
              f"cosine {cos:.6f}, max |diff| / max |logit| {rel:.2e}")
        check(f"row {s}: same first token", toks[0] == in_batch[0], f"{toks[0]} vs {in_batch[0]}")
        note(f"  row {s}: greedy tokens agree for {agree}/{G} steps"
             + ("" if agree == G else f" (alone {toks}, in batch {in_batch})"))
    single.teardown()

    verdict = "ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"
    print(verdict)
    if out:
        out.write_text("## Correctness on Kimi (batch vs rows alone)\n\n" + "\n".join(lines)
                       + f"\n\n**{verdict}**\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
