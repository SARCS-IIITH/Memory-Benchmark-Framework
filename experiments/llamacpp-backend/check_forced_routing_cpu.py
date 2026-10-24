"""CPU check of the nsbench forced-routing patch to llama.cpp (forced-routing.patch here).

Run with the patched environment; it never touches the GPU (CUDA_VISIBLE_DEVICES is emptied
before llama_cpp loads) and builds its own tiny model, so it is safe while a benchmark runs:

    ~/envs/nsbench-moeroute/bin/python experiments/llamacpp-backend/check_forced_routing_cpu.py

Each llama.cpp run happens in a fresh subprocess, because the patch counts passes per process
(as every ncu application-replay pass will see it). What is checked:

1. The ids the patch generates (printed under LLAMA_MOE_FORCE_ROUTING_DEBUG=1) equal the hf
   hook's own ``RoutingController._forced`` for the same pass number and token count, for
   fixed and disjoint, over a multi-token prefill and several decode steps.
2. The graph really uses them: two tiny qwen3moe models identical except for the weights of
   experts k..E-1 give bit-identical logits under fixed routing (experts 0..k-1 only) and
   different logits under natural routing.
3. Natural routing creates no forced-routing input (no debug lines), so with the variable
   unset or "natural" the graph is the unpatched one.
4. The last layer, which llama.cpp runs on the output token only, gets that token's ids
   (out_rows=1 on the prefill) without tripping the shape assert.
5. The nsbench backend refuses forced routing on an unpatched build and accepts it here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
GGUF_PY = Path.home() / "llama.cpp" / "gguf-py"

N_EXPERT, TOP_K, N_LAYER = 16, 4, 2
PROMPT = [5, 9, 3, 17, 11, 2, 30]          # 7-token prefill, then DECODE steps
DECODE = [7, 13, 21]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---- tiny model -----------------------------------------------------------------------

def write_tiny_moe(path: Path, expert_seed: int, n_vocab: int = 32) -> None:
    """A 2-layer qwen3moe with 16 experts, top-4. ``expert_seed`` only changes experts >= TOP_K."""
    import numpy as np

    sys.path.insert(0, str(GGUF_PY))
    import gguf

    n_embd, n_head, head_dim, n_ff = 64, 4, 16, 32
    rng = np.random.default_rng(0)

    def w(*shape, r=rng):
        return (r.standard_normal(shape) * 0.2).astype(np.float32)

    wr = gguf.GGUFWriter(str(path), "qwen3moe")
    wr.add_context_length(64)
    wr.add_embedding_length(n_embd)
    wr.add_block_count(N_LAYER)
    wr.add_feed_forward_length(n_ff)
    wr.add_head_count(n_head)
    wr.add_head_count_kv(n_head)
    wr.add_key_length(head_dim)
    wr.add_value_length(head_dim)
    wr.add_rope_freq_base(10000.0)
    wr.add_layer_norm_rms_eps(1e-6)
    wr.add_expert_count(N_EXPERT)
    wr.add_expert_used_count(TOP_K)
    wr.add_expert_feed_forward_length(n_ff)
    wr.add_tokenizer_model("llama")
    wr.add_token_list([b"<unk>", b"<s>", b"</s>"] + [f"t{i}".encode() for i in range(n_vocab - 3)])
    wr.add_token_scores([0.0] * n_vocab)
    wr.add_token_types([2, 3, 3] + [1] * (n_vocab - 3))
    wr.add_bos_token_id(1)
    wr.add_eos_token_id(2)

    wr.add_tensor("token_embd.weight", w(n_vocab, n_embd))
    wr.add_tensor("output_norm.weight", np.ones(n_embd, np.float32))
    wr.add_tensor("output.weight", w(n_vocab, n_embd))
    other = np.random.default_rng(expert_seed)
    for i in range(N_LAYER):
        wr.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(n_embd, np.float32))
        wr.add_tensor(f"blk.{i}.attn_q.weight", w(n_head * head_dim, n_embd))
        wr.add_tensor(f"blk.{i}.attn_k.weight", w(n_head * head_dim, n_embd))
        wr.add_tensor(f"blk.{i}.attn_v.weight", w(n_head * head_dim, n_embd))
        wr.add_tensor(f"blk.{i}.attn_output.weight", w(n_embd, n_head * head_dim))
        wr.add_tensor(f"blk.{i}.attn_q_norm.weight", np.ones(head_dim, np.float32))
        wr.add_tensor(f"blk.{i}.attn_k_norm.weight", np.ones(head_dim, np.float32))
        wr.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(n_embd, np.float32))
        # Router strongly prefers the high experts, so natural routing actually uses the
        # experts that differ between the two models.
        gate = w(N_EXPERT, n_embd)
        gate[TOP_K:] += 0.5
        wr.add_tensor(f"blk.{i}.ffn_gate_inp.weight", gate)
        for name, shape in (("ffn_gate_exps", (N_EXPERT, n_ff, n_embd)),
                            ("ffn_up_exps", (N_EXPERT, n_ff, n_embd)),
                            ("ffn_down_exps", (N_EXPERT, n_embd, n_ff))):
            t = w(*shape)
            t[TOP_K:] = (other.standard_normal((N_EXPERT - TOP_K, *shape[1:])) * 0.2).astype(np.float32)
            wr.add_tensor(f"blk.{i}.{name}.weight", t)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()


# ---- one llama.cpp run, in a subprocess ----------------------------------------------

CHILD = r"""
import ctypes, json, os, sys
import numpy as np
import llama_cpp
model_path, prompt, decode = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
llama_cpp.llama_backend_init()
mp = llama_cpp.llama_model_default_params(); mp.n_gpu_layers = 0
model = llama_cpp.llama_model_load_from_file(model_path.encode(), mp)
cp = llama_cpp.llama_context_default_params()
cp.n_ctx = 64; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_threads = 2; cp.n_threads_batch = 2
ctx = llama_cpp.llama_init_from_model(model, cp)
n_vocab = llama_cpp.llama_vocab_n_tokens(llama_cpp.llama_model_get_vocab(model))
out = []
for toks in [prompt] + [[t] for t in decode]:
    arr = (llama_cpp.llama_token * len(toks))(*toks)
    assert llama_cpp.llama_decode(ctx, llama_cpp.llama_batch_get_one(arr, len(toks))) == 0
    logits = llama_cpp.llama_get_logits_ith(ctx, -1)
    out.append([float(logits[i]) for i in range(n_vocab)])
print("LOGITS " + json.dumps(out))
"""


def run(model: Path, mode: str | None) -> tuple[list, list[dict]]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LLAMA_MOE_FORCE_ROUTING_DEBUG="1")
    env.pop("LLAMA_MOE_FORCE_ROUTING", None)
    if mode is not None:
        env["LLAMA_MOE_FORCE_ROUTING"] = mode
    proc = subprocess.run([sys.executable, "-c", CHILD, str(model), json.dumps(PROMPT),
                           json.dumps(DECODE)], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"llama.cpp run ({mode}) failed:\n{proc.stderr[-3000:]}")
    logits = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("LOGITS "))[7:])
    calls = []
    for line in proc.stderr.splitlines():
        if line.startswith("nsbench_moe_route "):
            f = dict(kv.split("=", 1) for kv in line.split()[1:])
            calls.append({"call": int(f["call"]), "tokens": int(f["tokens"]),
                          "out_rows": int(f["out_rows"]),
                          "ids": [int(x) for x in f["ids"].split(",")]})
    return logits, calls


def hf_ids(mode: str, call: int, tokens: int) -> list[int]:
    """The hf hook's forced ids for this pass, from RoutingController._forced itself."""
    import torch

    from nsight_bench.compat.routing import RoutingController

    rc = RoutingController.__new__(RoutingController)
    rc.mode, rc.num_experts, rc._calls = mode, N_EXPERT, [call]
    idx = torch.zeros(tokens, TOP_K, dtype=torch.long)
    forced, _ = rc._forced(0, idx, torch.ones(tokens, TOP_K))
    return forced.reshape(-1).tolist()


def main() -> int:
    import llama_cpp

    from nsight_bench.backends.llamacpp import forced_routing_supported

    print(f"llama_cpp from {llama_cpp.__file__}")
    print("Patched library")
    check("llama_nsbench_forced_routing exported", forced_routing_supported())
    if failures:
        print("Not the patched build; run with ~/envs/nsbench-moeroute/bin/python")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.gguf", Path(tmp) / "b.gguf"
        write_tiny_moe(a, expert_seed=1)
        write_tiny_moe(b, expert_seed=2)

        print("Natural routing leaves the graph alone")
        nat_a, nat_calls = run(a, None)
        nat_a2, nat_calls2 = run(a, "natural")
        nat_b, _ = run(b, None)
        check("unset: no forced-routing input built", nat_calls == [])
        check("'natural': no forced-routing input built", nat_calls2 == [])
        check("unset and 'natural' give identical logits", nat_a == nat_a2)
        check("natural routing reaches experts >= k (models a, b differ)", nat_a != nat_b)

        print("Generated ids match the hf hook")
        for mode in ("fixed", "disjoint"):
            logits, calls = run(a, mode)
            check(f"{mode}: one pass per llama_decode ({1 + len(DECODE)})",
                  [c["call"] for c in calls] == list(range(1 + len(DECODE))),
                  str([(c['call'], c['tokens']) for c in calls]))
            check(f"{mode}: prefill pass has {len(PROMPT)} tokens and 1 output row",
                  bool(calls) and calls[0]["tokens"] == len(PROMPT) and calls[0]["out_rows"] == 1)
            mismatched = [c["call"] for c in calls
                          if c["ids"] != hf_ids(mode, c["call"], c["tokens"])]
            check(f"{mode}: ids equal RoutingController._forced on every pass", not mismatched,
                  f"mismatch on calls {mismatched}" if mismatched else
                  f"e.g. call 1: {calls[1]['ids'] if len(calls) > 1 else '?'}")
            check(f"{mode}: logits differ from natural", logits != nat_a)

        print("The graph uses the forced ids")
        fixed_a, _ = run(a, "fixed")
        fixed_b, _ = run(b, "fixed")
        check("fixed: models differing only in experts >= k give identical logits",
              fixed_a == fixed_b)
        dis_a, _ = run(a, "disjoint")
        dis_b, _ = run(b, "disjoint")
        check("disjoint: same two models differ (it reaches experts >= k)", dis_a != dis_b)
        again, _ = run(a, "disjoint")
        check("disjoint: a fresh process repeats the same logits (replay-safe)", again == dis_a)

    print("Backend guard")
    from nsight_bench.config import ModelConfig, WorkloadConfig
    from nsight_bench.backends import llamacpp as lc

    wl = WorkloadConfig.load(REPO / "configs/workloads/moe/moe-decode-natural-b1.yaml")
    wl.routing = "fixed"
    real = lc.forced_routing_supported
    lc.forced_routing_supported = lambda: False
    try:
        cfg = ModelConfig.load(REPO / "configs/models/kimi-linear-48b-q8-gguf.yaml")
        try:
            lc.LlamaCppBackend(cfg, wl).load()
            check("unpatched build: forced routing refused", False, "load() did not raise")
        except ValueError as e:
            check("unpatched build: forced routing refused", "forced-routing patch" in str(e))
    finally:
        lc.forced_routing_supported = real

    print("All checks passed." if not failures else f"{len(failures)} FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
