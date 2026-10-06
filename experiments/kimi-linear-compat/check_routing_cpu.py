"""Check forced and observed MoE routing (nsight_bench/compat/routing.py) on the CPU.

Builds one Kimi-Linear MoE layer from the checkpoint's own modelling code -- the real
KimiMoEGate and moe_infer, shrunk to 16 small experts -- so nothing needs the GPU or the
weights. Also checks the routing-aware decode expectation against the full Kimi config.

    cd ~/Memory-Benchmark-Framework
    ~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_routing_cpu.py [KIMI_DIR]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # prove nothing here needs the GPU

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KIMI_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "kimi-fp8-meta")

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from nsight_bench import compat
    from nsight_bench.compat.routing import expected_distinct_experts
    from nsight_bench.config import ModelConfig, WorkloadConfig

    compat.apply_remote_code_shims()
    cfg = AutoConfig.from_pretrained(KIMI_DIR, trust_remote_code=True)
    cfg.num_experts, cfg.num_experts_per_token = 16, 4
    cfg.hidden_size, cfg.moe_intermediate_size = 32, 16
    block_cls = get_class_from_dynamic_module("modeling_kimi.KimiSparseMoeBlock", KIMI_DIR)

    torch.manual_seed(0)
    block = block_cls(cfg).eval()
    with torch.no_grad():
        block.gate.e_score_correction_bias.zero_()
        for p in block.parameters():          # some tensors are created uninitialised
            if not torch.isfinite(p).all():
                p.normal_(0, 0.02)
    holder = torch.nn.ModuleDict({"layer0": block})
    E, k = 16, 4
    x = torch.randn(1, 6, 32)                      # one pass of 6 tokens

    def run():
        with torch.inference_mode():
            return block(x)

    def gate(inputs):
        with torch.inference_mode():
            return block.gate(inputs)

    print("1. Gate discovery")
    ctl = compat.install_routing(holder, "natural", observe=True)
    check("KimiMoEGate found", ctl is not None and len(ctl.gates) == 1,
          type(block.gate).__name__)
    check("experts and top_k read from the gate", ctl.num_experts == E and ctl.top_k == k,
          f"{ctl.num_experts} experts, top-{ctl.top_k}")
    ctl.remove()

    print("\n2. Natural routing is untouched")
    reference = run()
    natural_idx, natural_w = gate(x)
    ctl = compat.install_routing(holder, "natural", observe=True)
    observed = run()
    check("output bit-identical with the observer installed", torch.equal(reference, observed))
    phase = ctl.summary()["phases"]["prefill"]
    want = len(set(natural_idx.flatten().tolist()))
    check("observer counts the router's distinct experts",
          phase["distinct_experts_per_pass"]["mean"] == want, f"{want} of {E}")
    check("observer records tokens per pass", phase["tokens_per_pass"] == 6)
    ctl.remove()
    check("removing the hook restores the model", torch.equal(run(), reference))

    print("\n3. Fixed routing")
    ctl = compat.install_routing(holder, "fixed", observe=True)
    idx, weight = gate(x)
    check("every token routed to experts 0..k-1",
          torch.equal(idx, torch.arange(k).expand(6, k)))
    check("weights equal, at the router's own per-token total",
          torch.allclose(weight.sum(-1), natural_w.sum(-1))
          and torch.allclose(weight, weight[:, :1].expand_as(weight)),
          f"{weight[0, 0].item():.4f} each")
    forced_out = run()
    check("output changes, shape does not",
          forced_out.shape == reference.shape and not torch.equal(forced_out, reference))
    check("observer sees exactly k distinct experts",
          ctl.summary()["phases"]["prefill"]["distinct_experts_per_pass"]["max"] == k)
    ctl.remove()

    print("\n4. Disjoint routing")
    ctl = compat.install_routing(holder, "disjoint", observe=True)
    first = gate(x)[0]
    second = gate(x)[0]
    one_token = gate(x[:, :1])[0]
    check("no expert shared within a pass while k*T <= E",
          len(set(first[:4].flatten().tolist())) == 16, "4 tokens x 4 = all 16 experts")
    check("a pass larger than E covers every expert",
          len(set(first.flatten().tolist())) == E, "6 tokens x 4 = 24 > 16")
    check("each token's k experts are distinct",
          all(len(set(row.tolist())) == k for row in first))
    check("the expert block rotates between passes", not torch.equal(first, second))
    check("a 1-token pass gets k distinct experts", len(set(one_token.flatten().tolist())) == k)
    out = run()
    check("full layer runs under disjoint routing", torch.isfinite(out).all().item())
    ctl.remove()

    print("\n5. Install guards")
    dense = torch.nn.Sequential(torch.nn.Linear(4, 4))
    check("natural on a dense model installs nothing",
          compat.install_routing(dense, "natural", observe=True) is None)
    try:
        compat.install_routing(dense, "fixed")
        check("forced routing on a model without a gate raises", False)
    except RuntimeError:
        check("forced routing on a model without a gate raises", True)
    try:
        WorkloadConfig(routing="sideways")
        check("WorkloadConfig rejects an unknown routing", False)
    except ValueError:
        check("WorkloadConfig rejects an unknown routing", True)
    loaded = {
        p.stem: WorkloadConfig.load(p).routing
        for p in sorted((REPO / "configs/workloads/moe").glob("*.yaml"))
    }
    check("MoE workload files load with their routing",
          len(loaded) == 22 and all(name.split("-")[2] == r for name, r in loaded.items()),
          f"{len(loaded)} files")

    print("\n6. Routing-aware decode expectation (full Kimi config)")
    table = {1: (8, 8, 8), 8: (8, 57.0, 64), 16: (8, 101.9, 128), 32: (8, 163.0, 256)}
    for tokens, (fixed, natural, disjoint) in table.items():
        got = tuple(expected_distinct_experts(m, tokens, 256, 8)
                    for m in ("fixed", "natural", "disjoint"))
        check(f"distinct experts at T={tokens}",
              got[0] == fixed and abs(got[1] - natural) < 0.5 and got[2] == disjoint,
              "fixed {:.0f}, natural {:.1f}, disjoint {:.0f}".format(*got))
    model = ModelConfig.load(REPO / "configs/models/kimi-linear-48b-fp8.yaml")
    roles = model.weight_bytes_by_role
    routed = roles["routed_experts"]
    always = sum(roles.values()) - roles.get("embedding", 0) - routed
    base, _ = model.decode_read_weight_bytes()
    check("batch-1 natural unchanged (routed experts at k/E)",
          base == always + int(routed * 8 / 256), f"{base / 1e9:.2f} GB")
    b32 = {m: model.decode_read_weight_bytes(32, m)[0] for m in ("fixed", "natural", "disjoint")}
    check("batch 32 orders fixed < natural < disjoint",
          b32["fixed"] < b32["natural"] < b32["disjoint"],
          ", ".join(f"{m} {v / 1e9:.2f} GB" for m, v in b32.items()))
    check("batch-32 disjoint reads every routed expert", b32["disjoint"] == always + routed)
    check("fixed does not grow with batch", b32["fixed"] == base)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: " + "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
