"""Check step B -- the analytic side of the analysis -- for hybrid MLA/KDA MoE models.

CPU only, no weights loaded (safetensors headers are read; tensors are not). Checks:

1. The decoding-state formula against **measured** live caches: the reduced Kimi-Linear of
   check_harness_gpu.py (1 MLA + 3 KDA layers, 72 tokens: 1,474,560 B KV and 6 MiB state,
   exact) and the real model in sanity_real_model.py (84 tokens: 11.5 / 40.0 / 1.9 MiB).
2. Dense models are unchanged: Qwen3-0.6B's KV equals the old 2*L*kv*hd formula at every
   length, and its weight expectation is still the resident size.
3. The weight expectation for Kimi comes from the checkpoint's tensors and is consistent
   everywhere it is used (ModelConfig and the physics check agree).
4. A full assemble + report pass over a synthetic Kimi run renders the new rows.

    cd ~/Memory-Benchmark-Framework
    ~/envs/nsbench/bin/python experiments/kimi-linear-compat/check_analysis_cpu.py [SMOKE_RUN_DIR]

SMOKE_RUN_DIR is any finished run directory to borrow measured artefacts from for step 4
(default: the newest runs/*__smoke). It is copied, never modified.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FAILURES: list[str] = []
MiB = 2 ** 20


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(REPO))
    from nsight_bench.config import ModelConfig

    kimi = ModelConfig.load(REPO / "configs/models/kimi-linear-48b-fp8.yaml")
    qwen = ModelConfig.load(REPO / "configs/models/qwen3-0.6b.yaml")

    print("1. Decoding-state formula vs measured caches")
    tiny = replace(kimi, num_layers=4, linear_attn_layers=[0, 1, 2])   # 3 KDA + 1 MLA
    parts = tiny.kv_cache_breakdown(72)
    check("reduced Kimi, 72 tokens: MLA KV exact", parts.get("kv") == 1_474_560,
          f"{parts.get('kv'):,} B (measured 1,474,560)")
    check("reduced Kimi: KDA recurrent state exact", parts.get("recurrent_state") == 3 * 2 * MiB,
          f"{parts.get('recurrent_state'):,} B (measured 6,291,456)")
    parts = kimi.kv_cache_breakdown(84)
    for kind, measured in (("kv", 11.5), ("recurrent_state", 40.0), ("conv_state", 1.9)):
        got = parts.get(kind, 0) / MiB
        check(f"real Kimi, 84 tokens: {kind}", abs(got - measured) < 0.06,
              f"{got:.2f} MiB (measured {measured} MiB)")
    latent = kimi.mla_latent_kv_bytes(8192)
    check("MLA latent is 576 elements/token/layer", latent == 7 * 8192 * 576 * 2,
          f"{latent / 1e6:.1f} MB at 8k vs {kimi.kv_cache_breakdown(8192)['kv'] / 1e6:,.0f} MB expanded")

    print("\n2. Dense model unchanged")
    old = lambda L: 2 * qwen.num_layers * qwen.num_key_value_heads * qwen.head_dim * L * 2  # noqa: E731
    check("Qwen KV == old formula at 1..32768 tokens",
          all(qwen.kv_cache_bytes(L) == old(L) for L in (1, 84, 512, 8192, 32768)))
    check("Qwen has no MLA/hybrid fields", not (qwen.is_mla or qwen.is_hybrid))
    check("Qwen active_weight_bytes is still the checkpoint size",
          qwen.active_weight_bytes() == qwen.estimated_weight_bytes())

    print("\n3. Kimi weight expectation from tensor roles")
    roles = kimi.weight_bytes_by_role
    check("roles sum to the checkpoint (within header bytes)",
          abs(sum(roles.values()) - kimi.weight_bytes_on_disk) < 10e6,
          f"{sum(roles.values()) / 1e9:.3f} vs {kimi.weight_bytes_on_disk / 1e9:.3f} GB")
    read, how = kimi.decode_read_weight_bytes()
    want = roles["other"] + roles["lm_head"] + roles["routed_experts"] * 8 / 256
    check("decode read = other + lm_head + 8/256 routed", abs(read - want) < 2, f"{read / 1e9:.3f} GB")
    check("ModelConfig.active_weight_bytes agrees", kimi.active_weight_bytes() == read)
    check("quant bits read from config_groups", kimi.bits_per_weight == 8)
    no_roles = replace(kimi, weight_bytes_by_role={})
    check("fallback without roles still works", no_roles.active_weight_bytes() > 0,
          f"{no_roles.active_weight_bytes() / 1e9:.2f} GB (estimate)")

    print("\n4. Assemble + report over a synthetic Kimi run")
    runs = sorted((REPO / "runs").glob("*__smoke"))
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (runs[-1] if runs else None)
    if source is None or not (source / "manifest.json").exists():
        check("a finished run to borrow artefacts from", False, "none found; skipping step 4")
    else:
        _synthetic_report(source, kimi)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: " + "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


def _synthetic_report(source: Path, kimi) -> None:
    """Borrow a real run's measured artefacts, swap in Kimi's model and cache, re-assemble.

    The measured byte counts stay those of the borrowed run, so the ratio is meaningless --
    the point is that every new code path runs and every new row renders.
    """
    from nsight_bench.analysis.assemble import assemble
    from nsight_bench.report.markdown import render_markdown

    with tempfile.TemporaryDirectory(prefix="kimi-synth-") as tmp:
        run = Path(tmp) / source.name
        shutil.copytree(source, run, ignore=shutil.ignore_patterns("*.ncu-rep", "*.nsys-rep"))

        cfg = json.loads((run / "run_config.json").read_text())
        cfg["model"] = kimi.to_dict()
        (run / "run_config.json").write_text(json.dumps(cfg))

        # The manifest records absolute artefact paths and a readable recorded path wins
        # (assemble._artifact), so on the same machine the copy would read the source run's
        # files. Point every recorded path into the copy instead.
        text = (run / "manifest.json").read_text().replace(str(source.resolve()), str(run))
        manifest = json.loads(text)
        for collection in (manifest.get("ncu") or {}).get("collections", []):
            collection["truncated"] = False
        (run / "manifest.json").write_text(json.dumps(manifest))

        context = cfg["workload"]["prompt_tokens"] + cfg["workload"]["generate_tokens"]
        parts = kimi.kv_cache_breakdown(context)
        for name in ("baseline_result.json",):
            path = run / "metrics" / name
            payload = json.loads(path.read_text())
            payload["result"]["kv_cache_bytes"] = sum(parts.values())
            payload["backend"]["cache_state_bytes"] = parts
            payload["backend"]["parameter_bytes_resident"] = kimi.weight_bytes_on_disk
            payload["backend"]["buffer_bytes_resident"] = 0
            path.write_text(json.dumps(payload))

        analysis = assemble(run)
        exp = analysis.phases["decode_step"].expectation
        read, _ = kimi.decode_read_weight_bytes()
        check("physics check uses the tensor-role read", exp is not None and exp.weight_bytes == read,
              f"{exp.weight_bytes / 1e9:.3f} GB, source '{exp.weight_bytes_source}'" if exp else "")
        check("MoE total is the resident size", exp is not None and exp.total_weight_bytes == kimi.weight_bytes_on_disk)
        check("KDA state write-back added", exp is not None and exp.state_write_bytes == 40 * MiB,
              f"{exp.state_write_bytes / MiB:.0f} MiB" if exp else "")
        check("expected = weights + measured state + write-back", exp is not None and
              exp.expected_bytes == read + sum(parts.values()) + 40 * MiB)
        fp = analysis.footprint
        check("footprint carries the cache breakdown", fp.cache_state_bytes == parts)
        check("footprint carries MLA expanded and latent sizes",
              fp.mla_latent_kv_bytes == kimi.mla_latent_kv_bytes(context)
              and fp.mla_expanded_kv_bytes == parts["kv"])

        md = render_markdown(analysis)
        for needle in ("Kimi Delta Attention", "MLA (kv_lora_rank 512", "routed-active",
                       "Recurrent state written back", "MLA latent equivalent",
                       "recurrent_state"):
            check(f"markdown report shows '{needle}'", needle in md)
        from nsight_bench.report.html import render_html
        html = render_html(analysis)
        check("html report renders with the new rows",
              "Recurrent state written back" in html and "MLA latent equivalent" in html)


if __name__ == "__main__":
    raise SystemExit(main())
