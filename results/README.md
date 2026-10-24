# Results

Finished benchmark runs, grouped by what was measured. New runs from `nsbench run` still go to
`runs/` (the harness default); move them here once a set is done. `runs/_calibration` stays
in `runs/` because the harness reads it.

## Folder naming

```
results/<model>-<total params>[-a<active params>]-<precision>[-gguf]__<framework>/<experiment>/<run id>
```

- **model / params**: `qwen3-0.6b` = Qwen3 0.6B dense; `kimi-linear-48b-a3b` = Kimi-Linear 48B total, ~3B active (MoE).
- **precision**: weight format as loaded (`bf16`, `fp8` = FP8-dynamic safetensors, `q8_0` = llama.cpp Q8_0). `-gguf` marks a GGUF file.
- **framework**: `hf-transformers` = `--backend hf`; `llamacpp` = `--backend llamacpp` or `llama-bench`.
- **run id**: the harness's own `<UTC timestamp>__<model config>__<workload>__<backend>__<tag>`, unchanged.

## Index

| Folder | Model | Framework | Runs | Size | Start here |
|---|---|---|---|---|---|
| [qwen3-0.6b-bf16__hf-transformers](qwen3-0.6b-bf16__hf-transformers/) | Qwen3-0.6B, BF16 safetensors | HF Transformers | 5 | 3.7 GB | per-run `report.md` |
| [qwen3-0.6b-bf16-gguf__llamacpp](qwen3-0.6b-bf16-gguf__llamacpp/) | Qwen3-0.6B, BF16 GGUF | llama.cpp | 2 + llama-bench | 0.3 GB | [docs/09](../docs/09-llamacpp-comparison.md) |
| [kimi-linear-48b-a3b-fp8__hf-transformers](kimi-linear-48b-a3b-fp8__hf-transformers/) | Kimi-Linear-48B-A3B, FP8 dynamic | HF Transformers | 26 | 124 GB | [moe-queue/results.md](kimi-linear-48b-a3b-fp8__hf-transformers/moe-queue/results.md), [docs/07](../docs/07-moe-routing.md) |
| [kimi-linear-48b-a3b-q8_0-gguf__llamacpp](kimi-linear-48b-a3b-q8_0-gguf__llamacpp/) | Kimi-Linear-48B-A3B, Q8_0 GGUF | llama.cpp | 38 | 22 GB | [results.md](kimi-linear-48b-a3b-q8_0-gguf__llamacpp/results.md), [docs/10](../docs/10-kimi-llamacpp-moe-sweep.md) |

### What is in each experiment folder

**qwen3-0.6b-bf16__hf-transformers**
- `smoke/`: harness smoke tests (model config `smoke-model` = Qwen3-0.6B), 2026-10-05.
- `l2-sampling-check/`: nsys L2 sampling check, 2026-10-05 ([docs/08](../docs/08-nsys-l2-sampling.md)).
- `llamacpp-comparison/`: HF side of the HF-vs-llama.cpp comparison, 2026-10-09 ([docs/09](../docs/09-llamacpp-comparison.md)).

**qwen3-0.6b-bf16-gguf__llamacpp**
- `bringup/`: first `--backend llamacpp` runs in the testbench, plus their logs.
- `llama-bench/`: stock `llama-bench` decode/prefill numbers (JSON, Markdown, meta).

**kimi-linear-48b-a3b-fp8__hf-transformers**
- `early/`: first bring-up runs (default, balanced, long-context workloads).
- top-level run folders: the 22-workload MoE routing sweep (natural / fixed / disjoint routing × decode b1–b32, prefill p16–p512, long context p16384; longctx re-run without ncu).
- `moe-queue/`: queue logs, `results.md` table, `findings.md`.

**kimi-linear-48b-a3b-q8_0-gguf__llamacpp**: `results.md` = all 22 MoE workloads in one table, plus power and energy.
- `bringup/`: first Kimi Q8 llama.cpp runs (tier choices, 4-stage, application replay), 2026-10-11.
- `natural-b1/`: natural routing, batch 1 (decode, prefill, long context).
- `forced-b1/`: fixed / disjoint forced routing, batch 1.
- `batch/`: decode at batch 8, 16, 32 for all three routing modes.
- `validation-forced-routing/`, `validation-batch/`: correctness checks before the forced-routing and batch sets.
- each set's `queue/` holds its queue log, per-run logs and a per-set `results.md`.

## Regenerating tables

```
python scripts/summarize_kimi_q8_all.py   # kimi-linear-48b-a3b-q8_0-gguf__llamacpp/results.md
python scripts/summarize_moe_queue.py     # kimi-linear-48b-a3b-fp8__hf-transformers/moe-queue/results.md
```
