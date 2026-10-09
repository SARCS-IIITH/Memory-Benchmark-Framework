#!/bin/bash
# Phase 1 of docs/09-llamacpp-comparison.md: llama.cpp's own speed on Qwen3-0.6B, matched to
# the testbench's decode-focused workload (128 prompt -> 128 generated, batch 1, bf16).
#
# Two invocations, because llama-bench applies -d (context depth) to every test it runs:
#   prefill: pp128 from an empty context            (testbench prefill)
#   decode:  tg128 after 128 tokens already cached  (testbench decode after the 128-token prompt)
# Run inside tmux, with the GPU idle.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
GGUF=${GGUF:-$HOME/ai-models/gguf/Qwen3-0.6B-bf16.gguf}
OUT_DIR=${OUT_DIR:-$HOME/Memory-Benchmark-Framework/runs/llamacpp-qwen}
REPS=${REPS:-5}
TS=$(date +%Y%m%d-%H%M)
BENCH="$LLAMA_DIR/build/bin/llama-bench"

mkdir -p "$OUT_DIR"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "GPU is in use by $busy process(es); not running." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
fi

{
  echo "timestamp: $TS"
  echo "llama.cpp commit: $(cat "$LLAMA_DIR/PINNED_COMMIT" 2>/dev/null || echo unknown)"
  echo "gguf: $GGUF ($(stat -c %s "$GGUF") bytes)"
  echo "repetitions: $REPS"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv
} | tee "$OUT_DIR/llamabench-$TS.meta.txt"

run() {  # $1 = label, rest = llama-bench args
  local label=$1; shift
  echo "== $label: llama-bench $*"
  # One run: JSON to stdout, the readable table to stderr, so both hold the same numbers.
  "$BENCH" -m "$GGUF" -ngl 99 -r "$REPS" -o json -oe md "$@" \
    > "$OUT_DIR/llamabench-$TS-$label.json" 2> "$OUT_DIR/llamabench-$TS-$label.log"
  grep '^|' "$OUT_DIR/llamabench-$TS-$label.log" | tee "$OUT_DIR/llamabench-$TS-$label.md"
}

run prefill -p 128 -n 0 -d 0
run decode  -p 0 -n 128 -d 128
echo "LLAMABENCH DONE -> $OUT_DIR/llamabench-$TS-*"
