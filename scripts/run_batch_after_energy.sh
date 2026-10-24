#!/usr/bin/env bash
# One-off (2026-10-13): the 9 batch > 1 MoE workloads (natural / fixed / disjoint x b8 / b16 / b32)
# on Kimi-Linear Q8 GGUF via llama.cpp, after the forced-routing set, the folder move and the
# energy pass are all done (so nothing is measuring energy when the queue takes the GPU).
#   tmux new-session -d -s kimi-q8-batch "./scripts/run_batch_after_energy.sh; exec bash"
# Each run: full 4 stages, ncu tier 1 + 2 (application replay), then its GPU energy pass
# (run_kimi_q8_llamacpp.sh, ENERGY=1). All on ~/envs/nsbench-moeroute: forced routing needs the
# patched llama.cpp, and the patched build was validated identical for natural routing
# (runs/kimi-q8-llamacpp/validation-forced-routing/validation.md). Batch > 1 verified on Kimi:
# runs/kimi-q8-llamacpp/validation-batch/batch-validation.md.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SET_DIR=runs/kimi-q8-llamacpp/batch
mkdir -p "${SET_DIR}"
echo "$(date '+%F %T') waiting for the energy pass (runs/kimi-q8-llamacpp/energy-pass.log)"
until grep -q "energy pass done" runs/kimi-q8-llamacpp/energy-pass.log 2>/dev/null; do sleep 60; done
WORKLOADS="moe-decode-natural-b8 moe-decode-fixed-b8 moe-decode-disjoint-b8
    moe-decode-natural-b16 moe-decode-fixed-b16 moe-decode-disjoint-b16
    moe-decode-natural-b32 moe-decode-fixed-b32 moe-decode-disjoint-b32" \
SET_DIR="${SET_DIR}" ENV_DIR="${HOME}/envs/nsbench-moeroute" TAG=llamacpp-batch \
    ./scripts/run_kimi_q8_llamacpp_queue.sh
