#!/usr/bin/env bash
# Benchmark one checkpoint across every workload shape, then compare the shapes.
#
#   ./scripts/run_bench.sh /path/to/checkpoint [name]
#
# Sweeping the workload rather than the model answers a different question from
# run_sweep.sh: how does *this* model's memory behaviour change as the prompt gets longer and
# the KV cache grows? That is where the L2 hit rate collapses and per-token traffic climbs.
#
# Environment:
#   PROFILE     profiler config (default configs/profiles/standard.yaml)
#   WORKLOADS   space-separated workload configs to run

set -uo pipefail

ENV_DIR="${ENV_DIR:-/home/sarcs/envs/samarthamp}"
NSBENCH="${ENV_DIR}/bin/nsbench"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <model-dir> [name]" >&2
    exit 1
fi

MODEL_DIR="$1"
NAME="${2:-$(basename "${MODEL_DIR%/}")}"
PROFILE="${PROFILE:-configs/profiles/standard.yaml}"
WORKLOADS="${WORKLOADS:-configs/workloads/prefill-focused.yaml configs/workloads/balanced.yaml configs/workloads/decode-focused.yaml}"

CONFIG="configs/models/${NAME}.yaml"
echo "==> Discovering ${NAME}"
"${NSBENCH}" discover "${MODEL_DIR}" --name "${NAME}" --out "${CONFIG}" || exit 1

TAG="shapes-${NAME}"
FAILED=()

for WORKLOAD in ${WORKLOADS}; do
    SHAPE=$(basename "${WORKLOAD}" .yaml)
    echo
    echo "=============================================================="
    echo "==> ${NAME} / ${SHAPE}"
    echo "=============================================================="
    if ! "${NSBENCH}" run \
        --model "${CONFIG}" \
        --workload "${WORKLOAD}" \
        --profile "${PROFILE}" \
        --tag "${TAG}"; then
        echo "!! ${SHAPE} failed; continuing" >&2
        FAILED+=("${SHAPE}")
    fi
done

echo
echo "==> Comparing workload shapes"
mapfile -t RUN_DIRS < <(ls -d runs/*__"${TAG}" 2>/dev/null)
if [[ ${#RUN_DIRS[@]} -gt 0 ]]; then
    "${NSBENCH}" compare "${RUN_DIRS[@]}" --out "runs/_comparison_${TAG}"
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed shapes: ${FAILED[*]}" >&2
    exit 1
fi
