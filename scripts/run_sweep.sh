#!/usr/bin/env bash
# Benchmark several checkpoints under the same workload, then compare them.
#
#   ./scripts/run_sweep.sh /path/to/model-a /path/to/model-b ...
#
# Each checkpoint is discovered, benchmarked and reported independently, and a comparison
# matrix is built at the end. One model failing does not stop the sweep -- the failure is
# recorded and the rest continue, because a nine-hour sweep should not be lost to a tenth
# checkpoint that will not load.
#
# Environment:
#   WORKLOAD   workload config      (default configs/workloads/decode-focused.yaml)
#   PROFILE    profiler config      (default configs/profiles/standard.yaml)
#   TAG        label for the runs   (default "sweep")

set -uo pipefail

ENV_DIR="${ENV_DIR:-/home/sarcs/envs/samarthamp}"
NSBENCH="${ENV_DIR}/bin/nsbench"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

WORKLOAD="${WORKLOAD:-configs/workloads/decode-focused.yaml}"
PROFILE="${PROFILE:-configs/profiles/standard.yaml}"
TAG="${TAG:-sweep}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <model-dir> [<model-dir> ...]" >&2
    exit 1
fi

echo "==> Preflight"
"${NSBENCH}" preflight || {
    echo "Preflight reported a blocking problem. Fix it before sweeping." >&2
    exit 1
}

FAILED=()
STARTED_AT=$(date +%s)

for MODEL_DIR in "$@"; do
    NAME=$(basename "${MODEL_DIR%/}")
    echo
    echo "=============================================================="
    echo "==> ${NAME}"
    echo "=============================================================="

    CONFIG="configs/models/${NAME}.yaml"
    if ! "${NSBENCH}" discover "${MODEL_DIR}" --name "${NAME}" --out "${CONFIG}"; then
        echo "!! discover failed for ${NAME}; skipping" >&2
        FAILED+=("${NAME} (discover)")
        continue
    fi

    if ! "${NSBENCH}" run \
        --model "${CONFIG}" \
        --workload "${WORKLOAD}" \
        --profile "${PROFILE}" \
        --tag "${TAG}"; then
        echo "!! run failed for ${NAME}; continuing with the rest" >&2
        FAILED+=("${NAME} (run)")
    fi
done

echo
echo "==> Comparison"
mapfile -t RUN_DIRS < <(ls -d runs/*__"${TAG}" 2>/dev/null)
if [[ ${#RUN_DIRS[@]} -gt 0 ]]; then
    "${NSBENCH}" compare "${RUN_DIRS[@]}" --out "runs/_comparison_${TAG}"
else
    echo "No runs to compare." >&2
fi

ELAPSED=$(( $(date +%s) - STARTED_AT ))
echo
printf 'Sweep finished in %dh %dm.\n' $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60))
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
fi
