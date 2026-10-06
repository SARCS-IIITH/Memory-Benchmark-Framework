#!/usr/bin/env bash
# End-to-end check that every stage of the harness works on this machine.
#
# Uses the smallest model available so the whole pipeline -- calibration, baseline, nsys,
# ncu, analysis, both reports -- runs in a few minutes rather than an hour. Run this after
# changing anything in the harness, and before starting a long benchmark session.

set -euo pipefail

ENV_DIR="${ENV_DIR:-${HOME}/envs/nsbench}"
NSBENCH="${ENV_DIR}/bin/nsbench"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

# A small model keeps the run short. Override with SMOKE_MODEL=/path/to/checkpoint.
DEFAULT_SMOKE_MODEL=$(ls -d "${HOME}"/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ \
    "${STORE:-/opt/ai-models}"/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ 2>/dev/null | head -1 || true)
SMOKE_MODEL="${SMOKE_MODEL:-${DEFAULT_SMOKE_MODEL}}"

if [[ -z "${SMOKE_MODEL}" || ! -d "${SMOKE_MODEL}" ]]; then
    echo "No smoke-test model found." >&2
    echo "Set SMOKE_MODEL=/path/to/a/small/checkpoint and re-run." >&2
    exit 1
fi

echo "==> 1/4  Preflight: what can this machine measure?"
"${NSBENCH}" preflight

echo
echo "==> 2/4  Discover: ${SMOKE_MODEL}"
"${NSBENCH}" discover "${SMOKE_MODEL}" --name smoke-model --out configs/models/_smoke.yaml

echo
echo "==> 3/4  Calibrate: is the memory derivation sound right now?"
"${NSBENCH}" calibrate --megabytes 128

echo
echo "==> 4/4  Full run (plumbing check)"
# --max-kernels is deliberately far below the real kernel count. A transformer decode step
# runs ~58 launches per block regardless of how many tokens are generated, so without a cap
# even the smallest workload profiles ~1600 kernels and takes half an hour. The run will
# report PARTIAL DATA -- that is expected and correct here: this step checks that every
# stage runs and produces its artefacts, not that the numbers are complete.
#
# --tiers 1,2 turns tier 1 back on (it is off by default) so the smoke test keeps
# exercising both paths: tier 1's byte totals, and tier 2 ranked from the nsys timeline.
"${NSBENCH}" run \
    --model configs/models/_smoke.yaml \
    --profile configs/profiles/quick.yaml \
    --tiers 1,2 \
    --prompt-tokens 128 \
    --generate-tokens 8 \
    --repeat 2 \
    --warmup 1 \
    --max-kernels 120 \
    --no-sweep \
    --tag smoke

echo
echo "NOTE: the PARTIAL DATA warning above is expected -- the smoke test caps ncu at 120"
echo "      kernels for speed. Real runs use the default cap and report full coverage."

echo
echo "Smoke test complete. Latest run:"
ls -dt runs/*__smoke | head -1
