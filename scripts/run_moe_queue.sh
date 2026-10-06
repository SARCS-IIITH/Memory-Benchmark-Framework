#!/usr/bin/env bash
# Run the 22 MoE routing workloads (configs/workloads/moe/) on Kimi-Linear, one after another.
#
#   tmux new-session -d -s moe-queue "./scripts/run_moe_queue.sh"
#
# Order: batch-1 decode first, because its three routing modes must agree (a self-check of the
# routing hook; docs/07-moe-routing.md section 4.1). Then the decode batch sweep, then prefill,
# then long context last, since it is the most likely to fail on memory.
#
# Tier-2 ncu runs only on the 8 runs marked "ncu" below. The rest pass --skip-ncu, and their
# bytes come from nsys-sampled L2 traffic (configs/profiles/moe.yaml).
#
# One run failing does not stop the queue. Every start and finish is appended to
# runs/moe-queue/queue.log as one line, so progress can be read, or the queue resumed, after
# a disconnect. Runs already marked DONE in queue.log are skipped on a restart.
#
# Before each run the GPU must have no compute processes; the queue waits (polling each
# minute) rather than start on a busy GPU, since the GPU is shared.

set -uo pipefail

ENV_DIR="${ENV_DIR:-${HOME}/envs/nsbench}"
NSBENCH="${ENV_DIR}/bin/nsbench"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

MODEL="${MODEL:-configs/models/kimi-linear-48b-fp8.yaml}"
PROFILE="${PROFILE:-configs/profiles/moe.yaml}"
QDIR="runs/moe-queue"
QLOG="${QDIR}/queue.log"
mkdir -p "${QDIR}"

# name  ncu|skip
QUEUE=(
    "moe-decode-natural-b1 ncu"
    "moe-decode-fixed-b1 ncu"
    "moe-decode-disjoint-b1 ncu"
    "moe-decode-natural-b8 skip"
    "moe-decode-fixed-b8 skip"
    "moe-decode-disjoint-b8 skip"
    "moe-decode-natural-b16 skip"
    "moe-decode-fixed-b16 skip"
    "moe-decode-disjoint-b16 skip"
    "moe-decode-natural-b32 ncu"
    "moe-decode-fixed-b32 ncu"
    "moe-decode-disjoint-b32 ncu"
    "moe-prefill-natural-p16 skip"
    "moe-prefill-fixed-p16 skip"
    "moe-prefill-disjoint-p16 skip"
    "moe-prefill-natural-p64 skip"
    "moe-prefill-fixed-p64 skip"
    "moe-prefill-disjoint-p64 skip"
    "moe-prefill-natural-p512 ncu"
    "moe-prefill-fixed-p512 skip"
    "moe-prefill-disjoint-p512 skip"
    "moe-longctx-natural-p16384 ncu"
)

log() { echo "$(date '+%F %T') $*" | tee -a "${QLOG}"; }

gpu_busy() {
    [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
}

log "QUEUE START model=${MODEL} profile=${PROFILE} runs=${#QUEUE[@]}"
index=0
for entry in "${QUEUE[@]}"; do
    index=$((index + 1))
    read -r name mode <<< "${entry}"
    if grep -q " DONE ${name} " "${QLOG}" 2>/dev/null; then
        log "SKIP ${name} (already DONE)"
        continue
    fi
    while gpu_busy; do
        log "WAIT ${name}: GPU has compute processes; retrying in 60 s"
        sleep 60
    done

    extra=()
    [[ "${mode}" == "skip" ]] && extra=(--skip-ncu)
    run_log="${QDIR}/${index}-${name}.log"
    log "START ${index}/${#QUEUE[@]} ${name} ncu=${mode} log=${run_log}"
    started=$(date +%s)
    "${NSBENCH}" run \
        --model "${MODEL}" \
        --workload "configs/workloads/moe/${name}.yaml" \
        --profile "${PROFILE}" \
        --no-sweep \
        --tag moe \
        "${extra[@]}" > "${run_log}" 2>&1
    status=$?
    run_dir=$(grep -o 'runs/[^ ]*__moe' "${run_log}" | tail -1)
    minutes=$(( ($(date +%s) - started) / 60 ))
    if [[ ${status} -eq 0 ]]; then
        log "DONE ${name} exit=0 ${minutes}min run=${run_dir:-unknown}"
    else
        log "FAIL ${name} exit=${status} ${minutes}min run=${run_dir:-unknown} (see ${run_log})"
    fi
done
log "QUEUE END"
