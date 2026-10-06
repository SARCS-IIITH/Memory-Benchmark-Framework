#!/usr/bin/env bash
# Re-run the MoE long-context workload (run 22) WITHOUT ncu, after the 2026-10-06 OOM crash.
#
#   tmux new-session -d -s moe-longctx "./scripts/rerun_moe_longctx.sh; exec bash"
#
# ncu tier 2 on the 16k-token prefill exhausted the unified memory pool and took the DGX down
# (progress.md caveat 12). Baseline and nsys already passed on this workload, so this re-run
# uses --skip-ncu.
#
# Memory guard: if MemAvailable stays under GUARD_MB for GUARD_SECONDS, the run's process tree
# is killed. A failed run is better than another machine-wide outage on a shared box.
#
# Logs START / DONE / FAIL to runs/moe-queue/queue.log in the queue's own format, so
# scripts/summarize_moe_queue.py picks the result up.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NSBENCH="${HOME}/envs/nsbench/bin/nsbench"
NAME=moe-longctx-natural-p16384
QLOG=runs/moe-queue/queue.log
RUN_LOG="runs/moe-queue/22-${NAME}-rerun-skip-ncu.log"
GUARD_MB="${GUARD_MB:-2000}"
GUARD_SECONDS="${GUARD_SECONDS:-60}"

log() { echo "$(date '+%F %T') $*" | tee -a "${QLOG}"; }

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    echo "GPU is busy; not starting." >&2
    exit 1
fi

log "START 22/22 ${NAME} ncu=skip (re-run after OOM crash) log=${RUN_LOG}"
started=$(date +%s)
"${NSBENCH}" run \
    --model configs/models/kimi-linear-48b-fp8.yaml \
    --workload "configs/workloads/moe/${NAME}.yaml" \
    --profile configs/profiles/moe.yaml \
    --no-sweep --tag moe --skip-ncu > "${RUN_LOG}" 2>&1 &
run_pid=$!

low_since=0
guard_tripped=0
while kill -0 "${run_pid}" 2>/dev/null; do
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if (( avail < GUARD_MB )); then
        (( low_since == 0 )) && low_since=$(date +%s)
        if (( $(date +%s) - low_since >= GUARD_SECONDS )); then
            echo "$(date '+%F %T') MEMORY GUARD: ${avail} MB available for ${GUARD_SECONDS}s; killing the run" \
                | tee -a "${RUN_LOG}"
            pkill -TERM -P "${run_pid}"; kill -TERM "${run_pid}"
            sleep 10
            pkill -KILL -f "nsight_bench.worker"; kill -KILL "${run_pid}" 2>/dev/null
            guard_tripped=1
            break
        fi
    else
        low_since=0
    fi
    sleep 5
done
wait "${run_pid}" 2>/dev/null
status=$?
(( guard_tripped )) && status=137

run_dir=$(grep -o 'runs/[^ ]*__moe' "${RUN_LOG}" | tail -1)
minutes=$(( ($(date +%s) - started) / 60 ))
if [[ ${status} -eq 0 ]]; then
    log "DONE ${NAME} exit=0 ${minutes}min run=${run_dir:-unknown}"
else
    log "FAIL ${NAME} exit=${status} ${minutes}min run=${run_dir:-unknown} (see ${RUN_LOG})"
fi
