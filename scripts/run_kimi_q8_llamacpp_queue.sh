#!/usr/bin/env bash
# The MoE workloads llama.cpp can run as written (batch 1, natural routing) on Kimi-Linear
# Q8_0 GGUF, one after another, each a full 4-stage run with ncu tier 1 + 2 (application
# replay). Each run goes through scripts/run_kimi_q8_llamacpp.sh, so each gets its own log,
# /proc/meminfo samples, memory guard, and the check that tier 1 is on.
#
#   tmux new-session -d -s kimi-q8-queue "./scripts/run_kimi_q8_llamacpp_queue.sh; exec bash"
#
# Long context last: it is the only one whose memory has not been measured yet.
#
# Everything for one queue lives in one set folder, SET_DIR (default
# runs/kimi-q8-llamacpp/natural-b1): the run directories themselves, and queue/ with the status
# log, summary and per-run logs. Status: every start, wait and finish is appended to
# <SET_DIR>/queue/queue.log as one line; runs already marked DONE there are skipped on a
# restart. Summary: after each start and finish, scripts/summarize_kimi_q8_queue.py rewrites
# <SET_DIR>/queue/results.md (one row per run). Each run's own log + memory samples:
# <SET_DIR>/queue/<n>-<workload>.log and .mem.csv. One run failing does not stop the queue. Before each run the queue waits
# (polling each minute) for the GPU to have no compute processes, since the GPU is shared.
#
# Written up in docs/10-kimi-llamacpp-moe-sweep.md.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${TAG:-llamacpp-queue}"
SET_DIR="${SET_DIR:-runs/kimi-q8-llamacpp/natural-b1}"
QDIR="${SET_DIR}/queue"
QLOG="${QDIR}/queue.log"
mkdir -p "${QDIR}"

# Override with WORKLOADS="name1 name2 ..." (and ENV_DIR=~/envs/nsbench-moeroute for
# fixed/disjoint routing, which needs the patched llama.cpp).
DEFAULT_QUEUE="moe-decode-natural-b1 moe-prefill-natural-p16 moe-prefill-natural-p64
    moe-prefill-natural-p512 moe-longctx-natural-p16384"
QUEUE=(${WORKLOADS:-${DEFAULT_QUEUE}})

log() { echo "$(date '+%F %T') $*" | tee -a "${QLOG}"; }

# runs/kimi-q8-queue/results.md: one table row per finished run, regenerated after every
# start and finish (same layout idea as runs/kimi-fp8-safetensors-hf/moe-queue/results.md from the HF queue).
summarize() {
    "${HOME}/envs/nsbench/bin/python" scripts/summarize_kimi_q8_queue.py "${QDIR}" > /dev/null 2>&1 \
        || echo "$(date '+%F %T') summarize FAILED" >> "${QDIR}/summarize-errors.log"
}

log "QUEUE START model=kimi-linear-48b-q8-gguf backend=llamacpp tag=${TAG} env=${ENV_DIR:-${HOME}/envs/nsbench} runs=${#QUEUE[@]}"
index=0
for wl in "${QUEUE[@]}"; do
    index=$((index + 1))
    if grep -q " DONE ${wl} " "${QLOG}" 2>/dev/null; then
        log "SKIP ${wl} (already DONE)"
        continue
    fi
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
        log "WAIT ${wl}: GPU has compute processes; retrying in 60 s"
        sleep 60
    done
    run_log="${QDIR}/${index}-${wl}.log"
    log "START ${index}/${#QUEUE[@]} ${wl} log=${run_log}"
    summarize
    started=$(date +%s)
    WORKLOAD="${wl}" TAG="${TAG}" RUN_LOG="${run_log}" OUTPUT_ROOT="${SET_DIR}" \
        ./scripts/run_kimi_q8_llamacpp.sh
    status=$?
    minutes=$(( ($(date +%s) - started) / 60 ))
    run_dir=$(grep -o "runs/[^ ]*__${TAG}" "${run_log}" 2>/dev/null | tail -1)
    if (( status == 0 )); then
        log "DONE ${wl} exit=0 ${minutes}min run=${run_dir:-unknown} log=${run_log}"
    else
        log "FAIL ${wl} exit=${status} ${minutes}min run=${run_dir:-unknown} log=${run_log}"
    fi
    summarize
done
log "QUEUE END"
summarize
