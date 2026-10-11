#!/usr/bin/env bash
# Full 4-stage run (calibration, baseline, nsys, ncu) of Kimi-Linear Q8_0 GGUF through
# the llamacpp backend, on the same workload and profile as the HF FP8 MoE run 1
# (moe-decode-natural-b1). Profile defaults to configs/profiles/moe.yaml; override with PROFILE=.
#
#   tmux new-session -d -s kimi-q8-4stage "./scripts/run_kimi_q8_llamacpp.sh; exec bash"
#   TIERS=1,2 TAG=llamacpp-t1 ./scripts/run_kimi_q8_llamacpp.sh      # add ncu tier 1
#
# ncu uses application replay here automatically (ncu.replay_mode: auto + the llamacpp
# backend), so memory stays at an unprofiled run's ~57 GiB; tier 1 + 2 take ~48 min.
#
# Alongside the run it samples /proc/meminfo every 5 s into <log>.mem.csv, tagged with the
# stage that is running, so the memory profile of each stage (ncu replay in particular) is
# recorded rather than inferred.
#
# Memory guard: if MemAvailable stays under GUARD_MB for GUARD_SECONDS, the run's process tree
# is killed (same guard as scripts/rerun_moe_longctx.sh).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NSBENCH="${HOME}/envs/nsbench/bin/nsbench"
TAG="${TAG:-llamacpp-4stage}"
PROFILE="${PROFILE:-configs/profiles/moe.yaml}"
TIERS="${TIERS:-}"
STAMP=$(date +%Y%m%d-%H%M)
RUN_LOG="runs/kimi-q8-${TAG}-${STAMP}.log"
MEM_CSV="${RUN_LOG%.log}.mem.csv"
GUARD_MB="${GUARD_MB:-2000}"
GUARD_SECONDS="${GUARD_SECONDS:-60}"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    echo "GPU is busy; not starting." >&2
    exit 1
fi

echo "profile: ${PROFILE}"
echo "log: ${RUN_LOG}"
echo "mem: ${MEM_CSV}"
started=$(date +%s)
"${NSBENCH}" run \
    --model configs/models/kimi-linear-48b-q8-gguf.yaml \
    --backend llamacpp \
    --workload configs/workloads/moe/moe-decode-natural-b1.yaml \
    --profile "${PROFILE}" \
    ${TIERS:+--tiers "${TIERS}"} \
    --tag "${TAG}" > "${RUN_LOG}" 2>&1 &
run_pid=$!

# Which stage is live: the worker's --mode, and for ncu the collection (newest ncu_tier*
# log in the run directory) plus the application-replay pass, counted from ncu's
# "Connected to process" lines (one per process launch).
stage() {
    local args
    args=$(pgrep -af "nsight_bench.worker" | grep -v pgrep | head -1)
    if [[ -z "${args}" ]]; then
        echo "other,,"
        return
    fi
    local mode
    mode=$(sed -n 's/.*--mode \([a-z_0-9]*\).*/\1/p' <<<"${args}")
    if [[ "${mode}" == ncu ]]; then
        local rd log coll pass
        rd=$(ls -td runs/*__"${TAG}" 2>/dev/null | head -1)
        log=$(ls -t "${rd}"/logs/ncu_tier*.stdout.log 2>/dev/null | head -1)
        coll=$(basename "${log:-none}" .stdout.log)
        pass=$(grep -c "Connected to process" "${log}" 2>/dev/null)
        echo "ncu,${coll},${pass}"
    else
        echo "${mode:-worker},,"
    fi
}

echo "time,elapsed_s,stage,ncu_collection,ncu_pass,mem_total_mb,mem_available_mb,mem_free_mb,cached_mb,committed_as_mb,swap_used_mb" > "${MEM_CSV}"
low_since=0
guard_tripped=0
while kill -0 "${run_pid}" 2>/dev/null; do
    read -r total avail free cached committed swaptotal swapfree < <(awk '
        /^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} /^MemFree:/ {f=$2} /^Cached:/ {c=$2}
        /^Committed_AS:/ {ca=$2} /^SwapTotal:/ {st=$2} /^SwapFree:/ {sf=$2}
        END {printf "%d %d %d %d %d %d %d\n", t/1024, a/1024, f/1024, c/1024, ca/1024, st/1024, sf/1024}' /proc/meminfo)
    echo "$(date '+%T'),$(( $(date +%s) - started )),$(stage),${total},${avail},${free},${cached},${committed},$(( swaptotal - swapfree ))" >> "${MEM_CSV}"
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
echo "=== finished (exit ${status}) after $(( ($(date +%s) - started) / 60 )) min ==="
