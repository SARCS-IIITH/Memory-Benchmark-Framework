#!/usr/bin/env bash
# Full 4-stage run (calibration, baseline, nsys, ncu) of Kimi-Linear Q8_0 GGUF through
# the llamacpp backend. Workload defaults to moe-decode-natural-b1 (same as the HF FP8 MoE
# run 1); override with WORKLOAD=<name in configs/workloads/moe/>. Profile defaults to
# configs/profiles/moe.yaml; override with PROFILE=.
#
#   tmux new-session -d -s kimi-q8-4stage "./scripts/run_kimi_q8_llamacpp.sh; exec bash"
#   WORKLOAD=moe-prefill-natural-p64 ./scripts/run_kimi_q8_llamacpp.sh
#   TIERS=2 TAG=llamacpp-t2 ./scripts/run_kimi_q8_llamacpp.sh        # exact tiers, no tier 1
#
# ncu uses application replay and tier 1 + 2 here automatically (ncu.replay_mode: auto and
# ncu.auto_tier1 + the llamacpp backend), so memory stays at an unprofiled run's ~57 GiB;
# ~48 min for a 128-token workload. Unless TIERS= is given, the run is stopped if its header
# does not show tier 1 (REQUIRE_TIER1=0 to skip that check).
#
# Alongside the run it samples /proc/meminfo every 5 s into <log>.mem.csv, tagged with the
# stage that is running, so the memory profile of each stage (ncu replay in particular) is
# recorded rather than inferred.
#
# Memory guard: if MemAvailable stays under GUARD_MB for GUARD_SECONDS, the run's process tree
# is killed (same guard as scripts/rerun_moe_longctx.sh).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ENV_DIR=~/envs/nsbench-moeroute for forced-routing workloads (patched llama.cpp).
ENV_DIR="${ENV_DIR:-${HOME}/envs/nsbench}"
NSBENCH="${ENV_DIR}/bin/nsbench"
TAG="${TAG:-llamacpp-4stage}"
PROFILE="${PROFILE:-configs/profiles/moe.yaml}"
TIERS="${TIERS:-}"
# Where the run directory is created (nsbench --output). A queue points this at its set folder.
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
# Extra nsbench run flags, e.g. EXTRA_ARGS=--skip-ncu for an nsys-only run.
EXTRA_ARGS="${EXTRA_ARGS:-}"
# After a successful run, measure its GPU energy (scripts/measure_energy.py -> metrics/energy.json,
# an unprofiled pass of ~1-2 min). ENERGY=0 to skip; skipped for nsys-only runs (--skip-ncu).
ENERGY="${ENERGY:-1}"
WORKLOAD="${WORKLOAD:-moe-decode-natural-b1}"
REQUIRE_TIER1="${REQUIRE_TIER1:-$([[ -z "${TIERS}" ]] && echo 1 || echo 0)}"
STAMP=$(date +%Y%m%d-%H%M)
RUN_LOG="${RUN_LOG:-runs/kimi-q8-${TAG}-${WORKLOAD}-${STAMP}.log}"
MEM_CSV="${RUN_LOG%.log}.mem.csv"
GUARD_MB="${GUARD_MB:-2000}"
GUARD_SECONDS="${GUARD_SECONDS:-60}"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    echo "GPU is busy; not starting." >&2
    exit 1
fi

echo "workload: ${WORKLOAD}"
echo "profile: ${PROFILE}"
echo "log: ${RUN_LOG}"
echo "mem: ${MEM_CSV}"
started=$(date +%s)
# Unbuffered, so the header reaches the log at once (the tier check below reads it).
PYTHONUNBUFFERED=1 "${NSBENCH}" run \
    --model configs/models/kimi-linear-48b-q8-gguf.yaml \
    --backend llamacpp \
    --workload "configs/workloads/moe/${WORKLOAD}.yaml" \
    --profile "${PROFILE}" \
    ${TIERS:+--tiers "${TIERS}"} \
    --output "${OUTPUT_ROOT}" \
    ${EXTRA_ARGS} \
    --tag "${TAG}" > "${RUN_LOG}" 2>&1 &
run_pid=$!

# The header ("ncu tiers  [...]") is printed before any stage starts.
if (( REQUIRE_TIER1 )); then
    for _ in $(seq 120); do
        grep -q "ncu tiers" "${RUN_LOG}" 2>/dev/null && break
        kill -0 "${run_pid}" 2>/dev/null || break
        sleep 1
    done
    if ! grep -q "ncu tiers  \[1, " "${RUN_LOG}" 2>/dev/null; then
        echo "$(date '+%F %T') TIER CHECK: header does not show ncu tier 1; stopping the run" \
            | tee -a "${RUN_LOG}"
        grep "ncu tiers" "${RUN_LOG}"
        pkill -TERM -P "${run_pid}"; kill -TERM "${run_pid}" 2>/dev/null
        wait "${run_pid}" 2>/dev/null
        exit 3
    fi
    echo "tier check: $(grep "ncu tiers" "${RUN_LOG}")"
fi

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
        rd=$(ls -td "${OUTPUT_ROOT}"/*__"${TAG}" 2>/dev/null | head -1)
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
if (( status == 0 && ENERGY )) && [[ "${EXTRA_ARGS}" != *--skip-ncu* ]]; then
    run_dir=$(sed -n 's/^Run -> //p' "${RUN_LOG}" | tail -1)
    if [[ -n "${run_dir}" ]]; then
        echo "energy pass: ${run_dir}"
        "${ENV_DIR}/bin/python" scripts/measure_energy.py "${run_dir}" 2>&1 | tail -3 | tee -a "${RUN_LOG}"
    fi
fi
echo "=== finished (exit ${status}) after $(( ($(date +%s) - started) / 60 )) min ==="
