#!/usr/bin/env bash
# After the natural-routing sweep (scripts/run_kimi_q8_llamacpp_queue.sh, runs/kimi-q8-queue)
# ends: move it into its set folder, validate the forced-routing llama.cpp build on the GPU,
# and only if every check passes, queue the 8 batch-1 fixed/disjoint workloads.
#
#   tmux new-session -d -s forced-after-queue "./scripts/run_forced_routing_after_queue.sh; exec bash"
#
# 1. Wait (polling each minute) for QUEUE END in runs/kimi-q8-queue/queue.log.
# 2. scripts/move_run_set.py -> runs/kimi-q8-llamacpp-sweep (run dirs + queue/).
# 2b. Batch > 1 verification on Kimi (gates nothing; for the user to review before any batch
#    workload is queued), in runs/kimi-q8-llamacpp-batch-validation: check_batch_gpu.py (batch 4
#    vs each row alone), nsys-only natural-b8 x2 and disjoint-b8 x2 (kernel sequences repeat),
#    natural-b32 x1 (4,096-token prefill runs; memory). Report: batch-validation.md.
# 3. Validation runs in runs/kimi-q8-llamacpp-forced-validation, on ~/envs/nsbench-moeroute:
#    moe-decode-natural-b1 with ncu tier 1 (patch off must change nothing), and
#    moe-decode-fixed-b1 / moe-decode-disjoint-b1 twice each, nsys only (replay safety).
# 4. experiments/llamacpp-backend/check_forced_routing_gpu.py -> validation.md; stop on failure.
# 5. The forced queue: runs/kimi-q8-llamacpp-forced-routing (run dirs + queue/), ~7 h.
#
# Progress: runs/kimi-q8-llamacpp-forced-validation/validation.log (one line per step).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${HOME}/envs/nsbench/bin/python"
MOE_ENV="${HOME}/envs/nsbench-moeroute"
SWEEP_Q="runs/kimi-q8-queue"
SWEEP_SET="runs/kimi-q8-llamacpp-sweep"
ABORTED="runs/20261012T182459Z__kimi-linear-48b-q8-gguf__moe-decode-natural-b1__llamacpp__llamacpp-queue"
VAL="runs/kimi-q8-llamacpp-forced-validation"
BVAL="runs/kimi-q8-llamacpp-batch-validation"
FORCED="runs/kimi-q8-llamacpp-forced-routing"
FORCED_WORKLOADS="moe-decode-fixed-b1 moe-decode-disjoint-b1
    moe-prefill-fixed-p16 moe-prefill-disjoint-p16 moe-prefill-fixed-p64 moe-prefill-disjoint-p64
    moe-prefill-fixed-p512 moe-prefill-disjoint-p512"

mkdir -p "${VAL}/logs" "${BVAL}/logs"
VLOG="${VAL}/validation.log"
log() { echo "$(date '+%F %T') $*" | tee -a "${VLOG}"; }

wait_gpu() {
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
        sleep 60
    done
}

# 1-2. The sweep: wait for it, then give it its own folder.
if [[ -d "${SWEEP_SET}/queue" ]]; then
    log "sweep already in ${SWEEP_SET}"
else
    log "waiting for QUEUE END in ${SWEEP_Q}/queue.log"
    until tail -1 "${SWEEP_Q}/queue.log" 2>/dev/null | grep -q "QUEUE END"; do sleep 60; done
    log "sweep ended: $(grep -c ' DONE ' "${SWEEP_Q}/queue.log") DONE, $(grep -c ' FAIL ' "${SWEEP_Q}/queue.log") FAIL"
    if ! "${PY}" scripts/move_run_set.py "${SWEEP_Q}" "${SWEEP_SET}" --extra "${ABORTED}" >> "${VLOG}" 2>&1; then
        log "MOVE FAILED (see above); stopping"
        exit 1
    fi
    "${PY}" scripts/summarize_kimi_q8_queue.py "${SWEEP_SET}/queue" > /dev/null 2>&1
    log "sweep moved to ${SWEEP_SET}"
fi
REF=$(grep -o "DONE moe-decode-natural-b1 .*run=\S*" "${SWEEP_SET}/queue/queue.log" | sed 's/.*run=//' | tail -1)
if [[ -z "${REF}" || ! -d "${REF}" ]]; then
    log "no finished moe-decode-natural-b1 in the sweep to compare against; stopping"
    exit 1
fi
log "reference: ${REF}"

# Validation runs.
one_run() {   # tag workload tiers extra_args [env_dir] [output_root]
    local tag=$1 wl=$2 tiers=$3 extra=$4 env=${5:-${MOE_ENV}} out=${6:-${VAL}} status
    if ls -d "${out}"/*__"${tag}" > /dev/null 2>&1; then
        log "SKIP ${tag} (run dir exists)"
        return 0
    fi
    wait_gpu
    log "START ${tag} (${wl}${tiers:+, tiers ${tiers}}${extra:+, ${extra}})"
    WORKLOAD="${wl}" TAG="${tag}" TIERS="${tiers}" EXTRA_ARGS="${extra}" REQUIRE_TIER1=0 \
        ENV_DIR="${env}" OUTPUT_ROOT="${out}" RUN_LOG="${out}/logs/${tag}.log" \
        ./scripts/run_kimi_q8_llamacpp.sh >> "${VLOG}" 2>&1
    status=$?
    log "$( (( status == 0 )) && echo DONE || echo "FAIL exit=${status}") ${tag}"
    return "${status}"
}
# 2b. Batch > 1 on Kimi: report only, never stops the chain.
if [[ ! -f "${BVAL}/batch-validation.md" ]]; then
    wait_gpu
    log "START batch correctness check (batch 4 vs rows alone)"
    "${PY}" experiments/llamacpp-backend/check_batch_gpu.py --out "${BVAL}/correctness.md" \
        > "${BVAL}/logs/check_batch_gpu.log" 2>&1
    log "batch correctness: $(tail -1 "${BVAL}/logs/check_batch_gpu.log")"
    one_run batch-natural-b8-nsys-a moe-decode-natural-b8 "" "--skip-ncu" "${HOME}/envs/nsbench" "${BVAL}"
    one_run batch-natural-b8-nsys-b moe-decode-natural-b8 "" "--skip-ncu" "${HOME}/envs/nsbench" "${BVAL}"
    one_run batch-disjoint-b8-nsys-a moe-decode-disjoint-b8 "" "--skip-ncu" "${MOE_ENV}" "${BVAL}"
    one_run batch-disjoint-b8-nsys-b moe-decode-disjoint-b8 "" "--skip-ncu" "${MOE_ENV}" "${BVAL}"
    one_run batch-natural-b32-nsys moe-decode-natural-b32 "" "--skip-ncu" "${HOME}/envs/nsbench" "${BVAL}"
    {
        echo "# Batch > 1 on Kimi-Linear Q8 (llama.cpp): GPU verification"
        echo
        cat "${BVAL}/correctness.md" 2>/dev/null || echo "correctness check produced no report (see logs/check_batch_gpu.log)"
        for pair in natural-b8 disjoint-b8; do
            echo; echo "## ${pair}: kernel sequences, two runs"; echo; echo '```'
            "${PY}" experiments/llamacpp-backend/compare_kernel_sequences.py \
                $(ls -d "${BVAL}"/*__batch-${pair}-nsys-a "${BVAL}"/*__batch-${pair}-nsys-b 2>/dev/null) 2>&1
            echo '```'
        done
        echo; echo "## natural-b32: one run (4,096-token prefill)"; echo; echo '```'
        "${PY}" experiments/llamacpp-backend/compare_kernel_sequences.py \
            $(ls -d "${BVAL}"/*__batch-natural-b32-nsys 2>/dev/null) 2>&1
        echo '```'
        b32_log="${BVAL}/logs/batch-natural-b32-nsys.mem.csv"
        [[ -f "${b32_log}" ]] && awk -F, 'NR>1{u=($6-$7)/1024; if(u>m)m=u}END{printf "\nPeak system memory in use during the b32 run: %.1f GiB\n", m}' "${b32_log}"
    } > "${BVAL}/batch-validation.md"
    log "batch verification written: ${BVAL}/batch-validation.md (nothing queued from it)"
fi

one_run moeroute-natural-t1 moe-decode-natural-b1 1 "" || { log "natural validation run failed; stopping"; exit 1; }
for mode in fixed disjoint; do
    for rep in a b; do
        one_run "moeroute-${mode}-nsys-${rep}" "moe-decode-${mode}-b1" "" "--skip-ncu" \
            || { log "${mode} validation run failed; stopping"; exit 1; }
    done
done
rundir() { ls -d "${VAL}"/*__"$1" | tail -1; }

# 4. The gate.
if "${PY}" experiments/llamacpp-backend/check_forced_routing_gpu.py \
        --reference "${REF}" --natural "$(rundir moeroute-natural-t1)" \
        --fixed "$(rundir moeroute-fixed-nsys-a)" "$(rundir moeroute-fixed-nsys-b)" \
        --disjoint "$(rundir moeroute-disjoint-nsys-a)" "$(rundir moeroute-disjoint-nsys-b)" \
        --out "${VAL}/validation.md" >> "${VLOG}" 2>&1; then
    log "VALIDATION PASSED (${VAL}/validation.md); starting the forced-routing queue"
else
    log "VALIDATION FAILED (${VAL}/validation.md); forced-routing queue NOT started"
    exit 1
fi

# 5. The forced-routing queue.
WORKLOADS="${FORCED_WORKLOADS}" SET_DIR="${FORCED}" ENV_DIR="${MOE_ENV}" TAG=llamacpp-forced \
    ./scripts/run_kimi_q8_llamacpp_queue.sh
log "forced-routing queue ended: see ${FORCED}/queue/results.md"
