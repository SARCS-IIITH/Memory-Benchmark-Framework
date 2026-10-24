#!/usr/bin/env bash
# One-off (2026-10-13): GPU energy for the batch-1 runs that the first energy pass missed
# (it stalled waiting on its own GPU context and skipped the forced-routing runs; both fixed in
# scripts/measure_energy.py). Waits for the batch queue to log QUEUE END so it never overlaps a
# run, then measures every run under natural-b1/ and forced-b1/ without metrics/energy.json
# (the batch runs measure their own) and refreshes the tables.
#   tmux new-session -d -s kimi-q8-energy2 "./scripts/run_energy_pass_after_batch.sh; exec bash"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=runs/kimi-q8-llamacpp/energy-pass.log
QLOG=runs/kimi-q8-llamacpp/batch/queue/queue.log
log() { echo "$(date '+%F %T') $*" | tee -a "${LOG}"; }
log "second energy pass: waiting for QUEUE END in ${QLOG}"
until tail -1 "${QLOG}" 2>/dev/null | grep -q "QUEUE END"; do sleep 60; done
runs=$(ls -d runs/kimi-q8-llamacpp/natural-b1/2026*__* runs/kimi-q8-llamacpp/forced-b1/2026*__*)
log "second energy pass over $(echo "${runs}" | wc -l) run folders (existing energy.json kept)"
"${HOME}/envs/nsbench/bin/python" scripts/measure_energy.py ${runs} 2>&1 \
    | grep --line-buffered -vE "^(llama|load|print_info|\.)|GPU busy" | tee -a "${LOG}"
for q in runs/kimi-q8-llamacpp/*/queue; do
    "${HOME}/envs/nsbench/bin/python" scripts/summarize_kimi_q8_queue.py "${q}" > /dev/null 2>&1
done
"${HOME}/envs/nsbench/bin/python" scripts/summarize_kimi_q8_all.py > /dev/null 2>&1
log "second energy pass done; tables refreshed"
