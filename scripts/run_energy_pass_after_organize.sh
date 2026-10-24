#!/usr/bin/env bash
# One-off (2026-10-13): once scripts/organize_kimi_q8_sets.py has moved the forced-routing set
# under runs/kimi-q8-llamacpp/, measure GPU energy for every finished batch-1 run there
# (scripts/measure_energy.py -> metrics/energy.json) and refresh the results tables.
#   tmux new-session -d -s kimi-q8-energy "./scripts/run_energy_pass_after_organize.sh; exec bash"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=runs/kimi-q8-llamacpp/energy-pass.log
log() { echo "$(date '+%F %T') $*" | tee -a "${LOG}"; }
log "waiting for the organizer (runs/kimi-q8-llamacpp/organize.log ends with 'done')"
until grep -q "; done$" runs/kimi-q8-llamacpp/organize.log 2>/dev/null; do sleep 60; done
runs=$(ls -d runs/kimi-q8-llamacpp/natural-b1/2026*__* runs/kimi-q8-llamacpp/forced-b1/2026*__* 2>/dev/null)
log "energy pass over $(echo "${runs}" | wc -l) run folders (those without metrics/summary.json are skipped)"
"${HOME}/envs/nsbench/bin/python" scripts/measure_energy.py ${runs} 2>&1 \
    | grep --line-buffered -vE "^(llama|load|print_info|\.)" | tee -a "${LOG}"
for q in runs/kimi-q8-llamacpp/*/queue; do
    "${HOME}/envs/nsbench/bin/python" scripts/summarize_kimi_q8_queue.py "${q}" > /dev/null 2>&1
done
"${HOME}/envs/nsbench/bin/python" scripts/summarize_kimi_q8_all.py > /dev/null 2>&1
log "energy pass done; tables refreshed (runs/kimi-q8-llamacpp/results.md, Power and energy)"
