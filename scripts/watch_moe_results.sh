#!/usr/bin/env bash
# Regenerate runs/moe-queue/results.md whenever runs/moe-queue/queue.log changes, i.e. after
# every run starts, finishes or fails. Exits after the final regeneration once the queue logs
# QUEUE END. Run inside tmux:  tmux new-session -d -s moe-results "./scripts/watch_moe_results.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
QLOG=runs/moe-queue/queue.log
last=""
while true; do
    now=$(stat -c %Y "${QLOG}" 2>/dev/null)
    if [[ "${now}" != "${last}" ]]; then
        "${HOME}/envs/nsbench/bin/python" scripts/summarize_moe_queue.py > /dev/null 2>&1 \
            && echo "$(date '+%F %T') results.md regenerated ($(grep -c '^| moe-' runs/moe-queue/results.md) runs)" \
            || echo "$(date '+%F %T') summarize FAILED"
        last="${now}"
        grep -q "QUEUE END" "${QLOG}" && { echo "queue ended; watcher exiting"; exit 0; }
    fi
    sleep 60
done
