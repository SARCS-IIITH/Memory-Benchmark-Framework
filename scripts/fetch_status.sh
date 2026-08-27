#!/usr/bin/env bash
# Download progress at a glance. Run bare for a snapshot, or `watch -n30 ./scripts/fetch_status.sh`.
#
# Rate is measured over a 20-second sample rather than averaged since the transfer began,
# because the number worth acting on is what the link is doing *now* -- an average is still
# reporting healthy hours after a stall.
set -uo pipefail
STORE="${STORE:-/opt/ai-models}"
NSBENCH="${NSBENCH:-/home/samarthamp/envs/samarthamp/bin/nsbench}"

if pgrep -f "nsight_bench.cli fetch" >/dev/null; then
    echo "fetch: RUNNING (pid $(pgrep -f 'nsight_bench.cli fetch' | head -1))"
else
    echo "fetch: not running"
fi

before=$(du -sb "$STORE" 2>/dev/null | cut -f1)
sleep 20
after=$(du -sb "$STORE" 2>/dev/null | cut -f1)
rate=$(( (after - before) / 20 ))

printf "rate:   %.1f MB/s over the last 20s\n" "$(echo "$rate/1000000" | bc -l)"
if [[ $rate -gt 0 ]]; then
    remaining=$(( 284000000000 - after ))
    printf "eta:    %.1f h for the remaining %.0f GB\n" \
        "$(echo "$remaining/$rate/3600" | bc -l)" "$(echo "$remaining/1000000000" | bc -l)"
else
    echo "eta:    stalled or complete -- no bytes written in the last 20s"
fi
echo
"$NSBENCH" models
