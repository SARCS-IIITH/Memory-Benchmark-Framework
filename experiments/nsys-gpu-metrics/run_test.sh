#!/usr/bin/env bash
# For each candidate metric file: profile the known-size copy test, export, then analyse.
cd "$(dirname "$0")"
PY=$HOME/envs/nsbench/bin/python
for key in ${KEYS:-l2_sectors}; do
  echo "=== $key"
  nsys profile --force-overwrite=true -o "out_$key" --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --gpu-metrics-devices=0 --gpu-metrics-set="file:$PWD/$key.config" --gpu-metrics-frequency=10000 \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    $PY copy_test.py > "nsys_$key.log" 2>&1
  echo "nsys exit $?"; grep -iE "error|warn|expected|unsupported|invalid" "nsys_$key.log" | head -5
  [ -f "out_$key.nsys-rep" ] || continue
  nsys export --type sqlite --force-overwrite true -o "out_$key.sqlite" "out_$key.nsys-rep" >/dev/null 2>&1
  $PY analyse.py "out_$key.sqlite"
done
