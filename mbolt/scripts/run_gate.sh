#!/bin/zsh
# Phase 0 gate sequence. Run AFTER trace capture completes and server is stopped.
set -e
cd /Users/dor/Documents/code/GPUopt/mbolt
PY=.venv/bin/python
SIM=.venv/bin/mbolt-sim
MODEL=/Users/dor/Documents/code/GPUopt/models/Qwen3-30B-A3B-UD-IQ3_XXS.gguf
TRACE=/Users/dor/Documents/code/GPUopt/traces/routing.bin
RES=/Users/dor/Documents/code/GPUopt/results

echo "=== trace stats ==="
$SIM trace-stats $TRACE

echo "=== clustering (train on first 80%) ==="
$SIM cluster $TRACE -o $RES/perms.json --train-frac 0.8

echo "=== drive characterization (cold) ==="
$SIM evict $MODEL --ceiling 999999 2>/dev/null || true   # unconditional probe print
$SIM drive $MODEL

echo "=== gate: warm LRU 32 slots/layer, 5 runs (PRIMARY) ==="
$SIM gate $MODEL $TRACE $RES/perms.json -o $RES/gate_warm32.json --cache 32 --runs 5 --warmup 64 --tokens 192

echo "=== gate: cold, 3 runs ==="
$SIM gate $MODEL $TRACE $RES/perms.json -o $RES/gate_cold.json --cache 0 --runs 3 --tokens 48

echo "=== sensitivity: warm 16 and 64 slots, 3 runs, key layouts ==="
$SIM gate $MODEL $TRACE $RES/perms.json -o $RES/gate_warm16.json --cache 16 --runs 3 --warmup 64 --tokens 192 \
    --layouts baseline,chain,chain+pipeline,interleave
$SIM gate $MODEL $TRACE $RES/perms.json -o $RES/gate_warm64.json --cache 64 --runs 3 --warmup 64 --tokens 192 \
    --layouts baseline,chain,chain+pipeline,interleave

echo "=== chart ==="
$PY scripts/gate_chart.py $RES/gate_warm32.json $RES/gate_cold.json -o $RES/gate_chart.png

echo "=== DONE ==="
