#!/bin/zsh
# Full mbolt pipeline for Qwen3-Next-80B-A3B-Instruct UD-IQ3_XXS.
# Stages are idempotent-ish: pass a stage name to start there
# (capture|cluster|gate|rewrite|verify|e2e). Logs to results/qwen80/.
set -e
cd /Users/dor/Documents/code/GPUopt/mbolt
PY=.venv/bin/python
SIM=.venv/bin/mbolt-sim
BASE=/Users/dor/Documents/code/GPUopt
MODEL=$BASE/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.gguf
OPT=$BASE/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.mbolt.gguf
TRACE=$BASE/traces/routing_qwen80.bin
RES=$BASE/results/qwen80
SRV=$BASE/llama.cpp/build/bin/llama-server
mkdir -p $RES
STAGE=${1:-capture}

case $STAGE in
capture)
  echo "=== map ==="
  $SIM map $MODEL
  echo "=== capture (120 prompts x 256 tokens) ==="
  MBOLT_TRACE=$TRACE $SRV -m $MODEL --port 8091 --no-warmup -np 1 -c 8192 \
      > $RES/server.log 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 60); do
    curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && break
    sleep 5
  done
  MBOLT_PORT=8091 MBOLT_PROMPT_COUNT=120 MBOLT_MAX_TOKENS=256 \
      $PY scripts/capture_traces.py 2>&1 | tail -3
  kill $SRV_PID; sleep 3
  ;&
cluster)
  echo "=== cluster (train 80%) ==="
  $SIM cluster $TRACE -o $RES/perms.json --train-frac 0.8
  ;&
gate)
  echo "=== sim gate: warm 128/512 slots, 4 runs ==="
  $SIM gate $MODEL $TRACE $RES/perms.json -o $RES/gate_warm128.json \
      --cache 128 --runs 4 --warmup 48 --tokens 128
  echo "=== sim gate: cold, 2 runs ==="
  $SIM gate $MODEL $TRACE $RES/perms.json -o $RES/gate_cold.json \
      --cache 0 --runs 2 --tokens 32
  ;&
rewrite)
  echo "=== rewrite chain+pipeline ==="
  .venv/bin/mbolt $MODEL $RES/perms.json -o $OPT
  ;&
verify)
  echo "=== byte-verify ==="
  $PY scripts/byte_verify.py $MODEL $OPT $RES/perms.json
  echo "=== routing equivalence ==="
  $PY scripts/routing_equiv.py $MODEL $OPT $RES/perms.json
  ;&
e2e)
  echo "=== E2E streaming benchmark (24GB mlocked holder) ==="
  $PY scripts/e2e_bench2.py $MODEL $OPT 24 2>&1 | tee $RES/e2e.log
  echo "=== PIPELINE DONE ==="
  ;;
esac
