#!/bin/zsh
# mbolt correctness CI - runs on every commit (hardware-bound: needs the model
# files and a Metal-capable Mac; wire to a self-hosted runner).
#
# Gates:
#  1. byte-verify        - rewritten bytes are exactly the intended permutation
#  2. identity proof     - identity-perm rewrite is token-identical (50 prompts, greedy)
#  3. routing equivalence- layer-0/1 expert selections map EXACTLY through the perm
#  4. KLD yardstick      - output perturbation of the permuted model stays below
#                          the same engine's CPU-vs-Metal backend delta
set -e
cd "$(dirname $0)/.."
PY=.venv/bin/python
MODELS=/Users/dor/Documents/code/GPUopt/models
RES=/Users/dor/Documents/code/GPUopt/results
ORIG=$MODELS/Qwen3-30B-A3B-UD-IQ3_XXS.gguf
OPT=$MODELS/Qwen3-30B-A3B-UD-IQ3_XXS.mbolt.gguf
IDENT=$MODELS/Qwen3-30B-A3B-UD-IQ3_XXS.ident.gguf
PPLX=/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-perplexity
WIKI=/Users/dor/Documents/code/GPUopt/traces/wikitext-2-raw/wiki.test.raw

echo "== 1/4 byte-verify =="
$PY scripts/byte_verify.py $ORIG $OPT $RES/perms.json

echo "== 2/4 identity token-identity proof =="
$PY - "$RES/perms.json" "$ORIG" "$IDENT" <<'EOF'
import json, sys, os
sys.path.insert(0, 'src')
from mbolt.rewrite import rewrite
doc = json.load(open(sys.argv[1]))
ident = [list(range(doc['n_expert'])) for _ in doc['layers']]
rewrite(sys.argv[2], sys.argv[3], ident, layout_name='identity+pipeline', pack_pipeline=True)
print('identity rewrite done')
EOF
$PY scripts/correctness_proof.py $ORIG $IDENT

echo "== 3/4 routing equivalence =="
$PY scripts/routing_equiv.py $ORIG $OPT $RES/perms.json

echo "== 4/4 KLD yardstick (permuted vs backend-switch envelope) =="
$PPLX -m $ORIG -f $WIKI --save-all-logits /tmp/mbolt_ci_base.bin -c 512 --chunks 8 2>/dev/null | tail -1
MB_KLD=$($PPLX -m $OPT -f $WIKI --kl-divergence-base /tmp/mbolt_ci_base.bin --kl-divergence -c 512 --chunks 8 2>/dev/null | grep "Mean    KLD" | awk '{print $3}')
BE_KLD=$($PPLX -m $ORIG -ngl 0 -f $WIKI --kl-divergence-base /tmp/mbolt_ci_base.bin --kl-divergence -c 512 --chunks 8 2>/dev/null | grep "Mean    KLD" | awk '{print $3}')
echo "permuted-model KLD: $MB_KLD ; backend-switch KLD: $BE_KLD"
$PY -c "import sys; m, b = float('$MB_KLD'), float('$BE_KLD'); assert m <= b, f'FAIL: permutation noise {m} exceeds backend-switch envelope {b}'; print('PASS KLD yardstick')"

echo "ALL CORRECTNESS GATES PASSED"
