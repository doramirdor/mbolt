"""CI check 3: routing equivalence through the permutation.

Teacher-forces the same text through original and rewritten models with
MBOLT_TRACE enabled, then verifies the selected expert sets map through the
permutation. Layers 0-1 see bit-identical inputs, so their mapping must be
EXACTLY 100% - any deviation is a semantic bug in the rewrite. Deeper layers
drift by engine FP-reduction noise (reported, threshold loose).
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-perplexity"
TEXT = "/Users/dor/Documents/code/GPUopt/traces/kl_text.txt"


def trace_run(model, out):
    env = dict(os.environ, MBOLT_TRACE=out)
    subprocess.run([BIN, "-m", model, "-f", TEXT, "-c", "512"],
                   env=env, capture_output=True, check=True, timeout=1200)


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from mbolt.trace import read_trace

    orig, opt, perms_path = sys.argv[1], sys.argv[2], sys.argv[3]
    perm_key = sys.argv[4] if len(sys.argv) > 4 else "chain_perm"
    doc = json.load(open(perms_path))
    E = doc["n_expert"]
    inv = []
    for l in doc["layers"]:
        p = np.array(l[perm_key])
        ip = np.empty(E, np.int64)
        ip[p] = np.arange(E)
        inv.append(ip)

    with tempfile.TemporaryDirectory() as td:
        ta, tb = f"{td}/a.bin", f"{td}/b.bin"
        trace_run(orig, ta)
        trace_run(opt, tb)
        a, b = read_trace(ta), read_trace(tb)

    rates = []
    for layer in range(a.n_layers):
        A, B = a.prefill_per_layer[layer], b.prefill_per_layer[layer]
        n = min(len(A), len(B))
        mism = sum(
            set(inv[layer][A[t]].tolist()) != set(B[t].tolist()) for t in range(n)
        )
        rates.append(mism / n)

    print(f"routing mismatch by layer: L0={rates[0]:.4%} L1={rates[1]:.4%} "
          f"median={np.median(rates):.2%} max={max(rates):.2%}")
    assert rates[0] == 0.0 and rates[1] == 0.0, (
        "FAIL: layer-0/1 routing does not map exactly through the permutation - semantic bug"
    )
    assert max(rates) < 0.15, "FAIL: deep-layer drift implausibly large"
    print("PASS routing-equiv: layer-0/1 mapping exact; deeper drift = FP-order noise")


if __name__ == "__main__":
    main()
