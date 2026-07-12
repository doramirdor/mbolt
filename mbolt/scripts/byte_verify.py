"""CI check 1: byte-exact data movement of the rewriter.

Verifies rewritten expert slice p == original slice perm[p] for every expert
tensor, router rows likewise, and all other tensors byte-identical. No
inference needed; ~seconds (mmap + sampled full-tensor compares).
"""

import json
import random
import sys

import numpy as np
from gguf import GGUFReader


def main():
    orig_path, opt_path, perms_path = sys.argv[1], sys.argv[2], sys.argv[3]
    perm_key = sys.argv[4] if len(sys.argv) > 4 else "chain_perm"
    doc = json.load(open(perms_path))
    perms = {l["layer"]: np.array(l[perm_key]) for l in doc["layers"]}
    E = doc["n_expert"]

    ot = {t.name: t for t in GGUFReader(orig_path).tensors}
    pt = {t.name: t for t in GGUFReader(opt_path).tensors}
    assert set(ot) == set(pt), "tensor name sets differ"

    rng = random.Random(0)
    layers = sorted(perms)
    sample = set(rng.sample(layers, min(8, len(layers))))
    checked_perm = checked_same = 0
    for name, t in ot.items():
        a = t.data.reshape(-1).view(np.uint8)
        b = pt[name].data.reshape(-1).view(np.uint8)
        assert a.nbytes == b.nbytes, f"{name}: size differs"
        is_exps = "_exps.weight" in name
        is_router = "ffn_gate_inp." in name
        if is_exps or is_router:
            layer = int(name.split(".")[1])
            if layer not in sample:
                continue
            p = perms[layer]
            A = a.reshape(E, a.nbytes // E)
            B = b.reshape(E, b.nbytes // E)
            assert (B == A[p]).all(), f"{name}: permuted bytes mismatch"
            checked_perm += 1
        else:
            if rng.random() < 0.15 or a.nbytes < (1 << 20):
                assert (a == b).all(), f"{name}: bytes differ"
                checked_same += 1
    print(f"PASS byte-verify: {checked_perm} permuted tensors exact, "
          f"{checked_same} passthrough tensors identical")


if __name__ == "__main__":
    main()
