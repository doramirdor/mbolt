"""mbolt: profile-guided layout optimization for GGUF models.

Usage:
  mbolt model.gguf perms.json -o model.opt.gguf [--layout chain+pipeline]

perms.json is the output of `mbolt-sim cluster` (routing-profile clustering).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from .rewrite import rewrite


def main():
    ap = argparse.ArgumentParser(prog="mbolt", description=__doc__)
    ap.add_argument("model")
    ap.add_argument("perms", help="perms.json from mbolt-sim cluster")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--layout", default="chain+pipeline",
                    choices=["chain+pipeline", "chain", "clique+pipeline", "clique", "heat"])
    args = ap.parse_args()

    doc = json.load(open(args.perms))
    key = {"chain+pipeline": "chain_perm", "chain": "chain_perm",
           "clique+pipeline": "clique_perm", "clique": "clique_perm",
           "heat": "heat_perm"}[args.layout]
    perms = [l[key] for l in doc["layers"]]
    heat = [l["heat"] for l in doc["layers"]]
    cliques = [l["top_cliques"] for l in doc["layers"]]
    pack = args.layout.endswith("+pipeline")

    t0 = time.time()
    stats = rewrite(args.model, args.output, perms, heat=heat, top_cliques=cliques,
                    layout_name=args.layout, pack_pipeline=pack)
    dt = time.time() - t0
    src, dst = os.path.getsize(args.model), os.path.getsize(args.output)
    print(f"rewrote {stats['tensors']} tensors "
          f"({stats['permuted_exps']} expert tensors, {stats['permuted_router']} router tensors) "
          f"in {dt:.1f}s")
    print(f"size: {src / 1e9:.3f} GB -> {dst / 1e9:.3f} GB "
          f"(+{100 * (dst - src) / src:.3f}% overhead)")


if __name__ == "__main__":
    main()
