"""Fair physical I/O-floor comparison: stock vs interleaved file, cold,
alternating order, pressure-evict before every measurement."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/dor/Documents/code/GPUopt/mbolt/src")
from gguf import GGUFReader
from mbolt.gguf_map import load_model_map
from mbolt.layouts import Layout, build_layout
from mbolt.replay import replay
from mbolt.sim_cli import _pressure_evict
from mbolt.trace import read_trace

ORIG = "/Users/dor/Documents/code/GPUopt/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.gguf"
ILV = "/Users/dor/Documents/code/GPUopt/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.ilv.gguf"
E = 512


def ilv_layout(chain):
    r = GGUFReader(ILV)
    blobs = {int(t.name.split(".")[1]): t for t in r.tensors if t.name.endswith("ffn_ilv_exps.weight")}
    lay = Layout("ilv-physical", None)
    for layer, t in blobs.items():
        stride = int(t.n_bytes) // E
        pos = np.empty(E, np.int64)
        pos[chain[layer]] = np.arange(E)
        off = int(t.data_offset) + pos * stride
        for kind in ("up", "gate", "down"):
            lay.offset[(layer, kind)] = off
            lay.sizes[(layer, kind)] = stride
    return lay


def main():
    t = read_trace("/Users/dor/Documents/code/GPUopt/traces/routing_qwen80.bin")
    perms = json.load(open("/Users/dor/Documents/code/GPUopt/results/qwen80/perms.json"))
    start = perms["train_tokens"]
    chain = [np.array(l["chain_perm"]) for l in perms["layers"]]
    mm_o = load_model_map(ORIG)
    base_lay = build_layout("baseline", mm_o)
    lay_ilv = ilv_layout(chain)

    res = {"orig": [], "ilv": []}
    for rep in range(3):
        win = t.decode[start + rep * 90 : start + rep * 90 + 80]
        order = [("orig", ORIG, base_lay), ("ilv", ILV, lay_ilv)]
        if rep % 2:
            order = order[::-1]
        for name, path, lay in order:
            _pressure_evict()
            r = replay(path, lay, win, cache_slots=0, measure_tokens=32)
            s = r.stats()
            res[name].append(s["io_ms_median"])
            print(f"rep {rep} {name:>5}: {s['io_ms_median']:7.2f} ms/tok, "
                  f"{s['reads_per_token']:6.1f} reads/tok, {s['mb_per_token']:5.1f} MB/tok", flush=True)
    mo, mi = np.median(res["orig"]), np.median(res["ilv"])
    print(f"\nmedians: orig {mo:.1f} ms/tok, ilv {mi:.1f} ms/tok -> speedup {mo/mi:.3f}x")


if __name__ == "__main__":
    main()
