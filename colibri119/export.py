"""colibri#119 deliverables for JustVugg's atlas ask (2026-07-21).

Part 1: co-activation profile export — heat + within-layer pairs/clusters over
        a flat expert index (atlas-composable, #175 style). Worked example on
        the two Qwen traces; same schema applies to any routing trace.
Part 2: readahead efficiency — bytes-per-useful-expert, stock layout vs
        interleave+chain, across readahead window sizes. Pure geometry against
        the real GGUF tensor table (bytes are bytes; same principle as the
        published replay sim). Cold per-token accounting, held-out trace tail.

Run: mbolt/.venv/bin/python colibri119/export.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mbolt", "src"))
from mbolt.gguf_map import EXPERT_KINDS, load_model_map
from mbolt.layouts import build_layout
from mbolt.trace import read_trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "colibri-119")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "qwen3-30b-a3b": {
        "gguf": os.path.join(ROOT, "models", "Qwen3-30B-A3B-UD-IQ3_XXS.gguf"),
        "trace": os.path.join(ROOT, "traces", "routing.bin"),
        "perms": os.path.join(ROOT, "results", "perms.json"),
        "quant": "IQ3_XXS",
    },
    "qwen3-next-80b-a3b": {
        "gguf": os.path.join(ROOT, "models", "Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.gguf"),
        "trace": os.path.join(ROOT, "traces", "routing_qwen80.bin"),
        "perms": os.path.join(ROOT, "results", "qwen80", "perms.json"),
        "quant": "IQ3_XXS",
    },
}

PAIR_MIN_COUNT = 5
PAIR_MIN_LIFT = 1.5
PAIR_CAP_PER_LAYER = 2000
MAX_TOKENS_READAHEAD = 2000
WINDOWS = [16 * 1024, 128 * 1024, 512 * 1024, 2 * 1024 * 1024]


def coact_pairs(tokens: np.ndarray, n_expert: int) -> list[dict]:
    """Full-trace heat + thresholded co-activation pairs, per layer."""
    n_tok, n_layers, k = tokens.shape
    layers = []
    for layer in range(n_layers):
        ids = tokens[:, layer, :]
        heat = np.bincount(ids.ravel(), minlength=n_expert).astype(np.int64)
        co = np.zeros((n_expert, n_expert), np.float64)
        for a in range(k):
            for b in range(a + 1, k):
                np.add.at(co, (ids[:, a], ids[:, b]), 1.0)
        co = co + co.T  # symmetric: co[i,j] = tokens where i and j both in top-k
        pi = heat / n_tok
        expected = np.outer(pi, pi) * n_tok
        iu, ju = np.triu_indices(n_expert, k=1)
        counts = co[iu, ju]
        exp_u = expected[iu, ju]
        ok = (counts >= PAIR_MIN_COUNT) & (exp_u > 1e-9)
        lift = np.zeros_like(counts)
        lift[ok] = counts[ok] / exp_u[ok]
        keep = ok & (lift >= PAIR_MIN_LIFT)
        order = np.argsort(-counts[keep])[:PAIR_CAP_PER_LAYER]
        ki, kj, kc, kl = iu[keep][order], ju[keep][order], counts[keep][order], lift[keep][order]
        layers.append(
            {
                "layer": layer,
                "heat": heat.tolist(),
                "pairs": [
                    [int(i), int(j), int(c), round(float(l), 2)]
                    for i, j, c, l in zip(ki, kj, kc, kl)
                ],
            }
        )
    return layers


def clusters_from_perms(perms_layer: dict) -> list[list[int]]:
    """Rebuild full cluster memberships from clique_perm + cluster_sizes."""
    perm, sizes = perms_layer["clique_perm"], perms_layer["cluster_sizes"]
    out, pos = [], 0
    for s in sizes:
        out.append(perm[pos : pos + s])
        pos += s
    assert pos == len(perm)
    return out


def export_profile(name: str, cfg: dict) -> None:
    tr = read_trace(cfg["trace"])
    perms = json.load(open(cfg["perms"]))
    E, L = tr.n_expert, tr.n_layers
    layers = coact_pairs(tr.decode, E)
    for lrec, prec in zip(layers, perms["layers"]):
        lrec["clusters"] = clusters_from_perms(prec)
        lrec["chain_order"] = prec["chain_perm"]
        lrec["heat_entropy_bits"] = round(prec["heat_entropy_bits"], 3)
    doc = {
        "model": name,
        "quant": cfg["quant"],
        "n_layers": L,
        "n_expert_per_layer": E,
        "total_experts": L * E,
        "k": tr.k,
        "n_decode_tokens": int(tr.decode.shape[0]),
        "flat_index": "gid = layer * n_expert_per_layer + expert_id (swap for your Brain-page convention)",
        "definitions": {
            "heat": "count[e] = decode tokens where e is in the layer's top-k (full trace)",
            "pairs": f"[i, j, count, lift]; count = decode tokens where i and j both in the layer's top-k; lift = P(ij)/(P(i)P(j)); within-layer only; thresholds count>={PAIR_MIN_COUNT}, lift>={PAIR_MIN_LIFT}, top {PAIR_CAP_PER_LAYER}/layer by count",
            "clusters": "greedy-modularity communities of the co-activation graph, heat-ordered (trained on first 80% of the trace — the physically verified layout artifacts)",
            "chain_order": "suggested adjacency order: within each cluster, greedy max-co-weight path; chain_order[p] = expert id stored at slot p",
        },
        "cross_layer_note": "pairs are within-layer by construction: unconditional cross-layer co-activation is ~uniform in our measurements (adjacent-layer Jaccard 0.017 vs 0.016 shuffle, replicated on 3 architectures)",
        "layers": layers,
    }
    path = os.path.join(OUT, f"coactivation_{name}.json")
    json.dump(doc, open(path, "w"))
    n_pairs = sum(len(l["pairs"]) for l in layers)
    print(f"[profile] {name}: {L}x{E} experts, {tr.decode.shape[0]} decode tokens, "
          f"{n_pairs} pairs kept, {os.path.getsize(path)/1e6:.1f} MB -> {path}")


def token_windows(starts: np.ndarray, ends: np.ndarray, w: int) -> tuple[int, int]:
    """(fetched_bytes, n_reads) for w-aligned window fetch of the given ranges."""
    ws = starts // w
    we = (ends - 1) // w
    order = np.argsort(ws, kind="stable")
    ws, we = ws[order], we[order]
    run_max = np.maximum.accumulate(we)
    breaks = ws[1:] > run_max[:-1] + 1  # adjacent windows chain into one sequential read
    n_reads = 1 + int(breaks.sum())
    # covered windows = sum over merged runs of (run_end - run_start + 1)
    run_starts = np.concatenate(([0], np.flatnonzero(breaks) + 1))
    run_ends = np.concatenate((np.flatnonzero(breaks), [len(ws) - 1]))
    covered = int((run_max[run_ends] - ws[run_starts] + 1).sum())
    return covered * w, n_reads


def readahead_table(name: str, cfg: dict) -> dict:
    mm = load_model_map(cfg["gguf"])
    perms = json.load(open(cfg["perms"]))
    chain = [l["chain_perm"] for l in perms["layers"]]
    layouts = {
        "stock": build_layout("baseline", mm),
        "interleave+chain": build_layout("interleave", mm, chain),
    }
    tr = read_trace(cfg["trace"])
    n = tr.decode.shape[0]
    tail = tr.decode[int(n * 0.8):][:MAX_TOKENS_READAHEAD]
    slice_bytes = {(l, k): et.slice_bytes for (l, k), et in mm.experts.items()}
    expert_bytes = sum(mm.experts[(0, k)].slice_bytes for k in EXPERT_KINDS)

    rows = []
    for lname, lay in layouts.items():
        for w in WINDOWS:
            fetched = reads = useful = demands = 0
            for t in range(len(tail)):
                starts_l, ends_l = [], []
                for layer in range(tr.n_layers):
                    ids = np.unique(tail[t, layer])
                    demands += len(ids)
                    for kind in EXPERT_KINDS:
                        offs = lay.offset[(layer, kind)][ids]
                        sz = slice_bytes[(layer, kind)]
                        useful += sz * len(ids)
                        starts_l.append(offs)
                        ends_l.append(offs + sz)
                f, r = token_windows(np.concatenate(starts_l), np.concatenate(ends_l), w)
                fetched += f
                reads += r
            nt = len(tail)
            rows.append(
                {
                    "layout": lname,
                    "readahead_kib": w // 1024,
                    "reads_per_token": round(reads / nt, 1),
                    "fetched_mb_per_token": round(fetched / nt / 1e6, 1),
                    "useful_mb_per_token": round(useful / nt / 1e6, 1),
                    "efficiency_pct": round(100 * useful / fetched, 1),
                    "bytes_per_useful_expert_kib": round(fetched / demands / 1024, 1),
                }
            )
            print(f"[readahead] {name} {lname:>16} W={w//1024:>5}KiB: "
                  f"{rows[-1]['reads_per_token']:>7} reads/tok, "
                  f"{rows[-1]['fetched_mb_per_token']:>6} MB fetched/tok, "
                  f"eff {rows[-1]['efficiency_pct']:>5}%, "
                  f"{rows[-1]['bytes_per_useful_expert_kib']:>7} KiB/useful-expert")
    return {
        "model": name,
        "quant": cfg["quant"],
        "n_layers": tr.n_layers,
        "n_expert_per_layer": tr.n_expert,
        "k": tr.k,
        "expert_bytes_kib": round(expert_bytes / 1024, 1),
        "eval_tokens": len(tail),
        "eval_window": "held-out tail (last 20% of decode trace; layouts trained on first 80%)",
        "accounting": "cold per-token: every demanded slice fetched fresh each token, no cross-token retention; readahead = W-aligned windows covering each demanded range, unioned per token; useful = exact demanded slice bytes",
        "rows": rows,
    }


if __name__ == "__main__":
    tables = []
    for name, cfg in MODELS.items():
        export_profile(name, cfg)
        tables.append(readahead_table(name, cfg))
    json.dump(tables, open(os.path.join(OUT, "readahead_efficiency.json"), "w"), indent=1)
    print(f"-> {os.path.join(OUT, 'readahead_efficiency.json')}")
