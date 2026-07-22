"""GLM-5.2 co-activation atlas overlay from pierre427's routing-trace export.

Input:  glm52-routing-trace-export/v2 (self-describing JSON; rank-preserved
        top-k ids + selection scores, abs<->moe_order map, per-prompt topic
        labels). Source: pierre427's mlx-lm fork release glm52-routing-atlas-20260722.
Output: atlas-format overlay over the 19,456-expert gid space
        (gid = moe_order * 256 + expert_id; MTP slot 75 reserved-not-traced),
        same per-layer schema as the Qwen worked profiles in results/colibri-119/
        plus a per-pair topic-category support count for the #175
        canonical-vs-topic-conditional split.

Profile discipline: DECODE rows only (matches the published Qwen profiles;
prefill routes differently and would blur the pair counts at this volume).

Run: mbolt/.venv/bin/python colibri119/glm_overlay.py <trace.json> <out.json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mbolt", "src"))
from mbolt.cluster import LayerStats, chain_perm, cluster_layer

PAIR_MIN_COUNT = 5
PAIR_MIN_LIFT = 1.5
PAIR_CAP_PER_LAYER = 2000
CAT_COFIRE_MIN = 2  # a pair "fires in a category" if it co-fires >= this many times there


def main(trace_path: str, out_path: str) -> None:
    d = json.load(open(trace_path))
    assert d["schema"] == "glm52-routing-trace-export/v2", d["schema"]
    E = d["routing"]["n_expert_per_layer"]
    k = d["routing"]["k"]
    n_rows = d["tokens"]["total_rows"]

    prompts = d["conditional"]["prompts"]
    decode_mask = np.zeros(n_rows, bool)
    cat_of_row = np.full(n_rows, -1)
    cats = sorted({p["category"] for p in prompts})
    cat_idx = {c: i for i, c in enumerate(cats)}
    for p in prompts:
        d0 = p["start_row"] + p["prefill_rows"]
        assert d0 + p["decode_step_rows"] == p["end_row"]
        decode_mask[d0 : p["end_row"]] = True
        cat_of_row[p["start_row"] : p["end_row"]] = cat_idx[p["category"]]
    n_dec = int(decode_mask.sum())
    assert n_dec == d["tokens"]["decode_step_rows"]

    traced = sorted(d["layer_map"]["traced_layers"], key=lambda t: t["moe_order"])
    out_layers = []
    total_pairs = 0
    frac2 = []
    for t in traced:
        L = d["layers"][str(t["abs_layer"])]
        ids_all = np.asarray(L["executed_topk"], np.int64)
        assert ids_all.shape == (n_rows, k) and ids_all.min() >= 0 and ids_all.max() < E
        ids = ids_all[decode_mask]
        heat = np.bincount(ids.ravel(), minlength=E).astype(np.float64)
        co = np.zeros((E, E), np.float64)
        for a in range(k):
            for b in range(a + 1, k):
                np.add.at(co, (ids[:, a], ids[:, b]), 1.0)
        co = co + co.T
        # per-category co-fire counts for pair support
        cat_co = np.zeros((len(cats), E, E), np.float32)
        cat_dec = cat_of_row[decode_mask]
        for ci in range(len(cats)):
            cids = ids[cat_dec == ci]
            for a in range(k):
                for b in range(a + 1, k):
                    np.add.at(cat_co[ci], (cids[:, a], cids[:, b]), 1.0)
        cat_co = cat_co + cat_co.transpose(0, 2, 1)

        pi = heat / n_dec
        expected = np.outer(pi, pi) * n_dec
        iu, ju = np.triu_indices(E, k=1)
        counts = co[iu, ju]
        ok = (counts >= PAIR_MIN_COUNT) & (expected[iu, ju] > 1e-9)
        lift = np.zeros_like(counts)
        lift[ok] = counts[ok] / expected[iu, ju][ok]
        keep = ok & (lift >= PAIR_MIN_LIFT)
        order = np.argsort(-counts[keep])[:PAIR_CAP_PER_LAYER]
        ki, kj, kc, kl = iu[keep][order], ju[keep][order], counts[keep][order], lift[keep][order]
        support = (cat_co[:, ki, kj] >= CAT_COFIRE_MIN).sum(axis=0)
        pairs = [
            [int(i), int(j), int(c), round(float(l), 2), int(s)]
            for i, j, c, l, s in zip(ki, kj, kc, kl, support)
        ]
        total_pairs += len(pairs)
        if ok.sum():
            frac2.append(float((lift[ok] > 2.0).mean()))

        st = LayerStats(heat=heat, co=co, n_tokens=n_dec)
        perm, clusters = cluster_layer(st)
        out_layers.append(
            {
                "abs_layer": t["abs_layer"],
                "moe_order": t["moe_order"],
                "gid_base": t["gid_base"],
                "heat": heat.astype(int).tolist(),
                "heat_entropy_bits": round(st.heat_entropy_bits, 3),
                "lift": st.lift_stats(),
                "pairs": pairs,
                "clusters": clusters,
                "chain_order": chain_perm(st, clusters),
            }
        )
        if t["moe_order"] % 15 == 0:
            print(f"  moe_order {t['moe_order']:2d} (abs {t['abs_layer']}): "
                  f"{len(pairs)} pairs, {len(clusters)} clusters, "
                  f"H={st.heat_entropy_bits:.2f}/8.00 bits", flush=True)

    doc = {
        "artifact": "GLM-5.2 co-activation atlas overlay (unconditional + per-pair topic support)",
        "source_trace": "glm52-routing-trace-export/v2 (pierre427, mlx-lm fork release glm52-routing-atlas-20260722; MLX enable_offload pager, greedy decode, 10 labeled prompts)",
        "flat_index": d["layer_map"]["flat_index"],
        "total_experts": d["layer_map"]["n_experts_in_model"],
        "n_expert_per_layer": E,
        "k": k,
        "profile_rows": {"used": "decode only", "decode_rows": n_dec,
                         "prefill_rows_excluded": int(n_rows - n_dec)},
        "categories": cats,
        "definitions": {
            "heat": "decode tokens where expert e in the layer's top-k",
            "pairs": f"[i, j, count, lift, category_support]; count/lift as in the Qwen profiles (count>={PAIR_MIN_COUNT}, lift>={PAIR_MIN_LIFT}, top {PAIR_CAP_PER_LAYER}/layer); category_support = number of the {len(cats)} topic categories where the pair co-fires >={CAT_COFIRE_MIN} times (the #175 canonical-vs-conditional axis: high support = canonical candidate)",
            "clusters": "greedy-modularity communities of the decode co-activation graph, heat-ordered",
            "chain_order": "within-cluster greedy max-co-weight path (directional at this volume; no layout action implied — see thread)",
        },
        "caveats": [
            f"volume: {n_dec} decode rows across 10 prompts (~vs 75,773 in the Qwen 30B profile) — treat lift as directional; at this N the count>={PAIR_MIN_COUNT} threshold itself implies lift ~>=4 for a random-rate pair, so surviving pairs are strong ones",
            "scores in the source trace are SELECTION scores (sigmoid+bias, bias-dominated) — margin analyses should use rank gaps of that quantity, not treat it as gating weight (see source score_note)",
            "MTP slot (moe_order 75, gid 19200..19455) reserved-not-traced: not exercised by greedy decode",
        ],
        "reserved_not_traced": d["layer_map"]["reserved_not_traced"],
        "layers": out_layers,
    }
    json.dump(doc, open(out_path, "w"))
    print(f"\n{total_pairs} pairs total across {len(out_layers)} layers; "
          f"median frac(lift>2 | count>={PAIR_MIN_COUNT}) = {np.median(frac2):.2f}; "
          f"{os.path.getsize(out_path)/1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
