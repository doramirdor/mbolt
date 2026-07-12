"""Co-activation clustering of experts.

Per layer: nodes = experts, edge weight = P(i and j both routed for the same
token). Greedy modularity clustering; clusters ordered by total heat, experts
within a cluster ordered by heat. Output: permutation (new file order of
expert ids) per layer.

Also computes flatness diagnostics (heat entropy, co-activation lift) so we
don't cluster noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class LayerStats:
    heat: np.ndarray          # [E] activation counts
    co: np.ndarray            # [E, E] pair co-activation counts
    n_tokens: int

    @property
    def heat_entropy_bits(self) -> float:
        p = self.heat / max(1, self.heat.sum())
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum())

    def lift_stats(self) -> dict:
        """lift(i,j) = P(ij) / (P(i) P(j)); ~1 everywhere means a flat graph."""
        n = self.n_tokens
        if n == 0:
            return {}
        pi = self.heat / n
        pij = self.co / n
        expected = np.outer(pi, pi)
        mask = (expected > 1e-9) & ~np.eye(len(pi), dtype=bool) & (self.co >= 5)
        lifts = pij[mask] / expected[mask]
        if len(lifts) == 0:
            return {}
        return {
            "median": float(np.median(lifts)),
            "p90": float(np.percentile(lifts, 90)),
            "p99": float(np.percentile(lifts, 99)),
            "max": float(lifts.max()),
            "frac_gt_1.5": float((lifts > 1.5).mean()),
            "frac_gt_2": float((lifts > 2.0).mean()),
        }


def coactivation_stats(tokens: np.ndarray, n_expert: int) -> list[LayerStats]:
    """tokens: [N, n_layers, k] expert ids."""
    n_tok, n_layers, k = tokens.shape
    out = []
    for layer in range(n_layers):
        ids = tokens[:, layer, :]  # [N, k]
        heat = np.bincount(ids.ravel(), minlength=n_expert).astype(np.float64)
        co = np.zeros((n_expert, n_expert), np.float64)
        # accumulate all ordered pairs within each token's expert set
        for a in range(k):
            for b in range(k):
                if a == b:
                    continue
                np.add.at(co, (ids[:, a], ids[:, b]), 1.0)
        out.append(LayerStats(heat=heat, co=co, n_tokens=n_tok))
    return out


def cluster_layer(stats: LayerStats) -> tuple[list[int], list[list[int]]]:
    """Returns (perm, clusters). perm[p] = expert id placed at position p."""
    E = len(stats.heat)
    G = nx.Graph()
    G.add_nodes_from(range(E))
    co = stats.co
    for i in range(E):
        for j in range(i + 1, E):
            if co[i, j] > 0:
                G.add_edge(i, j, weight=co[i, j] / max(1, stats.n_tokens))
    communities = nx.community.greedy_modularity_communities(G, weight="weight")
    clusters = [sorted(c, key=lambda e: -stats.heat[e]) for c in communities]
    clusters.sort(key=lambda c: -sum(stats.heat[e] for e in c))
    perm = [e for c in clusters for e in c]
    assert sorted(perm) == list(range(E))
    return perm, [list(c) for c in clusters]


def heat_perm(stats: LayerStats) -> list[int]:
    """Pure heat ordering (no clustering) - isolates hot/cold separation gains."""
    return [int(e) for e in np.argsort(-stats.heat, kind="stable")]


def chain_perm(stats: LayerStats, clusters: list[list[int]]) -> list[int]:
    """Within each cluster, greedy max-weight path: co-activated experts land
    ADJACENT so that same-token misses merge into single reads."""
    co = stats.co
    perm: list[int] = []
    for c in clusters:
        remaining = set(c)
        cur = max(remaining, key=lambda e: stats.heat[e])
        remaining.remove(cur)
        chain = [cur]
        while remaining:
            nxt = max(remaining, key=lambda e: (co[cur, e], stats.heat[e]))
            remaining.remove(nxt)
            chain.append(nxt)
            cur = nxt
        perm.extend(chain)
    return perm


def cluster_all(tokens: np.ndarray, n_expert: int) -> dict:
    """Full pipeline: stats + permutations for every layer."""
    layer_stats = coactivation_stats(tokens, n_expert)
    result = {
        "n_layers": len(layer_stats),
        "n_expert": n_expert,
        "n_tokens": int(tokens.shape[0]),
        "layers": [],
    }
    for layer, st in enumerate(layer_stats):
        perm, clusters = cluster_layer(st)
        result["layers"].append(
            {
                "layer": layer,
                "clique_perm": perm,
                "chain_perm": chain_perm(st, clusters),
                "heat_perm": heat_perm(st),
                "heat": st.heat.tolist(),
                "heat_entropy_bits": st.heat_entropy_bits,
                "max_entropy_bits": float(np.log2(n_expert)),
                "lift": st.lift_stats(),
                "n_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "top_cliques": [c[:16] for c in clusters[:8]],
            }
        )
    return result
