# Draft reply — colibri #119 (second message, answering JustVugg's two questions)

---

Great questions — both have measured answers, and your framing (attack the denominator, stacks with quantization) is exactly how we think about it.

**1. Clustering method + generalization.** Per layer: coincidence counts `C[i,j]` = number of tokens where experts i and j are both in the top-k, over the trace. Then greedy modularity communities (plain networkx) on the count-weighted graph, communities ordered by total heat — and within each community a **greedy max-weight chain** (repeatedly append the expert with the highest co-count with the last-placed one). The chain step matters more than the clustering: what read-merging rewards is *literal adjacency*, and chain-ordering beat both heat-sorted and clique-sorted variants in our ablations (`chain` 1.16× vs `heat` 1.11× cold, reorder-only).

Generalization, two levels:
- **Temporal holdout**: all published replay numbers are on the last 20% of the trace, clustered on the first 80%. Held: 1.25× (reorder) / 2.23× (interleave) cold on the 80B.
- **Cross-domain** (your #47 concern — measured today on the 30B): cluster on **general-instruction tokens only** (Dolly, 45k tokens), evaluate on **code tokens only** (CodeAlpaca, held-out): a dolly-trained layout still delivers **1.22×** on the pure-code eval (same-window ratios, N=3). Controls on identical eval windows: code-trained 1.46×, mixed-trained 1.61×. So a fully out-of-domain profile transfers (~55% of the achievable gain), and domain-matched tracing buys the rest. One wrinkle worth knowing: mixed-trained *beat* in-domain-trained — it saw 4× the tokens, and co-activation estimates keep improving with trace volume. I'd still recommend tracing on a mix that resembles the deployment workload — the trace is cheap (one env var, no output change).

**2. Interleave vs co-activation reorder — mostly orthogonal, and your instinct about which is bigger is right.** Decomposition on the 80B (cold, physical files, held-out trace):

| | reads/token | speedup |
|---|---|---|
| stock | 1,418 | — |
| co-activation reorder only (3 tensors separate) | 1,104 | 1.25× |
| interleave (up\|gate\|down contiguous per expert, chain-ordered) | ~370 | **2.23×** |

The 3-reads→1-read merge is the dominant term: it alone accounts for 1,418→~473 by arithmetic; the measured ~370 means co-activation adjacency merges another ~20% of reads *across* experts on top. We did not run interleave-with-random-order, so treat "orthogonal" as approximate — but the arithmetic bounds it tightly. End-to-end with our explicit-read prefetcher in llama.cpp: reorder-only 8.0 tok/s vs interleave 9.0 tok/s (stock 5.8), and 1.63× at a tighter memory squeeze.

**Formats — everything you need is in the repo:**
- Trace format (`MBLT v1`, 12-byte header + `[layer, k, n_tokens, ids…]` records): documented in [`mbolt/patches/llama.cpp-mbolt-trace.patch`](https://github.com/doramirdor/mbolt/blob/main/mbolt/patches/llama.cpp-mbolt-trace.patch) header. For colibri you don't need our patch — just append the top-k ids per layer per token from your router, it's k×n_layers×4 bytes/token.
- Permutations: [`results/perms.json`](https://github.com/doramirdor/mbolt/blob/main/results/perms.json) (`layers[i].chain_perm` = expert id stored at each position; heat + top cliques included). The verified-at-100% 80B maps are in `results/qwen80/perms.json`.
- The rewritten file also carries `mbolt.perm`, `mbolt.heat`, `mbolt.tier_hint` as GGUF metadata — the heat array is the same signal as your `.coli_usage`, so the containers could share one profiling pass: your usage log + an upper-triangular co-count per layer is all the input the layout pass needs.

**Concrete offer:** send me (or point me at) a routing trace from a colibri run on GLM-4.5-Air — your `.coli_usage` extended with top-k ids per token, any format — and I'll return the per-layer permutations + predicted read-count reduction from our replay simulator before you write a line of converter code. Or if you'd rather go straight to prototype: the layout pass is ~200 lines ([`mbolt/src/mbolt/cluster.py`](https://github.com/doramirdor/mbolt/blob/main/mbolt/src/mbolt/cluster.py) + [`layouts.py`](https://github.com/doramirdor/mbolt/blob/main/mbolt/src/mbolt/layouts.py)), MIT, take whatever's useful.

One caveat worth carrying over from our llama.cpp experience: layout only pays if the engine issues **explicit reads** (colibri already does — that's why your engine is the natural home for this; llama.cpp's mmap fault path is layout-blind and we had to add a prefetcher to harvest anything).
