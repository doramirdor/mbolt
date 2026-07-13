# mbolt Phase 0 — I/O Simulator Gate Report

**Date:** 2026-07-12 · **Machine:** MacBook Pro (M5 Pro, 18-core, 64 GB), APPLE SSD AP1024Z (1 TB internal NVMe), macOS Darwin 25.5.0
**Model:** Qwen3-30B-A3B-UD-IQ3_XXS.gguf (unsloth), 12.89 GB — 48 MoE layers × 128 experts, top-8 routing. Expert tensors = 92.2 % of file bytes; slices 588–816 KiB, all 4096-byte-aligned sizes.

## What was built

1. **Trace capture** — `MBOLT_TRACE=<file>` env-gated eval callback in llama.cpp `common/` (new `mbolt-trace.{h,cpp}` + 2-line hook, imatrix-style capture of the ids tensor `src[2]` of each layer's `ffn_moe_up` MUL_MAT_ID node). Works under Metal. Binary format `MBLT v1`.
2. **Offset mapping** — gguf-py-based map of every expert slice byte range.
3. **Replay engine** — physically replays the exact per-token byte-range read sequence a streaming engine would issue, against the real GGUF, at virtual offsets of each candidate layout. Per-layer read sort + adjacent-merge (gap 0), 16 KiB-aligned `pread`s, QD1. LRU cache model, N slots/layer, expert = cache unit.
4. **Cache discipline** — macOS `F_NOCACHE` does **not** bypass already-cached pages and `purge` needs sudo, so: memory-pressure eviction (~50 GB anonymous touch) + a random-read speed probe before every run; auto re-evict when the probe exceeds 1.4× the established cold device speed. Contamination during the gate runs was caught and auto-corrected by this mechanism (re-evict events were printed to console only; the saved gate JSONs record the per-file cold device speed).
5. **Clustering** — per-layer co-activation graph → greedy modularity communities, clusters ordered by heat; `chain` variant orders experts within a cluster by greedy max-weight path (optimizes *adjacency* of co-activated pairs, which is what read-merging rewards).

## Workload

200 prompts (120 Dolly across all categories incl. context-grounded, 80 CodeAlpaca), Qwen3 thinking mode, temp 0.6 / top-p 0.95, ≤384 completion tokens. 75,973 generated tokens → **75,773 decode-pass routing records** + 14,851 prefill tokens. Clustering trained on the first 80 % (60,618 tokens); **all replay numbers are on the held-out 20 % tail** (15,155 tokens, windows disjoint per run).

## Routing structure (pre-condition for clustering)

- Heat entropy: 6.07–6.63 bits per layer (max 7.0) — real hot/cold skew, not uniform.
- Co-activation lift: median ≈ 0.8 (bulk shows no positive co-activation; computed over pairs with ≥5 co-activations, per-layer medians 0.56–0.89), but pooled over all 48 layers p90 = 3.8, p99 = 17, and **21 % of expert pairs co-activate at >2× independence** (per-layer 17–29 %; mid-layer 24, the `trace-stats` printout: p90 = 4.0, p99 = 17, 23 %) → genuine clique structure. The "flat graph → stop" honesty condition does not trigger.

## Measured gate results (physical replay, held-out tokens)

Speedups are same-window ratios vs baseline, median over runs; N=5 (warm-32), N=3 (others).

| layout (Phase-1-achievable) | warm 32/128 | cold | warm 16/128 | warm 64/128 |
|---|---|---|---|---|
| heat | 1.06× | 1.11× | — | — |
| pipeline | 1.07× | 0.95× | — | — |
| clique | 1.11× | 1.13× | — | — |
| chain | 1.14× | 1.16× | 1.26× | 0.96× |
| clique+pipeline | 1.23× | 1.19× | — | — |
| **chain+pipeline** | **1.23×** | **1.27×** | 1.19× | 1.12× |
| *interleave (Phase-2 only)* | *1.59×* | *1.78×* | *2.37×* | *1.36×* |

Absolute I/O-bound decode floor (warm-32, median): baseline 42.5 tok/s → chain+pipeline 53.7 tok/s. Cold: 7.1 → 8.0 tok/s (interleave 12.3).

Mechanism check: read counts fall exactly as designed — cold reads/token 1089 (baseline) → 775 (chain, −29 %) → 258 (interleave, −76 %); bytes/token unchanged (~760 MB cold, ~154 MB warm-32). Wall-clock follows read count; `heat` and `pipeline` alone do little, so the win is co-activation structure, not hot/cold sorting — consistent with the thesis.

## Per-drive projection

Linear I/O model `t = reads × L + bytes / B` fit to all 88 measured replay points: L = 73 µs, B = 11.3 GB/s, R² = 0.956. Projections (warm-32 workload; **modeled, not measured** — conservative for chain+pipeline: the model under-predicts the measured gain on this Mac, 1.10× vs 1.23×, because it ignores locality/readahead effects; slightly optimistic for interleave, 1.67× modeled vs 1.59× measured):

| drive class | chain+pipeline | interleave |
|---|---|---|
| this Mac (fit) | 1.10× (measured 1.23×) | 1.67× (measured 1.59×) |
| PCIe4 NVMe | 1.07× | 1.42× |
| PCIe3 NVMe | 1.05× | 1.28× |
| USB/SATA SSD | 1.03× | 1.12× |

**Finding that contradicts an assumption in the plan:** the plan expected slower drives to *amplify* gains ("random-read penalty is larger"). With ~600 KiB expert slices the opposite holds: slow drives are bandwidth-bound (bytes dominate, layout can't cut bytes), while fast-NVMe/Apple-silicon machines are per-read-overhead-bound at QD1 — which is where layout wins. The deployment sweet spot is exactly the fast-SSD/unified-memory machines that stream big MoEs.

## Gate verdict

Best Phase-1-achievable layout (chain+pipeline): **1.23× warm (primary), 1.27× cold** — in the spec's judgment zone (1.15–1.3×). No second physical drive class was attached; the fitted model substitutes (labeled as such) and says slower drives dampen rather than amplify for this slice size.

**Decision: PROCEED to Phase 1**, on these grounds:
1. 1.23–1.27× is real and directionally reproducible — every chain+pipeline run beats baseline (warm-32 per-run 1.14–1.58×, N=5; cold 1.13–1.40×, N=3; medians 1.23×/1.27×) — on held-out data, mechanism-confirmed. Run-to-run spread is substantial (CV ≈ 11–14 %), so the point estimate carries roughly ±0.1× uncertainty.
2. Phase 1 is the vehicle for validating the simulator against end-to-end reality (spec: a mispredicting simulator is itself a finding).
3. The measured Phase-2 headroom (interleave: 1.36–2.37×, biggest exactly where memory is tightest) justifies the pipeline: it requires per-expert (or interleave-aware) tensor layout, which current llama.cpp cannot load (verified: legacy `blk.N.ffn_up.E` names remain only in the arch name table; no loader path) — an engine/format follow-up built on Phase 1's rewriter.

**Phase-1 target layout: `chain+pipeline`** (expert permutation per layer by clique+greedy-chain order, + expert tensors packed contiguously in execution order).

## Simulator honesty caveats

- QD1 synchronous replay models mmap page-fault streaming (current llama.cpp behavior); async-prefetch engines would see different (likely smaller) per-read-overhead gains.
- Warm-cache hit rates (0.78 @ 32 slots) come from the LRU model, not a real engine's residency policy.
- macOS page cache was the dominant measurement hazard; every reported run passed a cold-probe check, with contaminated states auto-re-evicted before measuring (probe/evict events were console output; the JSONs persist the cold device baselines).
- 188 of 200 completions (94 %) hit the 384-token cap (thinking mode); the remaining 12 stopped early at 230–361 tokens — still a decode-heavy workload, which is the regime under test.
