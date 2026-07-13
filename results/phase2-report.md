# mbolt Phase 2 — Interleaved Expert Layout, Prefetcher Harvest, and the Field

**Date:** 2026-07-13 · Machine: M5 Pro MacBook Pro / APPLE SSD AP1024Z · Model: Qwen3-Next-80B-A3B-Instruct UD-IQ3_XXS (48×512 experts, top-10, 33.1 GB)

## 1. The interleaved layout (per-expert tensor adjacency, no tensor split)

Phase 0 measured interleave (each expert's up|gate|down contiguous) at 1.75–2.29× replay speedup but "not loadable by llama.cpp." Phase 2 makes it loadable **without per-expert tensor explosion**: the rewriter (`mbolt --layout interleave`) packs each layer into one blob (`blk.N.ffn_ilv_exps.weight`, expert stride = up+gate+down slice sizes, chain order), and a ~90-line llama.cpp loader patch reconstructs the three `ffn_*_exps` tensors as **strided views** (custom `nb[2]`; `mul_mat_id` addresses experts via `nb02`, so compute needs no changes). Stock llama.cpp refuses the file cleanly (missing tensor names). +0.003 % file size.

Two engine-side sharp edges found and fixed:
- The scheduler's large-batch **offload** heuristic copies weights to Metal as contiguous tensors — overflow on strided views (SIGABRT in batch prefill). Fix: Metal `offload_op` now refuses non-contiguous `src0`.
- Expert views assigned to Metal directly are unsupported for now: load-time guard with a clear message (`-ot ".ffn_.*_exps.=CPU"` or `-ngl 0`) instead of a mid-decode crash.

## 2. Correctness status: experimental

- Blob movement byte-exact (slice p == orig slice perm[p], all kinds, sampled layers); router rows likewise.
- Teacher-forced routing: **layer-0 mapping exact (0/1026)** — permutation semantics correct.
- Unlike the contiguous rewrite, layer-1+ drifts (2.3 % at L1, median ~15 %): the strided views take a different CPU matmul accumulation path, so every expert output moves by ulps. Same FP-noise class, larger amplitude.
- Isolated same-config KLD (wikitext, 4k positions): **0.0036, top-1 97.2 %, PPL ratio 1.0031** vs the 80B backend-switch envelope 0.0029 / 97.8 % / 1.0027. **1.26× the envelope — marginally above our strict CI gate**, hence "experimental": quality impact indistinguishable from the contiguous rewrite (+0.3 % PPL), FP noise comparable to switching backends, but not under it.

## 3. Measured wins

**I/O floor (physical files, cold, fair 3-rep alternating, held-out trace):**

| | reads/token | ms/token | speedup |
|---|---|---|---|
| stock file | 1418 | 301.7 | — |
| interleave file | ~370 | 135.4 | **2.23×** (virtual sim predicted 2.29×) |

**End-to-end llama.cpp (CPU experts, 24 GB mlocked squeeze, 128 tokens, 3 reps, medians):**

| config | tok/s | vs stock |
|---|---|---|
| stock file, stock engine | 5.80 | — |
| stock file + prefetcher | 7.60 | 1.31× |
| chain+pipeline + prefetcher | 8.00 | 1.38× |
| interleave, fault streaming | 5.70 | 0.98× (faults stay layout-blind) |
| **interleave + prefetcher** | **9.00** | **1.55×** |

The mechanism is visible in the prefetcher's own counters: ~8.5k reads vs ~28.6k for the same ~13 GB — 3.3× fewer, ~1.5 MB average reads. Layout and explicit reads are complements: neither alone moves tok/s much on this hardware; together 1.55×.

## 4. Negative result: predictive cross-layer prefetch — rejected

Routing on 512-expert qwen3next is too diffuse for speculation: top-64-heat covers 34 % of selections; a train-split transition matrix predicts L+1 at 51 % coverage with ~2× byte overfetch. On a bandwidth-bound stream that's a regression. (colibri reports 71.6 % router-lookahead recall on GLM-5.2's 256 experts — predictability varies strongly by model; worth re-testing per target.)

## 5. The field (researched 2026-07-13)

- **colibri** (github.com/JustVugg/colibri, ~8.2k stars): pure-C GLM-5.2 expert streamer — per-layer LRU, learned pinning, router-lookahead. **No disk-layout optimization.** Caveat for outreach: their experts are ~19 MB, where transfer dwarfs per-read overhead — the co-activation *profile* (for pinning/lookahead-coalescing) is worth more to them than adjacency itself.
- **MLX ecosystem**: expert streaming exists only as PoCs + unanswered feature requests (mlx-lm #1438, omlx #986). No layout discussion anywhere.
- **llama.cpp**: open mmap-vs-direct-I/O debate (#18758) missing the layout variable entirely.
- **Prior art**: no published work does profile-guided on-disk MoE expert layout. Nearest: LLM-in-a-Flash (static row-column bundling, neuron-level), MoE-Infinity (profiles spent on caching, not layout). The niche is open.

Outreach drafts (not posted; require approval): `results/outreach/*.md`.

## 6. Honesty ledger

- All numbers this page: same-session alternating runs, pressure-evicted between runs (9–16 GB file-backed residual under the 24 GB mlock; full eviction needs sudo), N=3, medians with per-run values in `results/qwen80/ilv_e2e.log` and `ilv_replay_fair.log`. Single machine.
- E2E baseline varied between benchmark sessions (5.8 here vs 10.0 in the Phase-1 prefetch run — system/SSD state differs across days); every ratio quoted is within-session.
- Interleave is CPU-experts-only; Metal-resident expert views unimplemented (guarded, not silent).
- Two runs in `ilv_e2e.log` show "pf: off" where the stats line wasn't captured from verbose output; their tok/s values are retained (medians unaffected).
