# Draft: colibri issue/discussion post (github.com/JustVugg/colibri)

**Title: Profile-guided expert layout + co-activation profiles — data from a GGUF-side experiment (mbolt)**

Colibri optimizes *what stays in RAM* (LRU, pinning, router-lookahead). I've been working on the complementary axis: *where experts sit on disk*, so the misses that remain become fewer, larger, more sequential reads. Sharing data because your converter is the natural place for a layout pass, and one of my findings directly concerns your expert size.

**What I did (mbolt, BOLT/PGO-for-weights on GGUF):** trace MoE routing over real prompts → per-layer co-activation clustering → rewrite the model file so co-activated experts are adjacent (and, in a second mode, so each expert's up|gate|down matrices form one contiguous block). Weights byte-exact, routing exactly permutation-equivalent (verified layer-0 mapping 100.000%).

**Measured on Qwen3-Next-80B IQ3_XXS (48×512 experts, ~400KB slices, Apple M5 Pro):**
- Cold streaming I/O floor, physical files, held-out trace: **2.23× faster decode I/O** (reads/token 1418 → ~370) from interleave+co-activation layout alone.
- With an explicit-read prefetcher in llama.cpp (reads the selected experts' slices as merged ranges when the router's top-k lands): +13-14% end-to-end on both layouts; layout effect on tok/s is small on fast NVMe because compute dominates — it grows exactly where your users live (slow NVMe, deep memory pressure).

**The honest caveat for colibri specifically:** your experts are ~19MB at int4 — at that granularity, transfer time dwarfs per-read overhead, so *adjacency* buys little on raw reads. Where I think this data helps you anyway:
1. **Co-activation profiles beat heat for pinning.** In my traces, per-layer co-activation cliques are strong (p99 lift ≈ 17-20) while pure heat ranking is weak. If your pinned hot-store is heat-ranked today, clique-aware pinning should raise hit rates at equal RAM.
2. **Lookahead + adjacency = coalesced reads.** Your router-lookahead (71.6% recall) issues per-expert reads; if predicted co-activated experts were adjacent in the container, those become single larger sequential reads — cheap to add in the offline converter.
3. **Sub-expert layouts.** If you ever split experts (e.g., per-matrix or quant-block tiers), layout starts to matter at your scale too; the gate math in my repo predicts when.

Everything (I/O simulator, gate methodology, cold-cache discipline for macOS, negative results incl. why speculative prefetch didn't pay for 512-expert routing) is in the repo: https://github.com/doramirdor/mbolt. Happy to run the pipeline on a GLM-5.2 trace if you can point me at a routing log format, or to add a layout pass to your converter behind a flag.
