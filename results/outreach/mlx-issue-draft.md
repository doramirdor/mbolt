# Draft: comment for mlx-lm #1438 / omlx #986 (SSD expert streaming requests)

**Title: On-disk expert layout is the missing layer for SSD streaming — measurements + a format that ships its own profile**

Both threads ask for SSD expert streaming (LRU + prefetch). One design input from a GGUF-side experiment (mbolt) that's cheap to adopt early and painful to retrofit: **decide expert order on disk from a routing profile before you build the streamer.**

Measured on Qwen3-Next-80B-A3B IQ3_XXS (48×512 experts, M5 Pro):
- Experts are 95 % of file bytes; per-token decode touches 10 experts × 48 layers.
- Reordering experts by co-activation (cliques from a 25k-token routing trace) + interleaving each expert's up|gate|down into one contiguous block cuts cold streaming reads/token 1418 → ~370 and decode I/O time **2.23×** — measured by physically replaying held-out routing traces against the real files.
- In llama.cpp with an explicit-read prefetcher (read the top-k experts' slices when the router lands), the interleaved file decodes **1.55× faster end-to-end** than the stock file on stock streaming, under identical memory pressure.
- Anti-result worth knowing: mmap/page-fault streaming is layout-blind (identical tok/s and pageins across layouts in our CPU-mode apples-to-apples runs). If your streamer reads *slices* explicitly — which the PoCs cited in these threads do — you get the layout win; if it demand-pages, you don't.

Why this matters for MLX specifically: Apple SSDs have huge sequential-vs-scattered gaps at QD1, and unified memory means the streamer's reads land directly in the compute buffer. The safetensors container gives you free rein over expert order — the layout pass is offline, bit-exact per expert, and ~200 lines given a routing trace.

The mbolt file format also embeds the profile itself (`mbolt.*` metadata: per-expert heat, tier hints, top co-activation cliques, the permutation) so a streamer can make pinning/prefetch decisions without re-profiling. Happy to adapt the trace→cluster→rewrite pipeline to safetensors if there's interest: https://github.com/doramirdor/mbolt.
