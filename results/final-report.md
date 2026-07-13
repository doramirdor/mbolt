# mbolt — Final Report

**One-liner:** BOLT/PGO for model binaries. Trace which MoE experts a real workload activates together, rewrite the GGUF so those bytes sit together on disk, and teach llama.cpp to read them as a few big sequential chunks instead of thousands of scattered page faults.

**Machine for every number:** M5 Pro MacBook Pro (64 GB), APPLE SSD AP1024Z, llama.cpp b9977. Models: Qwen3-30B-A3B and Qwen3-Next-80B-A3B (48×512 experts, 33 GB, experts = 95 % of file bytes), both IQ3_XXS.

## What was built (TLDR ladder)

1. **Tracer** — 176-line env-gated llama.cpp patch logs per-token expert routing (`MBOLT_TRACE`).
2. **Simulator** — physically replays a routing trace's exact byte reads against the real file under any candidate layout, with cold-cache discipline (macOS page cache is the #1 way to lie to yourself; every number passed a cold probe).
3. **Rewriter** (`mbolt`) — reorders expert slices by co-activation cliques + greedy chain, permutes router rows to match, packs and 4096-aligns. Bit-exact weights (byte-verified on sampled tensors: 8/48 layers of expert+router tensors plus passthrough samples), +0.003 % size.
4. **Prefetcher** (`MBOLT_PREFETCH`) — when the router picks its top-k, reads exactly those experts' slices as merged ranges into the page cache before the matmuls fault. Works on any GGUF, no rewrite needed. Optional parallel reads (`MBOLT_PREFETCH_THREADS`).
5. **Interleave format** (Phase 2) — each expert's up|gate|down matrices become one contiguous block via one blob per layer + a ~90-line loader patch that rebuilds the standard tensors as strided views (`mul_mat_id` already supports the stride). One expert miss = one contiguous ~1.3 MB read (issued reads average ~1.5 MB after the prefetcher merges adjacent experts). CPU-experts only for now; stock llama.cpp refuses the file cleanly.

## Results (all measured; ratios within-session, alternating runs, N=3 medians — except the 2.29× simulator prediction and the chain+pipeline I/O-floor row, which come from the 2-run sim gate and a separate session)

**I/O floor** (cold replay of held-out traces against the physical files, 80B):

| layout | reads/token | speedup |
|---|---|---|
| stock file | 1418 | — |
| chain+pipeline | 1097 | 1.25× |
| **interleave** | **~370** | **2.23×** (simulator predicted 2.29× — 3 % off) |

**End-to-end llama.cpp** (80B, CPU experts, 24 GB mlocked squeeze, 128-token decode):

| config | tok/s | vs stock |
|---|---|---|
| stock file, stock engine | 5.8 | — |
| stock + prefetcher | 7.6 | 1.31× |
| chain+pipeline + prefetcher | 8.0 | 1.38× |
| interleave, no prefetcher | 5.7 | 0.98× |
| **interleave + prefetcher** | **9.0** | **1.55×** |

Cross-session replication: a second session (~90 min later; absolute tok/s drifts with SSD/system state) measured interleave+pf / stock+pf = 1.15× vs 1.18× in the first — ratios hold.

**Queue-depth sweep** (parallel prefetch reads): stock layout +11 % at QD4; interleave shows no further gain past QD1 (~66 large reads/token). Queue depth and layout are partial substitutes here — but interleave@QD1 matches stock@QD4 with 3.4× fewer I/O ops and zero extra threads (read threads compete with compute threads; QD8 ran unstable).

**Correctness** (the suite ships as CI; token-identity and KLD gates run on the 30B, byte-verify and routing-equivalence on both models):
- Weights byte-exact on every sampled tensor; routing maps 100.000 % through the permutation at layer 0 (semantics proven).
- Token-identity **cannot be guaranteed** under nontrivial expert permutation on current engines — measured: 3/50 greedy completions diverge (30B, llama.cpp Metal). The control ladder (engine-deterministic ✓, identity-rewrite token-identical ✓, nontrivial permutation diverges) isolates the cause to FP-level sensitivity to expert index order; the leading candidate seeds (positional-order softmax normalizer, top-k tie-breaking) were not directly instrumented. An engine-side id-remap would restore bit-identity.
- Measured output noise: chain+pipeline (30B) = **5× below** the same engine's CPU↔Metal backend delta (KLD 0.00096 vs 0.0048). Interleave (80B) = KLD 0.0036 vs that model's own backend envelope 0.0029 — **1.24× above it** (strided views change the matmul accumulation path) → labeled **experimental**. PPL impact +0.3 % for both.
- +0.003 % file-size overhead for every layout.

**Negative results (kept):** predictive cross-layer prefetch rejected (transition matrices reach 51 % coverage at 2× byte overfetch — a regression when bandwidth-bound); single-stream "async one-layer-ahead" prefetch retracted before building (no lookahead exists without prediction — the router output arrives microseconds before the weights are needed); mmap fault streaming is layout-blind in every apples-to-apples CPU-mode run (tok/s parity and, where recorded, identical pageins across layouts; the Metal-offload runs differed, but that is a buffer-mapping artifact analyzed in the Phase-1 report, not a layout effect).

**Field position** (researched 2026-07-13): colibri (the 8.2k-star GLM-5.2 expert streamer) has no disk-layout optimization; MLX expert streaming exists only as unanswered feature requests; no published prior art on profile-guided MoE disk layout (nearest: LLM-in-a-Flash, MoE-Infinity — both spend their effort elsewhere). Outreach drafts staged in `results/outreach/`, pending approval + a public repo.

## Verdict

- **Proven:** profile-guided expert layout is real and big at the storage level — 2.23× measured on physical files, simulator-validated to a few percent, weights bit-exact on every sampled tensor, routing semantics exact.
- **Shippable today:** the prefetcher — +13-31 % decode on memory-squeezed MoE streaming for *any* GGUF, ~200 lines total with the tracer, no file rewrite. It is also the vehicle that converts layout into tok/s (1.55× with interleave).
- **The honest catch:** this MacBook is close to the worst place to *see* the win. Every streaming config converges at a ~9.4-9.8 tok/s system ceiling set by compute (42 ms/token) plus fault/page-cache overhead — at that rate the stream moves only ~1 GB/s, nowhere near the drive's ~3.5 GB/s, so the NVMe itself is never the binding constraint here. On slower drives, thread-starved boxes, or any engine that streams synchronously, the same layout is the difference in kind, not degree.
- **Where this goes:** publish repo → prefetcher upstream to llama.cpp → outreach with the co-activation *profile* (not just the layout) to colibri/MLX — the niche is verified open.
