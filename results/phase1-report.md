# mbolt Phase 1 — Rewriter, Correctness, and End-to-End Reality

**Date:** 2026-07-12 · **Machine:** MacBook Pro (M5 Pro, 18-core, 64 GB), APPLE SSD AP1024Z, macOS Darwin 25.5.0
**Models:** Qwen3-30B-A3B UD-IQ3_XXS (12.9 GB, 48×128 experts, top-8) · Qwen3-Next-80B-A3B-Instruct UD-IQ3_XXS (33.1 GB, 48×512 experts, top-10; experts = 95.4 % of file)

## 1. The rewriter

`mbolt model.gguf perms.json -o model.opt.gguf` — permutes expert slices within each `ffn_{up,gate,down}_exps` tensor by the routing-profile clustering (greedy-modularity cliques + max-co-activation chain order), applies the same permutation to `ffn_gate_inp` rows, packs expert tensors contiguously in execution order, sets `general.alignment=4096` (every expert slice starts page-aligned; slice sizes verified multiples of 4096), and writes `mbolt.*` KV metadata (version, layout, permutations, per-position heat, tier hints, top cliques). Unmodified llama.cpp loads the output and ignores the extra keys (verified). 30B: +0.003 % size (verified from on-disk files; rewrite wall time not logged). 80B: 63 s, +0.003 % size. Refuses loudly on unrecognized expert-indexed tensors.

## 2. Correctness — what "bit-exact" turned out to mean

The spec demanded token-identical greedy output (tolerance exactly 0). Measured: 3/50 prompts diverged mid-stream. The diagnosis chain, each step isolating one variable:

| control | result |
|---|---|
| same model twice (engine determinism) | 50/50 token-identical |
| identity-perm rewrite (packing+alignment+KV only) | 50/50 token-identical |
| byte-verify (slice p == orig slice perm[p], router rows likewise) | exact on every sampled tensor: 8/48 randomly sampled layers of expert+router tensors, plus all small and ~15 % of large passthrough tensors byte-identical |
| routing equivalence, teacher-forced, layers 0/1 (L0 input bit-identical; L1 input already ulp-perturbed by L0) | **100.000 % maps through the perm** (0/1026 mismatches; L0 exactness required, L1 enforced as a tripwire) |
| routing mismatch by depth | 0 % (L0–L1) → 0.19 % (L2), peaks ~5 % (L30), falls to 2.7 % (L47) — depth-correlated FP drift |

Conclusion: the permutation semantics are exact. The controls exclude everything except **floating-point-level sensitivity to expert index order**. The leading candidate seeds are the router's full-width softmax normalizer (llama.cpp sums `exp()` over all experts in positional order before top-k) and index-order tie-breaking in argsort top-k — we did not instrument router probabilities directly. The ulp-scale seed is chaos-amplified over 48 layers, occasionally flipping a rank-8/9 boundary expert and eventually a near-tie greedy token. **Token-identity cannot be guaranteed under nontrivial expert permutation on current engines** — measured here as 3/50 greedy 64-token completions diverging (llama.cpp, Metal); an engine-side id-remap (router left unpermuted) would restore bit-identity.

The honest equivalence bound, measured on wikitext (12k teacher-forced positions):

| comparison | mean KLD | same top-1 | PPL ratio |
|---|---|---|---|
| orig vs its own saved logits (floor) | 1e-6 | 100.000 % | 1.000000 |
| **mbolt permuted vs orig** | **0.00096** | **98.87 %** | 1.0031 |
| orig-on-CPU vs orig-on-Metal (backend switch) | 0.00484 | 96.73 % | 1.0036 |

The permutation perturbs outputs **5× less than switching backend** on the same engine — noise every llama.cpp user already accepts implicitly. CI (`scripts/ci_correctness.sh`) enforces 4 gates on every commit: byte-verify, identity token-identity, layer-0/1 routing equivalence == 100 %, KLD ≤ backend-switch envelope. All 4 pass on the 30B; on the 80B, byte-verify and layer-0/1 routing equivalence pass (L0 = L1 = 0.0000 %) — the identity token-identity and KLD gates were run on the 30B only.

## 3. The I/O result (physical, both files)

Replaying the held-out routing trace cold against the two **real files** on disk (id→position translated for the rewritten file):

| | reads/token | ms/token | I/O-bound floor |
|---|---|---|---|
| original 80B file | 1418 | 255.7 | 3.91 tok/s |
| mbolt 80B file | 1104 | 204.1 | 4.90 tok/s |

**1.25× from `mv`-ing bytes** — and the physical file reproduces the simulator's virtual-offset prediction to 1 % (206.1 vs 204.1 ms/tok, identical read counts). Replay speedups across regimes (median, held-out): 30B warm 1.23× / cold 1.27×; 80B warm-128 1.09× / cold 1.25×. Interleave (per-expert up|gate|down adjacency, needs Phase-2 tensor split): 1.36–2.37× across all six measured regimes (30B warm-16 2.37×, warm-32 1.59×, warm-64 1.36×, cold 1.78×; 80B warm-128 1.75×, cold 2.29×).

## 4. End-to-end on today's llama.cpp — parity, and why

E2E protocol: experts forced to CPU (`-ot ".ffn_.*_exps.=CPU"`), 24 GB mlocked holder so the 33 GB model cannot fit in page cache, page cache pressure-evicted before every run down to a 9–16 GB file-backed residual (full eviction is not reachable under the 24 GB mlock without sudo `purge`), greedy, alternating order.

- **Metal offload mode:** original 22.0 vs mbolt 15.1 tok/s — *the original was not streaming.* Pageins: orig 66.6 GB (≈2× the 33.1 GB file) vs mbolt 39.6 GB. In the stock interleaved file the Metal-mapped buffer spans the whole 33 GB (dense tensors are scattered through it) and its residency keeps expert pages warm; mbolt segregates experts into an honestly evictable region. An engine/OS buffer-mapping artifact, not a layout-quality result — but a real deployment observation for stock llama.cpp on Apple silicon.
- **CPU mode (both files plain mmap, apples-to-apples):** orig 11.1/12.2 vs mbolt 11.2/11.4 tok/s, pageins identical (~42 GB) — **parity**. Cached compute ceiling 23.8 tok/s, so the runs were ~50 % I/O-bound; the replay-predicted ~+11 % end-to-end never materializes.

Root cause (inferred, not directly instrumented): llama.cpp's mmap path demand-pages experts in **16 KiB pages with kernel readahead**. The CPU-mode results are consistent with this being layout-blind — no speed gain and identical pagein volume (~42 GB) despite the mbolt file's adjacency — though fault sizes and readahead-cluster sizes were not instrumented directly. The 1.25× exists at the device level (measured) but the fault mechanism does not express it.

**The simulator "mispredicted" E2E, and that is the finding the spec anticipated:** the sim models an explicit-read streaming engine (sorted, merged slice reads — what colibri-class engines do); llama.cpp's mmap path is not that engine. Layout gains are harvestable by (a) explicit-read expert streaming (colibri port — Phase 2), (b) a llama.cpp streaming patch that reads missing expert slices explicitly instead of faulting, or (c) any engine adopting the `mbolt.*` metadata (tier hints + permutations ship in the file).

## 5. Honesty ledger

- Every number above: pressure-evicted and probed before each run (`purge` needs sudo); under the 24 GB mlock the E2E probes showed a 9–16 GB file-backed residual rather than full coldness — reported as-is. Replay numbers passed the cold-probe check against device baseline. N runs listed, held-out trace, same-window ratios. Machine and drive named. Single machine only — the spec's 2-machine protocol was not satisfiable here.
- The 30B E2E could not reach a genuine streaming regime on 64 GB (46 GB unmlocked squeeze: swapped out, null result 0.94×; ≥48 GB mlocked: destabilized the machine twice). The 80B (33 GB) streams legitimately under a safe 24 GB mlocked holder.
- 80B warm-128 replay (1.09×) is weaker than 30B warm-32 (1.23×): thinner co-activation statistics per pair (25k trace tokens over 130k pairs) and noisier 128-token windows. Cold results are tight on both.
- All capture completions hit the token cap (thinking mode); routing is decode-heavy by construction — the regime under test.
- Chaos-divergence means greedy outputs of rewritten models can differ from originals after tens of tokens (measured: 3/50 on llama.cpp Metal); quality deltas are bounded by the backend-switch envelope above. Token-identity claims for permuted MoE files should be treated as unverified until measured on the specific engine.

## 6. What ships

- `mbolt/` Python package: `mbolt-sim` (map / trace-stats / cluster / drive / evict / gate) + `mbolt` (rewriter CLI)
- `patches/llama.cpp-mbolt-trace.patch`: env-gated MoE routing tracer (`MBOLT_TRACE=file`), 176 lines
- `scripts/ci_correctness.sh` + byte_verify / correctness_proof / routing_equiv: the 4-gate suite
- Rewritten artifacts for both models + gate/launch charts + this report

## 7. Next (Phase 2, not started per spec)

Per-expert tensor split (or engine-side interleave awareness) to unlock the measured 1.36–2.37×; colibri-format port + benchmark; explicit-read streaming patch for llama.cpp; second machine + slower-drive-class measurements.
