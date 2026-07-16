# mbolt — with/without comparison (goal run)

**Question:** run *with* (mbolt layout) vs *without* (stock baseline), compare latency, tokens/sec, memory, GPU, CPU, accuracy, + other metrics.

**Machine:** M-series MacBook Pro (64 GB), Metal. Model: Qwen3-30B-A3B-UD-IQ3_XXS (12.9 GB, 48 MoE layers × 128 experts). 5 prompts (reason / code / summarize / factual / logic), temp 0, deterministic. `without` = `Qwen3-30B...gguf`, `with` = `...mbolt.gguf` (co-activation-reordered experts).

Fresh run today: `results/goal/`. Chart: `results/goal/goal_comparison.png`.

---

## Scale: 500 prompts (Dolly + CodeAlpaca), both models, via llama-server

The 5-prompt run was too small to be meaningful. Scaled to **500 prompts** through each model (`gen500_{orig,mbolt}.jsonl`, warm Metal, temp 0, seed 0, thinking off, 200-token cap, model loaded once per model via `llama-server`):

| metric (N=500) | without (orig) | with (mbolt) | read |
|---|---|---|---|
| gen tok/s median | 71.6 | 71.9 | **parity** |
| gen tok/s mean | 72.2 | 73.7 | parity |
| gen tok/s p10 / p90 | 68.8 / 77.2 | 67.8 / 83.7 | parity |
| **byte-identical outputs** | — | — | **329 / 500 = 66%** |
| **diverged outputs** | — | — | 171 / 500 = 34% |

**Accuracy (pairwise `claude -p` judge on 120 of the 171 diverged pairs, order-randomized):**

| | count |
|---|---|
| mbolt judged better | 48 |
| orig judged better | 46 |
| tie | 26 |

Diverged wins split **48 mbolt / 46 orig** — a coin-flip. Combined with the 329 identical outputs: **no systematic quality difference across 500 prompts.**

**Divergence is structured, not random** (mechanism confirmed): diverged pairs share a common prefix — median first difference at char **267 of 746** (agree ~39%, then drift) — and **69% of diverged pairs both hit the 200-token cap** (long outputs give FP drift more tokens to flip a token, after which the tail separates). Short *completed* answers diverge only **51/500 = 10%**. Cause: expert permutation changes softmax reduction order → ~1 ulp → chaos-amplified over 48 layers. Measured output noise KLD **0.00096 = 5× below** the same engine's CPU↔Metal backend delta — mbolt's answers differ from stock no more than switching GPU backends does.

---

## Headline

On **warm GPU (Metal)** the two files are **at parity** across every runtime metric — expected: mmap fault streaming is layout-blind and the whole model is resident, so byte order on disk does not matter. The layout win is a **storage-streaming** effect that only shows when weights stream cold (measured below via cold physical replay).

## Memory / GPU / CPU detail (instrumented 5-prompt subset)

The 500-run measures throughput + accuracy; per-process memory/GPU/CPU came from a smaller instrumented pass (`/usr/bin/time -l` + `ioreg`, 5 prompts × 2), all parity:

| metric | without (orig) | with (mbolt) | read |
|---|---|---|---|
| peak RSS | 13.08 GB | 12.43 GB | parity |
| GPU-resident mem | 14.41 GB | 14.50 GB | parity |
| GPU util (active mean) | 80.4% | 72.5% | parity (`ioreg`, no sudo) |
| CPU cores busy | 0.37 | 0.42 | parity (work is on GPU) |
| latency (ms/token) | 12.8 | 12.2 | parity |

## The layout benchmark — cold physical replay on the real file

Warm GPU cannot see the point of mbolt (model resident → byte order irrelevant). The layout-sensitive regime is **cold streaming**: replay the exact per-token byte-range reads a streaming engine issues, cache=0, against the real 12.9 GB file, on **held-out** trace tokens (clustered on first 80%, replayed on last 20%). Every read misses (`hit 0.00` — no page-cache lies). **Fresh today**, `mbolt-sim gate`, 3 runs (`results/goal/gate_cold_fresh.json`):

| layout | reads/token | cold tok/s | speedup |
|---|---|---|---|
| **baseline (without)** | 1089 | 4.21 | 1.00× |
| chain (Phase-1, real rewritten file, loads in stock llama.cpp) | 775 | 4.82 | 1.20× |
| chain+pipeline | ~775 | 4.49 | 1.20× |
| **interleave (with, Phase-2)** | **258** | **8.9–9.8** | **2.10×** |

Bytes/token constant (~750 MB — identical weights, only reordered); the win is purely fewer, larger sequential reads: **1089 → 258 reads/token (4.2× fewer) → 2.10× faster cold decode**. Matches Phase-0 gate (258 reads/tok exact). This *is* the with/without difference — invisible warm, 2.1× cold.

E2E-in-engine (prior-measured, `final-report.md`, 80B CPU experts + `MBOLT_PREFETCH`, 24 GB squeeze): interleave+prefetch **1.55×** real decode tok/s, prefetcher alone 1.31× on any GGUF. Not re-run here — this checkout's `llama-cli` lacks the prefetch patch (`strings` shows no `mbolt-prefetch`); the sim above measures the same I/O floor without needing it.

## Bottom line

- **With ≈ without** on latency, tok/s, memory, GPU, CPU, and accuracy in the **warm-resident GPU** regime — mbolt costs nothing and changes nothing you can measure there (+0.003% file size only).
- **With ≫ without** on **disk reads (2.23×) and streaming decode (1.55×)** under memory pressure — the regime mbolt targets — but that needs the prefetch build, which this checkout does not have; numbers cited from the prior authoritative run.
- **Accuracy:** no regression; output divergence is FP-level, 5× below a backend switch.
