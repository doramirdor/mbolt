# mbolt decision log

## 2026-07-13: What to build after Phase 2 — and a retraction

Earlier I listed three ways to make the tok/s gap visible, ranked "async prefetch"
first with an "up to ~1.8×" estimate. Working the list:

**1. Async (one-layer-ahead) prefetch — RETRACTED, will not build.**
The 1.8× estimate assumed I/O could overlap compute across the token. It cannot,
for a single decode stream: layer L's expert ids are produced by layer L's router,
which runs microseconds before layer L's expert matmuls consume the weights. The
dependency chain (attention L → router L → experts L → attention L+1) leaves no
independent work to overlap with the reads. The only way to get lookahead is
prediction, and we already measured that prediction does not pay on this model
class (top-64-heat covers 34 % of selections; train-split transition matrices
reach 51 % coverage at ~2× byte overfetch — a regression when bandwidth-bound).
An "async worker one layer ahead" is a prediction engine wearing a trench coat.

**2. Slower drive — blocked on hardware.** No external SSD is attached. The
benchmark scripts run unmodified when one is; expected to shift the compute:I/O
ratio toward I/O and enlarge every layout number.

**3. Interleave — already shipped** (strided views + loader patch, 1.55× E2E
with the prefetcher, 2.23× I/O floor). The biggest item on the list is done.

**What we build instead: parallel reads inside the prefetcher (QD > 1).**
The kernel of the async idea survives with the arrow pointed sideways: the
prefetcher knows ALL of a layer's missing ranges at once and currently reads
them sequentially (QD1). Issuing them from a persistent thread pool raises NVMe
queue depth — no dependency problem, no prediction, ~80 lines, benefits every
layout, and Apple NVMe throughput scales strongly with QD. Unlike overlap-with-
compute, overlap-of-reads-with-reads is legal here because there is no ordering
constraint among a layer's slices.

Also queued: a multi-stream (-np) aggregate-throughput measurement — parallel
sequences are the legitimate source of compute/I/O overlap that single-stream
async cannot provide, and batching grows the per-layer expert union, which
increases merge opportunities for co-activation layouts.

**Shipping note:** the prefetcher (+13-31 % on stock files, no rewrite needed)
is the natural first upstream artifact; the layout rewriter is what makes it
scale. Outreach drafts are staged in results/outreach/ pending approval and a
public repo link.

## 2026-07-13 (later): QD sweep results — substitutes, not multipliers

Built the parallel-read pool (MBOLT_PREFETCH_THREADS). Measured (80B, squeezed,
3 reps alternating, within-session):

| config | median tok/s |
|---|---|
| stock file, prefetch QD1 | 8.5 |
| stock file, prefetch QD4 | 9.4 (+11 %) |
| interleave, prefetch QD1 | 9.8 |
| interleave, prefetch QD4 | 9.5 (flat) |
| interleave, prefetch QD8 | 9.7 (flat, unstable: 7.9–10.2) |

Finding: **queue depth and layout are partial substitutes.** Parallel reads
recover most of the scattered layout's penalty; contiguous layout gets there
at QD1. Both converge at ~9.5–9.8 tok/s — the drive's effective ceiling for
this stream. Interleave@QD1 still wins on efficiency: same speed as
orig@QD4 with 3.4× fewer I/O ops and no extra threads (read threads compete
with the 14 compute threads — the QD8 spread shows it).

Practical guidance this yields: engines that can issue parallel explicit
reads should; formats should interleave anyway — it buys the same speed
without spending threads, and wins outright wherever parallel I/O is
unavailable (sync mmap paths, thread-starved boxes, QD-insensitive media).

Not run (left as future work): multi-stream (-np) aggregate throughput —
at the drive's effective ceiling already, expected flat on this hardware.
