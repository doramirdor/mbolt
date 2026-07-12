# mbolt — Profile-Guided Layout Optimization for GGUF Models

## Mission
Build a compiler pass for model binaries: take (model.gguf, routing profile) → emit a
bit-exact model whose on-disk layout matches the workload's access pattern.
BOLT/PGO for LLM weights.

**Thesis under test:** in expert-streaming / offloaded inference, decode speed is
bounded by storage access patterns that the checkpoint's arbitrary tensor order forces.
Reordering bytes by measured co-activation converts random reads to sequential reads.

## Phase 0 — I/O Simulator (THE GATE — build nothing else until this passes)
Goal: predict the speedup for a given (model, layout, drive) WITHOUT moving any weights.

1. **Trace capture.** Patch llama.cpp to log MoE routing: for each token, per layer,
   the top-k expert indices selected. Smallest viable patch: hook where the router
   output is argmax'd/top-k'd in the MoE forward (llm_build path for qwen3moe arch),
   append `(layer, [expert_ids])` per token to a binary log. Env-gated: MBOLT_TRACE=file.
   Alternative if patching is painful: use ik_llama.cpp or add a ggml callback on the
   ffn_gate_inp tensor output.
   Model: Qwen3-30B-A3B (or Qwen3.6-35B-A3B) at IQ2/IQ3 — small experts, streams on
   consumer hardware. Run ~200 real prompts (agent traces preferred) → routing log.

2. **Offset mapping.** Parse the GGUF tensor-info table (gguf-py GGUFReader): compute the
   exact byte range of every expert slice: offset(tensor) + expert_idx * slice_bytes,
   for each of ffn_up_exps / ffn_gate_exps / ffn_down_exps per layer.

3. **Replay engine.** For a routing log, generate the exact sequence of byte-range reads
   a streaming engine would issue per token, under a configurable cache model
   (LRU with N slots/layer — mirror colibri's model; also cache-off = cold mode).
   Then physically replay that read sequence against the real GGUF file with
   O_DIRECT + posix_fadvise(DONTNEED) between runs (drop page cache: run as script
   with sync + echo 3 > /proc/sys/vm/drop_caches between configs; on macOS use purge).
   Measure wall-clock per simulated token, reads/sec, achieved MB/s.

4. **Layout candidates to simulate** (pure offset arithmetic — no file rewriting):
   a. `baseline` — offsets as-is
   b. `clique` — experts reordered per layer by co-activation clustering (see algorithm)
   c. `pipeline` — hot backbone in layer-execution order across layers, cold at tail
   d. `clique+pipeline` — both
5. **Clustering algorithm.** Build per-layer co-activation graph: nodes = experts,
   edge weight = P(i and j both routed for same token). Cluster with a greedy
   modularity/agglomerative pass (networkx is fine; this is small — 128-256 nodes/layer).
   Order clusters by total heat; order experts within cluster by heat.
6. **GATE:** simulated speedup of best layout vs baseline, warm-cache mode, on the dev
   machine's NVMe. ≥1.3x → proceed to Phase 1. <1.15x → project dead, write up the
   negative result with the measured numbers. Between → judgment call, test on a
   second drive class (external SSD / slower NVMe) where random-read penalty is larger.

## Phase 1 — The Rewriter (only after gate passes)
`mbolt model.gguf profile.bin -o model.opt.gguf`

1. **Expert permutation (bit-exactness core).** Per layer: permutation π from Phase 0
   clustering. Physically reorder expert slices within each ffn_*_exps tensor, AND apply
   the inverse permutation to the corresponding rows of the router projection tensor
   (ffn_gate_inp weight — verify exact tensor name and orientation per arch; dims are
   stored reversed in GGUF). Handle quant-block granularity: slices must remain valid
   quant blocks (they do — expert slice = whole rows; verify block alignment per type).
2. **Correctness proof (non-negotiable, before any perf claims):**
   - Load original + rewritten in llama.cpp, same seed, greedy, 50 diverse prompts →
     token-for-token identical output. Any divergence = bug, full stop.
   - Also teacher-forcing logit comparison on 32 positions (tolerance: exactly 0 —
     permutation is exact arithmetic, not approximation).
3. **Alignment pass.** Re-emit with general.alignment=4096; pad expert slice starts to
   page boundaries. Measure file-size overhead (should be <1%). Verify llama.cpp loads it.
4. **Metadata pass.** Write mbolt.* KV keys: version, coactivation summary (top cliques),
   per-expert heat (uint32 array per layer), tier_hint. Verify old llama.cpp ignores
   them cleanly (it must, by spec — but verify).
5. **End-to-end benchmark.** Same protocol as Phase 0 cache-drop discipline; llama.cpp
   with expert offload (-ot ".ffn_.*_exps.=CPU" for the RAM-streaming regime, and/or
   --no-mmap off, restricted RAM via cgroup to force streaming). Report tok/s
   baseline vs optimized, 5 runs each, median + spread. On at least 2 machines.

## Phase 2 — stretch goals (separate build, do not start early)
- colibri integration: their format differs (safetensors-derived container) — port the
  permutation pass, post benchmark to their repo (issue thread, before/after on GLM-5.2)
- Per-expert precision: split ffn_*_exps into per-expert tensors (llama.cpp legacy path
  supports separate expert tensors? verify) or propose packed mixed type; heat decides
  IQ4 vs IQ2 per expert; floor's behavioral gate certifies the mix
- mlx-lm / oMLX expert-streaming feature requests reference this file format

## Honesty rules
- Every performance number: cold-cache disciplined, N=5, median, both machines named
  with drive model. Page cache is the #1 way to fool yourself here.
- The correctness proof (token-identical) ships in CI, runs on every commit.
- If clique structure doesn't exist in the routing data (flat co-activation graph),
  report entropy stats and stop — don't cluster noise.
- Simulator predictions vs Phase 1 measured reality get compared in the writeup;
  a simulator that mispredicts is itself a finding.

## Deliverables
- Phase 0: `mbolt-sim` + routing-trace llama.cpp patch + the gate chart
  (predicted tok/s by layout, per drive)
- Phase 1: `mbolt` CLI + correctness CI + benchmark table + writeup
- The launch artifact: one chart — same machine, same model, same engine,
  tok/s before/after `mv`-ing bytes.
