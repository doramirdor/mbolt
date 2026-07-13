# mbolt — profile-guided layout optimization for GGUF models

BOLT/PGO for LLM weights: measure which MoE experts a real workload co-activates,
then rewrite the model file so those bytes sit together on disk. Bit-exact weights,
permutation-equivalent routing, ~0% size overhead.

```
mbolt-sim map model.gguf                      # expert slice offsets
MBOLT_TRACE=route.bin llama-server -m ...     # capture routing (patched llama.cpp)
mbolt-sim cluster route.bin -o perms.json     # co-activation cliques + chain order
mbolt-sim gate model.gguf route.bin perms.json -o gate.json   # physical I/O replay
mbolt model.gguf perms.json -o model.opt.gguf # rewrite (chain+pipeline layout)
scripts/ci_correctness.sh                     # 4-gate correctness suite
```

## Results (M5 Pro MacBook Pro, APPLE SSD AP1024Z)

- **I/O floor, measured on the physical files** (Qwen3-Next-80B IQ3_XXS, cold,
  held-out trace): 1418 → 1104 reads/token, 255.7 → 204.1 ms/token — **1.25×
  from moving bytes**. Replay across regimes: 1.09–1.27× (chain+pipeline);
  interleave layout (needs per-expert tensors): **1.59–2.29×**.
- **Correctness**: weights byte-exact; routing maps 100.000% through the
  permutation at equal inputs; output perturbation (KLD 0.00096, top-1 98.9%)
  is **5× below** the same engine's CPU↔Metal backend delta. Token-identity
  under any expert permutation is impossible in principle on current engines
  (softmax reduction order) — see `results/phase1-report.md`.
- **End-to-end on stock llama.cpp: parity.** Its 16KiB page-fault streaming is
  layout-blind; explicit-read engines (colibri-class) are required to harvest
  the measured gains. Full mechanism analysis in the report.

## Repo map

- `mbolt/` — package (`gguf_map`, `trace`, `cluster`, `layouts`, `replay`, CLIs)
- `mbolt/patches/llama.cpp-mbolt-trace.patch` — routing tracer (env-gated)
- `mbolt/scripts/` — capture, gate, correctness suite, benchmarks, charts
- `results/phase0-gate.md`, `results/phase1-report.md` — the write-ups
- `results/launch_chart.png`, `results/gate_chart.png` — the charts
- `mbolt-build-prompt.md` — original build spec

Measurement discipline: macOS `F_NOCACHE` does not bypass already-cached pages
and `purge` needs sudo — every number here passed a cold-probe check
(random-read speed vs established device baseline) after memory-pressure
eviction, N runs, medians, held-out traces, same-window ratios.
