# mbolt

Profile-guided layout optimization for GGUF MoE models — BOLT/PGO, but for weights.

Trace which experts your workload co-activates, rewrite the GGUF so those bytes sit
together on disk, decode with explicit-read prefetch: **1.55–1.63× faster** memory-squeezed
MoE streaming in llama.cpp, **2.23×** at the storage level. Weights byte-exact, +0.003% size.

```bash
pip install mbolt

mbolt-sim cluster route.bin -o perms.json         # co-activation -> permutations
mbolt model.gguf perms.json -o model.opt.gguf --layout interleave
mbolt-sim gate model.gguf route.bin perms.json -o gate.json   # predict before rewriting
```

Trace capture and the explicit-read prefetcher need a small llama.cpp patch
(`MBOLT_TRACE` / `MBOLT_PREFETCH` env vars) shipped in the repo.

Full docs, benchmark reports, and the llama.cpp patch:
**https://github.com/doramirdor/mbolt**

MIT license.
