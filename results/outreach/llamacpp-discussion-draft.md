# Draft: llama.cpp discussion post (for ggml-org/llama.cpp#18758 "Mmap faster than direct I/O for MoE models")

**Title: Data: on-disk expert layout + explicit slice reads for MoE streaming (measurements + patches)**

This thread debates mmap vs direct I/O for MoE streaming. I measured a third variable that changes the answer: **on-disk expert layout**, plus a small explicit-read prefetcher. Numbers from Qwen3-Next-80B-A3B IQ3_XXS (48 layers × 512 experts, experts = 95.4% of file bytes), M5 Pro / APPLE SSD AP1024Z:

1. **mmap fault streaming is layout-blind.** I rewrote the GGUF so co-activated experts are adjacent (profile-guided, from routing traces; weights byte-exact, router rows permuted to match — outputs equivalent within backend-switch FP noise). Under a memory squeeze, decode tok/s and pagein volume were identical to the stock file. 16KiB faults + kernel readahead simply cannot express layout locality.

2. **Explicit slice reads beat faults by +13-14% end-to-end** — on both stock and rewritten files. The patch hooks the eval callback: when `ffn_moe_topk` lands, it `pread()`s the selected experts' non-resident slices (mincore-checked) as sorted, gap-merged ranges into the page cache before the expert matmuls fault. env-gated, ~200 LOC including the tracer.

3. **Layout pays at the I/O level, big.** Replaying the exact per-token read pattern cold against the physical files: co-activation-ordered layout cuts reads/token 1418 → 775; an interleaved layout (each expert's up|gate|down contiguous, loaded via strided views — small loader patch, `mul_mat_id` already addresses experts via nb02) cuts it to ~370: **2.23× faster cold decode I/O**. On fast Apple NVMe the end-to-end conversion is eaten by compute (42ms/token) — on the 1-2GB/s drives discussed in this thread, the I/O share is the bottleneck and this is where the tok/s should move.

Patches + rewriter + I/O simulator + full honesty ledger (incl. what did NOT work: predictive prefetch, and why token-identity is impossible under any expert permutation): <REPO LINK>.

Suggested takeaway for the mmap-vs-O_DIRECT debate: the win isn't the syscall, it's (a) reading *slices* instead of faulting pages, and (b) a file layout that makes those slices contiguous. Both compose with either I/O path.
