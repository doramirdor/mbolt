"""The launch artifact: same machine, same model, same engine -
what moving bytes does and where today's engine loses it.

Panel 1: I/O-bound decode floor, physically measured on the two real files
         (original vs mbolt-rewritten), cold, held-out trace replay.
Panel 2: llama.cpp end-to-end today (mmap fault streaming, CPU mode, 24GB
         mlocked squeeze) - parity, plus the cached compute ceiling.
Panel 3: measured replay speedups across regimes on both models (what an
         explicit-read streaming engine can harvest).
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "/Users/dor/Documents/code/GPUopt/results"

fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

# ---- Panel 1: physical before/after on Qwen3-Next-80B ----
ax = axes[0]
labels = ["original file\n(stock layout)", "mbolt file\n(chain+pipeline)"]
ms = [255.65, 204.05]  # measured cold replay on the two physical files
toks = [1000 / m for m in ms]
bars = ax.bar(labels, toks, color=["#8c8c8c", "#2c7fb8"], width=0.55)
for b, tk, m in zip(bars, toks, ms):
    ax.text(b.get_x() + b.get_width() / 2, tk + 0.08, f"{tk:.2f} tok/s\n({m:.0f} ms/tok)",
            ha="center", fontsize=10)
ax.text(0.5, 0.92, "1.25x", transform=ax.transAxes, ha="center",
        fontsize=22, fontweight="bold", color="#2c7fb8")
ax.set_ylabel("I/O-bound decode floor (tok/s)")
ax.set_title("what moving bytes buys: expert-read I/O floor\n"
             "Qwen3-Next-80B IQ3_XXS, cold, held-out routing trace,\n"
             "physically replayed on each real file (1418 -> 1104 reads/tok)", fontsize=10)
ax.set_ylim(0, max(toks) * 1.3)
ax.grid(axis="y", alpha=0.3)

# ---- Panel 2: llama.cpp end-to-end ladder (Phase 2 session, same-day runs) ----
ax = axes[1]
configs = ["stock file\nstock engine", "stock file\n+ prefetcher", "chain+pipeline\n+ prefetcher",
           "interleave\nfaults only", "interleave\n+ prefetcher"]
vals = [5.8, 7.6, 8.0, 5.7, 9.0]  # medians, 3 reps alternating, CPU experts, 24GB mlocked squeeze
colors = ["#8c8c8c", "#74a9cf", "#2c7fb8", "#c9b37e", "#d4a017"]
x = np.arange(len(configs))
bars = ax.bar(x, vals, 0.62, color=colors)
for xi, v in zip(x, vals):
    sp = v / vals[0]
    ax.text(xi, v + 0.12, f"{v:.1f}\n({sp:.2f}x)", ha="center", fontsize=8.5)
ax.set_xticks(x, configs, fontsize=8)
ax.set_ylim(0, 11.5)
ax.set_ylabel("end-to-end decode tok/s (squeezed, 80B)")
ax.set_title("llama.cpp end-to-end: layout and explicit reads are\n"
             "complements - faults alone are layout-blind (0.98x);\n"
             "interleave + prefetcher = 1.55x, reads/token 28.6k -> 8.5k\n"
             "per 128 tokens (avg read 450KB -> 1.5MB)", fontsize=9.5)
ax.grid(axis="y", alpha=0.3)

# ---- Panel 3: replay speedups across regimes/models ----
ax = axes[2]
regimes = ["30B warm\n(32/128)", "30B cold", "80B warm\n(128/512)", "80B cold"]
chainpipe = [1.231, 1.270, 1.089, 1.247]
interleave = [1.589, 1.780, 1.750, 2.292]
x = np.arange(len(regimes))
w = 0.34
ax.bar(x - w / 2, chainpipe, w, label="chain+pipeline (shipped)", color="#2c7fb8")
ax.bar(x + w / 2, interleave, w, label="interleave (shipped Phase 2, strided views; experimental)", color="#d4a017")
for xi, v in zip(x - w / 2, chainpipe):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, interleave):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(1.0, color="#888", lw=0.8)
ax.set_xticks(x, regimes, fontsize=9)
ax.set_ylabel("measured I/O speedup vs stock layout")
ax.set_title("measured replay speedups (median, held-out)\n"
             "explicit-read engines harvest these today;\n"
             "interleave shipped in Phase 2 (CPU experts)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "mbolt: profile-guided layout optimization for GGUF - M5 Pro MacBook Pro, APPLE SSD AP1024Z, llama.cpp b9977"
    " - bit-exact weights, permutation-equivalent routing (verified), output noise vs backend-switch envelope: 0.2x (chain+pipeline), 1.26x (interleave, experimental)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{RES}/launch_chart.png", dpi=150)
print("wrote", f"{RES}/launch_chart.png")
