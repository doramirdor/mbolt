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

# ---- Panel 2: llama.cpp end-to-end today ----
ax = axes[1]
groups = ["llama.cpp today\n(mmap faults, squeezed)", "cached ceiling\n(no squeeze)"]
orig_vals = [np.median([12.2, 11.1]), 23.8]
mb_vals = [np.median([11.2, 11.4]), 23.8]
x = np.arange(len(groups))
w = 0.34
ax.bar(x - w / 2, orig_vals, w, label="original", color="#8c8c8c")
ax.bar(x + w / 2, mb_vals, w, label="mbolt", color="#2c7fb8")
for xi, v in zip(x - w / 2, orig_vals):
    ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, mb_vals):
    ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(x, groups, fontsize=9)
ax.set_ylabel("end-to-end decode tok/s")
ax.set_title("llama.cpp end-to-end today: parity\n"
             "16KiB page-fault streaming + kernel readahead is\n"
             "layout-blind - explicit slice reads are required to harvest", fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---- Panel 3: replay speedups across regimes/models ----
ax = axes[2]
regimes = ["30B warm\n(32/128)", "30B cold", "80B warm\n(128/512)", "80B cold"]
chainpipe = [1.231, 1.270, 1.089, 1.247]
interleave = [1.589, 1.780, 1.750, 2.292]
x = np.arange(len(regimes))
ax.bar(x - w / 2, chainpipe, w, label="chain+pipeline (shipped)", color="#2c7fb8")
ax.bar(x + w / 2, interleave, w, label="interleave (needs per-expert tensors)", color="#d4a017")
for xi, v in zip(x - w / 2, chainpipe):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, interleave):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(1.0, color="#888", lw=0.8)
ax.set_xticks(x, regimes, fontsize=9)
ax.set_ylabel("measured I/O speedup vs stock layout")
ax.set_title("measured replay speedups (median, held-out)\n"
             "explicit-read engines harvest these today;\n"
             "interleave is the Phase-2 prize", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "mbolt: profile-guided layout optimization for GGUF - M5 Pro MacBook Pro, APPLE SSD AP1024Z, llama.cpp b6"
    " - bit-exact weights, permutation-equivalent routing (verified), output noise below backend-switch envelope",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{RES}/launch_chart.png", dpi=150)
print("wrote", f"{RES}/launch_chart.png")
