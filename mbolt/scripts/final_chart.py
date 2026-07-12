"""Final Phase 0 gate chart: measured warm + cold panels, plus per-drive projection."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "/Users/dor/Documents/code/GPUopt/results"
LAYOUT_ORDER = [
    "baseline", "heat", "pipeline", "clique", "chain",
    "clique+pipeline", "chain+pipeline", "interleave",
]


def load(path):
    doc = json.load(open(path))
    per = {}
    for r in doc["results"]:
        per.setdefault(r["layout"], []).append(r)
    return doc, per


def panel_measured(ax, path, label):
    doc, per = load(path)
    names = [n for n in LAYOUT_ORDER if n in per]
    base_by_run = {r["run"]: r["io_ms_median"] for r in per["baseline"]}
    toks, spreads, speedups = [], [], []
    for n in names:
        tok_vals = [r["implied_tok_s"] for r in per[n]]
        toks.append(np.median(tok_vals))
        spreads.append((np.median(tok_vals) - min(tok_vals), max(tok_vals) - np.median(tok_vals)))
        speedups.append(np.median([base_by_run[r["run"]] / r["io_ms_median"] for r in per[n]]))
    colors = ["#8c8c8c" if n == "baseline" else ("#d4a017" if n == "interleave" else "#2c7fb8") for n in names]
    y = np.arange(len(names))
    ax.barh(y, toks, xerr=np.array(spreads).T, color=colors, height=0.62, capsize=3)
    for i, (tk, sp) in enumerate(zip(toks, speedups)):
        t = f"{tk:.1f}" + ("" if names[i] == "baseline" else f"  ({sp:.2f}x)")
        ax.text(tk + max(toks) * 0.02, i, t, va="center", fontsize=9)
    ax.set_yticks(y, [n + (" *" if n == "interleave" else "") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("I/O-bound decode tok/s (median, held-out trace)")
    ax.set_title(label, fontsize=10)
    ax.set_xlim(0, max(toks) * 1.42)
    ax.grid(axis="x", alpha=0.3)


def panel_drives(ax):
    # linear I/O model fit to all 88 measured replay points
    pts = []
    for f in ["gate_warm32.json", "gate_cold.json", "gate_warm16.json", "gate_warm64.json"]:
        doc = json.load(open(f"{RES}/{f}"))
        for r in doc["results"]:
            pts.append((r["reads_per_token"], r["mb_per_token"] * 1e6, r["io_ms_median"] / 1e3))
    A = np.array([[a, b] for a, b, _ in pts])
    yv = np.array([t for _, _, t in pts])
    (L, invB), *_ = np.linalg.lstsq(A, yv, rcond=None)

    _, per = load(f"{RES}/gate_warm32.json")
    meas = {
        k: (np.median([r["reads_per_token"] for r in v]), np.median([r["mb_per_token"] for r in v]) * 1e6)
        for k, v in per.items()
    }
    drives = [
        ("this Mac (fit)", L, 1 / invB),
        ("PCIe4 NVMe", 90e-6, 5e9),
        ("PCIe3 NVMe", 90e-6, 3e9),
        ("USB/SATA SSD", 200e-6, 0.5e9),
    ]
    x = np.arange(len(drives))
    width = 0.28
    sel = ["chain", "chain+pipeline", "interleave"]
    cols = {"chain": "#74a9cf", "chain+pipeline": "#2c7fb8", "interleave": "#d4a017"}
    tb = [meas["baseline"][0] * l + meas["baseline"][1] / b for _, l, b in drives]
    for i, lay in enumerate(sel):
        sp = [
            tb[j] / (meas[lay][0] * l + meas[lay][1] / b)
            for j, (_, l, b) in enumerate(drives)
        ]
        ax.bar(x + (i - 1) * width, sp, width, label=lay + (" *" if lay == "interleave" else ""), color=cols[lay])
        for xi, s in zip(x + (i - 1) * width, sp):
            ax.text(xi, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="#888", lw=0.8)
    ax.axhline(1.3, color="#c0392b", lw=0.8, ls="--")
    ax.text(len(drives) - 0.55, 1.31, "gate 1.3x", color="#c0392b", fontsize=8)
    ax.set_xticks(x, [d for d, _, _ in drives], fontsize=9)
    ax.set_ylabel("projected speedup vs baseline (warm-32 workload)")
    ax.set_title("per-drive projection (linear I/O model, R²=0.96;\nconservative vs measured locality effects)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
panel_measured(axes[0], f"{RES}/gate_warm32.json",
               "measured, warm cache (LRU 32/128 experts/layer)\n5 runs x 192 held-out tokens, cold-verified F_NOCACHE")
panel_measured(axes[1], f"{RES}/gate_cold.json", "measured, cold (no expert cache)\n3 runs x 48 held-out tokens")
panel_drives(axes[2])
fig.suptitle(
    "mbolt Phase 0 gate - Qwen3-30B-A3B IQ3_XXS (12.9GB), routing trace: 200 prompts / 76k decode tokens\n"
    "M5 Pro MacBook Pro, APPLE SSD AP1024Z - physical replay of per-token expert reads at simulated layouts - "
    "* interleave requires per-expert tensors (Phase 2: not loadable by current llama.cpp)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{RES}/gate_chart.png", dpi=150)
print("wrote", f"{RES}/gate_chart.png")
