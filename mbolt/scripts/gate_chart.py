"""Render the Phase 0 gate chart from gate JSON results.

Usage: gate_chart.py results/gate_warm.json [results/gate_cold.json ...] -o results/gate_chart.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYOUT_ORDER = [
    "baseline", "heat", "clique", "chain", "pipeline",
    "clique+pipeline", "chain+pipeline", "interleave",
]


def load(path):
    doc = json.load(open(path))
    per_layout = {}
    for r in doc["results"]:
        per_layout.setdefault(r["layout"], []).append(r)
    return doc, per_layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gates", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--title", default="mbolt Phase 0 gate: predicted decode I/O by layout")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.gates), figsize=(7.5 * len(args.gates), 5.2), squeeze=False)

    for ax, path in zip(axes[0], args.gates):
        doc, per_layout = load(path)
        names = [n for n in LAYOUT_ORDER if n in per_layout]
        base_by_run = {r["run"]: r["io_ms_median"] for r in per_layout["baseline"]}

        toks = []
        spreads = []
        speedups = []
        for n in names:
            rs = per_layout[n]
            tok_vals = [r["implied_tok_s"] for r in rs]
            toks.append(np.median(tok_vals))
            spreads.append((np.median(tok_vals) - min(tok_vals), max(tok_vals) - np.median(tok_vals)))
            ratios = [base_by_run[r["run"]] / r["io_ms_median"] for r in rs]
            speedups.append(np.median(ratios))

        colors = ["#888888" if n == "baseline" else ("#d4a017" if n == "interleave" else "#2c7fb8") for n in names]
        y = np.arange(len(names))
        ax.barh(y, toks, xerr=np.array(spreads).T, color=colors, height=0.62, capsize=3)
        for i, (tk, sp) in enumerate(zip(toks, speedups)):
            label = f"{tk:.1f} tok/s" + (f"  ({sp:.2f}x)" if names[i] != "baseline" else "")
            ax.text(tk + max(toks) * 0.02, i, label, va="center", fontsize=9)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("implied decode tok/s (1000 / median I/O ms per token)")
        cache = doc["cache_slots"]
        mode = "cold (no cache)" if cache == 0 else f"warm (LRU {cache} slots/layer)"
        ax.set_title(f"{mode} — {doc['runs']} runs × {doc['tokens']} held-out tokens\n"
                     f"device random-read {doc.get('device_random_mbs', 0):.0f} MB/s", fontsize=10)
        ax.set_xlim(0, max(toks) * 1.35)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
