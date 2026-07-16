"""Comparison chart for the with/without goal run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statistics as st

G = Path("/Users/dor/Documents/code/GPUopt/results/goal")
d = json.loads((G / "metrics_ngl99.json").read_text())


def med(name, key):
    return st.median([r[key] for r in d[name] if r[key] >= 0])


ORIG = "#7f8c8d"
MBOLT = "#e67e22"

import statistics as _st
o5 = [json.loads(l) for l in (G / "gen500_orig.jsonl").open()]
m5 = [json.loads(l) for l in (G / "gen500_mbolt.jsonl").open()]
otps = sorted(r["gen_tps"] for r in o5 if r["gen_tps"] > 0)
mtps = sorted(r["gen_tps"] for r in m5 if r["gen_tps"] > 0)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.suptitle("mbolt with/without — Qwen3-30B, 500 prompts (Dolly+CodeAlpaca), temp 0",
             fontsize=13, weight="bold")

# panel 1: warm tok/s distribution over 500 prompts
ax = axes[0]
ax.hist(otps, bins=30, alpha=0.55, color=ORIG, label=f"orig  med {_st.median(otps):.1f}")
ax.hist(mtps, bins=30, alpha=0.55, color=MBOLT, label=f"mbolt med {_st.median(mtps):.1f}")
ax.axvline(_st.median(otps), color=ORIG, lw=1.5, ls="--")
ax.axvline(_st.median(mtps), color=MBOLT, lw=1.5, ls="--")
ax.set_xlabel("gen tok/s"); ax.set_ylabel("prompts")
ax.set_title("Warm GPU throughput, N=500: parity", fontsize=10)
ax.legend(fontsize=8)

# panel 2: accuracy over 500 prompts
jd = json.loads((G / "judge500.json").read_text())
n_id = 500 - jd["n_diverged"]
ax = axes[1]
cats = ["identical\n(66%)", "diverged:\norig better", "diverged:\nmbolt better", "diverged:\ntie"]
t = jd["tally"]
vals = [n_id, t["orig"], t["mbolt"], t["tie"]]
ax.bar(cats, vals, color=["#95a5a6", ORIG, MBOLT, "#bbbbbb"])
ax.set_title("Accuracy: LLM judge, 500 prompts", fontsize=10)
ax.set_ylabel("prompts"); ax.set_ylim(0, 360)
for i, v in enumerate(vals):
    ax.text(i, v + 5, str(v), ha="center", fontsize=9)
ax.text(0.5, -70, "diverged split 48 mbolt / 46 orig = coin-flip, no regression",
        transform=ax.transData, ha="center", fontsize=7.5, style="italic", color="#555")

# panel 3: MEANINGFUL benchmark - cold physical replay, real file, held-out
ax = axes[2]
cfg = ["baseline", "chain\n+pipeline", "interleave"]
tps = [4.21, 4.49, 8.9]
reads = [1089, 775, 258]
cols = [ORIG, "#b7772f", MBOLT]
ax.bar(cfg, tps, color=cols)
ax.set_title("MEANINGFUL: cold replay, real file (30B)", fontsize=10)
ax.set_ylabel("cold decode tok/s")
for i, (v, r) in enumerate(zip(tps, reads)):
    ax.text(i, v + 0.12, f"{v}\n{r} rd/tok", ha="center", fontsize=8)
ax.text(2, 8.9 + 0.9, "2.10x", ha="center", fontsize=11, weight="bold", color=MBOLT)
ax.set_ylim(0, 11)
ax.text(0.5, -1.7, "held-out trace, cache=0, every read cold (hit 0.00)",
        transform=ax.transData, ha="center", fontsize=7, style="italic", color="#777")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out = G / "goal_comparison.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
