"""Cross-domain generalization: cluster on one prompt domain, replay on another.

Answers colibri #119 Q1: does a co-activation layout learned on general
instructions (dolly) still pay on a code workload (codealpaca)?

Segments the 30B trace by prompt (order + per-prompt decode counts from
server.log), trains perms on dolly-only tokens, evaluates replay on
codealpaca-only tokens. Controls: mixed-trained perms (shipped perms.json)
and code-trained perms on the same eval set.
"""

import json
import re
import sys

import numpy as np

sys.path.insert(0, "/Users/dor/Documents/code/GPUopt/mbolt/src")
from mbolt.cluster import cluster_all
from mbolt.gguf_map import load_model_map
from mbolt.layouts import build_layout
from mbolt.replay import replay
from mbolt.sim_cli import _pressure_evict, _probe_random_read_mbs
from mbolt.trace import read_trace

BASE = "/Users/dor/Documents/code/GPUopt"
MODEL = f"{BASE}/models/Qwen3-30B-A3B-UD-IQ3_XXS.gguf"

# per-prompt decode-pass counts: completion_tokens - 1 (last token has no pass)
counts = []
for line in open(f"{BASE}/traces/server.log"):
    m = re.search(r"\beval time =\s+[0-9.]+ ms /\s+(\d+) tokens", line)
    if m and "prompt eval" not in line:
        counts.append(int(m.group(1)) - 1)
sources = [json.loads(l)["source"] for l in open(f"{BASE}/traces/prompts.jsonl")]
assert len(counts) == len(sources) == 200, (len(counts), len(sources))

t = read_trace(f"{BASE}/traces/routing.bin")
assert len(t.decode) == sum(counts), (len(t.decode), sum(counts))

dom = np.repeat([s.startswith("dolly") for s in sources], counts)
dolly, code = t.decode[dom], t.decode[~dom]
print(f"dolly tokens {len(dolly)}, codealpaca tokens {len(code)}", flush=True)

mm = load_model_map(MODEL)
perms_by = {}
for name, toks in [("dolly-trained", dolly), ("code-trained", code[: len(code) // 2])]:
    res = cluster_all(toks, t.n_expert)
    perms_by[name] = [l["chain_perm"] for l in res["layers"]]
    print(f"clustered {name} on {len(toks)} tokens", flush=True)
perms_by["mixed-trained"] = [
    l["chain_perm"] for l in json.load(open(f"{BASE}/results/perms.json"))["layers"]
]

# eval: codealpaca tokens NOT used for code-trained control (second half)
ev = code[len(code) // 2 :]
layouts = {"baseline": build_layout("baseline", mm)}
for name, p in perms_by.items():
    layouts[name] = build_layout("chain+pipeline", mm, p)

_pressure_evict()
print(f"cold probe: {_probe_random_read_mbs(MODEL):.0f} MB/s", flush=True)

WIN, WARM, MEAS, RUNS = 256, 64, 192, 3
results = {}
for run in range(RUNS):
    window = ev[run * WIN : (run + 1) * WIN]
    for name, lay in layouts.items():
        r = replay(MODEL, lay, window, cache_slots=32, warmup_tokens=WARM, measure_tokens=MEAS)
        s = r.stats()
        results.setdefault(name, []).append(s["io_ms_median"])
        print(f"run {run} {name:>14}: {s['io_ms_median']:7.2f} ms/tok, "
              f"{s['reads_per_token']:6.1f} reads/tok", flush=True)
    _pressure_evict()

print("\n=== speedup vs baseline on CODEALPACA-only held-out tokens (median of same-window ratios) ===")
for name in layouts:
    if name == "baseline":
        continue
    ratios = [results["baseline"][i] / results[name][i] for i in range(RUNS)]
    print(f"{name:>14}: {np.median(ratios):.3f}x  runs {[f'{x:.3f}' for x in ratios]}")
