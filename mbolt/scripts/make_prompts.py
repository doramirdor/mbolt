"""Build the 200-prompt capture set from locally cached datasets.

Mix: 120 Dolly (varied categories, incl. context-grounded) + 80 CodeAlpaca
(code tasks) - approximating an agent-ish mix of QA, extraction, and coding.
Deterministic selection (seed 42).
"""

import json
import random

DOLLY = "/Users/dor/.cache/huggingface/hub/datasets--databricks--databricks-dolly-15k/blobs/2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
CODE = "/Users/dor/.cache/huggingface/hub/datasets--sahil2801--CodeAlpaca-20k/blobs/4599591b17572755907bd945e34d25a956dcab09"
OUT = "/Users/dor/Documents/code/GPUopt/traces/prompts.jsonl"

rng = random.Random(42)

dolly = [json.loads(l) for l in open(DOLLY)]
code = json.load(open(CODE))

prompts = []

# Dolly: sample per category for diversity; cap context length
by_cat = {}
for r in dolly:
    by_cat.setdefault(r["category"], []).append(r)
per_cat = 120 // len(by_cat) + 1
picked = []
for cat, rows in sorted(by_cat.items()):
    rows = [r for r in rows if len(r.get("context", "")) < 4000]
    picked.extend(rng.sample(rows, min(per_cat, len(rows))))
rng.shuffle(picked)
for r in picked[:120]:
    text = r["instruction"]
    if r.get("context"):
        text = f"{r['context']}\n\n{r['instruction']}"
    prompts.append({"prompt": text, "source": f"dolly/{r['category']}"})

# CodeAlpaca
code_rows = [r for r in code if len(r.get("input", "")) < 2000]
for r in rng.sample(code_rows, 80):
    text = r["instruction"]
    if r.get("input"):
        text = f"{r['instruction']}\n\n{r['input']}"
    prompts.append({"prompt": text, "source": "codealpaca"})

rng.shuffle(prompts)
with open(OUT, "w") as f:
    for p in prompts:
        f.write(json.dumps(p) + "\n")
print(f"wrote {len(prompts)} prompts to {OUT}")
