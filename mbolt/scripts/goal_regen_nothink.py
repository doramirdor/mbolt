"""Regenerate answers with thinking disabled (/no_think) and a large token
budget so both models produce COMPLETE answers. The first judge pass was
confounded by the 256-token cap truncating a reasoning model mid-thought;
completed answers give a fair accuracy comparison under expert permutation.
"""
import json
import re
import subprocess
from pathlib import Path

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
ORIG = "/Users/dor/Documents/code/GPUopt/models/Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
MBOLT = "/Users/dor/Documents/code/GPUopt/models/Qwen3-30B-A3B-UD-IQ3_XXS.mbolt.gguf"
OUT = Path("/Users/dor/Documents/code/GPUopt/results/goal/metrics_nothink.json")
N = 400

PROMPTS = [
    ("reason", "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning. /no_think"),
    ("code", "Write a Python function that returns the nth Fibonacci number iteratively, then explain its time complexity. /no_think"),
    ("summarize", "Summarize the tradeoffs between B-trees and LSM-trees for a write-heavy workload, then recommend one for a time-series database and justify. /no_think"),
    ("factual", "Explain the CAP theorem and give one concrete example of a CP system and one AP system. /no_think"),
    ("logic", "Five houses in a row, each a different color. The green house is immediately left of the white house. Which house cannot be the white house? Explain. /no_think"),
]


def run(model, prompt):
    cmd = [BIN, "-m", model, "-p", prompt, "-n", str(N), "-st", "--temp", "0",
           "-c", "2048", "--no-warmup", "-ngl", "99"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    ans = out.stdout
    key = prompt.replace(" /no_think", "")
    if key in ans:
        ans = ans.split(key, 1)[1]
    ans = re.split(r"\n\[ Prompt:", ans)[0].strip()
    return ans


def main():
    res = {"orig": [], "mbolt": []}
    for pid, prompt in PROMPTS:
        for name, model in [("orig", ORIG), ("mbolt", MBOLT)]:
            a = run(model, prompt)
            res[name].append({"prompt_id": pid, "answer": a})
            print(f"{name:<6} {pid:<10} len={len(a)}", flush=True)
    OUT.write_text(json.dumps(res, indent=2))
    # divergence report
    print("\nidentical:")
    for i, (pid, _) in enumerate(PROMPTS):
        a, b = res["orig"][i]["answer"], res["mbolt"][i]["answer"]
        print(f"  {pid:<10} identical={a.strip()==b.strip()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
