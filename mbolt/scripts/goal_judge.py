"""Accuracy via pairwise LLM-as-judge (claude -p).
Reads saved orig/mbolt answers, presents each pair blind + order-randomized
(deterministic swap by prompt index), asks claude which answer is better.
mbolt is a weight permutation of orig -> expectation is TIE/equivalent quality.
A systematic loss for mbolt would flag an accuracy regression.
"""

import json
import subprocess
import sys
from pathlib import Path

METRICS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/dor/Documents/code/GPUopt/results/goal/metrics_ngl99.json")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Users/dor/Documents/code/GPUopt/results/goal/judge.json")

RUBRIC = """You are a strict evaluator. Two AI assistants (A and B) answered the same prompt.
Judge which answer is more correct, complete, and clear. Ignore length unless it hurts quality.
Reply with STRICT JSON only, no prose: {"winner":"A"|"B"|"TIE","reason":"<12 words"}"""


def judge(prompt, ans_a, ans_b):
    msg = (f"{RUBRIC}\n\n=== PROMPT ===\n{prompt}\n\n"
           f"=== ANSWER A ===\n{ans_a}\n\n=== ANSWER B ===\n{ans_b}\n\n"
           f"Return the JSON verdict now.")
    out = subprocess.run(["claude", "-p", msg], capture_output=True, text=True, timeout=300)
    txt = out.stdout.strip()
    # extract JSON
    i, j = txt.find("{"), txt.rfind("}")
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return {"winner": "PARSE_ERR", "reason": txt[:80]}


PROMPTS = {
    "reason": "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning.",
    "code": "Write a Python function that returns the nth Fibonacci number iteratively, then explain its time complexity.",
    "summarize": "Summarize the tradeoffs between B-trees and LSM-trees for a write-heavy workload, then recommend one for a time-series database and justify.",
    "factual": "Explain the CAP theorem and give one concrete example of a CP system and one AP system.",
    "logic": "Five houses in a row, each a different color. The green house is immediately left of the white house. Which house cannot be the white house? Explain.",
}


def main():
    data = json.loads(METRICS.read_text())
    orig = {r["prompt_id"]: r["answer"] for r in data["orig"]}
    mbolt = {r["prompt_id"]: r["answer"] for r in data["mbolt"]}

    results = []
    tally = {"orig": 0, "mbolt": 0, "tie": 0}
    for idx, (pid, prompt) in enumerate(PROMPTS.items()):
        # order-randomize: even idx -> A=orig, odd idx -> A=mbolt
        swap = idx % 2 == 1
        a_is = "mbolt" if swap else "orig"
        b_is = "orig" if swap else "mbolt"
        ans_a = (mbolt if swap else orig)[pid]
        ans_b = (orig if swap else mbolt)[pid]
        v = judge(prompt, ans_a, ans_b)
        w = v.get("winner")
        winner_model = "tie"
        if w == "A":
            winner_model = a_is
        elif w == "B":
            winner_model = b_is
        if winner_model in tally:
            tally[winner_model] += 1
        elif winner_model == "tie":
            tally["tie"] += 1
        results.append({"prompt_id": pid, "A": a_is, "B": b_is,
                        "verdict": v, "winner_model": winner_model,
                        "identical": ans_a.strip() == ans_b.strip()})
        print(f"{pid:<10} A={a_is:<5} B={b_is:<5} -> {w:<10} ({winner_model}) "
              f"identical={results[-1]['identical']}  {v.get('reason','')}", flush=True)

    OUT.write_text(json.dumps({"tally": tally, "results": results}, indent=2))
    print(f"\ntally: {tally}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
