"""Judge the DIVERGED subset of the 500-prompt run. Identical outputs are ties
by construction; only pairs that differ byte-level need a quality verdict.
Order-randomized (swap on odd index), claude -p, STRICT JSON. Caps at MAX_JUDGE
diverged pairs (deterministic first-N) to bound cost; reports the cap.
"""
import json
import subprocess
import sys
from pathlib import Path

G = Path("/Users/dor/Documents/code/GPUopt/results/goal")
PROMPTS = "/Users/dor/Documents/code/GPUopt/traces/prompts500.jsonl"
MAX_JUDGE = int(sys.argv[1]) if len(sys.argv) > 1 else 120

RUBRIC = """You are a strict evaluator. Two AI assistants (A and B) answered the same prompt.
Judge which answer is more correct, complete, and clear. Ignore length unless it hurts quality.
Reply with STRICT JSON only, no prose: {"winner":"A"|"B"|"TIE","reason":"<10 words"}"""


def judge(prompt, a, b):
    msg = f"{RUBRIC}\n\n=== PROMPT ===\n{prompt}\n\n=== ANSWER A ===\n{a}\n\n=== ANSWER B ===\n{b}\n\nReturn the JSON verdict now."
    out = subprocess.run(["claude", "-p", msg], capture_output=True, text=True, timeout=300)
    t = out.stdout.strip()
    i, j = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return {"winner": "PARSE_ERR", "reason": t[:60]}


def main():
    prompts = [json.loads(l)["prompt"] for l in open(PROMPTS)]
    orig = [json.loads(l) for l in (G / "gen500_orig.jsonl").open()]
    mbolt = [json.loads(l) for l in (G / "gen500_mbolt.jsonl").open()]
    idx = json.loads((G / "diverged_idx.json").read_text())
    capped = idx[:MAX_JUDGE]
    print(f"diverged={len(idx)}, judging {len(capped)} (cap {MAX_JUDGE})", flush=True)

    tally = {"orig": 0, "mbolt": 0, "tie": 0, "err": 0}
    results = []
    for rank, k in enumerate(capped):
        swap = rank % 2 == 1
        a_is = "mbolt" if swap else "orig"
        b_is = "orig" if swap else "mbolt"
        a = (mbolt if swap else orig)[k]["out"]
        b = (orig if swap else mbolt)[k]["out"]
        v = judge(prompts[k], a, b)
        w = v.get("winner")
        wm = a_is if w == "A" else b_is if w == "B" else "tie" if w == "TIE" else "err"
        tally[wm] = tally.get(wm, 0) + 1
        results.append({"idx": k, "A": a_is, "winner_model": wm, "verdict": v})
        if rank % 10 == 0:
            print(f"  {rank}/{len(capped)} idx={k} -> {wm}", flush=True)
    (G / "judge500.json").write_text(json.dumps({"tally": tally, "n_diverged": len(idx),
                                                 "n_judged": len(capped), "results": results}, indent=2))
    print(f"\ntally (diverged subset): {tally}")
    print(f"of ALL 500: tie/identical(non-diverged) + diverged-ties are equivalent")


if __name__ == "__main__":
    main()
