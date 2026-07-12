"""Correctness proof: original vs mbolt-rewritten model must produce
token-for-token identical greedy output on N diverse prompts.

Starts llama-server for each model in turn, replays the same prompts at
temperature 0, and compares completions exactly. Any divergence = bug.
"""

import json
import subprocess
import sys
import time
import urllib.request

PORT = 8099
BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-server"
PROMPTS = "/Users/dor/Documents/code/GPUopt/traces/prompts.jsonl"
N_PROMPTS = 50
MAX_TOKENS = 64


def wait_health(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(2)
    return False


def greedy(prompt: str) -> list:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        r = json.load(resp)
    return r["choices"][0]["message"].get("reasoning_content", "") or "", r["choices"][0]["message"]["content"]


def run_model(model_path: str) -> list:
    proc = subprocess.Popen(
        [BIN, "-m", model_path, "--port", str(PORT), "--no-warmup", "-np", "1", "-c", "4096"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_health(), f"server failed to start for {model_path}"
        outs = []
        prompts = [json.loads(l) for l in open(PROMPTS)][:N_PROMPTS]
        for i, p in enumerate(prompts):
            outs.append(greedy(p["prompt"]))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(prompts)}", flush=True)
        return outs
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def main():
    orig, opt = sys.argv[1], sys.argv[2]
    print(f"original: {orig}")
    a = run_model(orig)
    print(f"optimized: {opt}")
    b = run_model(opt)

    mismatches = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            mismatches += 1
            print(f"MISMATCH prompt {i}:")
            print(f"  orig: {x!r:.300}")
            print(f"  opt : {y!r:.300}")
    if mismatches:
        print(f"FAIL: {mismatches}/{len(a)} prompts diverged")
        sys.exit(1)
    print(f"PASS: {len(a)}/{len(a)} prompts token-identical (greedy, {MAX_TOKENS} tokens each)")


if __name__ == "__main__":
    main()
