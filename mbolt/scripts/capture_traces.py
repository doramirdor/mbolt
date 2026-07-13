"""Drive the traced llama-server through the prompt set, sequentially.

Server must be started separately with MBOLT_TRACE set. Sampling follows the
Qwen3 recommended settings (temp 0.6, top-p 0.95); max_tokens caps runaway
thinking traces.
"""

import json
import os
import sys
import time
import urllib.request

PORT = int(os.environ.get("MBOLT_PORT", "8089"))
PROMPTS = "/Users/dor/Documents/code/GPUopt/traces/prompts.jsonl"
MAX_TOKENS = int(os.environ.get("MBOLT_MAX_TOKENS", "384"))
PROMPT_COUNT = int(os.environ.get("MBOLT_PROMPT_COUNT", "0"))  # 0 = all


def chat(prompt: str) -> dict:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": MAX_TOKENS,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main():
    prompts = [json.loads(l) for l in open(PROMPTS)]
    if PROMPT_COUNT:
        prompts = prompts[:PROMPT_COUNT]
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    total_tokens = 0
    for i, p in enumerate(prompts[start:], start):
        r = chat(p["prompt"])
        usage = r.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)
        elapsed = time.time() - t0
        print(
            f"[{i+1}/{len(prompts)}] {p['source']:<20} "
            f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
            f"total_gen={total_tokens} elapsed={elapsed:.0f}s",
            flush=True,
        )
    print(f"done: {total_tokens} generated tokens in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
