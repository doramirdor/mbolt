"""Meaningful-scale with/without: 500 chat prompts through each model via
llama-server (model loaded ONCE per model, not per prompt). Deterministic
(temp 0, fixed seed). Captures per-prompt gen tok/s + output text.

Outputs -> results/goal/gen500_{orig,mbolt}.jsonl (resumable) and a combined
metrics summary. Accuracy judged separately on the DIVERGED subset only
(identical outputs are ties by construction), keeping claude -p calls bounded.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-server"
ORIG = "/Users/dor/Documents/code/GPUopt/models/Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
MBOLT = "/Users/dor/Documents/code/GPUopt/models/Qwen3-30B-A3B-UD-IQ3_XXS.mbolt.gguf"
PROMPTS = "/Users/dor/Documents/code/GPUopt/traces/prompts500.jsonl"
OUTDIR = Path("/Users/dor/Documents/code/GPUopt/results/goal")
PORT = 8099
N_PREDICT = 200


def wait_health(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server died during load")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.loads(r.read())["status"] == "ok":
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server health timeout")


def start_server(model):
    proc = subprocess.Popen(
        [BIN, "-m", model, "--port", str(PORT), "-c", "4096", "-ngl", "99",
         "--no-warmup", "-fa", "on"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_health(proc)
    return proc


def complete(prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt + " /no_think"}],
        "temperature": 0, "seed": 0, "n_predict": N_PREDICT, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    txt = d["choices"][0]["message"]["content"]
    tim = d.get("timings", {})
    return txt, tim.get("predicted_per_second", -1), tim.get("predicted_n", -1), \
        tim.get("prompt_per_second", -1)


def run_model(name, model, prompts):
    out = OUTDIR / f"gen500_{name}.jsonl"
    done = 0
    if out.exists():
        done = sum(1 for _ in out.open())
    if done >= len(prompts):
        print(f"[{name}] already complete ({done})", flush=True)
        return
    print(f"[{name}] starting server (resume from {done})", flush=True)
    proc = start_server(model)
    try:
        with out.open("a") as f:
            for i in range(done, len(prompts)):
                p = prompts[i]["prompt"]
                t0 = time.time()
                txt, tps, ntok, pps = complete(p)
                rec = {"i": i, "gen_tps": tps, "n_tok": ntok, "prompt_tps": pps,
                       "wall": round(time.time() - t0, 3), "out": txt}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if i % 25 == 0:
                    print(f"[{name}] {i}/{len(prompts)} {tps:.1f} tok/s", flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    print(f"[{name}] done", flush=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l) for l in open(PROMPTS)]
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("orig", "both"):
        run_model("orig", ORIG, prompts)
    if which in ("mbolt", "both"):
        run_model("mbolt", MBOLT, prompts)

    # summary
    import statistics as st
    print("\n=== 500-prompt summary ===")
    data = {}
    for name in ("orig", "mbolt"):
        f = OUTDIR / f"gen500_{name}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.open()]
        data[name] = recs
        tps = [r["gen_tps"] for r in recs if r["gen_tps"] > 0]
        print(f"{name}: n={len(recs)} gen tok/s median {st.median(tps):.1f} "
              f"mean {st.mean(tps):.1f} p10 {sorted(tps)[len(tps)//10]:.1f}")
    if "orig" in data and "mbolt" in data:
        n = min(len(data["orig"]), len(data["mbolt"]))
        div = sum(1 for k in range(n)
                  if data["orig"][k]["out"].strip() != data["mbolt"][k]["out"].strip())
        print(f"divergence: {div}/{n} = {100*div/n:.1f}% outputs differ (byte-level)")
        idx = [k for k in range(n)
               if data["orig"][k]["out"].strip() != data["mbolt"][k]["out"].strip()]
        (OUTDIR / "diverged_idx.json").write_text(json.dumps(idx))
        print(f"wrote diverged indices ({len(idx)}) for judge")


if __name__ == "__main__":
    main()
