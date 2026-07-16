"""Comprehensive with/without benchmark for the goal:
compare mbolt (with) vs baseline (without) on
  1. latency (TTFT + total wall)   2. tokens/sec
  3. memory (peak RSS + GPU mem)   4. GPU utilization %
  5. CPU utilization (avg cores)   6. accuracy (outputs saved for LLM judge)

Metal (GPU) mode, warm, temp 0, deterministic. One process per (model, prompt).
GPU util sampled via ioreg (no sudo). Peak RSS + CPU seconds via /usr/bin/time -l.
"""

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
ORIG = "/Users/dor/Documents/code/GPUopt/models/Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
MBOLT = "/Users/dor/Documents/code/GPUopt/models/Qwen3-30B-A3B-UD-IQ3_XXS.mbolt.gguf"
OUTDIR = Path("/Users/dor/Documents/code/GPUopt/results/goal")
N_TOKENS = 256
CTX = 2048

PROMPTS = [
    ("reason", "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning."),
    ("code", "Write a Python function that returns the nth Fibonacci number iteratively, then explain its time complexity."),
    ("summarize", "Summarize the tradeoffs between B-trees and LSM-trees for a write-heavy workload, then recommend one for a time-series database and justify."),
    ("factual", "Explain the CAP theorem and give one concrete example of a CP system and one AP system."),
    ("logic", "Five houses in a row, each a different color. The green house is immediately left of the white house. Which house cannot be the white house? Explain."),
]


def gpu_stats():
    """Return (device_util_pct, in_use_mem_bytes) from ioreg, no sudo."""
    out = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
                         capture_output=True, text=True).stdout
    util = re.search(r'"Device Utilization %"=(\d+)', out)
    mem = re.search(r'"In use system memory"=(\d+)', out)
    return (int(util.group(1)) if util else -1,
            int(mem.group(1)) if mem else -1)


class GpuSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = False
        self.utils = []
        self.mems = []

    def run(self):
        while not self.stop_flag:
            u, m = gpu_stats()
            if u >= 0:
                self.utils.append(u)
            if m >= 0:
                self.mems.append(m)
            time.sleep(self.interval)

    def summary(self):
        us = [u for u in self.utils if u > 0]
        return {
            "gpu_util_max": max(self.utils) if self.utils else -1,
            "gpu_util_mean_active": round(sum(us) / len(us), 1) if us else 0,
            "gpu_mem_peak_gb": round(max(self.mems) / 1e9, 2) if self.mems else -1,
            "samples": len(self.utils),
        }


def parse_time_l(stderr_txt):
    """Parse /usr/bin/time -l block: peak RSS bytes, user+sys CPU seconds."""
    rss = re.search(r'(\d+)\s+maximum resident set size', stderr_txt)
    user = re.search(r'([0-9.]+)\s+user', stderr_txt)
    sysc = re.search(r'([0-9.]+)\s+sys', stderr_txt)
    real = re.search(r'([0-9.]+)\s+real', stderr_txt)
    return {
        "peak_rss_gb": round(int(rss.group(1)) / 1e9, 2) if rss else -1,
        "cpu_user_s": float(user.group(1)) if user else -1,
        "cpu_sys_s": float(sysc.group(1)) if sysc else -1,
        "wall_s": float(real.group(1)) if real else -1,
    }


def run_once(model, prompt, ngl):
    sampler = GpuSampler()
    cmd = ["/usr/bin/time", "-l", BIN, "-m", model, "-p", prompt,
           "-n", str(N_TOKENS), "-st", "--temp", "0", "-c", str(CTX),
           "--no-warmup", "-ngl", str(ngl)]
    sampler.start()
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    wall = time.time() - t0
    sampler.stop_flag = True
    sampler.join(timeout=2)

    text = out.stdout + out.stderr
    prompt_ts = re.search(r"Prompt:\s*([0-9.]+)\s*t/s", text)
    gen_ts = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    # generated answer = everything after the echoed prompt, before the timing line
    ans = out.stdout
    if prompt in ans:
        ans = ans.split(prompt, 1)[1]
    ans = re.split(r"\n\[ Prompt:", ans)[0].strip()

    rec = {
        "prompt_ts": float(prompt_ts.group(1)) if prompt_ts else -1,
        "gen_ts": float(gen_ts.group(1)) if gen_ts else -1,
        "wall_s": round(wall, 2),
        "answer": ans,
    }
    rec.update(parse_time_l(text))
    rec.update(sampler.summary())
    # TTFT estimate: prompt eval time = prompt_tokens / prompt_ts (proxy; wall dominated by gen)
    return rec


def main():
    ngl = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    OUTDIR.mkdir(parents=True, exist_ok=True)
    configs = [("orig", ORIG), ("mbolt", MBOLT)]
    results = {name: [] for name, _ in configs}

    for pid, prompt in PROMPTS:
        for name, model in configs:
            print(f"[{name}] {pid} ...", flush=True)
            rec = run_once(model, prompt, ngl)
            rec["prompt_id"] = pid
            results[name].append(rec)
            print(f"  gen {rec['gen_ts']:.1f} t/s | prompt {rec['prompt_ts']:.0f} t/s "
                  f"| wall {rec['wall_s']:.1f}s | RSS {rec['peak_rss_gb']}GB "
                  f"| GPU {rec['gpu_util_mean_active']}%/{rec['gpu_util_max']}%max "
                  f"| GPUmem {rec['gpu_mem_peak_gb']}GB "
                  f"| CPU {rec['cpu_user_s']+rec['cpu_sys_s']:.1f}s", flush=True)

    (OUTDIR / f"metrics_ngl{ngl}.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUTDIR}/metrics_ngl{ngl}.json")

    # aggregate
    import statistics as st
    print("\n=== AGGREGATE (median across prompts) ===")
    hdr = f"{'metric':<22}{'orig':>12}{'mbolt':>12}{'delta':>10}"
    print(hdr)
    def med(name, key):
        return st.median([r[key] for r in results[name] if r[key] >= 0])
    for key, label in [("gen_ts", "gen tok/s"), ("prompt_ts", "prompt tok/s"),
                       ("wall_s", "wall s"), ("peak_rss_gb", "peak RSS GB"),
                       ("gpu_util_mean_active", "GPU util %"),
                       ("gpu_mem_peak_gb", "GPU mem GB")]:
        o, m = med("orig", key), med("mbolt", key)
        d = f"{(m/o-1)*100:+.1f}%" if o else "n/a"
        print(f"{label:<22}{o:>12.2f}{m:>12.2f}{d:>10}")
    for name in ("orig", "mbolt"):
        cpu = st.median([r["cpu_user_s"] + r["cpu_sys_s"] for r in results[name]])
        wall = st.median([r["wall_s"] for r in results[name]])
        print(f"{name} avg CPU cores busy: {cpu/wall:.2f}")


if __name__ == "__main__":
    main()
