"""Cold-start decode benchmark: first tokens after loading, page cache evicted.

This is where the layout must show end-to-end: every expert read misses, and
the misses are dominated by co-activated hot experts - exactly the slices the
mbolt layout packs adjacently, so the prefetcher's merged ranges collapse
into large sequential reads. Also a real UX metric: responsiveness right
after loading a big model.

No holder needed: cache is evicted before each run; 32 decode tokens finish
before the cache re-warms enough to matter.
"""

import os
import re
import subprocess
import sys

import numpy as np

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
PROMPT = "Explain the CAP theorem and give a concrete example of a CP and an AP system."
N_TOKENS = 32
N_REPS = 3


def vmstat_fb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("File-backed pages"):
            return int(line.split()[-1].rstrip(".")) * 16384 / 1e9
    return -1


def evict():
    hunks = []
    try:
        for i in range(56):
            hunks.append(np.ones(1 << 27, np.float64))
            if i % 6 == 5 and vmstat_fb() < 2.0:
                break
    except MemoryError:
        pass
    del hunks


def run_once(model: str, sidecar: str | None):
    env = dict(os.environ)
    env.pop("MBOLT_PREFETCH", None)
    if sidecar:
        env["MBOLT_PREFETCH"] = sidecar
    cmd = [BIN, "-m", model, "-p", PROMPT, "-n", str(N_TOKENS), "-st", "--temp", "0",
           "-ot", r".ffn_.*_exps.=CPU", "-c", "1024", "--no-warmup", "-ngl", "0", "-v"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, env=env)
    text = out.stdout + out.stderr
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    ts = float(m.group(1)) if m else -1
    pf = re.search(r"mbolt-prefetch:\s*(\d+) reads, ([0-9.]+) GB read", text)
    pf_stats = f"pf: {pf.group(1)} reads / {pf.group(2)} GB" if pf else "pf: off"
    return ts, pf_stats


def main():
    orig, opt, orig_pf, opt_pf = sys.argv[1:5]
    configs = [
        ("orig      ", orig, None),
        ("orig +pf  ", orig, orig_pf),
        ("mbolt     ", opt, None),
        ("mbolt+pf  ", opt, opt_pf),
    ]
    results = {name: [] for name, _, _ in configs}
    for rep in range(N_REPS):
        order = configs if rep % 2 == 0 else configs[::-1]
        for name, model, sidecar in order:
            evict()
            ts, pf_stats = run_once(model, sidecar)
            results[name].append(ts)
            print(f"rep {rep} {name}: {ts:6.2f} tok/s (first {N_TOKENS} tokens, cold) | {pf_stats}", flush=True)

    print()
    med = {}
    for name, vals in results.items():
        med[name] = np.median(vals)
        print(f"{name}: median {med[name]:.2f} tok/s, runs {sorted(round(v,2) for v in vals)}")
    print(f"\ncold-start layout win, explicit reads (mbolt+pf vs orig+pf): {med['mbolt+pf  ']/med['orig +pf  ']:.3f}x")
    print(f"cold-start full stack (mbolt+pf vs orig no-pf): {med['mbolt+pf  ']/med['orig      ']:.3f}x")
    print(f"cold-start prefetch effect on orig: {med['orig +pf  ']/med['orig      ']:.3f}x")
    print(f"cold-start prefetch effect on mbolt: {med['mbolt+pf  ']/med['mbolt     ']:.3f}x")


if __name__ == "__main__":
    main()
