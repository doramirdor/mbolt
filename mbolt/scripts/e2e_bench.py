"""End-to-end streaming benchmark: original vs mbolt-optimized GGUF.

Regime: llama.cpp with expert tensors forced to CPU (`-ot ".ffn_.*_exps.=CPU"`,
mmap-backed) while a RAM-squeeze holder keeps ~46 GB of incompressible
anonymous memory hot, so the page cache cannot hold the experts and decode
must stream them from SSD - the regime mbolt targets.

Protocol: greedy decode (both models emit identical tokens, so identical
routing -> comparable I/O), N_RUNS runs per model, alternating model order,
page-cache pressure held constant by the holder. Reports per-run decode t/s
parsed from llama-cli timings.
"""

import re
import subprocess
import sys
import time

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
PROMPT = ("Summarize the tradeoffs between B-trees and LSM-trees for a write-heavy "
          "workload, then recommend one for a time-series database and justify.")
N_TOKENS = 128
N_RUNS = 5

HOLDER_CODE = r"""
import numpy as np, time, sys
gb = int(sys.argv[1])
hunks = []
rng = np.random.default_rng(1)
for i in range(gb):
    a = rng.integers(0, 2**63, size=(1 << 27,), dtype=np.int64)  # 1 GiB incompressible
    hunks.append(a)
print("held", gb, "GB", flush=True)
view_stride = 2048  # touch every 16 KiB page
while True:
    for a in hunks:
        a[::view_stride] += 1
    time.sleep(3)
"""


def run_once(model: str) -> float:
    cmd = [BIN, "-m", model, "-p", PROMPT, "-n", str(N_TOKENS), "-st", "--temp", "0",
           "-ot", r".ffn_.*_exps.=CPU", "-c", "2048", "--no-warmup"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    text = out.stdout + out.stderr
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    if not m:
        print(text[-2000:])
        raise RuntimeError("no generation timing found")
    return float(m.group(1))


def main():
    orig, opt = sys.argv[1], sys.argv[2]
    hold_gb = int(sys.argv[3]) if len(sys.argv) > 3 else 46

    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER_CODE, str(hold_gb)],
        stdout=subprocess.PIPE, text=True)
    line = holder.stdout.readline()
    print(f"holder: {line.strip()}; settling 10s", flush=True)
    time.sleep(10)

    results = {orig: [], opt: []}
    try:
        for run in range(N_RUNS):
            order = [orig, opt] if run % 2 == 0 else [opt, orig]
            for model in order:
                ts = run_once(model)
                results[model].append(ts)
                name = "orig" if model == orig else "mbolt"
                print(f"run {run} {name:>6}: {ts:6.2f} tok/s", flush=True)
    finally:
        holder.terminate()

    import statistics
    for model, vals in results.items():
        name = "orig" if model == orig else "mbolt"
        print(f"{name:>6}: median {statistics.median(vals):.2f} tok/s, "
              f"runs {sorted(round(v, 2) for v in vals)}")
    sp = statistics.median(results[opt]) / statistics.median(results[orig])
    print(f"speedup (median): {sp:.3f}x")


if __name__ == "__main__":
    main()
