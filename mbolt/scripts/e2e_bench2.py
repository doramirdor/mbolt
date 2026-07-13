"""End-to-end streaming benchmark v2: mlock'd RAM squeeze.

v1 failed to engage the disk-bound regime: macOS swapped the holder's
anonymous pages instead of evicting the model's page cache. v2 mlocks the
holder memory (wired pages cannot swap), evicts residual page cache before
every timed run, and decodes longer for a stable streaming-phase estimate.
"""

import ctypes
import ctypes.util
import re
import subprocess
import sys
import time

import numpy as np

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
PROMPT = ("Summarize the tradeoffs between B-trees and LSM-trees for a write-heavy "
          "workload, then recommend one for a time-series database and justify.")
N_TOKENS = 192
N_RUNS = 5

HOLDER_CODE = r"""
import ctypes, ctypes.util, sys, time
import numpy as np
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
gb = int(sys.argv[1])
hunks, locked = [], 0
rng = np.random.default_rng(1)
for i in range(gb):
    a = rng.integers(0, 2**63, size=(1 << 27,), dtype=np.int64)  # 1 GiB incompressible
    hunks.append(a)
    if libc.mlock(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes)) == 0:
        locked += 1
print(f"held {gb} GB, mlocked {locked} GB", flush=True)
while True:
    time.sleep(5)
    for a in hunks[:: max(1, len(hunks) // 8)]:
        a[::2048] += 1  # keep unlocked hunks warm
"""


def file_backed_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "File-backed pages" in line:
            return int(line.split()[-1].rstrip(".")) * 16384 / 1e9
    return -1.0


def evict_page_cache():
    hunks = []
    try:
        for i in range(24):
            hunks.append(np.ones(1 << 27, np.float64))
            if i % 6 == 5 and file_backed_gb() < 2.5:
                break
    except MemoryError:
        pass
    finally:
        del hunks


def run_once(model: str) -> float:
    cmd = [BIN, "-m", model, "-p", PROMPT, "-n", str(N_TOKENS), "-st", "--temp", "0",
           "-ot", r".ffn_.*_exps.=CPU", "-c", "2048", "--no-warmup"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    text = out.stdout + out.stderr
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    if not m:
        print(text[-2000:])
        raise RuntimeError("no generation timing found")
    return float(m.group(1))


def main():
    orig, opt = sys.argv[1], sys.argv[2]
    hold_gb = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER_CODE, str(hold_gb)],
        stdout=subprocess.PIPE, text=True)
    print(f"holder: {holder.stdout.readline().strip()}", flush=True)
    time.sleep(5)

    results = {orig: [], opt: []}
    try:
        for run in range(N_RUNS):
            order = [orig, opt] if run % 2 == 0 else [opt, orig]
            for model in order:
                evict_page_cache()
                print(f"  (file-backed before run: {file_backed_gb():.1f} GB)", flush=True)
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
