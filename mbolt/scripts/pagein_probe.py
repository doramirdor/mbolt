"""Measure disk pageins during a fixed decode on each model under RAM squeeze.

If the packed (mbolt) layout triggers kernel readahead over-fetch, its
pageins-per-token will exceed the original's even though the logical expert
bytes are identical. Also captures output text for a sanity eyeball.
"""

import re
import subprocess
import sys
import time

import numpy as np

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
PROMPT = "Explain the CAP theorem and give a concrete example of a CP and an AP system."
N_TOKENS = 128

HOLDER_CODE = r"""
import ctypes, ctypes.util, sys, time
import numpy as np
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
gb = int(sys.argv[1])
hunks, locked = [], 0
rng = np.random.default_rng(1)
for i in range(gb):
    a = rng.integers(0, 2**63, size=(1 << 27,), dtype=np.int64)
    hunks.append(a)
    if libc.mlock(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes)) == 0:
        locked += 1
print(f"held {gb} GB, mlocked {locked} GB", flush=True)
while True:
    time.sleep(10)
"""


def vmstat(field: str) -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith(field):
            return int(line.split()[-1].rstrip("."))
    return -1


def evict():
    hunks = []
    try:
        for i in range(24):
            hunks.append(np.ones(1 << 27, np.float64))
            if i % 6 == 5 and vmstat("File-backed pages") * 16384 / 1e9 < 2.5:
                break
    except MemoryError:
        pass
    del hunks


def run_once(model: str):
    p0 = vmstat("Pageins")
    t0 = time.time()
    cmd = [BIN, "-m", model, "-p", PROMPT, "-n", str(N_TOKENS), "-st", "--temp", "0",
           "-ot", r".ffn_.*_exps.=CPU", "-c", "1024", "--no-warmup"] + EXTRA
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    dt = time.time() - t0
    p1 = vmstat("Pageins")
    text = out.stdout + out.stderr
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    ts = float(m.group(1)) if m else -1
    gen = out.stdout.strip().replace("\n", " ")[:160]
    return ts, (p1 - p0) * 16384 / 1e9, dt, gen


EXTRA: list = []


def main():
    global EXTRA
    orig, opt = sys.argv[1], sys.argv[2]
    EXTRA = sys.argv[3:]
    holder = subprocess.Popen([sys.executable, "-c", HOLDER_CODE, "24"],
                              stdout=subprocess.PIPE, text=True)
    print(holder.stdout.readline().strip(), flush=True)
    try:
        for rep in range(2):
            for name, model in [("orig", orig), ("mbolt", opt)] if rep % 2 == 0 else [("mbolt", opt), ("orig", orig)]:
                evict()
                ts, gb_in, dt, gen = run_once(model)
                print(f"rep {rep} {name:>6}: {ts:6.2f} tok/s | pageins {gb_in:6.2f} GB "
                      f"(total {dt:.0f}s incl load)", flush=True)
                print(f"    text: {gen}", flush=True)
    finally:
        holder.terminate()


if __name__ == "__main__":
    main()
