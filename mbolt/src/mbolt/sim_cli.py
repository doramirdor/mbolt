"""mbolt-sim: Phase 0 I/O simulator CLI.

Subcommands:
  map         model.gguf                      - expert slice map summary
  trace-stats trace.bin                       - trace + routing entropy stats
  cluster     trace.bin -o perms.json         - co-activation clustering
  drive       model.gguf                      - raw device read characteristics
  gate        model.gguf trace.bin perms.json - the Phase 0 gate benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

from .cluster import cluster_all
from .gguf_map import describe, load_model_map
from .layouts import ALL_LAYOUTS, build_layout
from .replay import replay
from .trace import read_trace


def cmd_map(args):
    print(describe(load_model_map(args.model)))


def cmd_trace_stats(args):
    t = read_trace(args.trace)
    print(t.summary())
    from .cluster import coactivation_stats

    stats = coactivation_stats(t.decode, t.n_expert)
    ent = [s.heat_entropy_bits for s in stats]
    print(f"heat entropy bits (max {np.log2(t.n_expert):.2f}): "
          f"min {min(ent):.2f} / median {np.median(ent):.2f} / max {max(ent):.2f}")
    mid = stats[len(stats) // 2]
    print(f"mid-layer lift stats: {mid.lift_stats()}")


def cmd_cluster(args):
    t = read_trace(args.trace)
    if len(t.decode) < args.min_tokens:
        sys.exit(f"only {len(t.decode)} decode tokens; need {args.min_tokens}")
    n_train = int(len(t.decode) * args.train_frac)
    res = cluster_all(t.decode[:n_train], t.n_expert)
    res["train_frac"] = args.train_frac
    res["train_tokens"] = n_train
    with open(args.output, "w") as f:
        json.dump(res, f)
    ents = [l["heat_entropy_bits"] for l in res["layers"]]
    lifts = [l["lift"].get("frac_gt_1.5", 0) for l in res["layers"]]
    ncl = [l["n_clusters"] for l in res["layers"]]
    print(f"clustered {res['n_layers']} layers over {res['n_tokens']} decode tokens")
    print(f"heat entropy bits: median {np.median(ents):.2f} (max possible {np.log2(res['n_expert']):.2f})")
    print(f"frac pairs with lift>1.5: median {np.median(lifts):.3f}")
    print(f"clusters per layer: median {np.median(ncl):.0f}")


def cmd_drive(args):
    """Raw device characteristics via the model file, F_NOCACHE reads."""
    import fcntl

    fd = os.open(args.model, os.O_RDONLY)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    fsize = os.path.getsize(args.model)
    rng = random.Random(7)

    # sequential: 4 GB in 8 MiB chunks
    chunk = 8 << 20
    n_seq = (4 << 30) // chunk
    t0 = time.perf_counter()
    for i in range(n_seq):
        os.pread(fd, chunk, i * chunk)
    seq_s = time.perf_counter() - t0
    print(f"sequential 8MiB reads: {4 / seq_s * 1024:.0f} MB/s")

    # random 600 KiB (expert-slice-sized) reads
    sz = 600 * 1024
    n_rand = 2000
    offs = [rng.randrange(0, (fsize - sz) // 16384) * 16384 for _ in range(n_rand)]
    t0 = time.perf_counter()
    for off in offs:
        os.pread(fd, sz, off)
    rand_s = time.perf_counter() - t0
    print(f"random 600KiB reads: {n_rand / rand_s:.0f} IOPS, "
          f"{n_rand * sz / rand_s / 1e6:.0f} MB/s")

    # random 4 KiB reads
    sz = 4096
    offs = [rng.randrange(0, (fsize - sz) // 16384) * 16384 for _ in range(n_rand)]
    t0 = time.perf_counter()
    for off in offs:
        os.pread(fd, sz, off)
    rand4_s = time.perf_counter() - t0
    print(f"random 4KiB reads: {n_rand / rand4_s:.0f} IOPS")
    os.close(fd)


def _try_purge():
    try:
        r = subprocess.run(["purge"], capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def _file_backed_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "File-backed pages" in line:
            return int(line.split()[-1].rstrip(".")) * 16384 / 1e9
    return -1.0


def _probe_random_read_mbs(path: str, n: int = 150) -> float:
    """Random expert-slice-sized F_NOCACHE reads; page-cache hits show up as
    an implausibly high rate for any NVMe."""
    import fcntl

    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    fsize = os.path.getsize(path)
    sz = 600 * 1024
    rng = random.Random(123)
    offs = [rng.randrange(0, (fsize - sz) // 16384) * 16384 for _ in range(n)]
    t0 = time.perf_counter()
    for off in offs:
        os.pread(fd, sz, off)
    dt = time.perf_counter() - t0
    os.close(fd)
    return n * sz / dt / 1e6


def _pressure_evict(target_file_backed_gb: float = 3.0, max_gb: int = 52):
    """Force clean file-backed pages out of the unified page cache by
    transient anonymous-memory pressure (purge(8) needs sudo)."""
    print(f"evicting page cache via memory pressure "
          f"(file-backed now {_file_backed_gb():.1f} GB)...", flush=True)
    hunks = []
    try:
        for i in range(max_gb):
            hunks.append(np.ones(1 << 27, np.float64))  # 1 GiB touched
            if i % 8 == 7 and _file_backed_gb() < target_file_backed_gb:
                break
    finally:
        del hunks
    print(f"eviction done (file-backed now {_file_backed_gb():.1f} GB)", flush=True)


def ensure_cold(path: str, ceiling_mbs: float = 6500.0):
    """Verify random reads run at plausible device speed; evict if page cache
    is serving them. Loud failure if eviction doesn't get us there."""
    mbs = _probe_random_read_mbs(path)
    print(f"cold-read probe: {mbs:.0f} MB/s (ceiling {ceiling_mbs:.0f})", flush=True)
    if mbs <= ceiling_mbs:
        return
    if not _try_purge():
        _pressure_evict()
    mbs = _probe_random_read_mbs(path)
    print(f"cold-read probe after evict: {mbs:.0f} MB/s", flush=True)
    if mbs > ceiling_mbs:
        raise RuntimeError(
            f"page cache still serving reads ({mbs:.0f} MB/s > {ceiling_mbs:.0f} MB/s ceiling); "
            "stop processes holding the model file and retry"
        )


def cmd_evict(args):
    ensure_cold(args.model, args.ceiling)


def cmd_prefetch_map(args):
    """Emit the MBOLT_PREFETCH sidecar: expert tensor offsets for the C++
    prefetcher (magic MBPF v1; see llama.cpp common/mbolt-trace.h)."""
    import struct

    from gguf import GGUFReader

    mm = load_model_map(args.model)
    reader = GGUFReader(args.model)
    arch = reader.get_field("general.architecture").contents()
    k = int(reader.get_field(f"{arch}.expert_used_count").contents())
    model_path = os.path.abspath(args.model).encode()

    with open(args.output, "wb") as f:
        f.write(struct.pack("<6I", 0x4650424D, 1, mm.n_layers, mm.n_experts, k, len(model_path)))
        f.write(model_path)
        for layer in range(mm.n_layers):
            for kind in ("up", "gate", "down"):
                et = mm.experts[(layer, kind)]
                f.write(struct.pack("<2Q", et.rec.offset, et.slice_bytes))
    print(f"wrote {args.output}: {mm.n_layers} layers x {mm.n_experts} experts (top-{k})")


def cmd_gate(args):
    mm = load_model_map(args.model)
    t = read_trace(args.trace)
    perms_doc = json.load(open(args.perms))
    clique_perms = [l["clique_perm"] for l in perms_doc["layers"]]
    chain_perms = [l["chain_perm"] for l in perms_doc["layers"]]
    heat_perms = [l["heat_perm"] for l in perms_doc["layers"]]

    perm_for = {
        "clique": clique_perms,
        "clique+pipeline": clique_perms,
        "chain": chain_perms,
        "chain+pipeline": chain_perms,
        "interleave": chain_perms,
        "heat": heat_perms,
    }
    wanted = args.layouts.split(",") if args.layouts else list(ALL_LAYOUTS)
    unknown = set(wanted) - set(ALL_LAYOUTS)
    assert not unknown, f"unknown layouts: {unknown}"
    if "baseline" not in wanted:
        wanted.insert(0, "baseline")
    layouts = {}
    for name in wanted:
        layouts[name] = build_layout(name, mm, perm_for.get(name))

    # replay only the held-out tail (tokens the clustering never saw)
    eval_start = perms_doc.get("train_tokens", 0)
    decode = t.decode[eval_start:]
    win = args.warmup + args.tokens
    n_windows = len(decode) // win
    assert n_windows >= args.runs, (
        f"held-out trace too short: {len(decode)} tokens (after skipping "
        f"{eval_start} train tokens) for {args.runs} windows of {win}"
    )
    print(f"evaluating on held-out tokens [{eval_start}:{eval_start + len(decode)}]")

    # establish genuinely-cold device speed, then verify per run
    if not _try_purge():
        _pressure_evict()
    device_mbs = _probe_random_read_mbs(args.model)
    ceiling = device_mbs * 1.4
    print(f"device random-read speed (cold, 600KiB): {device_mbs:.0f} MB/s; "
          f"re-evict ceiling {ceiling:.0f} MB/s")
    print(f"decode tokens {len(decode)}, {args.runs} runs, window={win} "
          f"({args.warmup} warmup + {args.tokens} measured), cache={args.cache} slots/layer")

    results = []
    order = list(layouts)
    for run in range(args.runs):
        window = decode[run * win : (run + 1) * win]
        random.Random(run).shuffle(order)
        ensure_cold(args.model, ceiling)
        for name in order:
            r = replay(
                args.model, layouts[name], window,
                cache_slots=args.cache, warmup_tokens=args.warmup,
                measure_tokens=args.tokens, merge_gap=args.merge_gap,
            )
            s = r.stats()
            s["run"] = run
            results.append(s)
            print(f"  run {run} {name:>16}: {s['io_ms_median']:7.2f} ms/tok median, "
                  f"{s['implied_tok_s']:6.2f} tok/s, {s['reads_per_token']:6.1f} reads/tok, "
                  f"{s['mb_per_token']:6.1f} MB/tok, hit {s['hit_rate']:.2f}", flush=True)

    doc = {
        "model": args.model,
        "cache_slots": args.cache,
        "warmup": args.warmup,
        "tokens": args.tokens,
        "merge_gap": args.merge_gap,
        "runs": args.runs,
        "device_random_mbs": device_mbs,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=1)

    # per-run speedup vs baseline, then median across runs
    print("\n=== speedup vs baseline (same-window ratios, median over runs) ===")
    base = {r["run"]: r["io_ms_median"] for r in results if r["layout"] == "baseline"}
    for name in layouts:
        ratios = [
            base[r["run"]] / r["io_ms_median"]
            for r in results
            if r["layout"] == name
        ]
        print(f"  {name:>16}: {np.median(ratios):.3f}x  (runs: {[f'{x:.3f}' for x in sorted(ratios)]})")


def main():
    ap = argparse.ArgumentParser(prog="mbolt-sim")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("map")
    p.add_argument("model")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("trace-stats")
    p.add_argument("trace")
    p.set_defaults(func=cmd_trace_stats)

    p = sub.add_parser("cluster")
    p.add_argument("trace")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--min-tokens", type=int, default=2000)
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="cluster on the first fraction of decode tokens; "
                        "gate replays the held-out tail")
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("drive")
    p.add_argument("model")
    p.set_defaults(func=cmd_drive)

    p = sub.add_parser("evict")
    p.add_argument("model")
    p.add_argument("--ceiling", type=float, default=6500.0)
    p.set_defaults(func=cmd_evict)

    p = sub.add_parser("prefetch-map")
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_prefetch_map)

    p = sub.add_parser("gate")
    p.add_argument("model")
    p.add_argument("trace")
    p.add_argument("perms")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--cache", type=int, default=32)
    p.add_argument("--warmup", type=int, default=64)
    p.add_argument("--tokens", type=int, default=192)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--merge-gap", type=int, default=0)
    p.add_argument("--layouts", default=None, help="comma-separated subset (baseline always included)")
    p.set_defaults(func=cmd_gate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
