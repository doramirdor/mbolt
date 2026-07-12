"""Replay engine: physically replays the byte-range read sequence a streaming
engine would issue for a routing trace, against the real GGUF file, under a
configurable per-layer LRU cache model.

Discipline:
  - file opened with F_NOCACHE (macOS) / O_DIRECT-equivalent so reads bypass
    the page cache
  - reads aligned to 16 KiB (Apple Silicon page size)
  - per layer, the engine knows all missing slices up front: ranges are
    sorted and adjacent ranges merged (gap <= merge_gap) before issuing
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .layouts import Layout
from .gguf_map import EXPERT_KINDS

ALIGN = 16384


@dataclass
class ReplayResult:
    layout: str
    cache_slots: int
    n_tokens: int
    io_ms_per_token: list[float] = field(default_factory=list)
    reads: int = 0
    bytes_read: int = 0
    hit_rate: float = 0.0

    def stats(self) -> dict:
        ms = np.array(self.io_ms_per_token)
        return {
            "layout": self.layout,
            "cache_slots": self.cache_slots,
            "n_tokens": self.n_tokens,
            "io_ms_median": float(np.median(ms)),
            "io_ms_mean": float(ms.mean()),
            "io_ms_p10": float(np.percentile(ms, 10)),
            "io_ms_p90": float(np.percentile(ms, 90)),
            "implied_tok_s": float(1000.0 / np.median(ms)) if np.median(ms) > 0 else 0.0,
            "reads_per_token": self.reads / self.n_tokens,
            "mb_per_token": self.bytes_read / self.n_tokens / 1e6,
            "achieved_mb_s": (self.bytes_read / 1e6) / (ms.sum() / 1000.0) if ms.sum() > 0 else 0.0,
            "hit_rate": self.hit_rate,
        }


class LRU:
    """Per-layer LRU of expert ids. capacity 0 = cache disabled (cold mode)."""

    def __init__(self, n_layers: int, capacity: int):
        self.capacity = capacity
        self.layers = [OrderedDict() for _ in range(n_layers)]
        self.hits = 0
        self.misses = 0

    def access(self, layer: int, expert: int) -> bool:
        """Returns True on hit. On miss, inserts (evicting LRU)."""
        if self.capacity == 0:
            self.misses += 1
            return False
        d = self.layers[layer]
        if expert in d:
            d.move_to_end(expert)
            self.hits += 1
            return True
        self.misses += 1
        d[expert] = None
        if len(d) > self.capacity:
            d.popitem(last=False)
        return False


def _merge_ranges(ranges: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    """ranges: sorted (offset, size). Merge overlapping/near-adjacent."""
    merged = []
    cur_off, cur_end = ranges[0][0], ranges[0][0] + ranges[0][1]
    for off, size in ranges[1:]:
        if off <= cur_end + merge_gap:
            cur_end = max(cur_end, off + size)
        else:
            merged.append((cur_off, cur_end - cur_off))
            cur_off, cur_end = off, off + size
    merged.append((cur_off, cur_end - cur_off))
    return merged


def replay(
    gguf_path: str,
    layout: Layout,
    tokens: np.ndarray,          # [N, n_layers, k] expert ids (decode trace)
    cache_slots: int,
    warmup_tokens: int = 64,
    measure_tokens: int = 192,
    merge_gap: int = 0,
    quiet: bool = False,
) -> ReplayResult:
    n_layers = tokens.shape[1]
    file_size = os.path.getsize(gguf_path)
    assert layout.max_end() <= file_size, (
        f"layout {layout.name} exceeds file: {layout.max_end()} > {file_size}"
    )

    fd = os.open(gguf_path, os.O_RDONLY)
    try:
        if sys.platform == "darwin":
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)

        lru = LRU(n_layers, cache_slots)
        result = ReplayResult(layout=layout.name, cache_slots=cache_slots, n_tokens=0)

        n_avail = len(tokens)
        if cache_slots == 0:
            warmup_tokens = 0
        total = min(n_avail, warmup_tokens + measure_tokens)

        # pre-extract offsets tables for speed
        off_tab = layout.offset
        size_tab = layout.sizes

        for t in range(total):
            measuring = t >= warmup_tokens
            if measuring:
                lru_hits0, lru_miss0 = lru.hits, lru.misses
            tok_ns = 0
            tok_reads = 0
            tok_bytes = 0
            for layer in range(n_layers):
                ids = tokens[t, layer]
                misses = [int(e) for e in ids if not lru.access(layer, int(e))]
                if not misses:
                    continue
                ranges = []
                for kind in EXPERT_KINDS:
                    offs = off_tab[(layer, kind)]
                    sz = size_tab[(layer, kind)]
                    for e in misses:
                        ranges.append((int(offs[e]), sz))
                ranges.sort()
                merged = _merge_ranges(ranges, merge_gap)
                t0 = time.perf_counter_ns()
                for off, size in merged:
                    a_off = off & ~(ALIGN - 1)
                    a_len = ((off + size + ALIGN - 1) & ~(ALIGN - 1)) - a_off
                    data = os.pread(fd, a_len, a_off)
                    expected = min(a_len, file_size - a_off)
                    if len(data) != expected:
                        raise IOError(f"short read at {a_off}")
                    tok_bytes += a_len
                tok_ns += time.perf_counter_ns() - t0
                tok_reads += len(merged)
            if measuring:
                result.io_ms_per_token.append(tok_ns / 1e6)
                result.reads += tok_reads
                result.bytes_read += tok_bytes
                result.n_tokens += 1
        if lru.hits + lru.misses > 0:
            result.hit_rate = lru.hits / (lru.hits + lru.misses)
        return result
    finally:
        os.close(fd)
