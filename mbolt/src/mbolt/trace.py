"""Parser for MBOLT_TRACE binary routing logs.

Format (little-endian int32 stream):
  header: magic "MBLT" (0x544C424D), version=1, n_expert
  record: layer, k, n_tokens, ids[n_tokens * k] (token-major)

Records stream in graph-execution order: layer 0..L-1 per forward pass.
Decode passes have n_tokens == 1 for every layer. In prefill passes the final
layer routes only the last token (llama.cpp drops unused outputs), so the
last-layer record has n_tokens == 1 while earlier layers have the full batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAGIC = 0x544C424D


@dataclass
class Trace:
    n_expert: int
    n_layers: int
    k: int
    # complete decode passes: [n_tokens, n_layers, k] expert ids
    decode: np.ndarray
    # prefill routing, per layer (ragged across layers because the final
    # layer only routes the last token of each batch): list of [N_l, k]
    prefill_per_layer: list[np.ndarray]

    def summary(self) -> str:
        pf = self.prefill_per_layer[0].shape[0] if self.prefill_per_layer else 0
        return (
            f"n_expert={self.n_expert} n_layers={self.n_layers} k={self.k} "
            f"decode_tokens={len(self.decode)} prefill_tokens={pf}"
        )


def read_trace(path: str) -> Trace:
    raw = np.fromfile(path, dtype=np.int32)
    assert raw[0] == MAGIC, f"bad magic {raw[0]:#x}"
    assert raw[1] == 1, f"unsupported version {raw[1]}"
    n_expert = int(raw[2])

    pos = 3
    n = len(raw)

    # group records into forward passes on layer-wrap
    passes: list[list[tuple[int, np.ndarray]]] = []
    cur: list[tuple[int, np.ndarray]] = []
    prev_layer = -1
    k_global = -1

    while pos + 3 <= n:
        layer, k, n_tokens = int(raw[pos]), int(raw[pos + 1]), int(raw[pos + 2])
        pos += 3
        count = n_tokens * k
        if pos + count > n:
            break  # truncated tail record (process killed mid-write)
        ids = raw[pos : pos + count].reshape(n_tokens, k)
        pos += count
        if k_global == -1:
            k_global = k
        if layer <= prev_layer and cur:
            passes.append(cur)
            cur = []
        cur.append((layer, ids))
        prev_layer = layer
    if cur:
        passes.append(cur)

    n_layers = max(len(p) for p in passes)

    decode_rows = []
    prefill_layers: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    for p in passes:
        if len(p) != n_layers:
            continue  # incomplete pass
        max_nt = max(ids.shape[0] for _, ids in p)
        if max_nt == 1:
            decode_rows.append(np.stack([ids[0] for _, ids in p])[None])  # [1, L, k]
        else:
            for layer, ids in p:
                prefill_layers[layer].append(ids)

    decode = (
        np.concatenate(decode_rows)
        if decode_rows
        else np.zeros((0, n_layers, k_global), np.int32)
    )
    prefill_per_layer = [
        np.concatenate(l) if l else np.zeros((0, k_global), np.int32) for l in prefill_layers
    ]

    return Trace(
        n_expert=n_expert,
        n_layers=n_layers,
        k=k_global,
        decode=decode,
        prefill_per_layer=prefill_per_layer,
    )
