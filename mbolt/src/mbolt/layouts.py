"""Layout candidates - pure offset arithmetic, no file rewriting.

A layout assigns every expert slice (layer, kind, expert_id) a virtual byte
offset. The replay engine reads the *real* GGUF at those offsets: bytes are
bytes, so the I/O cost profile matches the rewritten file exactly.

Layouts:
  baseline        - offsets as-is in the file
  clique          - experts permuted per layer by co-activation clustering
  heat            - experts permuted per layer by activation frequency only
  pipeline        - expert tensors packed contiguously in layer-execution
                    order (up|gate|down per layer), dense tensors pushed out
  clique+pipeline - both
  interleave      - per layer, per expert: up|gate|down adjacent, expert order
                    by clique perm. NOT achievable as a valid GGUF in Phase 1
                    (breaks tensor contiguity) - included to quantify the
                    Phase 2 headroom of per-expert tensors.
"""

from __future__ import annotations

import numpy as np

from .gguf_map import EXPERT_KINDS, ModelMap

PHASE1_LAYOUTS = ("baseline", "clique", "chain", "heat", "pipeline", "clique+pipeline", "chain+pipeline")
ALL_LAYOUTS = PHASE1_LAYOUTS + ("interleave",)


class Layout:
    """offset[(layer, kind)] = int64 array [n_expert] of virtual byte offsets;
    sizes[(layer, kind)] = slice size in bytes."""

    def __init__(self, name: str, mm: ModelMap):
        self.name = name
        self.mm = mm
        self.offset: dict[tuple[int, str], np.ndarray] = {}
        self.sizes: dict[tuple[int, str], int] = {}

    def max_end(self) -> int:
        return max(
            int(off.max()) + self.sizes[key]
            for key, off in self.offset.items()
        )


def _identity_perms(mm: ModelMap) -> list[list[int]]:
    return [list(range(mm.n_experts)) for _ in range(mm.n_layers)]


def build_layout(name: str, mm: ModelMap, perms: list[list[int]] | None = None) -> Layout:
    """perms[layer] = list where position p holds the expert id stored at
    slot p of the (virtual) tensor. Required for clique/heat variants."""
    lay = Layout(name, mm)
    E = mm.n_experts

    if name != "baseline" and name != "pipeline":
        assert perms is not None, f"layout {name} needs permutations"
    if perms is None:
        perms = _identity_perms(mm)

    # pos_of[layer][expert_id] = slot index after permutation
    pos_of = []
    for layer in range(mm.n_layers):
        p = np.empty(E, np.int64)
        p[np.asarray(perms[layer], np.int64)] = np.arange(E)
        pos_of.append(p)

    if name in ("baseline", "clique", "chain", "heat"):
        # tensor bases stay where they are in the real file
        for (layer, kind), et in mm.experts.items():
            pos = pos_of[layer] if name != "baseline" else np.arange(E)
            lay.offset[(layer, kind)] = et.rec.offset + pos * et.slice_bytes
            lay.sizes[(layer, kind)] = et.slice_bytes
        return lay

    if name in ("pipeline", "clique+pipeline", "chain+pipeline"):
        # pack expert tensors contiguously in execution order:
        # layer 0: up, gate, down; layer 1: ... - dense tensors displaced to tail
        cursor = 0
        for layer in range(mm.n_layers):
            for kind in EXPERT_KINDS:
                et = mm.experts[(layer, kind)]
                pos = pos_of[layer] if name != "pipeline" else np.arange(E)
                lay.offset[(layer, kind)] = cursor + pos * et.slice_bytes
                lay.sizes[(layer, kind)] = et.slice_bytes
                cursor += et.rec.nbytes
        return lay

    if name == "interleave":
        # per layer: expert p's up|gate|down slices adjacent, experts in perm order
        cursor = 0
        for layer in range(mm.n_layers):
            ets = {kind: mm.experts[(layer, kind)] for kind in EXPERT_KINDS}
            stride = sum(et.slice_bytes for et in ets.values())
            base = cursor
            kind_off = {}
            acc = 0
            for kind in EXPERT_KINDS:
                kind_off[kind] = acc
                acc += ets[kind].slice_bytes
            for kind in EXPERT_KINDS:
                lay.offset[(layer, kind)] = base + pos_of[layer] * stride + kind_off[kind]
                lay.sizes[(layer, kind)] = ets[kind].slice_bytes
            cursor += stride * E
        return lay

    raise ValueError(f"unknown layout {name}")
