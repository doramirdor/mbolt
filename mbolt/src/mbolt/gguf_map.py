"""GGUF expert-slice offset mapping.

Parses the GGUF tensor-info table and computes the exact byte range of every
expert slice: offset(tensor) + expert_idx * slice_bytes, for each of
ffn_up_exps / ffn_gate_exps / ffn_down_exps per layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from gguf import GGUFReader

EXPERT_KINDS = ("up", "gate", "down")

_EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight$")


@dataclass
class TensorRec:
    name: str
    type_name: str
    shape: tuple[int, ...]  # ne order: shape[0] is the innermost (row) dim
    offset: int             # absolute byte offset of tensor data in file
    nbytes: int


@dataclass
class ExpertTensor:
    layer: int
    kind: str               # up | gate | down
    rec: TensorRec
    n_expert: int
    slice_bytes: int        # bytes of one expert slice (whole rows, quant-block aligned)

    def expert_offset(self, e: int) -> int:
        return self.rec.offset + e * self.slice_bytes


@dataclass
class ModelMap:
    path: str
    file_size: int
    n_layers: int
    n_experts: int
    tensors: list[TensorRec] = field(default_factory=list)
    experts: dict[tuple[int, str], ExpertTensor] = field(default_factory=dict)

    @property
    def total_expert_bytes(self) -> int:
        return sum(t.rec.nbytes for t in self.experts.values())


def load_model_map(path: str) -> ModelMap:
    reader = GGUFReader(path)

    tensors: list[TensorRec] = []
    experts: dict[tuple[int, str], ExpertTensor] = {}
    n_layers = 0
    n_experts = 0

    # GGUFReader.tensors have .data_offset absolute in file (base + relative offset)
    for t in reader.tensors:
        shape = tuple(int(d) for d in t.shape)
        rec = TensorRec(
            name=t.name,
            type_name=t.tensor_type.name,
            shape=shape,
            offset=int(t.data_offset),
            nbytes=int(t.n_bytes),
        )
        tensors.append(rec)

        m = _EXPS_RE.match(t.name)
        if m:
            layer = int(m.group(1))
            kind = m.group(2)
            n_exp = shape[-1]  # ne[2] = expert dim (outermost)
            assert rec.nbytes % n_exp == 0, f"{t.name}: nbytes {rec.nbytes} not divisible by {n_exp}"
            experts[(layer, kind)] = ExpertTensor(
                layer=layer,
                kind=kind,
                rec=rec,
                n_expert=n_exp,
                slice_bytes=rec.nbytes // n_exp,
            )
            n_layers = max(n_layers, layer + 1)
            n_experts = max(n_experts, n_exp)

    import os

    return ModelMap(
        path=path,
        file_size=os.path.getsize(path),
        n_layers=n_layers,
        n_experts=n_experts,
        tensors=tensors,
        experts=experts,
    )


def describe(mm: ModelMap) -> str:
    lines = [
        f"file: {mm.path} ({mm.file_size / 1e9:.2f} GB)",
        f"layers with experts: {mm.n_layers}, experts/layer: {mm.n_experts}",
        f"total expert bytes: {mm.total_expert_bytes / 1e9:.2f} GB "
        f"({100 * mm.total_expert_bytes / mm.file_size:.1f}% of file)",
    ]
    # slice size distribution per kind
    for kind in EXPERT_KINDS:
        sizes = sorted({et.slice_bytes for (l, k), et in mm.experts.items() if k == kind})
        types = sorted({et.rec.type_name for (l, k), et in mm.experts.items() if k == kind})
        lines.append(f"  {kind:>5}: slice sizes {[f'{s/1024:.0f}KiB' for s in sizes]}, types {types}")
    # are expert tensors laid out in layer-execution order on disk?
    exp_offsets = [(et.rec.offset, et.layer, et.kind) for et in mm.experts.values()]
    exp_offsets.sort()
    layer_order = [l for _, l, _ in exp_offsets]
    monotone = all(a <= b for a, b in zip(layer_order, layer_order[1:]))
    lines.append(f"  expert tensors in layer order on disk: {monotone}")
    return "\n".join(lines)
