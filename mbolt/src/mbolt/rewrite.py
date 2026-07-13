"""mbolt rewriter: emit a bit-exact GGUF with profile-guided expert layout.

Per layer:
  - physically reorder expert slices within each ffn_{up,gate,down}_exps
    tensor by the profile permutation
  - apply the same permutation to the rows of the router projection
    (ffn_gate_inp.weight; expert e's logit row) and any expert-indexed bias
  - abort loudly on any other expert-indexed tensor (unknown semantics)

File layout ("pipeline"): expert tensors packed contiguously in
layer-execution order (up|gate|down per layer), all other tensors after,
general.alignment = 4096 so every expert slice starts on a page boundary
(slice sizes are verified multiples of the alignment).

Adds mbolt.* KV metadata (version, layout, permutations, per-position heat,
tier hints, top cliques) which llama.cpp ignores by spec.
"""

from __future__ import annotations

import json
import re

import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

MBOLT_VERSION = "0.1.0"
ALIGNMENT = 4096

_EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight$")
_ROUTER_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.(weight|bias)$")
# expert-indexed tensors we know how to permute; anything else with an
# n_expert dimension is an error
_EXP_BIAS_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|gate|down)_exps\.bias$")
_EXP_PROBS_B_RE = re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$")


def _flat_u8(t) -> np.ndarray:
    return t.data.reshape(-1).view(np.uint8)


def _permute_slices(t, n_expert: int, perm: np.ndarray) -> np.ndarray:
    """Reorder equal-size expert slices of a tensor's raw bytes; returns a
    new array shaped like t.data."""
    flat = _flat_u8(t)
    assert flat.nbytes == t.n_bytes
    assert flat.nbytes % n_expert == 0, f"{t.name}: {flat.nbytes} % {n_expert} != 0"
    sliced = flat.reshape(n_expert, flat.nbytes // n_expert)
    return sliced[perm].reshape(-1).view(t.data.dtype).reshape(t.data.shape)


def rewrite(
    src_path: str,
    dst_path: str,
    perms: list[list[int]],
    heat: list[list[float]] | None = None,
    top_cliques: list[list[list[int]]] | None = None,
    layout_name: str = "chain+pipeline",
    pack_pipeline: bool = True,
    interleave: bool = False,
) -> dict:
    reader = GGUFReader(src_path)

    arch = reader.get_field("general.architecture").contents()
    n_expert_f = reader.get_field(f"{arch}.expert_count")
    assert n_expert_f is not None, "not a MoE model"
    n_expert = int(n_expert_f.contents())
    n_layers = len(perms)
    perms_np = [np.asarray(p, np.int64) for p in perms]
    for p in perms_np:
        assert sorted(p.tolist()) == list(range(n_expert)), "invalid permutation"

    writer = GGUFWriter(dst_path, arch)
    writer.add_custom_alignment(ALIGNMENT)

    # ---- KVs: copy everything, then add mbolt.* ----
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == "general.alignment":
            continue  # replaced by add_custom_alignment
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    writer.add_key_value("mbolt.version", MBOLT_VERSION, GGUFValueType.STRING)
    writer.add_key_value("mbolt.layout", layout_name, GGUFValueType.STRING)
    writer.add_key_value("mbolt.n_layers", n_layers, GGUFValueType.UINT32)
    writer.add_key_value("mbolt.n_experts", n_expert, GGUFValueType.UINT32)
    # perm[l*E + p] = original expert id stored at position p of layer l
    flat_perm = np.concatenate(perms_np).astype(np.int32)
    writer.add_key_value("mbolt.perm", flat_perm.tolist(), GGUFValueType.ARRAY,
                         sub_type=GGUFValueType.INT32)
    if heat is not None:
        # heat in on-disk position order (heat of the expert stored at p)
        pos_heat = np.concatenate(
            [np.asarray(h, np.float64)[p] for h, p in zip(heat, perms_np)]
        )
        writer.add_key_value("mbolt.heat", pos_heat.astype(np.uint32).tolist(),
                             GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
        # tier hint per position: 0 hot / 1 warm / 2 cold (terciles of layer mass)
        tiers = []
        for h, p in zip(heat, perms_np):
            hp = np.asarray(h, np.float64)[p]
            csum = np.cumsum(hp) / max(1.0, hp.sum())
            tiers.append(np.digitize(csum, [1 / 3, 2 / 3]).astype(np.uint32))
        writer.add_key_value("mbolt.tier_hint", np.concatenate(tiers).tolist(),
                             GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
    if top_cliques is not None:
        writer.add_key_value("mbolt.coactivation_top_cliques", json.dumps(top_cliques),
                             GGUFValueType.STRING)
    if interleave:
        writer.add_key_value("mbolt.interleaved", True, GGUFValueType.BOOL)

    # ---- classify tensors ----
    exps: dict[tuple[int, str], object] = {}
    routers: dict[tuple[int, str], object] = {}
    others: list[object] = []
    for t in reader.tensors:
        m = _EXPS_RE.match(t.name)
        if m:
            exps[(int(m.group(1)), m.group(2))] = t
            continue
        m = _ROUTER_RE.match(t.name)
        if m:
            routers[(int(m.group(1)), m.group(2))] = t
            continue
        if _EXP_BIAS_RE.match(t.name) or _EXP_PROBS_B_RE.match(t.name):
            raise SystemExit(f"expert-indexed tensor {t.name} present but unhandled; refusing")
        # any unclassified expert-ish tensor means unknown semantics - refuse
        # (an n_expert-sized dim alone is not enough: head_dim can coincide)
        if t.name.startswith("blk.") and ("exps" in t.name or "exp_probs" in t.name):
            raise SystemExit(f"unrecognized expert tensor {t.name}; refusing to guess")
        others.append(t)

    assert len(exps) == n_layers * 3, f"expected {n_layers * 3} expert tensors, found {len(exps)}"

    # ---- emit order: interleave, pipeline packing, or original order ----
    plan: list[tuple[object, np.ndarray | None, str]] = []  # (tensor|layer, perm|None, mode)
    if interleave:
        # per layer one blob: expert e's up|gate|down slices adjacent, chain order.
        # loader reconstructs the three tensors as strided views (llama.cpp patch,
        # gated on mbolt.interleaved; stock llama.cpp refuses cleanly on the
        # missing ffn_*_exps names)
        types, ne0s, ne1s = [], [], []
        for layer in range(n_layers):
            for kind in ("up", "gate", "down"):
                t = exps[(layer, kind)]
                types.append(int(t.tensor_type))
                ne0s.append(int(t.shape[0]))
                ne1s.append(int(t.shape[1]))
            plan.append((layer, perms_np[layer], "ilv"))
        writer.add_key_value("mbolt.ilv.type", types, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
        writer.add_key_value("mbolt.ilv.ne0", ne0s, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
        writer.add_key_value("mbolt.ilv.ne1", ne1s, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
        for (layer, part), t in sorted(routers.items()):
            plan.append((t, perms_np[layer], "rows"))
        for t in others:
            plan.append((t, None, "copy"))
    elif pack_pipeline:
        for layer in range(n_layers):
            for kind in ("up", "gate", "down"):
                plan.append((exps[(layer, kind)], perms_np[layer], "slices"))
        for (layer, part), t in sorted(routers.items()):
            plan.append((t, perms_np[layer], "rows"))
        for t in others:
            plan.append((t, None, "copy"))
    else:
        for t in reader.tensors:
            m = _EXPS_RE.match(t.name)
            if m:
                plan.append((t, perms_np[int(m.group(1))], "slices"))
                continue
            m = _ROUTER_RE.match(t.name)
            if m:
                plan.append((t, perms_np[int(m.group(1))], "rows"))
                continue
            plan.append((t, None, "copy"))

    # verify page alignment of expert slices holds under chosen alignment
    for (layer, kind), t in exps.items():
        slice_bytes = t.n_bytes // n_expert
        assert slice_bytes % ALIGNMENT == 0, (
            f"{t.name}: slice {slice_bytes} not a multiple of {ALIGNMENT}; "
            "expert starts would not be page-aligned"
        )

    from gguf import GGMLQuantizationType

    def _ilv_build(layer: int, perm: np.ndarray) -> np.ndarray:
        parts = []
        for kind in ("up", "gate", "down"):
            t = exps[(layer, kind)]
            flat = _flat_u8(t)
            parts.append(flat.reshape(n_expert, flat.nbytes // n_expert)[perm])
        return np.hstack(parts).reshape(-1)  # [E, up+gate+down] -> interleaved bytes

    for item, perm, mode in plan:
        if mode == "ilv":
            layer = item
            nbytes = sum(exps[(layer, k)].n_bytes for k in ("up", "gate", "down"))
            writer.add_tensor_info(f"blk.{layer}.ffn_ilv_exps.weight", (nbytes,),
                                   np.dtype(np.int8), nbytes, GGMLQuantizationType.I8)
        else:
            writer.add_tensor_info(item.name, item.data.shape, item.data.dtype,
                                   item.data.nbytes, item.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    stats = {"tensors": len(plan), "permuted_exps": 0, "permuted_router": 0, "ilv_blobs": 0}
    for item, perm, mode in plan:
        if mode == "ilv":
            data = _ilv_build(item, perm).view(np.int8)
            stats["ilv_blobs"] += 1
            stats["permuted_exps"] += 3
        elif mode == "slices":
            data = _permute_slices(item, n_expert, perm)
            stats["permuted_exps"] += 1
        elif mode == "rows":
            # router: ne = [n_embd, n_expert] -> rows (ne[1]) are experts;
            # bias: ne = [n_expert] -> elements are experts
            data = _permute_slices(item, n_expert, perm)
            stats["permuted_router"] += 1
        else:
            data = item.data
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)

    writer.close()
    return stats
