"""Fallback ladder, rung 3 (design doc §05): a structural tree from
safetensors headers alone — tensor names, shapes, and dtypes via HTTP range
reads (or local files, or a GGUF tensor table). Explore-only: no classes,
no trace."""

from __future__ import annotations

import json
import os
import struct
from collections import Counter, defaultdict
from typing import Iterable

from huggingface_hub import HfApi, get_safetensors_metadata, parse_safetensors_file_metadata

from modelmap import collapse
from modelmap.hubio import with_retries
from modelmap.schema import SCHEMA_VERSION, Edge, Graph, Node

MAX_FILES = 300


def _collect_tensors(model_id: str, revision: str, token: str | None, notes: list[str]):
    """Tensor name → TensorInfo for the whole repo.

    Fast path: the standard root-level model.safetensors(.index.json). Fallback
    (diffusers-style pipelines, arbitrary layouts): scan every *.safetensors in
    the repo, namespacing tensor names by their folder so components don't
    collide ("transformer/…" → "transformer.blocks.0…")."""
    try:
        meta = with_retries(
            lambda: get_safetensors_metadata(model_id, revision=revision, token=token)
        )
        tensors = {}
        for fm in meta.files_metadata.values():
            tensors.update(fm.tensors)
        return tensors
    except Exception:
        pass

    files = [
        f
        for f in with_retries(
            lambda: HfApi(token=token).list_repo_files(model_id, revision=revision)
        )
        if f.endswith(".safetensors")
    ]
    if not files:
        raise ValueError(f"'{model_id}' has no safetensors files to build a weights view from")
    if len(files) > MAX_FILES:
        notes.append(f"repo has {len(files)} safetensors files; reading the first {MAX_FILES}")
        files = files[:MAX_FILES]

    tensors = {}
    failed = 0
    for f in files:
        prefix = f.rsplit("/", 1)[0].replace("/", ".") + "." if "/" in f else ""
        try:
            fm = with_retries(
                lambda f=f: parse_safetensors_file_metadata(
                    model_id, f, revision=revision, token=token
                )
            )
        except Exception:
            failed += 1
            continue
        for tname, info in fm.tensors.items():
            tensors[prefix + tname] = info
    if failed:
        notes.append(f"{failed} safetensors files could not be parsed")
    if not tensors:
        raise ValueError(f"could not read any safetensors headers from '{model_id}'")
    return tensors


TensorTriple = tuple[str, list[int], str]  # (name, shape, dtype)


def local_safetensors(path: str) -> list[TensorTriple]:
    """Tensor table of every *.safetensors under a local checkpoint dir (or a
    single file): the 8-byte header length + JSON header, no tensor data."""
    files = [path] if os.path.isfile(path) else sorted(
        os.path.join(root, f)
        for root, _, fs in os.walk(path)
        for f in fs
        if f.endswith(".safetensors")
    )
    out: list[TensorTriple] = []
    for fp in files[:MAX_FILES]:
        with open(fp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        rel = os.path.relpath(fp, path) if os.path.isdir(path) else ""
        prefix = rel.rsplit("/", 1)[0].replace("/", ".") + "." if "/" in rel else ""
        for tname, info in header.items():
            if tname == "__metadata__":
                continue
            out.append((prefix + tname, list(info["shape"]), str(info["dtype"]).lower()))
    return out


def weights_graph(
    model_id: str,
    revision: str = "main",
    token: str | None = None,
    notes: list[str] | None = None,
    tensors: Iterable[TensorTriple] | None = None,
    **extra,
) -> Graph:
    notes = list(notes or [])
    if tensors is None:
        tensors = [
            (tname, list(info.shape), str(info.dtype).lower())
            for tname, info in _collect_tensors(model_id, revision, token, notes).items()
        ]
    tensors = list(tensors)

    # "model.layers.0.self_attn.q_proj.weight" → module "…q_proj", tensor "weight"
    mod_weights: dict[str, dict[str, list[int]]] = defaultdict(dict)
    mod_dtypes: dict[str, Counter] = defaultdict(Counter)
    for tname, shape, dtype in tensors:
        mod, _, leaf = tname.rpartition(".")
        mod = mod or tname
        mod_weights[mod][leaf] = list(shape)
        mod_dtypes[mod][dtype] += 1

    params_by_node: dict[str, int] = defaultdict(int)
    node_ids: set[str] = set()
    for mod, ws in mod_weights.items():
        p = sum(_numel(s) for s in ws.values())
        parts = mod.split(".")
        for i in range(1, len(parts) + 1):
            nid = ".".join(parts[:i])
            node_ids.add(nid)
            params_by_node[nid] += p

    siblings: dict[str, list[str]] = defaultdict(list)
    for nid in node_ids:
        parent = nid.rsplit(".", 1)[0] if "." in nid else None
        siblings[parent].append(nid)
    order: dict[str, int] = {}
    for ks in siblings.values():
        ks.sort(key=_natural_key)
        for i, nid in enumerate(ks):
            order[nid] = i

    nodes = []
    for nid in sorted(node_ids, key=_natural_key):
        leaf = nid.rsplit(".", 1)[-1]
        dtype = mod_dtypes[nid].most_common(1)[0][0] if mod_dtypes.get(nid) else None
        nodes.append(Node(
            id=nid,
            kind=_classify_name(leaf, is_leaf=nid in mod_weights),
            cls="?",
            parent=nid.rsplit(".", 1)[0] if "." in nid else None,
            depth=nid.count(".") + 1,
            order=order[nid],
            params=params_by_node[nid],
            dtype=dtype,
            weight_shapes=mod_weights.get(nid),
        ))

    nodes, repeats = collapse.collapse_repeats(nodes)
    kids: dict[str, list[Node]] = defaultdict(list)
    for n in nodes:
        if n.parent is not None:
            kids[n.parent].append(n)
    edges = []
    for ks in kids.values():
        ks.sort(key=lambda n: n.order)
        edges.extend(Edge(src=a.id, dst=b.id) for a, b in zip(ks, ks[1:]))

    return Graph(
        schema_version=SCHEMA_VERSION,
        model_id=model_id,
        revision=revision,
        fidelity="weights",
        architecture=None,
        params_total=sum(_numel(shape) for _, shape, _ in tensors),
        config={},
        nodes=nodes,
        repeats=repeats,
        edges=edges,
        trace=[],
        notes=notes,
        **extra,
    )


def _classify_name(leaf: str, is_leaf: bool) -> str:
    l = leaf.lower()
    if "rotary" in l:
        return "module"
    if "embed" in l or l in ("wte", "wpe"):
        return "embedding"
    if l in ("lm_head", "score", "classifier"):
        return "head"
    if l in ("self_attn", "attn", "attention", "cross_attn"):
        return "attention"
    if l == "experts":
        return "moe"
    if l in ("mlp", "ffn", "intermediate") or "expert" in l:
        return "mlp"
    if "norm" in l or l == "ln_f" or l.startswith("ln_"):
        return "norm"
    if l.endswith("_proj") or l in ("fc1", "fc2", "dense", "proj", "gate", "query", "key", "value"):
        return "linear"
    return "module" if is_leaf else "container"


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _natural_key(nid: str):
    return [(0, int(p)) if p.isdigit() else (1, p) for p in nid.split(".")]
