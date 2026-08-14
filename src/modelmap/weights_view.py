"""Fallback ladder, rung 3 (design doc §05): a structural tree from
safetensors headers alone — tensor names, shapes, and dtypes via HTTP range
reads. Explore-only: no classes, no trace."""

from __future__ import annotations

from collections import Counter, defaultdict

from huggingface_hub import get_safetensors_metadata

from modelmap import collapse
from modelmap.schema import SCHEMA_VERSION, Edge, Graph, Node


def weights_graph(
    model_id: str,
    revision: str = "main",
    token: str | None = None,
    notes: list[str] | None = None,
) -> Graph:
    meta = get_safetensors_metadata(model_id, revision=revision, token=token)
    tensors = {}
    for fm in meta.files_metadata.values():
        tensors.update(fm.tensors)

    # "model.layers.0.self_attn.q_proj.weight" → module "…q_proj", tensor "weight"
    mod_weights: dict[str, dict[str, list[int]]] = defaultdict(dict)
    mod_dtypes: dict[str, Counter] = defaultdict(Counter)
    for tname, info in tensors.items():
        mod, _, leaf = tname.rpartition(".")
        mod = mod or tname
        mod_weights[mod][leaf] = list(info.shape)
        mod_dtypes[mod][str(info.dtype).lower()] += 1

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
        params_total=sum(_numel(list(t.shape)) for t in tensors.values()),
        config={},
        nodes=nodes,
        repeats=repeats,
        edges=edges,
        trace=[],
        notes=list(notes or []),
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
