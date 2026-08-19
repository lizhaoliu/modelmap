"""Compare two graph documents (design doc §15) — Python twin of
web/src/compare/align.ts, for `modelmap diff`, /api/compare and the MCP
compare_models tool.

Alignment is recursive from the roots: children pair by leaf name first
(HF naming has converged — model.layers.N.self_attn.q_proj is the same in
Llama, Qwen, Mistral, Gemma), then unpaired children pair by role (same
kind, in order — wte ↔ embed_tokens). What is left is added / removed.
Repeat stacks compare representative-to-representative plus counts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from modelmap.analytics import Index, build_index

CONFIG_KEYS = [
    "model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size", "max_position_embeddings",
    "rope_theta", "rope_scaling", "tie_word_embeddings", "num_experts", "num_experts_per_tok",
    "moe_intermediate_size", "sliding_window", "attention_bias", "hidden_act", "rms_norm_eps",
    "layer_norm_epsilon", "dtype", "torch_dtype", "n_layer", "n_head", "n_embd", "n_positions",
    "quantization_config",
]


@dataclass
class Change:
    field: str
    a: str | None
    b: str | None


@dataclass
class Pair:
    key: str
    a: str | None  # node id on side A
    b: str | None
    status: str  # same | changed | added | removed
    changes: list[Change] = field(default_factory=list)
    dirty: bool = False


@dataclass
class Alignment:
    pairs: list[Pair]
    counts: dict[str, int]
    config_diff: list[Change]

    def to_dict(self, *, changed_only: bool = True) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "config_diff": [asdict(c) for c in self.config_diff],
            "pairs": [asdict(p) for p in self.pairs if not changed_only or p.status != "same"],
        }


def _leaf(nid: str) -> str:
    return nid.rsplit(".", 1)[-1]


def _fmt(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"), sort_keys=True)
    return str(v)


def align(doc_a: dict, doc_b: dict, ia: Index | None = None, ib: Index | None = None) -> Alignment:
    ia = ia or build_index(doc_a)
    ib = ib or build_index(doc_b)
    pairs: list[Pair] = []

    def push(a: dict | None, b: dict | None) -> Pair:
        p = Pair(key=a["id"] if a else "+" + b["id"], a=a["id"] if a else None, b=b["id"] if b else None, status="same")
        if a and b:
            p.changes = diff_nodes(a, ia, b, ib)
            p.status = "changed" if p.changes else "same"
        else:
            p.status = "removed" if a else "added"
        pairs.append(p)
        return p

    def is_box(n: dict) -> bool:
        return n["kind"] in ("container", "module")

    def mark_subtree(n: dict, idx: Index, status: str) -> None:
        p = push(n, None) if status == "removed" else push(None, n)
        p.dirty = True
        for c in idx.children.get(n["id"]) or []:
            mark_subtree(c, idx, status)

    def align_children(pa: str, pb: str) -> bool:
        ca = list(ia.children.get(pa) or [])
        cb = list(ib.children.get(pb) or [])
        used_b: set[str] = set()
        matched: list[tuple[dict, dict]] = []
        for a in ca:
            b = next((x for x in cb if x["id"] not in used_b and _leaf(x["id"]) == _leaf(a["id"])), None)
            if b:
                matched.append((a, b))
                used_b.add(b["id"])
        matched_a = {a["id"] for a, _ in matched}
        rest_a = [a for a in ca if a["id"] not in matched_a]
        boxes_a = [a for a in rest_a if is_box(a)]
        boxes_b = [x for x in cb if x["id"] not in used_b and is_box(x)]
        for a in rest_a:
            b = None
            if is_box(a):
                if len(boxes_a) == 1 and len(boxes_b) == 1:
                    b = boxes_b[0]
            else:
                b = next((x for x in cb if x["id"] not in used_b and x["kind"] == a["kind"] and not is_box(x)), None)
            if b:
                matched.append((a, b))
                used_b.add(b["id"])
        matched.sort(key=lambda ab: ab[0]["order"])
        matched_a = {a["id"] for a, _ in matched}
        dirty = False
        for a, b in matched:
            p = push(a, b)
            child_dirty = align_children(a["id"], b["id"])
            p.dirty = p.status != "same" or child_dirty
            dirty = dirty or p.dirty
        for a in ca:
            if a["id"] not in matched_a:
                mark_subtree(a, ia, "removed")
                dirty = True
        for b in cb:
            if b["id"] not in used_b:
                mark_subtree(b, ib, "added")
                dirty = True
        return dirty

    roots_a = [n for n in doc_a["nodes"] if n["parent"] is None]
    roots_b = [n for n in doc_b["nodes"] if n["parent"] is None]
    if len(roots_a) == 1 and len(roots_b) == 1:
        p = push(roots_a[0], roots_b[0])
        p.dirty = align_children(roots_a[0]["id"], roots_b[0]["id"]) or p.status != "same"
    else:
        align_children("", "")

    counts = {"same": 0, "changed": 0, "added": 0, "removed": 0}
    for p in pairs:
        counts[p.status] += 1
    config_diff = []
    ca_, cb_ = doc_a.get("config") or {}, doc_b.get("config") or {}
    for k in CONFIG_KEYS:
        a, b = _fmt(ca_.get(k)), _fmt(cb_.get(k))
        if a != b and (a is not None or b is not None):
            config_diff.append(Change(k, a, b))
    return Alignment(pairs, counts, config_diff)


def diff_nodes(a: dict, ia: Index, b: dict, ib: Index) -> list[Change]:
    out: list[Change] = []

    def cmp(fld: str, x: Any, y: Any) -> None:
        fx, fy = _fmt(x), _fmt(y)
        if fx != fy:
            out.append(Change(fld, fx, fy))

    cmp("kind", a["kind"], b["kind"])
    cmp("class", a["cls"], b["cls"])
    cmp("params", a["params"], b["params"])
    cmp("dtype", a.get("dtype"), b.get("dtype"))
    wa, wb = a.get("weight_shapes") or {}, b.get("weight_shapes") or {}
    for k in sorted(set(wa) | set(wb)):
        cmp(f"weight {k}", wa.get(k), wb.get(k))
    aa, ab = a.get("attrs") or {}, b.get("attrs") or {}
    for k in sorted(set(aa) | set(ab)):
        if k.startswith("_"):
            continue
        cmp(k, aa.get(k), ab.get(k))
    cmp("repeats", (ia.repeat_by_rep.get(a["id"]) or {}).get("count"), (ib.repeat_by_rep.get(b["id"]) or {}).get("count"))
    ta, tb = ia.trace_by_node.get(a["id"]), ib.trace_by_node.get(b["id"])
    if ta and tb:
        cmp("input", (ta.get("inputs") or [None])[0], (tb.get("inputs") or [None])[0])
        cmp("output", (ta.get("outputs") or [None])[0], (tb.get("outputs") or [None])[0])
    return out


def diff_markdown(doc_a: dict, doc_b: dict, al: Alignment | None = None, *, limit: int = 80) -> str:
    """Human-readable diff: config changes, then changed / added / removed modules."""
    al = al or align(doc_a, doc_b)
    a_id, b_id = doc_a.get("model_id"), doc_b.get("model_id")
    lines = [f"# {a_id} vs {b_id}", ""]
    c = al.counts
    lines.append(
        f"{c['same']} modules identical · {c['changed']} changed · {c['added']} added in B · {c['removed']} removed from A"
    )
    lines.append("")
    if al.config_diff:
        lines += ["## Config", "", "| field | A | B |", "|---|---|---|"]
        lines += [f"| {ch.field} | {ch.a or '—'} | {ch.b or '—'} |" for ch in al.config_diff]
        lines.append("")
    changed = [p for p in al.pairs if p.status == "changed"]
    if changed:
        lines += ["## Changed modules", "", "| module | field | A | B |", "|---|---|---|---|"]
        n = 0
        for p in changed:
            for ch in p.changes:
                lines.append(f"| {p.a} | {ch.field} | {ch.a or '—'} | {ch.b or '—'} |")
                n += 1
                if n >= limit:
                    break
            if n >= limit:
                lines.append(f"| … | {sum(len(q.changes) for q in changed) - n} more | | |")
                break
        lines.append("")
    for status, title in (("added", "Only in B"), ("removed", "Only in A")):
        ps = [p for p in al.pairs if p.status == status]
        if ps:
            lines += [f"## {title}", ""]
            ids = [p.b if status == "added" else p.a for p in ps]
            # show only subtree roots (skip nodes whose parent is also listed)
            s = set(ids)
            roots = [i for i in ids if i and (i.rsplit(".", 1)[0] if "." in i else "") not in s]
            lines += [f"- `{i}`" for i in roots[:limit]]
            if len(roots) > limit:
                lines.append(f"- … {len(roots) - limit} more")
            lines.append("")
    return "\n".join(lines)
