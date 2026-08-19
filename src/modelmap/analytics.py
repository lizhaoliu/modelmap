"""Cost analytics and serving planner over a graph document (design doc §14, §17).

A Python twin of web/src/analytics/cost.ts so the CLI, the REST summary and
the MCP server report the same numbers the UI shows. Everything here is an
analytic estimate derived from traced shapes, module attributes and config —
weight-matmul + attention-core MACs, bytes from shapes × dtype. Every value
is reproducible from the document alone; no weights are ever read.

Keep the two implementations in lockstep: tests/test_analytics.py pins the
same fixture numbers that web/tests/cost.test.ts does.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# ----------------------------------------------------------------- index

VISION = re.compile(r"(^|\.)(visual|vision|vision_tower|vision_model|image_encoder)(\.|$)")
_VISION_STEP = re.compile(r"(^|\.)(visual|vision)")
LINEAR_ATTN = re.compile(r"(DeltaNet|Mamba|LinearAttention|GatedLinear|SSM|Retention)", re.I)

# bytes per element at a stored dtype; GGUF block quants are fractional
# (bits per weight / 8, including scales) — see gguf.py
DTYPE_BYTES: dict[str, float] = {
    "f64": 8, "f32": 4, "float32": 4, "f16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
    "f8_e4m3": 1, "f8_e5m2": 1, "float8_e4m3fn": 1, "float8_e5m2": 1, "i8": 1, "int8": 1,
    "u8": 1, "bool": 1, "i16": 2, "i32": 4, "i64": 8, "int4": 0.5, "u4": 0.5, "i4": 0.5,
    "nf4": 0.5, "fp4": 0.5,
    # GGUF quant types (bits per weight incl. scales / 8)
    "q4_0": 4.5 / 8, "q4_1": 5 / 8, "q5_0": 5.5 / 8, "q5_1": 6 / 8, "q8_0": 8.5 / 8, "q8_1": 9 / 8,
    "q2_k": 2.625 / 8, "q3_k": 3.4375 / 8, "q4_k": 4.5 / 8, "q5_k": 5.5 / 8, "q6_k": 6.5625 / 8,
    "q8_k": 8.5 / 8, "iq2_xxs": 2.0625 / 8, "iq2_xs": 2.3125 / 8, "iq2_s": 2.5 / 8,
    "iq3_xxs": 3.0625 / 8, "iq3_s": 3.4375 / 8, "iq1_s": 1.5625 / 8, "iq1_m": 1.75 / 8,
    "iq4_nl": 4.5 / 8, "iq4_xs": 4.25 / 8, "tq1_0": 1.6875 / 8, "tq2_0": 2.0625 / 8,
    "mxfp4": 4.25 / 8,
}

# what-if dtype choices (activations, and weights whose stored dtype is unknown)
WHATIF_DTYPES: dict[str, float] = {"bf16": 2, "f16": 2, "f32": 4, "f8": 1, "int8": 1, "int4": 0.5}


def bytes_of(dtype: str | None, fallback: float) -> float:
    if not dtype:
        return fallback
    return DTYPE_BYTES.get(dtype.lower(), fallback)


def _num(c: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = c.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def _prod(xs: list[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


@dataclass
class Index:
    by_id: dict[str, dict]
    children: dict[str | None, list[dict]]
    repeat_by_rep: dict[str, dict]
    repeats_by_parent: dict[str, list[dict]]
    trace_by_node: dict[str, dict]
    dim_labels: dict[int, str]
    trace_batch: int | None
    trace_seq: int | None


def dim_labels(doc: dict) -> dict[int, str]:
    """Value-matched semantic labels for tensor dims (twin of types.ts
    buildDimLabels): a value claimed by two labels is left unlabeled."""
    c = doc.get("config") or {}

    def num(k: str) -> float | None:
        v = c.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    hidden = num("hidden_size") or num("n_embd")
    heads = num("num_attention_heads") or num("n_head")
    cands: list[tuple[float | None, str]] = [
        (hidden, "hidden"), (num("intermediate_size"), "ffn"), (num("moe_intermediate_size"), "ffn"),
        (num("vocab_size"), "vocab"), (heads, "heads"), (num("num_key_value_heads"), "kv heads"),
        (num("head_dim"), "head dim"),
        (num("num_experts") or num("n_routed_experts") or num("num_local_experts"), "experts"),
        (num("max_position_embeddings") or num("n_positions"), "max pos"), (num("patch_size"), "patch"),
    ]
    if not num("head_dim") and hidden and heads and hidden % heads == 0:
        cands.append((hidden / heads, "head dim"))
    v = c.get("vision_config") or {}

    def vnum(k: str) -> float | None:
        x = v.get(k) if isinstance(v, dict) else None
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    cands += [
        (vnum("hidden_size"), "vision hidden"), (vnum("intermediate_size"), "vision ffn"),
        (vnum("num_heads") or vnum("num_attention_heads"), "heads"),
        (vnum("patch_size") or num("patch_size"), "patch"), (vnum("out_hidden_size"), "hidden"),
    ]
    ch, ps, tps = vnum("in_channels") or vnum("num_channels"), vnum("patch_size"), vnum("temporal_patch_size")
    if ch and ps and tps:
        cands.append((ch * tps * ps * ps, "patch values"))
    trace = doc.get("trace") or []
    first_text = next((t for t in trace if not _VISION_STEP.search(t["node"])), None)
    first = (first_text or {}).get("inputs", [None])[0] if first_text and first_text.get("inputs") else None
    if first and len(first) == 2:
        cands.append((first[1], "seq"))
    if first and len(first) == 4:
        cands += [(first[1], "ch"), (first[2], "px"), (first[3], "px")]
    vis_steps = [t for t in trace if _VISION_STEP.search(t["node"])]
    tower = min(vis_steps, key=lambda t: len(t["node"].split("."))) if vis_steps else None
    vis = tower["inputs"][0] if tower and tower.get("inputs") else None
    if vis and len(vis) == 2:
        cands.append((vis[0], "patches"))
    if vis and len(vis) == 4:
        cands += [(vis[1], "ch"), (vis[2], "px"), (vis[3], "px")]
    tower_out = tower["outputs"][-1] if tower and tower.get("outputs") else None
    if tower_out and len(tower_out) == 2 and (not vis or tower_out[0] != vis[0]):
        cands.append((tower_out[0], "image tokens"))

    out: dict[int, str] = {}
    clash: set[int] = set()
    for val, label in cands:
        if val is None or val < 3 or val in clash:
            continue
        key = int(val)
        prev = out.get(key)
        if prev and prev != label:
            del out[key]
            clash.add(key)
            continue
        out[key] = label
    return out


def build_index(doc: dict) -> Index:
    by_id: dict[str, dict] = {}
    children: dict[str | None, list[dict]] = {}
    for n in doc["nodes"]:
        by_id[n["id"]] = n
        key = n["parent"] if n["parent"] is not None else (None if n["id"] == "" else "")
        children.setdefault(key, []).append(n)
    for lst in children.values():
        lst.sort(key=lambda n: n["order"])
    repeat_by_rep = {r["representative"]: r for r in doc.get("repeats") or []}
    repeats_by_parent: dict[str, list[dict]] = {}
    for r in doc.get("repeats") or []:
        repeats_by_parent.setdefault(r["parent"], []).append(r)
    trace_by_node: dict[str, dict] = {}
    for t in doc.get("trace") or []:
        trace_by_node.setdefault(t["node"], t)
    trace = doc.get("trace") or []
    main = next((t for t in trace if not _VISION_STEP.search(t["node"])), trace[0] if trace else None)
    batch = main["inputs"][0][0] if main and main.get("inputs") and main["inputs"][0] else None
    labels = dim_labels(doc)
    seq = next((v for v, l in labels.items() if l == "seq"), None)
    return Index(by_id, children, repeat_by_rep, repeats_by_parent, trace_by_node, labels, batch, seq)


# ------------------------------------------------------------------ cost


@dataclass
class Assumptions:
    T: int = 4096
    B: int = 1
    dtype: str = "bf16"

    @property
    def bytes(self) -> float:
        return WHATIF_DTYPES.get(self.dtype, DTYPE_BYTES.get(self.dtype, 2))


@dataclass
class Cost:
    macs: float = 0.0
    other: float = 0.0
    param_bytes: float = 0.0
    active_params: float = 0.0
    act_bytes: float = 0.0
    max_act: float = 0.0
    max_act_node: str = ""
    kv_per_token: float = 0.0
    formula: str | None = None

    def add(self, o: Cost, m: float = 1) -> None:
        self.macs += o.macs * m
        self.other += o.other * m
        self.param_bytes += o.param_bytes * m
        self.active_params += o.active_params * m
        self.act_bytes += o.act_bytes * m
        self.kv_per_token += o.kv_per_token * m
        if o.max_act > self.max_act:
            self.max_act, self.max_act_node = o.max_act, o.max_act_node


@dataclass
class CostReport:
    by_node: dict[str, Cost]
    root: Cost
    assumptions: Assumptions
    kv_layers: int
    kv_skipped: int
    notes: list[str] = field(default_factory=list)
    multiplicity: dict[str, int] = field(default_factory=dict)


def compute_costs(doc: dict, index: Index | None = None, a: Assumptions | None = None) -> CostReport:
    index = index or build_index(doc)
    a = a or Assumptions()
    c = doc.get("config") or {}
    notes: list[str] = []
    T0 = index.trace_seq if index.trace_seq is not None else 7
    B0 = index.trace_batch if index.trace_batch is not None else 1

    mult: dict[str, int] = {}

    def mult_of(nid: str) -> int:
        if nid in mult:
            return mult[nid]
        n = index.by_id.get(nid)
        own = (index.repeat_by_rep.get(nid) or {}).get("count", 1)
        v = own * (mult_of(n["parent"]) if n and n.get("parent") is not None else 1)
        mult[nid] = v
        return v

    E = _num(c, "num_experts", "n_routed_experts", "num_local_experts") or 0
    K = _num(c, "num_experts_per_tok", "moe_top_k", "num_experts_per_token") or 0

    def under_experts(nid: str) -> bool:
        return bool(re.search(r"(^|\.)experts(\.|$)", nid)) and "shared_experts" not in nid

    def expert_frac(nid: str, weights: list[list[int]]) -> float:
        if not under_experts(nid) or not K:
            return 1.0
        fused = next((w for w in weights if len(w) == 3), None)
        e = fused[0] if fused else E
        return min(1.0, K / e) if e else 1.0

    def scale(shape: list[int], nid: str) -> list[int]:
        vis = bool(VISION.search(nid))
        out = []
        for i, d in enumerate(shape):
            if i == 0 and d == B0 and len(shape) > 1:
                out.append(a.B)
            elif not vis and d == T0:
                out.append(a.T)
            else:
                out.append(d)
        return out

    def tokens_of(shape: list[int] | None, nid: str) -> float:
        if not shape or len(shape) < 2:
            return a.B * a.T
        s = scale(shape, nid)
        t = _prod(s[:-1])
        if not (shape[0] == B0 and len(shape) > 1):
            t *= a.B
        return t

    heads = _num(c, "num_attention_heads", "n_head") or 0
    kv_heads = _num(c, "num_key_value_heads") or heads
    hidden = _num(c, "hidden_size", "n_embd") or 0
    head_dim = _num(c, "head_dim") or (hidden / heads if heads else 0)
    qk_dim = ((_num(c, "qk_nope_head_dim") or 0) + (_num(c, "qk_rope_head_dim") or 0)) or head_dim
    v_dim = _num(c, "v_head_dim") or head_dim
    kv_lora = _num(c, "kv_lora_rank")
    rope_dim = _num(c, "qk_rope_head_dim") or 0
    vc = c.get("vision_config") or {}
    vc = vc if isinstance(vc, dict) else {}
    v_heads = _num(vc, "num_heads", "num_attention_heads") or 0
    v_hidden = _num(vc, "hidden_size") or 0
    v_head_dim = v_hidden / v_heads if v_heads else 0

    # a tied lm_head shares the embedding matrix: real compute, but its
    # parameters are stored (and counted in params_total) only once
    tied_heads: set[str] = set()
    if c.get("tie_word_embeddings") is True:
        emb_shapes = {
            tuple(w) for n in doc["nodes"] if n["kind"] == "embedding"
            for w in (n.get("weight_shapes") or {}).values()
        }
        for n in doc["nodes"]:
            if n["kind"] == "head" and any(tuple(w) in emb_shapes for w in (n.get("weight_shapes") or {}).values()):
                tied_heads.add(n["id"])

    own: dict[str, Cost] = {}
    kv_layers = 0
    kv_skipped = 0
    for n in doc["nodes"]:
        nid = n["id"]
        io = index.trace_by_node.get(nid)
        weights = list((n.get("weight_shapes") or {}).values())
        cost = Cost(max_act_node=nid)
        own_params = 0 if nid in tied_heads else sum(_prod(w) for w in weights)
        frac = expert_frac(nid, weights)
        cost.param_bytes = own_params * bytes_of(n.get("dtype"), a.bytes)
        cost.active_params = own_params * frac
        outs = (io or {}).get("outputs") or []
        if outs:
            b = _prod(scale(outs[0], nid)) * a.bytes
            cost.act_bytes = b
            cost.max_act = b
        kids = index.children.get(nid) or []
        is_leaf = not kids
        formula: list[str] = []
        matmul = [w for w in weights if len(w) in (2, 3)]
        ins = (io or {}).get("inputs") or []
        if n["kind"] not in ("embedding", "norm", "conv") and matmul:
            tokens = tokens_of(ins[0] if ins else None, nid)
            for w in matmul:
                cost.macs += tokens * _prod(w) * frac
            formula.append(
                f"tokens × k/E × prod(W) = {tokens:,.0f} × {K:g}/{E:g} × …" if frac < 1
                else f"tokens × in × out = {tokens:,.0f} × …"
            )
        elif n["kind"] == "conv":
            w = next((x for x in weights if len(x) >= 3), None)
            if outs and w:
                out_elems = _prod(scale(outs[0], nid))
                per_out = _prod(w[1:])
                cost.macs = out_elems * per_out
                formula.append(f"out_elems × in_ch/groups × kernel = {out_elems:,} × {per_out:,}")
        elif n["kind"] == "attention":
            vis = bool(VISION.search(nid))
            if not LINEAR_ATTN.search(n["cls"]):
                h = v_heads if vis else heads
                dq = v_head_dim if vis else qk_dim
                dv = v_head_dim if vis else v_dim
                seq = (ins[0][0] if ins and ins[0] else 0) if vis else a.T
                cost.macs = a.B * h * seq * seq * (dq + dv)
                formula.append(
                    f"attention core: B × heads × T² × (d_qk + d_v) = {a.B} × {h:g} × {seq:,}² × {dq + dv:g}"
                )
                if not vis:
                    cost.kv_per_token = (
                        (kv_lora + rope_dim) * a.bytes if kv_lora else 2 * kv_heads * head_dim * a.bytes
                    )
                    kv_layers += mult_of(nid)
            else:
                kv_skipped += mult_of(nid)
                formula.append("linear attention: T-linear core not modeled")
        elif is_leaf and outs:
            cost.other = _prod(scale(outs[0], nid))
        if formula:
            cost.formula = "; ".join(formula)
        own[nid] = cost

    by_node: dict[str, Cost] = {}
    for n in sorted(doc["nodes"], key=lambda x: -x["depth"]):
        base = own[n["id"]]
        total = Cost(**asdict(base))
        for k in index.children.get(n["id"]) or []:
            kc = by_node.get(k["id"])
            if kc is None:
                continue
            total.add(kc, (index.repeat_by_rep.get(k["id"]) or {}).get("count", 1))
        by_node[n["id"]] = total

    roots = [n for n in doc["nodes"] if n["parent"] is None]
    root_node = next((n for n in roots if n["id"] == ""), roots[0] if roots else None)
    if root_node and not (len(index.children.get("") or []) > 1 and root_node["id"] != ""):
        root = by_node[root_node["id"]]
    else:
        root = Cost()
        for n in roots:
            root.add(by_node[n["id"]])
    if kv_skipped:
        notes.append(f"{kv_skipped} linear-attention layers hold no KV cache")
    if tied_heads:
        notes.append("lm_head is tied to the embedding matrix (stored once)")
    if E and K:
        notes.append(f"MoE: {K:g} of {E:g} experts run per token")
    for n in doc["nodes"]:
        mult_of(n["id"])
    return CostReport(by_node, root, a, kv_layers, kv_skipped, notes, mult)


# --------------------------------------------------------------- summary


def summarize(doc: dict, a: Assumptions | None = None) -> dict[str, Any]:
    """The headline numbers for a model: what `modelmap cost`, /api/summary
    and the MCP describe_model tool return."""
    a = a or Assumptions()
    index = build_index(doc)
    rep = compute_costs(doc, index, a)
    c = doc.get("config") or {}
    layers = _num(c, "num_hidden_layers", "n_layer")
    tokens = max(1, a.T * a.B)
    root = rep.root
    kinds: dict[str, int] = {}
    for n in doc["nodes"]:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    stacks = [
        {"parent": r["parent"], "count": r["count"], "representative": r["representative"]}
        for r in doc.get("repeats") or []
    ]
    return {
        "model_id": doc.get("model_id"),
        "revision": doc.get("revision"),
        "architecture": doc.get("architecture"),
        "model_type": c.get("model_type"),
        "fidelity": doc.get("fidelity"),
        "variant": doc.get("variant"),
        "variants": doc.get("variants") or [],
        "params_total": doc.get("params_total"),
        "active_params": round(root.active_params),
        "modules": len(doc["nodes"]),
        "trace_steps": len(doc.get("trace") or []),
        "repeat_stacks": stacks,
        "config": {
            k: c.get(k)
            for k in (
                "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "intermediate_size", "vocab_size", "max_position_embeddings",
                "num_experts", "n_routed_experts", "num_local_experts", "num_experts_per_tok",
                "moe_intermediate_size", "tie_word_embeddings", "dtype", "torch_dtype",
                "rope_theta", "sliding_window", "kv_lora_rank", "quantization_config",
            )
            if c.get(k) is not None
        },
        "layers": layers,
        "assumptions": asdict(a),
        "cost": {
            "macs_per_token": root.macs / tokens,
            "macs_per_forward": root.macs,
            "weight_bytes": root.param_bytes,
            "activation_bytes": root.act_bytes,
            "largest_activation_bytes": root.max_act,
            "largest_activation_node": root.max_act_node,
            "kv_bytes_per_token": root.kv_per_token,
            "kv_bytes_at_T": root.kv_per_token * tokens,
            "kv_layers": rep.kv_layers,
        },
        "notes": list(doc.get("notes") or []) + rep.notes,
    }


def module_rows(doc: dict, a: Assumptions | None = None, *, leaves_only: bool = False) -> list[dict[str, Any]]:
    """One row per shipped module (collapsed repeats carry their multiplicity)."""
    a = a or Assumptions()
    index = build_index(doc)
    rep = compute_costs(doc, index, a)
    tokens = max(1, a.T * a.B)
    rows = []
    for n in doc["nodes"]:
        kids = index.children.get(n["id"]) or []
        if leaves_only and kids:
            continue
        io = index.trace_by_node.get(n["id"]) or {}
        cost = rep.by_node[n["id"]]
        r = index.repeat_by_rep.get(n["id"])
        rows.append({
            "module": n["id"] or "(root)",
            "kind": n["kind"],
            "class": n["cls"],
            "depth": n["depth"],
            "repeats": r["count"] if r else 1,
            "multiplicity": rep.multiplicity.get(n["id"], 1),
            "params": n["params"],
            "params_total_with_repeats": n["params"] * rep.multiplicity.get(n["id"], 1),
            "dtype": n.get("dtype") or "",
            "weight_shapes": " ".join(f"{k}{v}" for k, v in (n.get("weight_shapes") or {}).items()),
            "input_shape": " ".join(str(s) for s in io.get("inputs") or []),
            "output_shape": " ".join(str(s) for s in io.get("outputs") or []),
            "macs_per_token": cost.macs / tokens,
            "weight_bytes": cost.param_bytes,
            "activation_bytes": cost.act_bytes,
            "kv_bytes_per_token": cost.kv_per_token,
            "attrs": " ".join(f"{k}={v}" for k, v in (n.get("attrs") or {}).items() if not k.startswith("_")),
            "source": (n.get("attrs") or {}).get("_src_url") or "",
        })
    return rows


# ----------------------------------------------------------- planner (§17)

GPU_PRESETS: dict[str, float] = {
    # usable HBM in GiB (marketing GB ≈ GiB here; headroom handled separately)
    "H100 80GB": 80, "H200 141GB": 141, "A100 80GB": 80, "A100 40GB": 40, "L40S 48GB": 48,
    "L4 24GB": 24, "A10G 24GB": 24, "RTX 4090 24GB": 24, "RTX 3090 24GB": 24, "RTX 5090 32GB": 32,
    "MI300X 192GB": 192, "B200 180GB": 180, "T4 16GB": 16, "Apple M-series (unified)": 0,
}


@dataclass
class PlanRequest:
    gpus: int = 1
    gpu_memory_gb: float = 80
    tp: int = 1
    pp: int = 1
    T: int = 4096
    B: int = 1
    dtype: str = "bf16"
    # fraction of memory kept free for framework overhead / fragmentation / workspace
    headroom: float = 0.10


@dataclass
class Stage:
    stage: int
    gpus: list[int]
    layers: list[int]  # inclusive range [first, last], or [] for none
    layer_count: int
    weight_bytes_per_gpu: float
    kv_bytes_per_gpu: float
    act_bytes_per_gpu: float
    total_bytes_per_gpu: float
    fits: bool
    boundary_bytes_out: float  # activation bytes sent to the next stage per forward


@dataclass
class Plan:
    request: PlanRequest
    fits: bool
    stages: list[Stage]
    weight_bytes: float
    kv_bytes: float
    act_bytes: float
    per_gpu_capacity_bytes: float
    max_context_tokens: int  # at B, largest T where every stage still fits (KV-limited)
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_serving(doc: dict, req: PlanRequest | None = None) -> Plan:
    """Tensor-/pipeline-parallel placement estimate.

    Weights (stored dtype) and KV cache split across TP ranks; PP splits the
    layer stack into contiguous groups so that weight bytes balance; the
    embedding joins the first stage and the head the last. Activations per GPU
    are the largest single activation (at T, B, dtype) plus the KV + weights.
    Boundary traffic is B × T × hidden × bytes per forward between stages.
    """
    req = req or PlanRequest()
    notes: list[str] = []
    index = build_index(doc)
    a = Assumptions(T=req.T, B=req.B, dtype=req.dtype)
    rep = compute_costs(doc, index, a)
    c = doc.get("config") or {}
    tp, pp = max(1, req.tp), max(1, req.pp)
    if tp * pp != req.gpus:
        notes.append(f"tp × pp = {tp * pp} ≠ gpus = {req.gpus}; planning for {tp * pp} GPUs")
    gpus = tp * pp
    cap = req.gpu_memory_gb * 2**30 * (1 - req.headroom)
    tokens = req.T * req.B
    hidden = _num(c, "hidden_size", "n_embd") or 0

    # the layer stack: the largest repeat (by count × params) among top-level stacks
    reps = sorted(doc.get("repeats") or [], key=lambda r: -(r["count"] * index.by_id[r["representative"]]["params"]))
    stack = reps[0] if reps else None
    # items to place, in execution order: [pre..., layer_0..layer_n, post...]
    items: list[tuple[str, float, float]] = []  # (label, weight bytes, kv per token)
    if stack:
        parent = stack["parent"]
        siblings = index.children.get(parent) or []
        rep_cost = rep.by_node[stack["representative"]]
        # other stacks in the same parent (DeepSeek: dense 3 + MoE 58) keep their own cost
        before_layers = True
        for s in siblings:
            r = index.repeat_by_rep.get(s["id"])
            if r:
                cost = rep.by_node[s["id"]]
                for i, m in enumerate(r["members"]):
                    items.append((f"{parent}.{m}", cost.param_bytes, cost.kv_per_token))
                before_layers = False
            else:
                cost = rep.by_node[s["id"]]
                items.append((s["id"] or "model", cost.param_bytes, cost.kv_per_token))
        # everything outside the stack's parent (embeddings, head, vision tower)
        outside_w = rep.root.param_bytes - rep.by_node[parent].param_bytes
        outside_kv = rep.root.kv_per_token - rep.by_node[parent].kv_per_token
        del rep_cost, before_layers
    else:
        notes.append("no repeated layer stack found; pipeline stages split the top-level modules")
        for s in index.children.get("") or []:
            cost = rep.by_node[s["id"]]
            items.append((s["id"], cost.param_bytes, cost.kv_per_token))
        outside_w = 0.0
        outside_kv = 0.0

    # split the embedding/head (outside) weights: first stage and last stage halves
    first_extra, last_extra = (outside_w / 2, outside_w / 2) if pp > 1 else (outside_w, 0.0)
    # balanced contiguous partition by weight bytes
    total_w = sum(w for _, w, _ in items)
    target = (total_w + outside_w) / pp
    groups: list[list[int]] = [[] for _ in range(pp)]
    acc = first_extra
    g = 0
    for i, (_, w, _) in enumerate(items):
        if g < pp - 1 and acc + w > target and groups[g]:
            g += 1
            acc = 0.0
        groups[g].append(i)
        acc += w
    # working activations: the largest single output outside the head — a
    # serving engine computes logits for sampled tokens only, so the
    # [B, T, vocab] logits tensor would overstate decode-time memory
    largest_act = max(
        (rep.by_node[n["id"]].max_act for n in doc["nodes"] if n["kind"] != "head" and not (index.children.get(n["id"]) or [])),
        default=0.0,
    )
    stages: list[Stage] = []
    fits = True
    for s in range(pp):
        idx = groups[s]
        w = sum(items[i][1] for i in idx) + (first_extra if s == 0 else 0) + (last_extra if s == pp - 1 else 0)
        kv = sum(items[i][2] for i in idx) * tokens + (outside_kv * tokens if s == 0 else 0)
        w_gpu, kv_gpu, act_gpu = w / tp, kv / tp, largest_act / tp
        total = w_gpu + kv_gpu + act_gpu
        layer_ids = [int(m.rsplit(".", 1)[-1]) for m in (items[i][0] for i in idx) if m.rsplit(".", 1)[-1].isdigit()]
        stages.append(Stage(
            stage=s,
            gpus=list(range(s * tp, (s + 1) * tp)),
            layers=[min(layer_ids), max(layer_ids)] if layer_ids else [],
            layer_count=len(layer_ids),
            weight_bytes_per_gpu=w_gpu,
            kv_bytes_per_gpu=kv_gpu,
            act_bytes_per_gpu=act_gpu,
            total_bytes_per_gpu=total,
            fits=total <= cap,
            boundary_bytes_out=(req.B * req.T * hidden * a.bytes) if s < pp - 1 else 0.0,
        ))
        fits = fits and total <= cap
    # KV-limited max context at this B: smallest over stages of (cap - weights - act) / kv-per-token
    max_ctx = math.inf
    for s, st in enumerate(stages):
        kv_tok = (st.kv_bytes_per_gpu / tokens) if tokens else 0
        free = cap - st.weight_bytes_per_gpu - st.act_bytes_per_gpu
        if kv_tok > 0:
            max_ctx = min(max_ctx, free / kv_tok / max(1, req.B))
        elif free < 0:
            max_ctx = 0
    max_ctx_i = 0 if max_ctx is math.inf else max(0, int(max_ctx))
    if not any(st.kv_bytes_per_gpu for st in stages):
        notes.append("no KV cache (no standard attention layers found); context is not memory-bound here")
        max_ctx_i = 0
    if tp > 1:
        heads = _num(c, "num_attention_heads", "n_head") or 0
        kvh = _num(c, "num_key_value_heads") or heads
        if heads and heads % tp:
            notes.append(f"{heads:g} attention heads do not divide evenly across tp={tp}")
        if kvh and kvh < tp:
            notes.append(f"only {kvh:g} KV heads: tp={tp} replicates K/V (KV memory per GPU is higher than shown)")
    if req.gpu_memory_gb <= 0:
        notes.append("GPU memory is 0 — unified-memory devices: compare totals against system RAM")
    notes.append(
        "activations = largest single non-logits activation at T, B (prefill peak; decode needs far less); "
        "weights at stored dtypes; KV at the activation dtype; no framework workspace beyond the headroom"
    )
    return Plan(
        request=req,
        fits=fits,
        stages=stages,
        weight_bytes=rep.root.param_bytes,
        kv_bytes=rep.root.kv_per_token * tokens,
        act_bytes=largest_act,
        per_gpu_capacity_bytes=cap,
        max_context_tokens=max_ctx_i,
        notes=notes,
    )


# ------------------------------------------------------------ formatting


def fmt_big(n: float, unit: str = "") -> str:
    a = abs(n)
    for div, s in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            v = n / div
            break
    else:
        v, s = n, ""
    digits = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
    return f"{v:.{digits}f} {s}{unit}".rstrip()


def fmt_bytes(n: float) -> str:
    a = abs(n)
    for div, s in ((2**40, "TB"), (2**30, "GB"), (2**20, "MB"), (1024, "KB")):
        if a >= div:
            v = n / div
            break
    else:
        v, s = n, "B"
    digits = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
    return f"{v:.{digits}f} {s}"


def fmt_params(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.{0 if n >= 1e11 else 2}f}B"
    if n >= 1e6:
        return f"{n / 1e6:.{0 if n >= 1e8 else 1}f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"
