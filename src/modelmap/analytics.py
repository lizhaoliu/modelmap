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


def text_config(doc: dict) -> dict[str, Any]:
    """The config with the language model's numbers in reach: multimodal
    configs (Qwen2.5-VL, Llama-4, Gemma-3, …) nest hidden_size / heads /
    layers under text_config — every analytic that reads them goes through
    here (twin of cost.ts textConfig)."""
    c = dict(doc.get("config") or {})
    tc = c.get("text_config")
    if isinstance(tc, dict):
        for k, v in tc.items():
            c.setdefault(k, v)
    return c


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
    c = text_config(doc)

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
    # serve the weights quantized to this dtype (int4, f8, …); None = as stored
    weights: str | None = None

    @property
    def bytes(self) -> float:
        return WHATIF_DTYPES.get(self.dtype, DTYPE_BYTES.get(self.dtype, 2))

    @property
    def weight_bytes(self) -> float | None:
        if not self.weights or self.weights == "stored":
            return None
        return WHATIF_DTYPES.get(self.weights, DTYPE_BYTES.get(self.weights))

    def weight_bytes_of(self, stored_dtype: str | None) -> float:
        """Bytes per weight element: the what-if precision when one is chosen,
        else the stored dtype (falling back to the activation dtype)."""
        wb = self.weight_bytes
        return wb if wb is not None else bytes_of(stored_dtype, self.bytes)


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
    c = text_config(doc)
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
        cost.param_bytes = own_params * a.weight_bytes_of(n.get("dtype"))
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


def _recipe(doc: dict) -> list[str]:
    from modelmap.insights import profile, recipe  # lazy: insights imports analytics

    try:
        return recipe(profile(doc))
    except Exception:  # a recipe is a nicety; never fail a summary for it
        return []


def summarize(doc: dict, a: Assumptions | None = None) -> dict[str, Any]:
    """The headline numbers for a model: what `modelmap cost`, /api/summary
    and the MCP describe_model tool return."""
    a = a or Assumptions()
    index = build_index(doc)
    rep = compute_costs(doc, index, a)
    c = text_config(doc)
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
        "recipe": _recipe(doc),
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
    weights: str | None = None  # serve weights quantized to this dtype; None = stored
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
    a = Assumptions(T=req.T, B=req.B, dtype=req.dtype, weights=req.weights)
    rep = compute_costs(doc, index, a)
    c = text_config(doc)
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
        + (f"weights quantized to {req.weights}" if a.weight_bytes is not None else "weights at stored dtypes")
        + "; KV at the activation dtype; no framework workspace beyond the headroom"
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


# ------------------------------------------- training planner (§22)

# LoRA target groups: leaf names of the linear modules adapters attach to.
# Fused 3-D expert weights are excluded — adapters on fused experts are rare
# and framework-specific.
LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "c_attn", "query", "key", "value", "dense", "out_proj"),
    "attn-mlp": (
        "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "c_attn", "query", "key", "value", "dense", "out_proj",
        "gate_proj", "up_proj", "down_proj", "c_fc", "c_proj", "fc1", "fc2", "wi", "wo",
    ),
    "all-linear": (),  # every ≥2-D matmul weight outside embeddings/heads
}

# bytes per parameter of optimizer state (+ fp32 master weights, the mixed-
# precision convention): AdamW keeps fp32 m+v (8) + master (4); the 8-bit
# variant quantizes m+v to 1 byte each but keeps the fp32 master
OPTIMIZER_BYTES: dict[str, float] = {"adamw": 12.0, "adamw8bit": 6.0}

# NF4 base weights: 4 bits + double-quantized absmax constants ≈ 0.55 B/param
QLORA_BASE_BYTES = 0.55


@dataclass
class TrainRequest:
    method: str = "lora"  # full | lora | qlora
    optimizer: str = "adamw"  # adamw | adamw8bit
    lora_rank: int = 16
    lora_targets: str = "attn-mlp"  # attention | attn-mlp | all-linear
    gpus: int = 1
    gpu_memory_gb: float = 80
    sharding: str = "none"  # none | zero2 | zero3 (data parallel across gpus)
    T: int = 2048
    B: int = 1  # micro-batch per GPU
    grad_checkpoint: bool = True
    flash_attention: bool = True
    headroom: float = 0.10
    gpu: str | None = None  # preset name, for the speed estimate


@dataclass
class TrainPlan:
    request: TrainRequest
    trainable_params: float
    total_params: float
    weight_bytes_per_gpu: float
    grad_bytes_per_gpu: float
    optimizer_bytes_per_gpu: float
    activation_bytes_per_gpu: float
    total_bytes_per_gpu: float
    per_gpu_capacity_bytes: float
    fits: bool
    max_microbatch: int  # largest per-GPU B that still fits at this T
    train_tokens_per_sec: float | None  # across all GPUs, when a preset is named
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lora_trainable(doc: dict, index: Index, rep: CostReport, targets: str, rank: int) -> tuple[float, int]:
    """(adapter params, matched module count) for LoRA at `rank`."""
    leaf_names = LORA_TARGETS.get(targets, ())
    total = 0.0
    matched = 0
    for n in doc["nodes"]:
        ws = n.get("weight_shapes") or {}
        w = ws.get("weight")
        if not w or len(w) != 2:
            continue
        if n["kind"] in ("embedding", "head", "norm"):
            continue
        leaf = n["id"].rsplit(".", 1)[-1]
        if targets != "all-linear" and leaf not in leaf_names:
            continue
        m = rep.multiplicity.get(n["id"], 1)
        out_f, in_f = w
        total += rank * (in_f + out_f) * m
        matched += m
    return total, matched


def plan_training(doc: dict, req: TrainRequest | None = None) -> TrainPlan:
    """Fine-tuning memory estimate (full / LoRA / QLoRA) with data-parallel
    sharding (ZeRO-2/3), gradient checkpointing and flash-attention toggles.

    Conventions (each carried into the notes): bf16 compute; grads bf16;
    optimizer = fp32 m+v + fp32 master (AdamW 12 B, 8-bit 6 B per trainable);
    activations at bf16 from the traced shapes; attention scores add
    heads × T² unless flash attention.
    """
    req = req or TrainRequest()
    notes: list[str] = []
    index = build_index(doc)
    a = Assumptions(T=req.T, B=req.B, dtype="bf16")
    rep = compute_costs(doc, index, a)
    c = text_config(doc)
    params = float(doc.get("params_total") or 0)
    gpus = max(1, req.gpus)
    cap = req.gpu_memory_gb * 2**30 * (1 - req.headroom)

    if req.method == "full":
        trainable = params
        weight_bytes = params * 2.0
        notes.append("full fine-tune: every parameter trains (bf16 weights)")
    else:
        adapters, matched = _lora_trainable(doc, index, rep, req.lora_targets, req.lora_rank)
        trainable = adapters
        if not matched:
            notes.append(f"no linear modules matched targets '{req.lora_targets}'")
        else:
            notes.append(f"LoRA r={req.lora_rank} on {matched} linear modules ({req.lora_targets})")
        if any((n.get("weight_shapes") or {}) and len(list((n.get("weight_shapes") or {}).values())[0]) == 3 for n in doc["nodes"]):
            notes.append("fused 3-D expert weights are not LoRA targets here")
        base = params * (QLORA_BASE_BYTES if req.method == "qlora" else 2.0)
        if req.method == "qlora":
            notes.append("QLoRA: frozen base at NF4 (≈0.55 B/param incl. quant constants)")
        weight_bytes = base + adapters * 2.0
    grad_bytes = trainable * 2.0
    opt_bytes = trainable * OPTIMIZER_BYTES.get(req.optimizer, 12.0)

    # activations per GPU (each GPU runs its own micro-batch)
    layers = _num(c, "num_hidden_layers", "n_layer") or 0
    hidden = _num(c, "hidden_size", "n_embd") or 0
    heads = _num(c, "num_attention_heads", "n_head") or 0
    block_act = 0.0
    reps = sorted(doc.get("repeats") or [], key=lambda r: -(r["count"] * index.by_id[r["representative"]]["params"]))
    if reps:
        block_act = rep.by_node[reps[0]["representative"]].act_bytes
    full_act = rep.root.act_bytes
    scores = 0.0 if req.flash_attention else layers * heads * req.B * req.T * req.T * 2.0
    if req.grad_checkpoint:
        act_bytes = layers * req.B * req.T * hidden * 2.0 + block_act + (0.0 if req.flash_attention else scores / max(layers, 1))
        notes.append("gradient checkpointing: layer inputs kept, one block's activations resident during recompute")
    else:
        act_bytes = full_act + scores
        notes.append("no gradient checkpointing: every traced activation held for backward")
    if not req.flash_attention:
        notes.append("without flash attention, softmax scores add heads × T² per layer")

    # data-parallel sharding
    shard = max(1, gpus)
    if req.sharding == "zero3":
        w_gpu, g_gpu, o_gpu = weight_bytes / shard, grad_bytes / shard, opt_bytes / shard
        notes.append("ZeRO-3 / FSDP: weights, grads and optimizer sharded across GPUs (gathering adds transient overhead)")
    elif req.sharding == "zero2":
        w_gpu, g_gpu, o_gpu = weight_bytes, grad_bytes / shard, opt_bytes / shard
        notes.append("ZeRO-2: grads and optimizer sharded; each GPU keeps full weights")
    else:
        w_gpu, g_gpu, o_gpu = weight_bytes, grad_bytes, opt_bytes
        if gpus > 1:
            notes.append("plain data parallel: every GPU holds a full replica")
    total = w_gpu + g_gpu + o_gpu + act_bytes
    fits = total <= cap

    fixed = w_gpu + g_gpu + o_gpu
    act_per_b = act_bytes / max(req.B, 1)
    max_b = int((cap - fixed) / act_per_b) if act_per_b > 0 and cap > fixed else 0

    tps: float | None = None
    if req.gpu and req.gpu in GPU_SPECS:
        spec = GPU_SPECS[req.gpu]
        # fwd+bwd ≈ 3 × forward FLOPs; FLOPs = 2 × MACs
        macs_tok = rep.root.macs / max(1, req.T * req.B)
        tps = (spec["tflops"] * 1e12 * TRAIN_MFU * gpus) / (6.0 * macs_tok)
        notes.append(f"speed assumes {int(TRAIN_MFU * 100)}% MFU on {req.gpu}; fwd+bwd ≈ 3× forward FLOPs")
    return TrainPlan(
        request=req, trainable_params=trainable, total_params=params,
        weight_bytes_per_gpu=w_gpu, grad_bytes_per_gpu=g_gpu, optimizer_bytes_per_gpu=o_gpu,
        activation_bytes_per_gpu=act_bytes, total_bytes_per_gpu=total,
        per_gpu_capacity_bytes=cap, fits=fits, max_microbatch=max_b,
        train_tokens_per_sec=tps, notes=notes,
    )


# ------------------------------------------- throughput estimates (§22)

# approximate dense-bf16 TFLOPs and memory bandwidth (GB/s) per preset;
# marketing sheets vary — these feed *estimates*, always labeled as such
GPU_SPECS: dict[str, dict[str, float]] = {
    "H100 80GB": {"tflops": 989, "bw": 3350}, "H200 141GB": {"tflops": 989, "bw": 4800},
    "A100 80GB": {"tflops": 312, "bw": 2039}, "A100 40GB": {"tflops": 312, "bw": 1555},
    "L40S 48GB": {"tflops": 362, "bw": 864}, "L4 24GB": {"tflops": 121, "bw": 300},
    "A10G 24GB": {"tflops": 70, "bw": 600}, "RTX 4090 24GB": {"tflops": 165, "bw": 1008},
    "RTX 3090 24GB": {"tflops": 71, "bw": 936}, "RTX 5090 32GB": {"tflops": 210, "bw": 1792},
    "MI300X 192GB": {"tflops": 1307, "bw": 5300}, "B200 180GB": {"tflops": 2250, "bw": 8000},
    "T4 16GB": {"tflops": 65, "bw": 320},
}
PREFILL_MFU = 0.40  # fraction of peak FLOPs a good serving stack reaches in prefill
DECODE_BW_EFF = 0.60  # fraction of peak bandwidth reached streaming weights in decode
TRAIN_MFU = 0.35


@dataclass
class Throughput:
    gpu: str
    prefill_tok_per_sec: float
    decode_tok_per_sec_b1: float
    decode_tok_per_sec_at_b: float
    batch: int
    bytes_read_per_token: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_throughput(
    doc: dict, gpu: str, *, tp: int = 1, T: int = 4096, B: int = 1, dtype: str = "bf16", weights: str | None = None,
) -> Throughput | None:
    """Roofline speed estimate on a named GPU preset: prefill is compute-bound
    (MACs vs peak FLOPs at PREFILL_MFU); decode is bandwidth-bound (active
    weight bytes + the KV cache read every token, vs peak bandwidth at
    DECODE_BW_EFF). TP aggregates both, ignoring interconnect."""
    spec = GPU_SPECS.get(gpu)
    if not spec:
        return None
    index = build_index(doc)
    rep = compute_costs(doc, index, Assumptions(T=T, B=B, dtype=dtype, weights=weights))
    tokens = max(1, T * B)
    macs_tok = rep.root.macs / tokens
    params = float(doc.get("params_total") or 1)
    active_frac = rep.root.active_params / params if params else 1.0
    active_bytes = rep.root.param_bytes * active_frac
    kv_read = rep.root.kv_per_token * T  # the whole cache, read once per decoded token
    per_seq_bytes = active_bytes + kv_read
    bw = spec["bw"] * 1e9 * DECODE_BW_EFF * tp
    flops = spec["tflops"] * 1e12 * PREFILL_MFU * tp
    prefill = flops / (2.0 * macs_tok)
    decode_b1 = bw / per_seq_bytes
    # at batch B the weight read amortizes; each sequence still reads its KV
    decode_at_b = (B * bw) / (active_bytes + B * kv_read)
    notes = [
        f"roofline estimate: prefill at {int(PREFILL_MFU * 100)}% MFU, decode at {int(DECODE_BW_EFF * 100)}% of "
        f"{spec['bw']:.0f} GB/s; interconnect and scheduler overhead not modeled",
    ]
    if active_frac < 0.95:
        notes.append(f"MoE: decode streams the ≈{active_frac * 100:.0f}% of weights that are active per token")
    return Throughput(
        gpu=gpu, prefill_tok_per_sec=prefill, decode_tok_per_sec_b1=decode_b1,
        decode_tok_per_sec_at_b=decode_at_b, batch=B, bytes_read_per_token=per_seq_bytes, notes=notes,
    )
