"""Architecture takeaways (design doc §27): the sentences a reviewer would
write after staring at two maps — derived, not hand-written.

`profile(doc)` reads the recipe off a graph document: attention kind (MHA /
GQA / MQA / MLA), KV bytes per token, MLP shape (gated 3-matrix SwiGLU vs the
2-matrix GELU block), positional scheme (RoPE / learned / relative bias /
none), norm family, MoE routing, biases, tying, context, vocab. `insights(a, b)`
turns the differences into short, quantified statements ("GQA 8/32 vs 4/28 →
KV cache per token 2.6× larger") that /api/compare, the compare page, the zoo
lineage arrows, `modelmap diff` and the MCP compare tool all share — the web
never re-derives them, so there is one wording everywhere.
"""

from __future__ import annotations

import re
from typing import Any

from modelmap.analytics import Assumptions, _num, build_index, compute_costs, fmt_bytes, fmt_params, text_config

_VISION = re.compile(r"(^|\.)(visual|vision|vision_tower|vision_model|image_encoder)(\.|$)")


def _block_nodes(doc: dict, index) -> list[dict]:
    """Every node inside the largest repeated text block (the decoder layer
    recipe) — leaves and the fused modules that own weights themselves."""
    reps = [r for r in doc.get("repeats") or [] if not _VISION.search(r["parent"])]
    if not reps:
        return [n for n in doc["nodes"] if n["id"] and not _VISION.search(n["id"])]
    rep = max(reps, key=lambda r: r["count"] * index.by_id[r["representative"]]["params"])
    root = rep["representative"]
    return [n for n in doc["nodes"] if n["id"].startswith(root + ".")]


def profile(doc: dict) -> dict[str, Any]:
    c = text_config(doc)
    index = build_index(doc)
    rep = compute_costs(doc, index, Assumptions(T=4096, B=1, dtype="bf16"))
    leaves = _block_nodes(doc, index)
    leaf_names = {n["id"].split(".")[-1] for n in leaves}
    classes = {n["cls"] for n in doc["nodes"]}
    p: dict[str, Any] = {}
    p["model_type"] = c.get("model_type")
    p["params"] = float(doc.get("params_total") or 0)
    p["active"] = float(rep.root.active_params)
    p["layers"] = _num(c, "num_hidden_layers", "n_layer", "num_layers")
    p["hidden"] = _num(c, "hidden_size", "n_embd", "d_model")
    p["experts"] = _num(c, "num_experts", "n_routed_experts", "num_local_experts")
    # a MoE block's width is the expert width; intermediate_size is the dense MLP (often unused)
    p["ffn"] = (_num(c, "moe_intermediate_size") if p["experts"] else None) or _num(c, "intermediate_size", "n_inner", "d_ff")
    p["heads"] = _num(c, "num_attention_heads", "n_head", "num_heads")
    kv = _num(c, "num_key_value_heads")
    p["kv_heads"] = kv if kv is not None else p["heads"]
    p["head_dim"] = _num(c, "head_dim") or (p["hidden"] / p["heads"] if p["hidden"] and p["heads"] else None)
    p["kv_lora_rank"] = _num(c, "kv_lora_rank")
    p["kv_bytes"] = float(rep.root.kv_per_token)
    if p["kv_lora_rank"]:
        p["attention"] = "MLA"
    elif p["heads"] and p["kv_heads"] == 1:
        p["attention"] = "MQA"
    elif p["heads"] and p["kv_heads"] and p["kv_heads"] < p["heads"]:
        p["attention"] = "GQA"
    elif p["heads"]:
        p["attention"] = "MHA"
    else:
        p["attention"] = None
    p["top_k"] = _num(c, "num_experts_per_tok", "moe_top_k")
    p["shared_experts"] = _num(c, "n_shared_experts")
    # MLP shape from the block recipe: a gate projection means 3 matrices
    # (SwiGLU-style). Experts may be fused into one module whose weights are
    # 3-D (gate_up_proj / down_proj) — read the weight names too.
    weight_names = {w for n in leaves for w in (n.get("weight_shapes") or {})}
    names = leaf_names | weight_names
    p["gated_mlp"] = ("gate_up_proj" in names) or (
        any(n in names for n in ("gate_proj", "w1")) and any(n in names for n in ("up_proj", "w3"))
    )
    act = next((n["cls"] for n in leaves if n["kind"] == "module" and n["cls"].lower().endswith("activation")), None)
    if act is None:
        act = c.get("hidden_act") or c.get("activation_function")
    p["activation"] = _act_name(act)
    # positional scheme
    if any("Rotary" in k for k in classes) or c.get("rope_theta") or c.get("rope_scaling") or (isinstance(c.get("rope_parameters"), dict) and c["rope_parameters"].get("rope_theta")):
        p["positions"] = "RoPE"
    elif any(n["kind"] == "embedding" and re.search(r"(^|\.)(wpe|position_embeddings|pos_embed|embed_positions)$", n["id"]) for n in doc["nodes"]):
        p["positions"] = "learned"
    elif "relative_attention_bias" in {n["id"].split(".")[-1] for n in doc["nodes"]}:
        p["positions"] = "relative bias"
    elif any("Alibi" in k or "ALiBi" in k for k in classes) or c.get("alibi"):
        p["positions"] = "ALiBi"
    else:
        p["positions"] = None
    # transformers ≥ 5 folds rope_theta / rope_scaling into rope_parameters
    rp = c.get("rope_parameters") if isinstance(c.get("rope_parameters"), dict) else {}
    p["rope_theta"] = _num(c, "rope_theta") or _num(rp, "rope_theta")
    rs = c.get("rope_scaling") if isinstance(c.get("rope_scaling"), dict) else rp
    kind = (rs.get("rope_type") or rs.get("type")) if isinstance(rs, dict) else None
    p["rope_scaling"] = None if kind in (None, "default") else kind
    p["rope_factor"] = _num(rs, "factor") if isinstance(rs, dict) else None
    # norms
    norm_cls = {n["cls"] for n in doc["nodes"] if n["kind"] == "norm"}
    p["norm"] = "RMSNorm" if any("RMS" in k for k in norm_cls) else "LayerNorm" if norm_cls else None
    p["qk_norm"] = "q_norm" in leaf_names and "k_norm" in leaf_names
    p["attn_bias"] = any(
        n["kind"] == "linear" and n["id"].split(".")[-1] in ("q_proj", "k_proj", "v_proj", "c_attn", "qkv_proj")
        and "bias" in (n.get("weight_shapes") or {})
        for n in leaves
    )
    p["tied"] = c.get("tie_word_embeddings") is True
    p["context"] = _num(c, "max_position_embeddings", "n_positions")
    p["vocab"] = _num(c, "vocab_size")
    p["sliding_window"] = _num(c, "sliding_window")
    p["vision"] = bool(c.get("vision_config")) or any(_VISION.search(n["id"]) for n in doc["nodes"])
    p["encoder_decoder"] = bool(c.get("is_encoder_decoder"))
    return p


def _act_name(a: Any) -> str | None:
    if not a:
        return None
    s = str(a)
    s = re.sub(r"Activation$", "", s)
    s = {"NewGELU": "GELU (tanh)", "GELUTanh": "GELU (tanh)", "gelu_new": "GELU (tanh)", "gelu_pytorch_tanh": "GELU (tanh)",
         "SiLU": "SiLU", "silu": "SiLU", "swish": "SiLU", "GELU": "GELU", "gelu": "GELU", "ReLU": "ReLU", "relu": "ReLU",
         "QuickGELU": "QuickGELU", "quick_gelu": "QuickGELU"}.get(s, s)
    return s


def recipe(p: dict[str, Any]) -> list[str]:
    """One model's recipe as chips: what the zoo and cards show."""
    out: list[str] = []
    if p.get("experts"):
        out.append(f"MoE {int(p['top_k'] or 0)}/{int(p['experts'])}" + (f" +{int(p['shared_experts'])} shared" if p.get("shared_experts") else ""))
    else:
        out.append("dense")
    if p.get("attention"):
        att = p["attention"]
        if att == "GQA" and p.get("heads") and p.get("kv_heads"):
            att += f" {int(p['heads'] / p['kv_heads'])}×"
        out.append(att)
    if p.get("gated_mlp"):
        out.append(f"gated {p['activation'] or ''} MLP".replace("  ", " ").strip())
    elif p.get("activation"):
        out.append(f"{p['activation']} MLP")
    if p.get("positions"):
        out.append(p["positions"] + (f" ({p['rope_scaling']})" if p.get("rope_scaling") else ""))
    if p.get("norm"):
        out.append(p["norm"] + (" + q/k norm" if p.get("qk_norm") else ""))
    if p.get("attn_bias"):
        out.append("QKV bias")
    if p.get("tied"):
        out.append("tied embeddings")
    if p.get("sliding_window"):
        out.append(f"sliding window {int(p['sliding_window']):,}")
    return out


def _ratio(x: float, y: float) -> str:
    if not x or not y:
        return ""
    r = y / x
    if r >= 1.05:
        return f"{r:.1f}× larger" if r < 10 else f"{r:.0f}× larger"
    if r <= 0.95:
        return f"{1 / r:.1f}× smaller" if 1 / r < 10 else f"{1 / r:.0f}× smaller"
    return "about the same"


def _ctx(v: float | None) -> str:
    if not v:
        return "—"
    return f"{int(v):,}" if v < 8192 else f"{int(v) // 1024}k" if v < (1 << 20) else f"{v / (1 << 20):g}M"


def insights(doc_a: dict, doc_b: dict) -> list[dict[str, Any]]:
    """What changed from A to B, as quantified sentences. Each item carries
    a topic key, the sentence, and the two raw values it was built from."""
    a, b = profile(doc_a), profile(doc_b)
    out: list[dict[str, Any]] = []

    def add(topic: str, text: str, va: Any = None, vb: Any = None) -> None:
        out.append({"topic": topic, "text": text, "a": va, "b": vb})

    # size
    if a["params"] and b["params"] and abs(b["params"] / a["params"] - 1) > 0.02:
        add("params", f"{fmt_params(b['params'])} vs {fmt_params(a['params'])} parameters ({_ratio(a['params'], b['params'])}).",
            fmt_params(a["params"]), fmt_params(b["params"]))
    a_sparse = a["experts"] and a["active"] < 0.9 * a["params"]
    b_sparse = b["experts"] and b["active"] < 0.9 * b["params"]
    if b_sparse and not a_sparse:
        add("moe", f"B is a mixture of experts: {int(b['top_k'] or 0)} of {int(b['experts'])} experts run per token"
            + (f" (+{int(b['shared_experts'])} shared)" if b.get("shared_experts") else "")
            + f", so {fmt_params(b['active'])} of its {fmt_params(b['params'])} parameters are active — about {b['active'] / b['params'] * 100:.0f}% of the weights per token, versus A's dense MLP where all {fmt_params(a['params'])} run.",
            "dense", f"MoE {int(b['top_k'] or 0)}/{int(b['experts'])}")
    elif a_sparse and not b_sparse:
        add("moe", f"A is a mixture of experts ({int(a['top_k'] or 0)} of {int(a['experts'])} per token, {fmt_params(a['active'])} active); B is dense — every one of its {fmt_params(b['params'])} parameters runs for each token.",
            f"MoE {int(a['top_k'] or 0)}/{int(a['experts'])}", "dense")
    elif a_sparse and b_sparse and (a["experts"] != b["experts"] or a["top_k"] != b["top_k"]):
        add("moe", f"Routing changed: {int(b['top_k'] or 0)} of {int(b['experts'])} experts per token vs {int(a['top_k'] or 0)} of {int(a['experts'])} — {fmt_params(b['active'])} active vs {fmt_params(a['active'])}.",
            f"{int(a['top_k'] or 0)}/{int(a['experts'])}", f"{int(b['top_k'] or 0)}/{int(b['experts'])}")
    # depth / width
    if a["layers"] and b["layers"] and a["hidden"] and b["hidden"] and (a["layers"] != b["layers"] or a["hidden"] != b["hidden"]):
        deeper = "deeper" if b["layers"] > a["layers"] else "shallower" if b["layers"] < a["layers"] else "as deep"
        wider = "wider" if b["hidden"] > a["hidden"] else "narrower" if b["hidden"] < a["hidden"] else "as wide"
        add("shape", f"B is {deeper} and {wider}: {int(b['layers'])} layers × {int(b['hidden']):,} hidden vs {int(a['layers'])} × {int(a['hidden']):,}.",
            f"{int(a['layers'])}×{int(a['hidden'])}", f"{int(b['layers'])}×{int(b['hidden'])}")
    if a["ffn"] and b["ffn"] and a["ffn"] != b["ffn"] and a["hidden"] and b["hidden"]:
        lab = lambda p: ("expert" if p["experts"] else "MLP") + " width"  # noqa: E731
        add("ffn", f"{lab(b)} {int(b['ffn']):,} ({b['ffn'] / b['hidden']:.2g}× hidden) vs {lab(a)} {int(a['ffn']):,} ({a['ffn'] / a['hidden']:.2g}× hidden).",
            int(a["ffn"]), int(b["ffn"]))
    # attention / KV
    if a["attention"] and b["attention"] and not (a["vision"] != b["vision"] and None in (a["heads"], b["heads"])):
        def att_desc(p: dict) -> str:
            if p["attention"] == "MLA":
                return f"MLA (K/V compressed to rank {int(p['kv_lora_rank'])})"
            if p["attention"] == "MHA":
                return f"MHA ({int(p['heads'])} heads, every one with its own K/V)"
            return f"{p['attention']} ({int(p['kv_heads'])} KV heads for {int(p['heads'])} query heads)"
        if a["attention"] != b["attention"] or a["kv_heads"] != b["kv_heads"] or a["heads"] != b["heads"] or a["kv_lora_rank"] != b["kv_lora_rank"]:
            kvtxt = ""
            if a["kv_bytes"] and b["kv_bytes"]:
                kvtxt = f" → KV cache per token {fmt_bytes(b['kv_bytes'])} vs {fmt_bytes(a['kv_bytes'])} ({_ratio(a['kv_bytes'], b['kv_bytes'])})"
            add("attention", f"Attention: {att_desc(b)} vs {att_desc(a)}{kvtxt}.", att_desc(a), att_desc(b))
        elif a["kv_bytes"] and b["kv_bytes"] and abs(b["kv_bytes"] / a["kv_bytes"] - 1) > 0.05:
            add("kv", f"Same attention scheme, but the KV cache per token is {fmt_bytes(b['kv_bytes'])} vs {fmt_bytes(a['kv_bytes'])} ({_ratio(a['kv_bytes'], b['kv_bytes'])}) — from the layer count and head size.",
                fmt_bytes(a["kv_bytes"]), fmt_bytes(b["kv_bytes"]))
    if a["head_dim"] and b["head_dim"] and a["head_dim"] != b["head_dim"]:
        add("head_dim", f"Head size {int(b['head_dim'])} vs {int(a['head_dim'])}.", int(a["head_dim"]), int(b["head_dim"]))
    if a["qk_norm"] != b["qk_norm"]:
        add("qk_norm", ("B adds per-head RMSNorm on q and k before RoPE (training-stability trick; A has none)." if b["qk_norm"]
                        else "B drops the per-head q/k norms that A applies before RoPE."), a["qk_norm"], b["qk_norm"])
    if a["attn_bias"] != b["attn_bias"]:
        add("attn_bias", ("B adds bias terms to the q/k/v projections; A's are bias-free." if b["attn_bias"]
                          else "B drops the q/k/v projection biases A carries — fewer parameters, same shape."), a["attn_bias"], b["attn_bias"])
    if (a["sliding_window"] or 0) != (b["sliding_window"] or 0):
        add("window", (f"B attends within a sliding window of {int(b['sliding_window']):,} tokens; A attends over the full context." if b["sliding_window"] and not a["sliding_window"]
                       else f"A attends within a {int(a['sliding_window']):,}-token window; B over the full context." if a["sliding_window"] and not b["sliding_window"]
                       else f"Sliding window {int(b['sliding_window']):,} vs {int(a['sliding_window']):,} tokens."), a["sliding_window"], b["sliding_window"])
    # MLP
    if a["gated_mlp"] != b["gated_mlp"] or (a["activation"] != b["activation"] and a["activation"] and b["activation"]):
        def mlp_desc(p: dict) -> str:
            if p["gated_mlp"]:
                return f"gated {p['activation'] or ''} MLP (3 matrices: gate, up, down)".replace("  ", " ")
            return f"{p['activation'] or 'plain'} MLP (2 matrices)"
        add("mlp", f"MLP block: {mlp_desc(b)} vs {mlp_desc(a)}.", mlp_desc(a), mlp_desc(b))
    # positions
    if a["positions"] != b["positions"] and a["positions"] and b["positions"]:
        why = {"RoPE": "relative positions inside attention, extrapolates with scaling", "learned": "an absolute table, hard ceiling at the trained length",
               "relative bias": "learned per-bucket biases on the scores", "ALiBi": "a linear distance penalty on the scores"}
        add("positions", f"Positions: {b['positions']} ({why.get(b['positions'], '')}) vs {a['positions']} ({why.get(a['positions'], '')}).", a["positions"], b["positions"])
    elif a["positions"] == "RoPE" and b["positions"] == "RoPE":
        if a["rope_theta"] and b["rope_theta"] and a["rope_theta"] != b["rope_theta"]:
            add("rope", f"RoPE base {b['rope_theta']:,.0f} vs {a['rope_theta']:,.0f} — a higher base stretches the usable context.", a["rope_theta"], b["rope_theta"])
        if a["rope_scaling"] != b["rope_scaling"]:
            def rs_desc(p: dict) -> str:
                if not p["rope_scaling"]:
                    return "no RoPE scaling"
                return f"{p['rope_scaling']} RoPE scaling" + (f" (×{p['rope_factor']:g})" if p.get("rope_factor") else "")
            add("rope_scaling", f"{rs_desc(b)[0].upper() + rs_desc(b)[1:]} vs {rs_desc(a)} — scaling is how a model trained short reaches a long context.", rs_desc(a), rs_desc(b))
    # norm
    if a["norm"] != b["norm"] and a["norm"] and b["norm"]:
        add("norm", f"Normalization: {b['norm']} vs {a['norm']}" + (" — RMSNorm drops the mean-centering and the bias, cheaper and now standard." if b["norm"] == "RMSNorm" else "."), a["norm"], b["norm"])
    # context / vocab / tying
    if a["context"] and b["context"] and a["context"] != b["context"]:
        add("context", f"Trained context {_ctx(b['context'])} vs {_ctx(a['context'])} tokens.", _ctx(a["context"]), _ctx(b["context"]))
    if a["vocab"] and b["vocab"] and abs(b["vocab"] / a["vocab"] - 1) > 0.02:
        add("vocab", f"Vocabulary {int(b['vocab']):,} vs {int(a['vocab']):,} tokens" + (" — a bigger vocabulary packs more text per token, at the cost of a larger embedding/head." if b["vocab"] > a["vocab"] else "."), int(a["vocab"]), int(b["vocab"]))
    if a["tied"] != b["tied"]:
        add("tied", ("B ties the output head to the input embedding (stored once); A keeps them separate." if b["tied"]
                     else "B untied the output head from the input embedding (a separate matrix); A shares them."), a["tied"], b["tied"])
    if a["vision"] != b["vision"]:
        add("vision", ("B adds a vision tower feeding the language model; A is text-only." if b["vision"] else "A has a vision tower; B is text-only."), a["vision"], b["vision"])
    if a["encoder_decoder"] != b["encoder_decoder"]:
        add("seq2seq", ("B is an encoder–decoder; A is decoder-only." if b["encoder_decoder"] else "B is decoder-only; A is an encoder–decoder."), a["encoder_decoder"], b["encoder_decoder"])
    # headline size first, then the structural story, then the bookkeeping
    rank = {t: i for i, t in enumerate(TOPIC_ORDER)}
    out.sort(key=lambda it: rank.get(it["topic"], len(TOPIC_ORDER)))
    return out


TOPIC_ORDER = [
    "params", "moe", "attention", "kv", "mlp", "positions", "rope", "rope_scaling", "norm", "qk_norm",
    "attn_bias", "window", "head_dim", "vision", "seq2seq", "shape", "ffn", "context", "vocab", "tied",
]


def insights_markdown(items: list[dict[str, Any]]) -> list[str]:
    return [f"- {it['text']}" for it in items]
