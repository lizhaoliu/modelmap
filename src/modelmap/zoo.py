"""The architecture zoo (design doc §23): structural facts across every
cached graph, browsable as a catalog (/api/models → /models) and as curated
family pages (/arch/<family>) whose lineage steps are live structural diffs.

Everything here is derived from cached graph documents — no Hub calls."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from modelmap import cache
from modelmap.analytics import LINEAR_ATTN, Assumptions, _num, build_index, compute_costs

log = logging.getLogger(__name__)

# model_type → family key (first-token heuristic with overrides)
FAMILY_OF: dict[str, str] = {
    "qwen2": "qwen", "qwen2_moe": "qwen", "qwen2_vl": "qwen", "qwen2_5_vl": "qwen",
    "qwen3": "qwen", "qwen3_moe": "qwen", "qwen3_vl": "qwen", "qwen3_vl_moe": "qwen",
    "llama": "llama", "llama4": "llama", "mllama": "llama",
    "mistral": "mistral", "mixtral": "mistral", "mistral3": "mistral",
    "deepseek_v2": "deepseek", "deepseek_v3": "deepseek", "deepseek_v4": "deepseek", "deepseek_vl": "deepseek",
    "gpt2": "gpt", "gpt_oss": "gpt", "gpt_neo": "gpt", "gpt_neox": "gpt", "gptj": "gpt",
    "gemma": "gemma", "gemma2": "gemma", "gemma3": "gemma", "gemma3_text": "gemma", "gemma4": "gemma", "gemma4_text": "gemma",
    "bert": "bert", "roberta": "bert", "distilbert": "bert", "modernbert": "bert",
    "t5": "t5", "umt5": "t5", "mt5": "t5",
    "whisper": "whisper",
    "phi": "phi", "phi3": "phi", "phi4": "phi", "phimoe": "phi",
}


def family_of(model_type: str | None) -> str | None:
    if not model_type:
        return None
    if model_type in FAMILY_OF:
        return FAMILY_OF[model_type]
    return model_type.split("_")[0].rstrip("0123456789.")


# Curated family pages: members in release order — each arrow in the lineage
# is rendered as a live structural diff. Every id must extract without a
# token and without trust_remote_code.
FAMILIES: list[dict[str, Any]] = [
    {
        "key": "qwen",
        "title": "Qwen",
        "blurb": "Alibaba's line shows the modern dense recipe converging — then going sparse. "
        "Qwen2 → 2.5 keeps the shape and retunes widths; Qwen3 drops the projection biases and adds "
        "per-head q/k norms; the MoE variants swap each MLP for 128 routed experts.",
        "members": [
            "Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B",
        ],
        "extra": ["Qwen/Qwen2.5-VL-3B-Instruct"],
    },
    {
        "key": "llama",
        "title": "Llama lineage",
        "blurb": "The architecture most others are measured against: RMSNorm, rotary embeddings, GQA, "
        "SwiGLU. SmolLM2 is the same design at 1/60th scale; most 'llama-style' models on the Hub "
        "differ only in widths and vocabulary.",
        "members": ["HuggingFaceTB/SmolLM2-135M", "NousResearch/Meta-Llama-3.1-8B"],
        "extra": ["Maykeye/TinyLLama-v0"],
    },
    {
        "key": "deepseek",
        "title": "DeepSeek",
        "blurb": "Aggressive attention-memory engineering: V3 compresses the KV cache with multi-head "
        "latent attention (61 layers, 256 experts); V4-Flash interleaves two block designs — "
        "sliding-window and compressed attention — with hyper-connections.",
        "members": ["deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V4-Flash"],
    },
    {
        "key": "gpt",
        "title": "GPT",
        "blurb": "Where the decoder recipe started: learned positions, LayerNorm with biases, fused "
        "QKV as a Conv1D. Comparing gpt2 with anything modern shows fifteen years of small deletions.",
        "members": ["openai-community/gpt2", "openai-community/gpt2-medium", "distilbert/distilgpt2"],
    },
    {
        "key": "mistral",
        "title": "Mistral",
        "blurb": "Llama-shaped with sliding-window attention; Mixtral made sparse MoE mainstream by "
        "swapping each MLP for 8 experts with top-2 routing.",
        "members": ["mistralai/Mistral-7B-v0.3", "mistralai/Mixtral-8x7B-v0.1"],
    },
    {
        "key": "bert",
        "title": "BERT & encoders",
        "blurb": "Encoder-only: bidirectional attention, no causal mask, no KV cache. RoBERTa is the "
        "same graph retrained; DistilBERT halves the layers.",
        "members": ["google-bert/bert-base-uncased", "FacebookAI/roberta-base", "distilbert/distilbert-base-uncased"],
    },
    {
        "key": "t5",
        "title": "T5",
        "blurb": "The encoder-decoder workhorse: relative position biases instead of absolute "
        "embeddings, cross-attention bridging the two stacks.",
        "members": ["google-t5/t5-base", "google/flan-t5-base"],
    },
    {
        "key": "whisper",
        "title": "Whisper",
        "blurb": "Speech as translation: a mel-spectrogram encoder (conv downsampling, then a plain "
        "transformer) feeding a text decoder through cross-attention.",
        "members": ["openai/whisper-base", "openai/whisper-large-v3-turbo"],
    },
]

ZOO_IDS: list[str] = [m for f in FAMILIES for m in f["members"] + f.get("extra", [])]


def structural_tags(doc: dict) -> list[str]:
    """Short structural facts the catalog filters on — derived, never typed in."""
    c = dict(doc.get("config") or {})
    tc = c.get("text_config")
    if isinstance(tc, dict):
        for k, v in tc.items():
            c.setdefault(k, v)
    tags: list[str] = []
    heads = _num(c, "num_attention_heads", "n_head")
    kv = _num(c, "num_key_value_heads")
    experts = _num(c, "num_experts", "n_routed_experts", "num_local_experts")
    topk = _num(c, "num_experts_per_tok", "moe_top_k")
    if experts:
        tags.append(f"moe {int(topk) if topk else '?'}/{int(experts)}")
    if _num(c, "kv_lora_rank"):
        tags.append("mla")
    elif heads and kv and kv < heads:
        tags.append(f"gqa {int(heads / kv)}×")
    elif heads and kv == heads:
        tags.append("mha")
    if c.get("vision_config"):
        tags.append("vlm")
    if c.get("audio_config") or (c.get("model_type") or "").startswith(("whisper", "qwen2_audio")):
        tags.append("audio")
    if c.get("is_encoder_decoder"):
        tags.append("seq2seq")
    if _num(c, "sliding_window"):
        tags.append("sliding-window")
    if any(LINEAR_ATTN.search(n.get("cls") or "") for n in doc.get("nodes") or []):
        tags.append("linear-attn")
    if c.get("tie_word_embeddings") is True:
        tags.append("tied-embeddings")
    q = c.get("quantization_config")
    if isinstance(q, dict) and q.get("quant_method"):
        tags.append(f"quant:{q['quant_method']}")
    if doc.get("variant"):
        tags.append(f"gguf:{doc['variant']}")
    reps = doc.get("repeats") or []
    by_parent: dict[str, int] = {}
    for r in reps:
        by_parent[r["parent"]] = by_parent.get(r["parent"], 0) + 1
    if any(v > 1 for v in by_parent.values()):
        tags.append("mixed-blocks")  # two+ repeated block designs in one stack
    ctx = _num(c, "max_position_embeddings", "n_positions")
    if ctx and ctx >= 32768:
        tags.append(f"ctx {int(ctx / 1024)}k" if ctx < 1 << 20 else f"ctx {ctx / (1 << 20):g}M")
    return tags


def catalog_entry(doc: dict) -> dict[str, Any]:
    c = dict(doc.get("config") or {})
    # VLMs nest the language model's numbers under text_config
    tc = c.get("text_config")
    if isinstance(tc, dict):
        for k, v in tc.items():
            c.setdefault(k, v)
    index = build_index(doc)
    rep = compute_costs(doc, index, Assumptions(T=4096, B=1, dtype="bf16"))
    return {
        "model_id": doc.get("model_id"),
        "architecture": doc.get("architecture"),
        "model_type": c.get("model_type"),
        "family": family_of(c.get("model_type")),
        "fidelity": doc.get("fidelity"),
        "params_total": doc.get("params_total"),
        "active_params": round(rep.root.active_params),
        "layers": _num(c, "num_hidden_layers", "n_layer"),
        "hidden": _num(c, "hidden_size", "n_embd"),
        "heads": _num(c, "num_attention_heads", "n_head"),
        "kv_heads": _num(c, "num_key_value_heads"),
        "context": _num(c, "max_position_embeddings", "n_positions"),
        "vocab": _num(c, "vocab_size"),
        "kv_bytes_per_token": rep.root.kv_per_token,
        "macs_per_token": rep.root.macs / 4096,
        "tags": structural_tags(doc),
    }


# ---- catalog over the cache, memoized (rebuilds when entries appear)

_cat_lock = threading.Lock()
_cat: dict[str, Any] = {"count": -1, "ts": 0.0, "entries": []}
_TTL_S = 300


def catalog() -> list[dict[str, Any]]:
    """One entry per cached graph. Rebuilt when the cache grows or every
    _TTL_S; each rebuild is a few ms per entry (decompress + analytics)."""
    n = cache.count()
    with _cat_lock:
        if n == _cat["count"] and time.time() - _cat["ts"] < _TTL_S:
            return _cat["entries"]
    best: dict[tuple, tuple[int, dict[str, Any]]] = {}
    for p in sorted(cache.cache_dir().glob("*.json.gz")):
        try:
            import gzip
            import json

            doc = json.loads(gzip.decompress(p.read_bytes()))
            # the cache keeps entries from older schema versions around;
            # one row per (model, variant), newest schema wins
            key = (doc.get("model_id"), doc.get("variant"))
            sv = int(doc.get("schema_version") or 0)
            if key in best and best[key][0] >= sv:
                continue
            best[key] = (sv, catalog_entry(doc))
        except Exception as e:  # a corrupt entry must not break the catalog
            log.warning("catalog skipped %s: %s", p.name, e)
    entries = [e for _, e in best.values()]
    entries.sort(key=lambda e: -(e.get("params_total") or 0))
    with _cat_lock:
        _cat.update(count=n, ts=time.time(), entries=entries)
    return entries


def families_payload() -> list[dict[str, Any]]:
    """The curated families with each member's catalog entry (when cached)."""
    by_id = {e["model_id"]: e for e in catalog()}
    out = []
    for f in FAMILIES:
        out.append({
            "key": f["key"],
            "title": f["title"],
            "blurb": f["blurb"],
            "members": [{"id": m, "entry": by_id.get(m)} for m in f["members"]],
            "extra": [{"id": m, "entry": by_id.get(m)} for m in f.get("extra", [])],
        })
    return out
