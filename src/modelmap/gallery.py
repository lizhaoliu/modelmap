"""Landing gallery: what's trending on the Hub right now, plus a few classics.

`trending()` asks the Hub for its trending list, keeps ungated `transformers`
repos with a declared architecture, collapses quantization / derivative
re-uploads (FP8, GGUF, MLX, "uncensored"…), and caches the result for an
hour. `CLASSICS` are hand-picked references that always extract cleanly and
are baked into the container image."""

from __future__ import annotations

import logging
import re
import threading
import time

log = logging.getLogger(__name__)

CLASSICS: list[dict[str, str]] = [
    {"id": "openai-community/gpt2", "blurb": "The classic 124M decoder — the cleanest first read"},
    {"id": "Qwen/Qwen3-8B", "blurb": "Modern dense LLM: GQA, RMSNorm, gated MLP, 36 layers"},
    {"id": "Qwen/Qwen3-235B-A22B", "blurb": "Mixture of experts: 128 experts per layer, 235B params"},
    {"id": "deepseek-ai/DeepSeek-V3.1", "blurb": "671B MoE with multi-head latent attention"},
    {"id": "google-bert/bert-base-uncased", "blurb": "Encoder-only: the original bidirectional transformer"},
    {"id": "Qwen/Qwen2.5-VL-3B-Instruct", "blurb": "Vision-language: a vision tower feeding an LLM"},
]
CLASSIC_IDS = [g["id"] for g in CLASSICS]
# kept for older callers
GALLERY = CLASSICS
GALLERY_IDS = CLASSIC_IDS

TRENDING_LIMIT = 8
_TTL_S = 3600
_VARIANT = re.compile(
    r"(fp8|nvfp4|fp4|int4|int8|4bit|8bit|gguf|mlx|awq|gptq|exl2|bnb|uncensored|heretic|abliterated|-i1-|quant)",
    re.I,
)
_TAGS_OK = {
    "text-generation", "image-text-to-text", "text2text-generation", "fill-mask",
    "image-classification", "automatic-speech-recognition", "audio-text-to-text",
    "feature-extraction", "sentence-similarity", "any-to-any", "video-text-to-text",
}

_cache: dict[str, object] = {"ts": 0.0, "items": []}
_lock = threading.Lock()


def _fetch() -> list[dict]:
    from huggingface_hub import HfApi

    models = HfApi().list_models(
        sort="trendingScore", limit=60,
        expand=["trendingScore", "gated", "pipeline_tag", "library_name", "downloads", "likes", "config"],
    )
    out: list[dict] = []
    seen_base: set[str] = set()
    for m in models:
        if getattr(m, "gated", False):
            continue
        if getattr(m, "library_name", None) != "transformers":
            continue
        cfg = getattr(m, "config", None) or {}
        archs = cfg.get("architectures") if isinstance(cfg, dict) else None
        if not archs:
            continue
        tag = getattr(m, "pipeline_tag", None)
        if tag and tag not in _TAGS_OK:
            continue
        if _VARIANT.search(m.id):
            continue
        # collapse variants of one base model: same org + name up to a size suffix
        base = re.sub(r"-(instruct|chat|base|it|thinking|preview|\d{4})\b.*$", "", m.id.lower())
        if base in seen_base:
            continue
        seen_base.add(base)
        out.append({
            "id": m.id,
            "pipeline_tag": tag,
            "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None),
            "trending_score": getattr(m, "trendingScore", None),
            "architecture": archs[0],
        })
        if len(out) >= TRENDING_LIMIT:
            break
    return out


def trending(force: bool = False) -> list[dict]:
    with _lock:
        fresh = time.time() - float(_cache["ts"]) < _TTL_S
        if fresh and not force:
            return list(_cache["items"])  # type: ignore[arg-type]
    try:
        items = _fetch()
    except Exception as e:  # Hub hiccup: serve the stale list rather than nothing
        log.warning("trending fetch failed: %s", e)
        with _lock:
            return list(_cache["items"])  # type: ignore[arg-type]
    with _lock:
        _cache["ts"] = time.time()
        _cache["items"] = items
    return list(items)


def trending_ids() -> list[str]:
    return [t["id"] for t in trending()]
