"""Curated, ungated models for the landing gallery and cache pre-warming.

Every entry must extract without a token so the hosted deployment can warm
them at startup. Kept small on purpose: the gallery is a first-ten-seconds
experience, not a catalog."""

GALLERY: list[dict[str, str]] = [
    {"id": "openai-community/gpt2", "blurb": "The classic 124M decoder — the cleanest first read"},
    {"id": "Qwen/Qwen3-8B", "blurb": "Modern dense LLM: GQA, RMSNorm, gated MLP, 36 layers"},
    {"id": "Qwen/Qwen3-235B-A22B", "blurb": "Mixture of experts: 128 experts per layer, 235B params"},
    {"id": "deepseek-ai/DeepSeek-V3.1", "blurb": "671B MoE with multi-head latent attention"},
    {"id": "NousResearch/Meta-Llama-3.1-8B", "blurb": "Llama 3.1 architecture (ungated mirror)"},
    {"id": "google-bert/bert-base-uncased", "blurb": "Encoder-only: the original bidirectional transformer"},
    {"id": "google/vit-base-patch16-224", "blurb": "Vision transformer: patches in, class logits out"},
    {"id": "Qwen/Qwen2.5-VL-3B-Instruct", "blurb": "Vision-language: a vision tower feeding an LLM"},
]

GALLERY_IDS = [g["id"] for g in GALLERY]
