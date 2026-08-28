"""Model-id normalization (M19): the shapes people actually paste — full Hub
URLs, ollama-style hf.co ids, typographic dashes from word processors, links
deep into a repo's file tree — all collapse to canonical owner/name[:variant]."""

import pytest

from modelmap.ids import normalize_model_id, parse_model_id, valid_hub_id


@pytest.mark.parametrize(
    ("pasted", "canonical"),
    [
        ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
        ("  Qwen/Qwen3-8B  ", "Qwen/Qwen3-8B"),
        ("https://huggingface.co/Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
        ("http://www.huggingface.co/Qwen/Qwen3-8B/", "Qwen/Qwen3-8B"),
        ("hf.co/Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
        # ollama's registry syntax, seen verbatim in the traffic logs
        ("hf.co/ryzdfm/some-model:Q4_K_M", "ryzdfm/some-model:Q4_K_M"),
        # links into the file tree
        ("https://huggingface.co/Qwen/Qwen3-8B/tree/main", "Qwen/Qwen3-8B"),
        ("https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json", "Qwen/Qwen3-8B"),
        ("https://huggingface.co/Qwen/Qwen3-8B?not-for-all-audiences=true", "Qwen/Qwen3-8B"),
        # a GGUF file path names the variant (seen in the logs: repo + quant stem)
        ("o/repo-GGUF/model-Q8_0.gguf", "o/repo-GGUF:model-Q8_0"),
        ("o/repo-GGUF/Ornith-1.5-35B-Q8_0", "o/repo-GGUF:Ornith-1.5-35B-Q8_0"),
        # a non-quant tail is repo-path noise, not a variant
        ("o/repo/model.safetensors", "o/repo"),
        ("a/b/c", "a/b"),
        # typographic dashes (U+2011 non-breaking hyphen, U+2013 en dash)
        ("Qwen/Qwen3.5‑27B", "Qwen/Qwen3.5-27B"),
        ("Qwen/Qwen3–8B", "Qwen/Qwen3-8B"),
        # zero-width junk from rich-text copies
        ("Qwen/​Qwen3-8B", "Qwen/Qwen3-8B"),
        # local ids pass through untouched
        ("local:/tmp/ckpt", "local:/tmp/ckpt"),
    ],
)
def test_normalize(pasted, canonical):
    assert normalize_model_id(pasted) == canonical


def test_parse_uses_the_canonical_spelling():
    src = parse_model_id("https://hf.co/Qwen/Qwen3-8B/tree/main")
    assert src.model_id == "Qwen/Qwen3-8B"  # the cache key and document id
    assert src.repo == "Qwen/Qwen3-8B"
    assert src.variant is None


def test_parse_gguf_path_becomes_variant():
    src = parse_model_id("hf.co/o/repo-GGUF/model-Q4_K_M.gguf")
    assert src.repo == "o/repo-GGUF"
    assert src.variant == "model-Q4_K_M"


def test_valid_hub_id_accepts_pasted_urls():
    assert valid_hub_id("https://huggingface.co/Qwen/Qwen3-8B")
    assert valid_hub_id("hf.co/o/name:Q4_0")
    assert not valid_hub_id("not a model id")
    assert not valid_hub_id(":::")


def test_gibberish_still_rejected():
    with pytest.raises(ValueError, match="must look like"):
        parse_model_id("owner/name with spaces")
