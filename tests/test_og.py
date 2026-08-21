"""Social cards (design doc §25): every shareable page draws a 1200×630 PNG
from the cached graph and the served index.html carries matching og:/twitter:
tags — crawlers never run JS, so the unfurl is decided server-side."""
import gzip
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from modelmap import og

FIX = Path(__file__).parent.parent / "web" / "tests" / "fixtures"


def load(name: str) -> dict:
    return json.loads(gzip.decompress((FIX / f"{name}.graph.json.gz").read_bytes()))


def _img(png: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(png))
    im.load()
    return im


@pytest.mark.parametrize("name", ["qwen3-8b", "deepseek-v3.1", "gpt2", "mixtral-8x7b"])
def test_model_card_is_a_full_size_png_that_actually_draws_the_model(name):
    doc = load(name)
    im = _img(og.render_model_card(doc))
    assert im.size == (og.W, og.H) and im.mode == "RGB"
    # not a blank canvas: the strip paints attention (pink) and MLP (teal) blocks
    px = im.getdata()
    pinkish = sum(1 for r, g, b in px if r > 90 and b > 60 and r > g + 20)
    tealish = sum(1 for r, g, b in px if g > r + 20 and b > r)
    assert pinkish > 2000 and tealish > 2000


def test_model_card_tags_and_stack_count_use_the_doc():
    # the rendering is deterministic; the same doc → the same bytes, a
    # different doc → different bytes (the text and strip depend on it)
    a = og.render_model_card(load("qwen3-8b"))
    assert a == og.render_model_card(load("qwen3-8b"))
    assert a != og.render_model_card(load("qwen2.5-7b"))


def test_other_cards_render():
    assert _img(og.render_default_card()).size == (og.W, og.H)
    fam = {"key": "qwen", "title": "Qwen", "blurb": "words " * 80,
           "members": ["Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B"]}
    entries = {"Qwen/Qwen3-8B": {"params_total": 8.19e9, "active_params": 8.19e9},
               "Qwen/Qwen3-30B-A3B": {"params_total": 30.5e9, "active_params": 3.3e9}}
    assert _img(og.render_family_card(fam, entries)).size == (og.W, og.H)
    assert _img(og.render_compare_card(load("qwen2.5-7b"), load("qwen3-8b"))).size == (og.W, og.H)


def test_meta_for_each_page_kind(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELMAP_CACHE", str(tmp_path))
    m = og.meta_for("/", {})
    assert m["image"].endswith("/og/default.png") and m["url"] == "https://modelmap.cc/"
    m = og.meta_for("/m/Qwen/Qwen3-8B", {"lens": "kv"})
    assert m["title"].startswith("Qwen/Qwen3-8B") and m["image"] == "https://modelmap.cc/og/m/Qwen/Qwen3-8B.png"
    assert m["url"] == "https://modelmap.cc/m/Qwen/Qwen3-8B"  # view state stays out of the canonical
    m = og.meta_for("/arch/qwen", {})
    assert "Qwen" in m["title"] and m["image"].endswith("/og/arch/qwen.png")
    m = og.meta_for("/arch/nope", {})
    assert m["image"].endswith("/og/default.png")
    m = og.meta_for("/compare/x/A...y/B", {})
    assert m["title"] == "x/A vs y/B" and "a=x/A&b=y/B" in m["image"] and m["url"].endswith("/compare/x/A...y/B")
    m = og.meta_for("/models", {}, site="https://maps.example.com")
    assert m["url"] == "https://maps.example.com/models"
    # ids are URL-encoded in the generated URLs
    m = og.meta_for("/m/o/has space", {})
    assert m["image"].endswith("/og/m/o/has%20space.png")


def test_inject_meta_escapes_and_replaces_title():
    html = '<head><title>modelmap</title><meta name="description" content="Interactive, animated architecture maps for any Hugging Face model" /></head><body></body>'
    out = og.inject_meta(html, og.meta_for("/m/<script>alert(1)</script>", {}))
    assert "<script>" not in out.split("<body>")[0]
    assert "&lt;script&gt;" in out
    assert out.count('property="og:image"') == 1 and 'twitter:card" content="summary_large_image"' in out
    assert "<title>&lt;script&gt;alert(1)&lt;/script&gt; — architecture map</title>" in out
    assert 'rel="canonical"' in out
