"""Takeaways (design doc §27): derived, quantified sentences about what
changed between two architectures — one wording shared by /api/compare, the
compare page, the zoo lineage, `modelmap diff` and the MCP tool."""
import gzip
import json
from pathlib import Path

from modelmap.insights import insights, profile, recipe

FIX = Path(__file__).parent.parent / "web" / "tests" / "fixtures"


def load(name: str) -> dict:
    return json.loads(gzip.decompress((FIX / f"{name}.graph.json.gz").read_bytes()))


def by_topic(items):
    return {it["topic"]: it["text"] for it in items}


def test_profiles_read_the_recipe_off_the_graph():
    q3 = profile(load("qwen3-8b"))
    assert q3["attention"] == "GQA" and q3["kv_heads"] == 8 and q3["heads"] == 32
    assert q3["gated_mlp"] and q3["activation"] == "SiLU" and q3["positions"] == "RoPE" and q3["norm"] == "RMSNorm"
    assert q3["qk_norm"] and not q3["attn_bias"] and not q3["tied"]
    assert recipe(q3) == ["dense", "GQA 4×", "gated SiLU MLP", "RoPE", "RMSNorm + q/k norm"]
    g = profile(load("gpt2"))
    assert g["attention"] == "MHA" and not g["gated_mlp"] and g["activation"] == "GELU (tanh)"
    assert g["positions"] == "learned" and g["norm"] == "LayerNorm" and g["attn_bias"] and g["tied"]
    assert recipe(g) == ["dense", "MHA", "GELU (tanh) MLP", "learned", "LayerNorm", "QKV bias", "tied embeddings"]
    ds = profile(load("deepseek-v3.1"))
    assert ds["attention"] == "MLA" and ds["experts"] == 256 and ds["top_k"] == 8 and ds["shared_experts"] == 1
    assert ds["gated_mlp"]  # fused experts: read off the 3-D weight names
    assert recipe(ds)[0] == "MoE 8/256 +1 shared" and "MLA" in recipe(ds)
    mx = profile(load("mixtral-8x7b"))
    assert mx["gated_mlp"] and mx["ffn"] == 14336  # expert width, not the unused dense intermediate_size


def test_qwen25_to_qwen3_takeaways_are_quantified():
    t = by_topic(insights(load("qwen2.5-7b"), load("qwen3-8b")))
    assert t["params"] == "8.19B vs 7.62B parameters (1.1× larger)."
    assert "2.6× larger" in t["attention"] and "144 KB vs 56.0 KB" in t["attention"]
    assert "8 KV heads for 32 query heads" in t["attention"]
    assert t["qk_norm"].startswith("B adds per-head RMSNorm")
    assert t["attn_bias"].startswith("B drops the q/k/v projection biases")
    assert t["shape"] == "B is deeper and wider: 36 layers × 4,096 hidden vs 28 × 3,584."
    assert t["context"] == "Trained context 40k vs 128k tokens."
    assert "mlp" not in t and "norm" not in t and "positions" not in t  # same recipe there


def test_dense_to_moe_and_mha_to_gqa_stories():
    t = by_topic(insights(load("qwen3-8b"), load("mixtral-8x7b")))
    assert t["moe"].startswith("B is a mixture of experts: 2 of 8 experts run per token")
    assert "12.88B of its 46.70B" in t["moe"] and "28%" in t["moe"]
    assert "mlp" not in t  # both gated SiLU
    t = by_topic(insights(load("gpt2"), load("qwen3-8b")))
    assert "MHA (12 heads" in t["attention"] and "GQA (8 KV heads" in t["attention"] and "4.0× larger" in t["attention"]
    assert t["mlp"] == "MLP block: gated SiLU MLP (3 matrices: gate, up, down) vs GELU (tanh) MLP (2 matrices)."
    assert t["positions"].startswith("Positions: RoPE") and "learned" in t["positions"]
    assert t["norm"].startswith("Normalization: RMSNorm vs LayerNorm")
    assert t["tied"].startswith("B untied")
    t = by_topic(insights(load("mixtral-8x7b"), load("deepseek-v3.1")))
    assert t["moe"] == "Routing changed: 8 of 256 experts per token vs 2 of 8 — 37.55B active vs 12.88B."
    assert t["attention"].startswith("Attention: MLA (K/V compressed to rank 512) vs GQA") and "1.9× smaller" in t["attention"]


def test_identical_and_ordering():
    assert insights(load("qwen3-8b"), load("qwen3-8b")) == []
    items = insights(load("gpt2"), load("qwen3-8b"))
    topics = [it["topic"] for it in items]
    assert topics[0] == "params" and topics.index("attention") < topics.index("shape") < topics.index("vocab")
    # reverse direction flips the wording, never crashes
    rev = by_topic(insights(load("qwen3-8b"), load("gpt2")))
    assert rev["tied"].startswith("B ties") and "4.0× smaller" in rev["attention"]
