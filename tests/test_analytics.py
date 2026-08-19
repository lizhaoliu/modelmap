"""Cost analytics / compare / export in Python must agree with the web
implementation: these pin the same fixture numbers web/tests/cost.test.ts and
align.test.ts do."""
import gzip
import json
from pathlib import Path

import pytest

from modelmap import compare, export
from modelmap.analytics import (
    Assumptions, PlanRequest, build_index, compute_costs, module_rows, plan_serving, summarize,
)

FIX = Path(__file__).parent.parent / "web" / "tests" / "fixtures"


def load(name: str) -> dict:
    return json.loads(gzip.decompress((FIX / f"{name}.graph.json.gz").read_bytes()))


def run(name: str, T: int = 7):
    doc = load(name)
    idx = build_index(doc)
    return doc, idx, compute_costs(doc, idx, Assumptions(T=T))


def own_params(n):
    return sum(__import__("math").prod(w) for w in (n.get("weight_shapes") or {}).values())


def test_moe_active_params_qwen3_235b():
    doc, _, rep = run("qwen3-235b-a22b")
    assert 20 < rep.root.active_params / 1e9 < 24
    assert round(rep.root.param_bytes / 2 / 1e9) == round(doc["params_total"] / 1e9)
    assert "8 of 128 experts" in " ".join(rep.notes)


@pytest.mark.parametrize("name", ["qwen3-8b", "gpt2"])
def test_dense_macs_per_token_matches_non_embedding_params(name):
    doc, _, rep = run(name, 7)
    per_token = rep.root.macs / 7
    embed = sum(own_params(n) for n in doc["nodes"] if n["kind"] == "embedding")
    head = sum(own_params(n) for n in doc["nodes"] if n["kind"] == "head")
    expected = doc["params_total"] - embed + (head if doc["config"].get("tie_word_embeddings") is True else 0)
    assert 0.9 < per_token / expected < 1.1


def test_kv_cache_gqa_and_mla():
    _, _, q = run("qwen3-8b")
    assert q.root.kv_per_token == 36 * 2 * 8 * 128 * 2
    assert q.kv_layers == 36
    _, _, d = run("deepseek-v3.1")
    assert d.root.kv_per_token == 61 * (512 + 64) * 2


def test_attention_share_grows_with_T():
    _, _, short = run("qwen3-8b", 128)
    _, _, long = run("qwen3-8b", 32768)
    share = lambda r: r.by_node["model.layers.0.self_attn"].macs / r.by_node["model.layers.0"].macs
    assert share(long) > 2 * share(short)
    ns = short.by_node["model.layers.0.input_layernorm"].act_bytes
    nl = long.by_node["model.layers.0.input_layernorm"].act_bytes
    assert abs(nl / ns - 32768 / 128) < 1e-6


def test_summary_and_rows():
    doc = load("qwen3-8b")
    s = summarize(doc, Assumptions(T=4096))
    assert s["model_id"] == "Qwen/Qwen3-8B"
    assert s["layers"] == 36 and s["cost"]["kv_layers"] == 36
    assert s["cost"]["kv_bytes_at_T"] == 36 * 2 * 8 * 128 * 2 * 4096
    assert s["repeat_stacks"][0]["count"] == 36
    rows = module_rows(doc)
    by = {r["module"]: r for r in rows}
    assert by["model.layers.0"]["repeats"] == 36 and by["model.layers.0.self_attn.q_proj"]["multiplicity"] == 36
    assert by["model.layers.0.self_attn.q_proj"]["params_total_with_repeats"] == 36 * 4096 * 4096
    leaves = module_rows(doc, leaves_only=True)
    assert all(r["module"] != "model.layers.0" for r in leaves) and len(leaves) < len(rows)


def test_exports_render():
    doc = load("qwen3-8b")
    csv_text, mt = export.render(doc, "csv")
    assert mt == "text/csv" and csv_text.splitlines()[0].startswith("module,kind,class")
    assert "model.layers.0.self_attn.q_proj" in csv_text
    md, mt = export.render(doc, "md", Assumptions(T=8192))
    assert mt == "text/markdown" and md.startswith("# Qwen/Qwen3-8B") and "KV cache" in md and "T = 8,192" in md
    dot, mt = export.render(doc, "dot", depth=3)
    assert dot.startswith("digraph") and "ltail=\"cluster_model.layers\"" in dot and "\"model.embed_tokens\" -> " in dot
    with pytest.raises(ValueError):
        export.render(doc, "xlsx")


def test_planner_tp_pp_fits_and_max_context():
    doc = load("qwen3-8b")
    one = plan_serving(doc, PlanRequest(gpus=1, gpu_memory_gb=24, tp=1, pp=1, T=4096, B=1))
    assert one.fits and len(one.stages) == 1
    # weights 16.4 GB + KV 0.56 GB < 24 × 0.9
    assert 15e9 < one.weight_bytes < 17e9
    assert one.max_context_tokens > 4096
    two = plan_serving(doc, PlanRequest(gpus=2, gpu_memory_gb=24, tp=1, pp=2, T=4096))
    assert len(two.stages) == 2 and two.stages[0].layers[0] == 0 and two.stages[1].layers[1] == 35
    assert two.stages[0].layer_count + two.stages[1].layer_count == 36
    assert two.stages[0].boundary_bytes_out == 4096 * 4096 * 2  # B·T·hidden·bytes
    assert two.max_context_tokens > one.max_context_tokens
    tp2 = plan_serving(doc, PlanRequest(gpus=2, gpu_memory_gb=24, tp=2, pp=1, T=4096))
    assert abs(tp2.stages[0].weight_bytes_per_gpu - one.stages[0].weight_bytes_per_gpu / 2) < 1
    tiny = plan_serving(doc, PlanRequest(gpus=1, gpu_memory_gb=8, T=4096))
    assert not tiny.fits and tiny.max_context_tokens == 0
    # MoE: DeepSeek-style two stacks still partition to all 61 layers
    ds = plan_serving(load("deepseek-v3.1"), PlanRequest(gpus=8, gpu_memory_gb=80, tp=1, pp=8, T=4096))
    assert sum(s.layer_count for s in ds.stages) == 61


def test_compare_qwen25_vs_qwen3():
    a, b = load("qwen2.5-7b"), load("qwen3-8b")
    al = compare.align(a, b)
    added = {p.b for p in al.pairs if p.status == "added"}
    assert {"model.layers.0.self_attn.q_norm", "model.layers.0.self_attn.k_norm"} <= added
    by = {p.a: p for p in al.pairs if p.a}
    ch = {c.field: (c.a, c.b) for c in by["model.layers.0"].changes}
    assert ch["repeats"] == ("28", "36")
    ch = {c.field: (c.a, c.b) for c in by["model.layers.0.self_attn.q_proj"].changes}
    assert ch["bias"] == ("True", "False")
    assert al.counts["removed"] == 0
    md = compare.diff_markdown(a, b, al)
    assert "Qwen2.5-7B vs Qwen/Qwen3-8B" in md and "q_norm" in md and "| num_hidden_layers | 28 | 36 |" in md


def test_compare_base_vs_finetune_is_clean():
    al = compare.align(load("qwen3-8b-base"), load("qwen3-8b"))
    assert al.counts["added"] + al.counts["removed"] == 0
    assert not [p for p in al.pairs if any(c.field.startswith(("params", "weight", "input", "output", "repeats", "class")) for c in p.changes)]


def test_compare_unrelated_pairs_by_role():
    al = compare.align(load("gpt2"), load("qwen3-8b"))
    wte = next(p for p in al.pairs if p.a == "transformer.wte")
    assert wte.b == "model.embed_tokens"
    assert al.counts["same"] < len(al.pairs) / 2


def test_tied_lm_head_is_stored_once():
    doc = load("gpt2")  # tie_word_embeddings: true
    s = summarize(doc)
    assert s["active_params"] == doc["params_total"]
    assert s["cost"]["weight_bytes"] == doc["params_total"] * 4  # f32 checkpoint
    assert any("tied" in n for n in s["notes"])
