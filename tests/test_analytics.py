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


def test_trust_remote_code_local_checkpoint(tmp_path):
    """A local repo whose config demands custom code: refused by default
    (weights view / error), fully traced with trust_remote_code=True."""
    import textwrap

    from modelmap.extract import extract_graph

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "toycustom",
        "architectures": ["ToyModel"],
        "auto_map": {"AutoConfig": "modeling_toy.ToyConfig", "AutoModel": "modeling_toy.ToyModel"},
        "hidden_size": 8,
    }))
    (tmp_path / "modeling_toy.py").write_text(textwrap.dedent("""
        import torch
        from torch import nn
        from transformers import PretrainedConfig, PreTrainedModel

        class ToyConfig(PretrainedConfig):
            model_type = "toycustom"
            def __init__(self, hidden_size=8, **kw):
                self.hidden_size = hidden_size
                super().__init__(**kw)

        class ToyModel(PreTrainedModel):
            config_class = ToyConfig
            def __init__(self, config):
                super().__init__(config)
                self.embed = nn.Embedding(16, config.hidden_size)
                self.fc = nn.Linear(config.hidden_size, config.hidden_size)
            def forward(self, input_ids=None, **kw):
                return self.fc(self.embed(input_ids))
    """))
    import torch
    from safetensors.torch import save_file

    save_file({"embed.weight": torch.zeros(16, 8), "fc.weight": torch.zeros(8, 8)}, str(tmp_path / "model.safetensors"))
    refused = extract_graph(f"local:{tmp_path}", allow_local=True)
    assert refused.fidelity == "weights" and "trust_remote_code" in " ".join(refused.notes)

    g = extract_graph(f"local:{tmp_path}", allow_local=True, trust_remote_code=True)
    assert g.fidelity == "full"
    assert {n.id for n in g.nodes} >= {"embed", "fc"}
    assert len(g.trace) >= 2


def test_training_planner_lora_qlora_full():
    doc = load("qwen3-8b")
    from modelmap.analytics import TrainRequest, plan_training

    lora = plan_training(doc, TrainRequest(method="lora", lora_rank=16, lora_targets="attn-mlp", gpus=1, gpu_memory_gb=24, T=2048))
    # r × (in+out) over q/k/v/o + gate/up/down across 36 layers = 43.6M
    assert round(lora.trainable_params / 1e6, 1) == 43.6
    assert lora.grad_bytes_per_gpu == lora.trainable_params * 2
    assert lora.optimizer_bytes_per_gpu == lora.trainable_params * 12
    assert lora.fits  # bf16 base 16.4 GB + adapters + activations < 21.6 GiB

    qlora = plan_training(doc, TrainRequest(method="qlora", lora_rank=16, gpus=1, gpu_memory_gb=24, T=2048))
    assert qlora.weight_bytes_per_gpu < lora.weight_bytes_per_gpu / 3  # NF4 base
    assert qlora.fits and qlora.max_microbatch >= 8

    full = plan_training(doc, TrainRequest(method="full", gpus=1, gpu_memory_gb=80, T=2048))
    # 16 B/param convention: 2 w + 2 g + 12 optim
    assert full.weight_bytes_per_gpu + full.grad_bytes_per_gpu + full.optimizer_bytes_per_gpu == doc["params_total"] * 16
    assert not full.fits
    z3 = plan_training(doc, TrainRequest(method="full", gpus=8, gpu_memory_gb=80, sharding="zero3", T=2048))
    assert z3.fits and z3.weight_bytes_per_gpu == full.weight_bytes_per_gpu / 8

    ckpt = plan_training(doc, TrainRequest(method="lora", gpus=1, gpu_memory_gb=24, T=2048, grad_checkpoint=True))
    nockpt = plan_training(doc, TrainRequest(method="lora", gpus=1, gpu_memory_gb=24, T=2048, grad_checkpoint=False))
    assert nockpt.activation_bytes_per_gpu > 3 * ckpt.activation_bytes_per_gpu
    noflash = plan_training(doc, TrainRequest(method="lora", gpus=1, gpu_memory_gb=24, T=2048, grad_checkpoint=False, flash_attention=False))
    # scores: 36 layers × 32 heads × 2048² × 2 B = 9.0 GiB on top
    assert noflash.activation_bytes_per_gpu - nockpt.activation_bytes_per_gpu == 36 * 32 * 2048 * 2048 * 2


def test_throughput_roofline():
    doc = load("qwen3-8b")
    from modelmap.analytics import estimate_throughput

    t = estimate_throughput(doc, "A100 80GB", T=4096)
    # decode B=1: 2039 GB/s × 0.6 / (16.38 GB weights + 0.59 GB KV) ≈ 72 tok/s
    assert 65 < t.decode_tok_per_sec_b1 < 80
    assert 6000 < t.prefill_tok_per_sec < 8500
    t8 = estimate_throughput(doc, "A100 80GB", T=4096, B=8)
    assert t8.decode_tok_per_sec_at_b > 5 * t.decode_tok_per_sec_b1  # weights amortize
    assert estimate_throughput(doc, "GTX 9999", T=4096) is None
    moe = estimate_throughput(load("qwen3-235b-a22b"), "H100 80GB", tp=8, T=4096)
    assert any("MoE" in n for n in moe.notes)
    # active fraction ≈ 22/235: decode reads ~44 GB not 470 GB
    assert moe.bytes_read_per_token < 60e9


def test_zoo_tags_families_and_catalog(tmp_path, monkeypatch):
    from modelmap import zoo

    doc = load("qwen3-235b-a22b")
    tags = zoo.structural_tags(doc)
    assert "moe 8/128" in tags and any(t.startswith("gqa") for t in tags)
    dense = zoo.structural_tags(load("qwen3-8b"))
    assert "gqa 4×" in dense and not any(t.startswith("moe") for t in dense)
    ds = zoo.structural_tags(load("deepseek-v3.1"))
    assert "mla" in ds and "mixed-blocks" in ds
    assert zoo.family_of("qwen3_moe") == "qwen" and zoo.family_of("deepseek_v4") == "deepseek"
    assert zoo.family_of("gemma3_text") == "gemma" and zoo.family_of("somefuture2") == "somefuture"

    # catalog: newest schema version wins, one row per model
    monkeypatch.setenv("MODELMAP_CACHE", str(tmp_path))
    from modelmap import cache

    old = dict(load("qwen3-8b"))
    old["schema_version"] = 1
    cache.put("Qwen/Qwen3-8B", "old", old)
    cache.put("Qwen/Qwen3-8B", "main", load("qwen3-8b"))
    cache.put("openai-community/gpt2", "main", load("gpt2"))
    zoo._cat.update(count=-1, ts=0.0)  # bust the memo
    cat = zoo.catalog()
    ids = [e["model_id"] for e in cat]
    assert sorted(ids) == ["Qwen/Qwen3-8B", "openai-community/gpt2"]
    q = next(e for e in cat if e["model_id"] == "Qwen/Qwen3-8B")
    assert q["layers"] == 36 and q["family"] == "qwen" and q["kv_bytes_per_token"] == 36 * 2 * 8 * 128 * 2
