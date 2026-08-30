"""GGUF header parsing, variant grouping, name mapping — on a synthetic file (no network)."""
import struct

import pytest

from modelmap import gguf


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_str(k, v):
    return _s(k) + struct.pack("<I", 8) + _s(v)


def _kv_u32(k, v):
    return _s(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _kv_f32(k, v):
    return _s(k) + struct.pack("<I", 6) + struct.pack("<f", v)


def _kv_str_array(k, items):
    return _s(k) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(items)) + b"".join(_s(i) for i in items)


def _tensor(name, dims, ggml_type):
    return _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", ggml_type) + struct.pack("<Q", 0)


def build_gguf(vocab=100, hidden=64, layers=2, ffn=128, heads=4, kv_heads=2, tied=False) -> bytes:
    kv = [
        _kv_str("general.architecture", "qwen3"), _kv_str("general.name", "Tiny"), _kv_u32("general.file_type", 15),
        _kv_u32("qwen3.block_count", layers), _kv_u32("qwen3.context_length", 4096), _kv_u32("qwen3.embedding_length", hidden),
        _kv_u32("qwen3.feed_forward_length", ffn), _kv_u32("qwen3.attention.head_count", heads),
        _kv_u32("qwen3.attention.head_count_kv", kv_heads), _kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_u32("qwen3.attention.key_length", hidden // heads), _kv_u32("qwen3.attention.value_length", hidden // heads),
        _kv_str_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(vocab)]),
    ]
    tensors = [_tensor("token_embd.weight", [hidden, vocab], 12), _tensor("output_norm.weight", [hidden], 0)]
    if not tied:
        tensors.append(_tensor("output.weight", [hidden, vocab], 14))
    for i in range(layers):
        tensors += [
            _tensor(f"blk.{i}.attn_q.weight", [hidden, hidden], 12), _tensor(f"blk.{i}.attn_k.weight", [hidden, hidden // 2], 12),
            _tensor(f"blk.{i}.attn_v.weight", [hidden, hidden // 2], 14), _tensor(f"blk.{i}.attn_output.weight", [hidden, hidden], 12),
            _tensor(f"blk.{i}.ffn_gate.weight", [hidden, ffn], 12), _tensor(f"blk.{i}.ffn_up.weight", [hidden, ffn], 12),
            _tensor(f"blk.{i}.ffn_down.weight", [ffn, hidden], 14 if i % 2 else 12), _tensor(f"blk.{i}.attn_norm.weight", [hidden], 0),
        ]
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv)) + b"".join(kv) + b"".join(tensors)


def test_parse_header_grows_window_and_reverses_dims():
    data = build_gguf()
    calls = []

    def fetch(off, n):
        calls.append((off, n))
        return data[off : off + n]

    gguf.INITIAL_FETCH, saved = 64, gguf.INITIAL_FETCH  # force several growth rounds
    try:
        h = gguf.read_header(fetch)
    finally:
        gguf.INITIAL_FETCH = saved
    assert len(calls) > 1 and h.version == 3
    assert h.architecture == "qwen3" and h.file_type == "Q4_K_M"
    assert h.kv["_tokens_count"] == 100 and "tokenizer.ggml.tokens" not in h.kv
    emb = next(t for t in h.tensors if t.name == "token_embd.weight")
    assert emb.shape == [100, 64] and emb.dtype == "q4_k" and emb.numel == 6400  # ggml ne is innermost-first
    assert next(t for t in h.tensors if t.name == "output.weight").dtype == "q6_k"


def test_truncated_header_raises():
    data = build_gguf()[:200]
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(lambda off, n: data[off : off + n])
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(lambda off, n: b"NOPE" + b"\x00" * 100)


def test_config_from_header_builds_causal_lm_config():
    h = gguf.read_header(lambda off, n: build_gguf()[off : off + n])
    cfg, notes = gguf.config_from_header(h)
    assert cfg.model_type == "qwen3" and cfg.hidden_size == 64 and cfg.num_hidden_layers == 2
    assert cfg.vocab_size == 100 and cfg.num_key_value_heads == 2 and cfg.head_dim == 16
    assert cfg.architectures == ["Qwen3ForCausalLM"] and cfg.tie_word_embeddings is False
    tied = gguf.read_header(lambda off, n: build_gguf(tied=True)[off : off + n])
    assert gguf.config_from_header(tied)[0].tie_word_embeddings is True
    assert any("Q4_K_M" in n for n in notes)


def test_unknown_architecture_is_an_error():
    data = build_gguf().replace(b"qwen3", b"weird")
    h = gguf.read_header(lambda off, n: data[off : off + n])
    with pytest.raises(gguf.GGUFError):
        gguf.config_from_header(h)


def test_variants_and_choice():
    files = [
        "README.md", "Q/Qwen3-8B-Q4_K_M.gguf", "Q/Qwen3-8B-Q8_0-00001-of-00002.gguf", "Q/Qwen3-8B-Q8_0-00002-of-00002.gguf",
        "Q/Qwen3-8B-UD-Q4_K_XL.gguf", "Q/Qwen3-8B-BF16.gguf", "Q/Qwen3-8B-IQ4_XS.gguf",
    ]
    v = gguf.variants_of(files)
    assert set(v) == {"Q4_K_M", "Q8_0", "UD-Q4_K_XL", "BF16", "IQ4_XS"}
    assert v["Q8_0"] == ["Q/Qwen3-8B-Q8_0-00001-of-00002.gguf", "Q/Qwen3-8B-Q8_0-00002-of-00002.gguf"]
    assert gguf.choose_variant(v, None) == "Q4_K_M"
    assert gguf.choose_variant(v, "q8_0") == "Q8_0"
    assert gguf.choose_variant(v, "iq4") == "IQ4_XS"
    with pytest.raises(gguf.GGUFError):
        gguf.choose_variant(v, "Q2_K")
    # files without a quant token: distinguishing stems become the labels
    odd = gguf.variants_of(["DS-Flash-MTP-chat-v2.gguf", "DS-Flash-MTP-chat-v3.gguf"])
    assert set(odd) == {"2", "3"} or set(odd) == {"chat-v2", "chat-v3"}
    assert gguf.variants_of(["model.safetensors"]) == {}


def test_hf_module_mapping_and_dtypes():
    assert gguf.hf_module_for("blk.3.attn_q.weight") == "model.layers.3.self_attn.q_proj"
    assert gguf.hf_module_for("blk.0.ffn_down_exps.weight") == "model.layers.0.mlp.experts"
    assert gguf.hf_module_for("token_embd.weight") == "model.embed_tokens"
    assert gguf.hf_module_for("output.weight") == "lm_head"
    assert gguf.hf_module_for("blk.1.mystery.weight") is None
    from modelmap.schema import Node

    nodes = [
        Node(id="model.layers.0.self_attn.q_proj", kind="linear", cls="Linear", parent="x", depth=4, order=0, params=1, weight_shapes={"weight": [64, 64]}, dtype="bfloat16"),
        Node(id="model.layers.0.mlp.experts", kind="moe", cls="E", parent="x", depth=3, order=0, params=1, weight_shapes={"w": [4, 4, 4]}, dtype="bfloat16"),
        Node(id="model.norm", kind="norm", cls="N", parent="", depth=1, order=0, params=1, weight_shapes={"weight": [64]}, dtype="bfloat16"),
    ]
    h = gguf.read_header(lambda off, n: build_gguf()[off : off + n])
    h.tensors.append(gguf.GGUFTensor("blk.0.ffn_up_exps.weight", [4, 4, 4], "q4_k", 64))
    h.tensors.append(gguf.GGUFTensor("blk.0.ffn_down_exps.weight", [4, 4, 4], "q6_k", 64))
    notes = []
    gguf.apply_gguf_dtypes(nodes, [h], notes)
    assert nodes[0].dtype == "q4_k"
    assert nodes[1].dtype == "q4_k"  # the coarser of the fused experts' types
    assert nodes[2].dtype == "f32"
    assert notes == []


def test_choose_variant_from_a_pasted_file_stem():
    """M19: 'owner/repo/model-Q8_0' pastes arrive as variant 'model-Q8_0';
    the contained quant label decides, longest label winning."""
    from modelmap.gguf import choose_variant

    variants = {"Q8_0": ["a-Q8_0.gguf"], "BF16": ["a-BF16.gguf"], "F16": ["a-F16.gguf"]}
    assert choose_variant(variants, "Ornith-1.5-35B-Q8_0") == "Q8_0"
    assert choose_variant(variants, "model-BF16") == "BF16"


def test_read_window_caps_a_range_ignoring_server():
    """A server that ignores Range answers 200 with the WHOLE file; the fetch
    must return exactly the requested window and stop reading — not buffer
    16 GB into the worker (seen in prod when a repo turned gated)."""
    from modelmap.gguf import _read_window

    consumed = {"n": 0}

    class FullFile200:
        status_code = 200

        def iter_bytes(self):
            for i in range(10_000_000):  # would be ~640 GB if fully consumed
                consumed["n"] += 1
                yield bytes([i % 251]) * 65536

    body = b"".join(bytes([i % 251]) * 65536 for i in range(4))
    out = _read_window(FullFile200(), offset=100, length=1000)
    assert out == body[100:1100]
    assert consumed["n"] < 20  # stopped almost immediately


def test_read_window_passes_through_a_proper_206():
    from modelmap.gguf import _read_window

    class Partial206:
        status_code = 206

        def iter_bytes(self):
            yield b"abc"
            yield b"defgh"

    # a 206 body already starts at the requested offset: no skipping
    assert _read_window(Partial206(), offset=500, length=6) == b"abcdef"


def test_status_errors_are_plain_and_picklable():
    """httpx's HTTPStatusError drags live sockets across the worker's pickle
    boundary and kills the process; fetch must raise plain errors whose
    wording routes to the friendly answers (gated / rate-limited)."""
    import pickle

    from modelmap.gguf import _status_error

    gated = _status_error(401, "model-Q4_K_M.gguf", "https://huggingface.co/o/r/resolve/main/model-Q4_K_M.gguf")
    assert "gated" in str(gated)
    limited = _status_error(429, "f.gguf", "u")
    assert "429 Too Many Requests" in str(limited)
    other = _status_error(500, "f.gguf", "u")
    assert "HTTP 500" in str(other)
    for e in (gated, limited, other):
        assert str(pickle.loads(pickle.dumps(e))) == str(e)
