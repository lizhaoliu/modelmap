import torch
from torch import nn

from modelmap.annotate import annotate, register_annotator
from modelmap.inputs import build_dummy_inputs, register_input_builder


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _M(nn.Module):
    main_input_name = "input_ids"


def test_seq2seq_gets_decoder_ids():
    kw, notes, name = build_dummy_inputs(_M(), _Cfg(is_encoder_decoder=True), 7)
    assert name == "text-seq2seq"
    assert set(kw) == {"input_ids", "decoder_input_ids"}
    assert kw["input_ids"].device.type == "meta" and kw["input_ids"].shape == (1, 7)


def test_speech_seq2seq_uses_mel_features():
    m = _M(); m.main_input_name = "input_features"
    kw, _, name = build_dummy_inputs(m, _Cfg(is_encoder_decoder=True, num_mel_bins=80, max_source_positions=1500), 5)
    assert name == "speech-seq2seq"
    assert tuple(kw["input_features"].shape) == (1, 80, 3000)
    assert "decoder_input_ids" in kw


def test_multimodal_takes_text_path():
    kw, notes, name = build_dummy_inputs(_M(), _Cfg(vision_config=object()), 7)
    assert name == "multimodal-text-path" and set(kw) == {"input_ids"}
    assert any("encoder towers traced separately" in n for n in notes)


def test_custom_builder_wins_when_first():
    @register_input_builder("custom", lambda m, c: getattr(c, "model_type", "") == "zzz", first=True)
    def _b(model, config, seq_len):
        return {"weird": torch.zeros((2, 2), device="meta")}

    kw, _, name = build_dummy_inputs(_M(), _Cfg(model_type="zzz"), 7)
    assert name == "custom" and "weird" in kw


def test_extra_repr_and_source_annotations():
    a = annotate("x", nn.Linear(4, 8, bias=False))
    assert a["in_features"] == "4" and a["out_features"] == "8" and a["bias"] == "False"
    assert a["_src"].startswith("torch/") and "github.com/pytorch/pytorch" in a["_src_url"]
    e = annotate("y", nn.Embedding(100, 16))
    assert e["num_embeddings"] == "100" and e["embedding_dim"] == "16"
    ln = annotate("z", nn.LayerNorm(16, eps=1e-6))
    assert ln["normalized_shape"] == "(16,)" and ln["eps"] == "1e-06"


def test_custom_annotator_merges_and_never_breaks():
    @register_annotator
    def _ok(name, module):
        return {"note": "hi"}

    @register_annotator
    def _boom(name, module):
        raise RuntimeError("annotators must not break extraction")

    assert annotate("q", nn.Identity())["note"] == "hi"
