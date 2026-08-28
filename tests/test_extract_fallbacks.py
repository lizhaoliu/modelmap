"""The fallback ladder's failure manners (M19): gated repos surface their
actionable message instead of degrading into "no headers"; unknown
architectures get an honest banner instead of transformers' pip-upgrade
advice; pickle-only repos say what they hold and why that's unreadable."""

from types import SimpleNamespace

import pytest

import modelmap.extract as extract
import modelmap.weights_view as weights_view
from modelmap.extract import _degrade_note, extract_graph

GatedRepoError = type("GatedRepoError", (Exception,), {})


def _gated_oserror() -> OSError:
    try:
        raise GatedRepoError("401 Client Error. Cannot access gated repo")
    except GatedRepoError as inner:
        try:
            raise OSError(
                "You are trying to access a gated repo.\nMake sure to have access to it "
                "at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct."
            ) from inner
        except OSError as e:
            return e


# ---------------------------------------------------------------- degrade notes


def test_unknown_model_type_note_is_honest():
    e = ValueError(
        "The checkpoint you are trying to load has model type `breeze` but Transformers "
        "does not recognize this architecture. This could be because of an issue with the "
        "checkpoint, or because your version of Transformers is out of date.\n\n"
        "You can update Transformers with the command `pip install --upgrade transformers`."
    )
    note = _degrade_note(e)
    assert "model type `breeze` isn't in transformers" in note
    assert "vendor" in note
    assert "pip install" not in note  # upgrading doesn't help for vendor-only archs


def test_trust_remote_code_note():
    assert "trust_remote_code" in _degrade_note(ValueError("requires trust_remote_code=True"))


def test_generic_note_drops_the_upgrade_boilerplate():
    e = ValueError("something odd. You can update Transformers with `pip install -U transformers`.")
    note = _degrade_note(e)
    assert note.startswith("not a transformers-loadable config (something odd")
    assert "pip install" not in note


# ------------------------------------------------------------ gated propagation


def test_gated_config_raises_instead_of_degrading(monkeypatch):
    """A gated config must NOT fall through to the weights view — every header
    read would 403 too, and the masked error hides the fix (accept + token)."""
    monkeypatch.setattr(
        extract.AutoConfig, "from_pretrained",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(_gated_oserror())),
    )
    with pytest.raises(OSError, match="gated repo"):
        extract_graph("meta-llama/Llama-3.2-1B-Instruct")


def test_weights_view_propagates_gated_headers(monkeypatch):
    """No config.json at all (LTX-style bundle): the per-file header reads hit
    GatedRepoError, which must propagate, not count as a parse failure."""
    monkeypatch.setattr(
        weights_view, "get_safetensors_metadata",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("not a safetensors repo")),
    )
    monkeypatch.setattr(
        weights_view.HfApi, "list_repo_files",
        lambda self, *a, **k: ["transformer/model.safetensors"],
    )
    monkeypatch.setattr(
        weights_view, "parse_safetensors_file_metadata",
        lambda *a, **k: (_ for _ in ()).throw(GatedRepoError("401 Cannot access gated repo")),
    )
    with pytest.raises(GatedRepoError):
        weights_view.weights_graph("Lightricks/LTX-2.5")


# ------------------------------------------------------- subfolder safetensors


def _info(shape, dtype="BF16"):
    return SimpleNamespace(shape=shape, dtype=dtype)


def test_subfolder_files_join_the_root_index(monkeypatch):
    """Breeze-TTS-style repos: the root shards hold the backbone, a subfolder
    holds the audio tokenizer — the map must show both."""
    root = SimpleNamespace(files_metadata={
        "model-00001-of-00001.safetensors": SimpleNamespace(
            tensors={"backbone.layers.0.w.weight": _info([4, 4])}
        ),
    })
    monkeypatch.setattr(weights_view, "get_safetensors_metadata", lambda *a, **k: root)
    monkeypatch.setattr(
        weights_view.HfApi, "list_repo_files",
        lambda self, *a, **k: ["model-00001-of-00001.safetensors", "audio_tokenizer/model.safetensors", "README.md"],
    )
    monkeypatch.setattr(
        weights_view, "parse_safetensors_file_metadata",
        lambda mid, f, **k: SimpleNamespace(tensors={"encoder.w.weight": _info([2, 2])}),
    )
    notes: list[str] = []
    tensors = weights_view._collect_tensors("o/tts", "main", None, notes)
    assert "backbone.layers.0.w.weight" in tensors
    assert "audio_tokenizer.encoder.w.weight" in tensors  # folder-namespaced


def test_listing_hiccup_keeps_the_indexed_tensors(monkeypatch):
    root = SimpleNamespace(files_metadata={
        "model.safetensors": SimpleNamespace(tensors={"w.weight": _info([2, 2])}),
    })
    monkeypatch.setattr(weights_view, "get_safetensors_metadata", lambda *a, **k: root)
    monkeypatch.setattr(
        weights_view.HfApi, "list_repo_files",
        lambda self, *a, **k: (_ for _ in ()).throw(TimeoutError("hub timeout")),
    )
    monkeypatch.setattr("modelmap.hubio.time.sleep", lambda _: None)
    tensors = weights_view._collect_tensors("o/m", "main", None, [])
    assert list(tensors) == ["w.weight"]


def test_colliding_variant_files_are_noted(monkeypatch):
    """LTX-style bundles ship one component at several precisions in one
    folder; the map keeps one copy and says so."""
    monkeypatch.setattr(
        weights_view, "get_safetensors_metadata",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no root index")),
    )
    monkeypatch.setattr(
        weights_view.HfApi, "list_repo_files",
        lambda self, *a, **k: ["t/model-bf16.safetensors", "t/model-int8.safetensors"],
    )
    monkeypatch.setattr(
        weights_view, "parse_safetensors_file_metadata",
        lambda mid, f, **k: SimpleNamespace(tensors={"blocks.0.w.weight": _info([2, 2])}),
    )
    notes: list[str] = []
    tensors = weights_view._collect_tensors("o/bundle", "main", None, notes)
    assert len(tensors) == 1
    assert any("more than one file" in n for n in notes)


# ---------------------------------------------------------- pickle-only repos


def test_pickle_only_repo_message(monkeypatch):
    from modelmap.extract import _weights_only
    from modelmap.ids import parse_model_id

    monkeypatch.setattr(
        weights_view, "weights_graph",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("'o/yolo' has no safetensors files to build a weights view from")),
    )
    with pytest.raises(ValueError, match=r"pickle checkpoints .* without downloading and executing"):
        _weights_only(parse_model_id("o/yolo"), "main", None, [], "pytorch")


def test_nothing_readable_message(monkeypatch):
    from modelmap.extract import _weights_only
    from modelmap.ids import parse_model_id

    monkeypatch.setattr(
        weights_view, "weights_graph",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("'o/onnx' has no safetensors files to build a weights view from")),
    )
    with pytest.raises(ValueError, match="no transformers-loadable config, no safetensors, and no GGUF"):
        _weights_only(parse_model_id("o/onnx"), "main", None, [], None)
