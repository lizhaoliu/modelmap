"""Dummy-input builders: how to make a model *run* on the meta device.

I/O shapes are recorded generically by forward hooks on every module; the
only family-specific knowledge is what kwargs make the forward execute.
Builders are tried in registration order; the first whose predicate matches
wins. Add your own with @register_input_builder (see EXTENDING.md), e.g. from
a module named in $MODELMAP_PLUGINS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

DEFAULT_SEQ_LEN = 7

Predicate = Callable[[Any, Any], bool]          # (model, config) -> bool
Builder = Callable[[Any, Any, int], dict[str, Any]]  # (model, config, seq_len) -> kwargs


@dataclass
class InputBuilder:
    name: str
    predicate: Predicate
    build: Builder
    notes: list[str] = field(default_factory=list)


_BUILDERS: list[InputBuilder] = []


def register_input_builder(name: str, predicate: Predicate, *, first: bool = False, notes: list[str] | None = None):
    """Decorator. `first=True` puts the builder ahead of the built-ins."""
    def deco(fn: Builder) -> Builder:
        b = InputBuilder(name, predicate, fn, list(notes or []))
        _BUILDERS.insert(0, b) if first else _BUILDERS.append(b)
        return fn
    return deco


def build_dummy_inputs(model, config, seq_len: int = DEFAULT_SEQ_LEN) -> tuple[dict[str, Any], list[str], str]:
    """Returns (kwargs, notes, builder_name)."""
    for b in _BUILDERS:
        try:
            if b.predicate(model, config):
                return b.build(model, config, seq_len), list(b.notes), b.name
        except Exception:  # a broken predicate must not take the ladder down
            continue
    return {"input_ids": _ids(seq_len)}, [], "text"


# ------------------------------------------------------------------ helpers

def _ids(n: int, start: int | None = None) -> torch.Tensor:
    ids = torch.zeros((1, n), dtype=torch.long, device="meta")
    return ids


def _main_input(model) -> str:
    return getattr(model, "main_input_name", "input_ids")


def _cfg(config, *keys, default=None):
    for k in keys:
        v = getattr(config, k, None)
        if v is not None:
            return v
    return default


def _pixel_shape(config) -> tuple[int, int, int, int]:
    vc = getattr(config, "vision_config", None) or config
    size = _cfg(vc, "image_size", default=224)
    h, w = (size[0], size[-1]) if isinstance(size, (tuple, list)) else (size, size)
    return (1, _cfg(vc, "num_channels", "in_channels", default=3), h, w)


def _mel_shape(config) -> tuple[int, int, int]:
    ac = getattr(config, "audio_config", None) or config
    mel = _cfg(ac, "num_mel_bins", "input_feat_per_channel", default=80)
    frames = 2 * _cfg(ac, "max_source_positions", default=1500)  # Whisper: 30 s → 3000 frames
    return (1, mel, frames)


# ----------------------------------------------------------------- built-ins

@register_input_builder("image", lambda m, c: _main_input(m) == "pixel_values")
def _image(model, config, seq_len):
    return {"pixel_values": torch.zeros(_pixel_shape(config), device="meta")}


@register_input_builder("audio-waveform", lambda m, c: _main_input(m) == "input_values")
def _waveform(model, config, seq_len):
    return {"input_values": torch.zeros((1, 16000), device="meta")}


@register_input_builder(
    "speech-seq2seq",
    lambda m, c: _main_input(m) == "input_features" and bool(getattr(c, "is_encoder_decoder", False)),
)
def _speech_seq2seq(model, config, seq_len):
    # Whisper & friends: log-mel features in, decoder token ids out
    return {
        "input_features": torch.zeros(_mel_shape(config), device="meta"),
        "decoder_input_ids": _ids(seq_len),
    }


@register_input_builder("audio-features", lambda m, c: _main_input(m) == "input_features")
def _audio_features(model, config, seq_len):
    return {"input_features": torch.zeros(_mel_shape(config), device="meta")}


@register_input_builder(
    "text-seq2seq",
    lambda m, c: bool(getattr(c, "is_encoder_decoder", False)) and _main_input(m) == "input_ids",
)
def _text_seq2seq(model, config, seq_len):
    # T5/BART/mBART/…: BART shifts labels itself, T5 insists on decoder ids — give both
    return {"input_ids": _ids(seq_len), "decoder_input_ids": _ids(seq_len)}


@register_input_builder(
    "multimodal-text-path",
    lambda m, c: any(getattr(c, k, None) is not None for k in ("vision_config", "audio_config")),
    notes=["text path traced with input_ids; encoder towers traced separately (see towers.py)"],
)
def _multimodal(model, config, seq_len):
    return {"input_ids": _ids(seq_len)}
