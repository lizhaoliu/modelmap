"""Hooked fake forward on the meta device (design doc §05, stage 3).

Meta tensors carry shape and dtype but no storage, so a full forward pass costs
milliseconds regardless of parameter count — and shape inference still runs,
which is exactly the data Flow mode needs.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch

from modelmap.schema import TraceStep

DEFAULT_SEQ_LEN = 7
MAX_SHAPES_PER_SIDE = 4
MAX_STEPS = 20_000


def run_trace(
    model, config, seq_len: int = DEFAULT_SEQ_LEN
) -> tuple[list[TraceStep], str, list[str]]:
    """Record per-module I/O shapes in execution order.

    Returns (steps, fidelity, notes): fidelity is "full" when the forward
    completed, "structural" when a meta-incompatible op faulted partway —
    steps up to the fault are kept (fallback ladder, rung 2).
    """
    steps: list[TraceStep] = []
    counter = itertools.count()

    def make_hook(name: str):
        def hook(module, args, kwargs, output):
            if len(steps) >= MAX_STEPS:
                return
            steps.append(TraceStep(
                step=next(counter),
                node=name,
                inputs=_tensor_shapes((args, kwargs)),
                outputs=_tensor_shapes(output),
            ))
        return hook

    handles = [
        m.register_forward_hook(make_hook(n), with_kwargs=True)
        for n, m in model.named_modules()
        if n
    ]
    inputs, notes = build_dummy_inputs(model, config, seq_len)
    fidelity = "full"
    try:
        with torch.no_grad():
            model(**inputs)
        if len(steps) >= MAX_STEPS:
            notes.append(
                f"trace truncated at {MAX_STEPS} steps (very high module count); "
                "flow replay covers the recorded prefix"
            )
    except Exception as e:  # data-dependent shapes, .item(), … — expected for e.g. MoE routing
        notes.append(
            f"traced forward faulted after {len(steps)} steps: {type(e).__name__}: {e}"
        )
        fidelity = "structural"
    finally:
        for h in handles:
            h.remove()

    # vision-language models: the text-only forward never runs the image
    # encoder, so trace it separately and put its steps first (vision.py)
    if getattr(config, "vision_config", None) is not None:
        from modelmap.vision import trace_vision_tower

        vsteps, vnotes = trace_vision_tower(model, config)
        notes = [n for n in notes if not n.startswith("vision tower not traced")] + vnotes
        if vsteps:
            for i, s in enumerate(vsteps + steps):
                s.step = i
            steps = vsteps + steps
    return steps, fidelity, notes


def build_dummy_inputs(model, config, seq_len: int) -> tuple[dict[str, Any], list[str]]:
    main = getattr(model, "main_input_name", "input_ids")
    if main == "pixel_values":
        return {"pixel_values": torch.zeros(_pixel_shape(config), device="meta")}, []
    if main == "input_values":  # audio encoders
        return {"input_values": torch.zeros((1, 16000), device="meta")}, []

    notes = []
    if getattr(config, "vision_config", None) is not None:
        # VLM pixel inputs are model-specific; the text path is traced here and
        # the vision tower separately (see vision.py) — this note is replaced
        notes.append("vision tower not traced (text-only dummy input)")
    return {
        "input_ids": torch.zeros((1, seq_len), dtype=torch.long, device="meta")
    }, notes


def _pixel_shape(config) -> tuple[int, int, int, int]:
    vc = getattr(config, "vision_config", None) or config
    size = getattr(vc, "image_size", 224)
    h, w = (size[0], size[-1]) if isinstance(size, (tuple, list)) else (size, size)
    return (1, getattr(vc, "num_channels", 3), h, w)


def _tensor_shapes(obj: Any) -> list[list[int]]:
    out: list[list[int]] = []
    _collect(obj, out)
    return out


def _collect(obj: Any, out: list[list[int]]) -> None:
    if len(out) >= MAX_SHAPES_PER_SIDE:
        return
    if isinstance(obj, torch.Tensor):
        out.append(list(obj.shape))
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _collect(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _collect(x, out)
    elif hasattr(obj, "to_tuple"):  # transformers ModelOutput
        _collect(obj.to_tuple(), out)
