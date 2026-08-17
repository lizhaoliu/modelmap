"""Hooked fake forward on the meta device (design doc §05, stage 3).

Meta tensors carry shape and dtype but no storage, so a full forward pass costs
milliseconds regardless of parameter count — and shape inference still runs,
which is exactly the data Flow mode needs.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch

from modelmap.inputs import DEFAULT_SEQ_LEN, build_dummy_inputs  # noqa: F401 (re-exported)
from modelmap.schema import TraceStep
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
    inputs, notes, builder = build_dummy_inputs(model, config, seq_len)
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

    # multimodal models: the text-only forward never runs the encoder towers,
    # so trace them separately and put their steps first (towers.py)
    if builder == "multimodal-text-path":
        from modelmap.towers import trace_towers

        tsteps, tnotes = trace_towers(model, config)
        notes = [n for n in notes if not n.startswith("text path traced")] + tnotes
        if tsteps:
            for i, s in enumerate(tsteps + steps):
                s.step = i
            steps = tsteps + steps
    return steps, fidelity, notes


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
