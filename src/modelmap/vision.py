"""Vision-tower tracing for vision-language models (design doc §05, fallback ladder).

The main traced forward feeds text only: multimodal inputs need model-specific
packing, and the position math inside vision towers is data-dependent
(`.tolist()` on patch grids), which the meta device cannot evaluate. So the
tower is traced separately, in two attempts:

  A. standalone on the meta device with a plausible pixel input — works for
     ViT/SigLIP-style towers with no data-dependent ops (zero memory);
  B. a *shallow twin*: the tower's own class instantiated from its config
     with depth 1 on the CPU (one block + patch embed + merger, tens of MB),
     run on real tiny inputs so every shape is exact; block-0 steps are
     replicated per block since all blocks are structurally identical.

Steps are prefixed with the tower's path in the full model and merged ahead
of the text steps (a real forward encodes the image first).
"""

from __future__ import annotations

import copy
import itertools
import logging
import re
from typing import Any

import torch
from torch import nn

from modelmap.schema import TraceStep

log = logging.getLogger(__name__)

_VISION_CLS = re.compile(r"(Vision|Visual|Siglip|Clip|ImageEncoder|Pixtral)", re.I)
_PROJECTOR = re.compile(r"(multi_modal_projector|mm_projector|connector|projector|vision_projection|visual_projection|aligner)$")
_DEPTH_ATTRS = ("depth", "num_hidden_layers", "num_layers")


def find_vision_tower(model: nn.Module) -> tuple[str, nn.Module] | None:
    """Outermost submodule that looks like an image encoder."""
    best: tuple[str, nn.Module] | None = None
    for name, m in model.named_modules():
        if not name or not _VISION_CLS.search(type(m).__name__):
            continue
        if not any(True for _ in m.parameters()):
            continue
        if best is None or name.count(".") < best[0].count("."):
            best = (name, m)
    return best


def _vision_config(config, tower: nn.Module):
    vc = getattr(tower, "config", None) or getattr(config, "vision_config", None)
    return vc


def _pixel_input(vc, device: str, dtype: torch.dtype) -> tuple[list[Any], dict[str, Any], str]:
    """Positional args + kwargs for the tower forward, and a description."""
    ch = getattr(vc, "in_channels", None) or getattr(vc, "num_channels", 3)
    if hasattr(vc, "spatial_merge_size"):  # Qwen-VL family: flattened patches + grid
        ps = getattr(vc, "patch_size", 14)
        merge = getattr(vc, "spatial_merge_size", 2)
        tps = getattr(vc, "temporal_patch_size", 2)
        gh = gw = 3 * merge  # a tiny image: (3·merge)² patches → 9 merged image tokens (36 ≠ any common patch size)
        pixels = torch.zeros((gh * gw, ch * tps * ps * ps), device=device, dtype=dtype)
        grid = torch.tensor([[1, gh, gw]], dtype=torch.long, device=device)
        return [pixels, grid], {}, f"{gh}×{gw} patches"
    size = getattr(vc, "image_size", 224)
    h, w = (size[0], size[-1]) if isinstance(size, (tuple, list)) else (size, size)
    return [torch.zeros((1, ch, h, w), device=device, dtype=dtype)], {}, f"{ch}×{h}×{w} image"


def _hook_all(module: nn.Module, prefix: str, sink: list[TraceStep], counter):
    from modelmap.trace import _tensor_shapes

    handles = []
    for name, m in module.named_modules():
        full = f"{prefix}.{name}" if name else prefix

        def hook(mod, args, kwargs, output, full=full):
            sink.append(TraceStep(
                step=next(counter), node=full,
                inputs=_tensor_shapes((args, kwargs)), outputs=_tensor_shapes(output),
            ))

        handles.append(m.register_forward_hook(hook, with_kwargs=True))
    return handles


def _run(module: nn.Module, args: list[Any], kwargs: dict[str, Any]) -> None:
    with torch.no_grad():
        try:
            module(*args, **kwargs)
        except TypeError:  # towers that only accept pixel_values=
            module(pixel_values=args[0], **kwargs)


def trace_vision_tower(model: nn.Module, config) -> tuple[list[TraceStep], list[str]]:
    found = find_vision_tower(model)
    if not found:
        return [], []
    name, tower = found
    vc = _vision_config(config, tower)
    if vc is None:
        return [], [f"vision tower {name}: no vision config to build an input from"]
    notes: list[str] = []
    dtype = next(tower.parameters()).dtype

    # ---- A: standalone on the meta device -------------------------------
    steps: list[TraceStep] = []
    handles = _hook_all(tower, name, steps, itertools.count())
    try:
        args, kwargs, desc = _pixel_input(vc, "meta", dtype)
        _run(tower, args, kwargs)
        notes.append(f"vision tower traced standalone on the meta device ({desc})")
        return steps + _trace_projector(model, config, name, steps), notes
    except Exception as e:
        log.debug("meta vision trace failed: %s", e)
    finally:
        for h in handles:
            h.remove()

    # ---- B: shallow twin on the CPU --------------------------------------
    depth_attr = next((a for a in _DEPTH_ATTRS if hasattr(vc, a)), None)
    real_depth = int(getattr(vc, depth_attr)) if depth_attr else 1
    vc2 = copy.deepcopy(vc)
    if depth_attr:
        setattr(vc2, depth_attr, 1)
    try:
        vc2._attn_implementation = "eager"
    except Exception:
        pass
    twin_dtype = torch.float32  # CPU-safe for every op; a few hundred MB at most
    for attempt_dtype in (torch.bfloat16, torch.float32):
        try:
            prev = torch.get_default_dtype()
            torch.set_default_dtype(attempt_dtype)
            try:
                with torch.device("cpu"):
                    twin = type(tower)(vc2)
            finally:
                torch.set_default_dtype(prev)
            twin.eval()
            twin_dtype = attempt_dtype
            steps = []
            handles = _hook_all(twin, name, steps, itertools.count())
            try:
                args, kwargs, desc = _pixel_input(vc2, "cpu", attempt_dtype)
                _run(twin, args, kwargs)
            finally:
                for h in handles:
                    h.remove()
            del twin
            break
        except Exception as e:  # e.g. bf16 conv unsupported on this CPU → retry fp32
            log.debug("twin (%s) failed: %s", attempt_dtype, e)
            steps = []
    if not steps:
        return [], [f"vision tower {name} could not be traced (no standalone or twin forward)"]

    real_names = {n for n, _ in model.named_modules()}
    steps = [s for s in steps if s.node in real_names]
    steps = _replicate_blocks(steps, real_names, real_depth)
    notes.append(
        f"vision tower traced with a shallow CPU twin ({desc}, {twin_dtype!s}); "
        f"1 of {real_depth} identical blocks run, its shapes replicated per block"
    )
    return steps + _trace_projector(model, config, name, steps), notes


def _trace_projector(model: nn.Module, config, tower_name: str, tower_steps: list[TraceStep]) -> list[TraceStep]:
    """CLIP/SigLIP-style models keep the vision→text projector outside the
    tower; run it on a tensor shaped like the tower's output. Best-effort:
    meta first, then a CPU twin of the projector class; silent on failure."""
    proj = next(
        ((n, m) for n, m in model.named_modules()
         if n and not n.startswith(tower_name) and _PROJECTOR.search(n)),
        None,
    )
    if not proj:
        return []
    pname, pmod = proj
    tower_out = next((s.outputs for s in reversed(tower_steps) if s.node == tower_name), None)
    if not tower_out:
        return []
    shape = list(tower_out[0])
    # CLIP-style "default" feature selection drops the CLS token before projecting
    if getattr(config, "vision_feature_select_strategy", None) == "default" and len(shape) == 3:
        shape[1] = max(1, shape[1] - 1)
    dtype = next(pmod.parameters()).dtype
    steps: list[TraceStep] = []
    for device, module_factory in (("meta", lambda: pmod), ("cpu", lambda: type(pmod)(config))):
        try:
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32 if device == "cpu" else prev)
            try:
                with torch.device(device):
                    module = module_factory()
            finally:
                torch.set_default_dtype(prev)
            module.eval()
            steps = []
            handles = _hook_all(module, pname, steps, itertools.count())
            try:
                x = torch.zeros(shape, device=device, dtype=dtype if device == "meta" else torch.float32)
                with torch.no_grad():
                    module(x)
            finally:
                for h in handles:
                    h.remove()
            if steps:
                real = {n for n, _ in model.named_modules()}
                return [s for s in steps if s.node in real]
        except Exception as e:
            log.debug("projector trace on %s failed: %s", device, e)
            steps = []
    return []


def _replicate_blocks(steps: list[TraceStep], real_names: set[str], depth: int) -> list[TraceStep]:
    """Twin traces block 0 only; repeat its steps for blocks 1..depth-1 in place,
    so execution order (block by block) and shapes stay exact."""
    if depth <= 1:
        return steps
    pat = re.compile(r"^(.*?\.)(\w+)\.0(\.|$)")
    m0 = next((pat.match(s.node) for s in steps if pat.match(s.node)), None)
    if not m0:
        return steps
    prefix, lst = m0.group(1), m0.group(2)
    block0 = f"{prefix}{lst}.0"
    is_b0 = lambda n: n == block0 or n.startswith(block0 + ".")
    first = next(i for i, s in enumerate(steps) if is_b0(s.node))
    last = max(i for i, s in enumerate(steps) if is_b0(s.node))
    run = steps[first : last + 1]
    out = steps[:first] + list(run)
    for i in range(1, depth):
        blk = f"{prefix}{lst}.{i}"
        if blk not in real_names:
            break
        out.extend(
            TraceStep(step=0, node=s.node.replace(block0, blk, 1), inputs=s.inputs, outputs=s.outputs)
            for s in run
        )
    out.extend(steps[last + 1 :])
    for i, s in enumerate(out):
        s.step = i
    return out
