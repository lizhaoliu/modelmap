"""Per-module metadata beyond shapes: what the module object already knows.

Annotators run over every module during extraction and return a flat
dict merged into the node's `attrs`. Built-ins harvest `extra_repr()`
(in_features, kernel_size, eps, …) and the defining source location (with a
GitHub link for transformers / torch classes). Register your own with
@register_annotator (see EXTENDING.md).
"""

from __future__ import annotations

import inspect
import os
import re
from typing import Any, Callable

from torch import nn

Annotator = Callable[[str, nn.Module], dict[str, Any] | None]  # (qualified name, module)
_ANNOTATORS: list[Annotator] = []


def register_annotator(fn: Annotator) -> Annotator:
    _ANNOTATORS.append(fn)
    return fn


def annotate(name: str, module: nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fn in _ANNOTATORS:
        try:
            r = fn(name, module)
        except Exception:  # never let an annotator break extraction
            continue
        if r:
            out.update({k: v for k, v in r.items() if v is not None and v != ""})
    return out


# ------------------------------------------------------------ extra_repr

_POSITIONAL = {  # torch classes whose extra_repr leads with positional values
    "Embedding": ("num_embeddings", "embedding_dim"),
    "Conv1d": ("in_channels", "out_channels"), "Conv2d": ("in_channels", "out_channels"),
    "Conv3d": ("in_channels", "out_channels"), "ConvTranspose2d": ("in_channels", "out_channels"),
    "LayerNorm": ("normalized_shape",), "GroupNorm": ("num_groups", "num_channels"),
    "BatchNorm1d": ("num_features",), "BatchNorm2d": ("num_features",),
    "MultiheadAttention": ("embed_dim", "num_heads"),
}


def _split_top_level(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


@register_annotator
def _extra_repr(name: str, module: nn.Module) -> dict[str, Any] | None:
    try:
        text = module.extra_repr()
    except Exception:
        return None
    if not text or len(text) > 400:
        return None
    out: dict[str, Any] = {}
    positional = list(_POSITIONAL.get(type(module).__name__, ()))
    pi = 0
    for part in _split_top_level(text):
        if "=" in part and re.match(r"^[A-Za-z_]\w*\s*=", part):
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            key = positional[pi] if pi < len(positional) else ("shape" if part.startswith("(") else f"arg{pi}")
            out[key] = part
            pi += 1
    return out


# ------------------------------------------------------------ source link

_ROOTS: list[tuple[str, str, str]] = []  # (abs package dir, github repo, ref)


def _roots() -> list[tuple[str, str, str]]:
    if _ROOTS:
        return _ROOTS
    try:
        import transformers

        _ROOTS.append((os.path.dirname(transformers.__file__), "huggingface/transformers",
                       f"v{transformers.__version__}"))
    except Exception:
        pass
    try:
        import torch

        ver = torch.__version__.split("+")[0]
        _ROOTS.append((os.path.dirname(torch.__file__), "pytorch/pytorch", f"v{ver}"))
    except Exception:
        pass
    return _ROOTS


@register_annotator
def _source(name: str, module: nn.Module) -> dict[str, Any] | None:
    cls = type(module)
    try:
        file = inspect.getsourcefile(cls)
        line = inspect.getsourcelines(cls)[1]
    except (TypeError, OSError):
        return None
    if not file:
        return None
    for root, repo, ref in _roots():
        if file.startswith(root):
            rel = os.path.relpath(file, root).replace(os.sep, "/")
            sub = "src/transformers/" if repo.endswith("transformers") else "torch/"
            return {
                "_src": f"{os.path.basename(root)}/{rel}:{line}",
                "_src_url": f"https://github.com/{repo}/blob/{ref}/{sub}{rel}#L{line}",
            }
    return {"_src": f"{os.path.basename(file)}:{line}"}


# ------------------------------------------------------------- plugins

def load_plugins() -> list[str]:
    """Import modules named in $MODELMAP_PLUGINS (comma-separated). Importing
    is how a plugin registers its input builders and annotators."""
    import importlib

    loaded = []
    for mod in filter(None, (m.strip() for m in os.environ.get("MODELMAP_PLUGINS", "").split(","))):
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except Exception as e:  # report, don't fail extraction
            loaded.append(f"{mod} (failed: {type(e).__name__}: {e})")
    return loaded
