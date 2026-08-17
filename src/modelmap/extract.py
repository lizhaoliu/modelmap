"""Model-id → Graph, without downloading weights (design doc §05).

Pipeline:
  1. AutoConfig.from_pretrained            — fetches config.json only (~2 KB)
  2. instantiate under torch.device("meta") — full nn.Module tree, no storage
  3. hooked fake forward (trace.py)         — execution order + tensor shapes
  4. collapse repeats (collapse.py) + serialize

Fallback ladder: full → structural (forward faulted; approximate order) →
weights view (instantiation impossible; tree from safetensors names alone).
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections import defaultdict

import torch
from torch import nn

import transformers
from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig, AutoModel

from modelmap import collapse, trace, weights_view
from modelmap.annotate import annotate, load_plugins
from modelmap.hubio import with_retries
from modelmap.schema import SCHEMA_VERSION, Edge, Graph, Node

log = logging.getLogger(__name__)


class UnsupportedArchitectureError(RuntimeError):
    pass


_MOE_CLS = re.compile(r"(SparseMoe|Experts$|MoE$|MoeBlock$|MoEBlock$|MoeMLP$|MoELayer$|SparseMoeBlock)", re.I)


def extract_graph(
    model_id: str,
    revision: str = "main",
    token: str | None = None,
    trust_remote_code: bool = False,
    seq_len: int = trace.DEFAULT_SEQ_LEN,
) -> Graph:
    notes: list[str] = []
    plugins = load_plugins()
    if plugins:
        notes.append("plugins: " + ", ".join(plugins))
    try:
        config = with_retries(
            lambda: AutoConfig.from_pretrained(
                model_id, revision=revision, token=token, trust_remote_code=trust_remote_code
            )
        )
    except ValueError as e:
        # two expected shapes: custom-code repos (execute arbitrary Python —
        # refused by default, §05) and repos transformers can't parse at all
        # (no model_type, diffusers-style pipelines, brand-new architectures);
        # both degrade to the weights view instead of erroring
        if "trust_remote_code" in str(e):
            notes.append("repo requires trust_remote_code; refused — weights view only")
        else:
            notes.append(f"not a transformers-loadable config ({e}) — weights view only")
        return weights_view.weights_graph(model_id, revision=revision, token=token, notes=notes)

    # eager attention is plain matmul+softmax and always has meta kernels;
    # sdpa/flash backends may not
    config._attn_implementation = "eager"

    try:
        model = _instantiate_meta(config, trust_remote_code)
    except UnsupportedArchitectureError as e:
        notes.append(f"could not instantiate architecture: {e}")
        return weights_view.weights_graph(model_id, revision=revision, token=token, notes=notes)

    nodes = _build_nodes(model)
    steps, fidelity, trace_notes = trace.run_trace(model, config, seq_len=seq_len)
    notes.extend(trace_notes)
    nodes, repeats = collapse.collapse_repeats(nodes)
    edges = _build_edges(nodes)
    _apply_dtypes(model_id, revision, token, config, nodes, notes)

    archs = getattr(config, "architectures", None) or []
    return Graph(
        schema_version=SCHEMA_VERSION,
        model_id=model_id,
        revision=revision,
        fidelity=fidelity,
        architecture=archs[0] if archs else type(model).__name__,
        params_total=sum(p.numel() for p in model.parameters()),
        config={k: v for k, v in config.to_dict().items() if not k.startswith("_")},
        nodes=nodes,
        repeats=repeats,
        edges=edges,
        trace=steps,
        notes=notes,
    )


def _instantiate_meta(config, trust_remote_code: bool):
    """Build the module tree on the meta device — milliseconds at any scale.

    Instantiates under the checkpoint's declared float dtype: some kernels
    (e.g. MoE grouped matmul) assert bf16 inputs even on meta tensors.
    """
    declared = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(declared, str):
        declared = getattr(torch, declared, None)
    use_dtype = declared if isinstance(declared, torch.dtype) and declared.is_floating_point else None

    builders = []
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            # the checkpoint's own class (e.g. Qwen3ForCausalLM) beats AutoModel,
            # which resolves to the headless base model
            builders.append((arch, lambda cls=cls: cls(config)))
    builders.append((
        "AutoModel",
        lambda: AutoModel.from_config(config, trust_remote_code=trust_remote_code),
    ))

    errors = []
    for label, build in builders:
        try:
            with torch.device("meta"), _default_dtype(use_dtype):
                model = build()
            model.eval()
            return model
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    raise UnsupportedArchitectureError("; ".join(errors))


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype | None):
    if dtype is None:
        yield
        return
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _build_nodes(model) -> list[Node]:
    order_in_parent: dict[str, int] = {}
    for name, module in model.named_modules():
        for i, (child_name, _) in enumerate(module.named_children()):
            order_in_parent[f"{name}.{child_name}" if name else child_name] = i

    nodes = []
    for name, module in model.named_modules():
        own = {pn: list(p.shape) for pn, p in module.named_parameters(recurse=False)}
        parent = None if not name else (name.rsplit(".", 1)[0] if "." in name else "")
        nodes.append(Node(
            id=name,
            kind=_classify(name, module),
            cls=type(module).__name__,
            parent=parent,
            depth=0 if not name else name.count(".") + 1,
            order=order_in_parent.get(name, 0),
            params=sum(p.numel() for p in module.parameters()),
            weight_shapes=own or None,
            attrs=annotate(name, module) or None,
        ))
    return nodes


def _classify(name: str, module: nn.Module) -> str:
    cls = type(module).__name__
    lcls = cls.lower()
    leaf = name.rsplit(".", 1)[-1].lower() if name else ""
    has_children = next(module.children(), None) is not None

    if "rotary" in lcls or "rotary" in leaf:
        return "module"  # rope is computation, not a lookup table
    if isinstance(module, nn.Embedding) or "embedding" in lcls or "embed" in leaf:
        return "embedding"
    if leaf in ("lm_head", "score", "classifier") or lcls.endswith("head"):
        return "head"
    if "attention" in lcls or leaf in ("self_attn", "attn", "attention", "cross_attn"):
        return "attention"
    if "norm" in lcls or "norm" in leaf:  # before moe: Qwen3MoeRMSNorm is a norm
        return "norm"
    # the MoE block itself, or the (fused or ModuleList) experts container —
    # not every class that merely carries "Moe" in its name (decoder layers, norms)
    if _MOE_CLS.search(cls) or leaf == "experts":
        return "moe"
    if (
        "mlp" in lcls or "feedforward" in lcls or "intermediate" in lcls
        or "expert" in lcls or leaf in ("mlp", "ffn")
    ):
        return "mlp"
    if isinstance(module, nn.Linear) or cls == "Conv1D":  # transformers Conv1D ≡ linear
        return "linear"
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "conv"
    return "container" if has_children else "module"


def _build_edges(nodes: list[Node]) -> list[Edge]:
    """Sibling chains in registration order.

    An approximation of dataflow (parallel branches like q/k/v read as
    sequential); the client refines rendering per kind, and the trace carries
    the true order. Revisit in M3 if templates aren't enough.
    """
    kids: dict[str, list[Node]] = defaultdict(list)
    for n in nodes:
        if n.parent is not None:
            kids[n.parent].append(n)
    edges = []
    for ks in kids.values():
        ks.sort(key=lambda n: n.order)
        edges.extend(Edge(src=a.id, dst=b.id) for a, b in zip(ks, ks[1:]))
    return edges


def _apply_dtypes(model_id, revision, token, config, nodes, notes) -> None:
    """Meta params default to fp32; real dtypes come from config and headers."""
    declared = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if declared is not None:
        d = str(declared).removeprefix("torch.")
        for n in nodes:
            if n.weight_shapes:
                n.dtype = d
    try:
        meta = with_retries(
            lambda: get_safetensors_metadata(model_id, revision=revision, token=token)
        )
    except Exception as e:
        notes.append(f"no safetensors metadata: {type(e).__name__}")
        return
    dtype_by_module: dict[str, str] = {}
    for fm in meta.files_metadata.values():
        for tname, info in fm.tensors.items():
            dtype_by_module[tname.rsplit(".", 1)[0]] = str(info.dtype).lower()
    for n in nodes:
        # checkpoint names may omit the base-model prefix (gpt2: "h.0.attn…"
        # for module "transformer.h.0.attn…"), so fall back to the stripped id
        d = dtype_by_module.get(n.id) or dtype_by_module.get(n.id.split(".", 1)[-1])
        if d:
            n.dtype = d
