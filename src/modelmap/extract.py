"""Model-id → Graph, without downloading weights (design doc §05).

Pipeline:
  1. AutoConfig.from_pretrained            — fetches config.json only (~2 KB)
  2. instantiate under torch.device("meta") — full nn.Module tree, no storage
  3. hooked fake forward (trace.py)         — execution order + tensor shapes
  4. collapse repeats (collapse.py) + serialize

Fallback ladder: full → structural (forward faulted; approximate order) →
weights view (instantiation impossible; tree from safetensors names alone).

Sources (design doc §16–§18):
  owner/name            a Hub repo (config.json + safetensors / pytorch bins)
  owner/name:Q4_K_M     a GGUF variant in a Hub repo — config rebuilt from the
                        GGUF header, real quant dtypes per module (gguf.py)
  local:/path/to/ckpt   a local checkpoint directory, file or .gguf — only
                        when the caller allows it (the CLI does; the hosted
                        server does not)
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections import Counter, defaultdict

import torch
from torch import nn

import transformers
from huggingface_hub import HfApi, get_safetensors_metadata
from transformers import AutoConfig, AutoModel

from modelmap import analytics, collapse, gguf, trace, weights_view
from modelmap.annotate import annotate, load_plugins
from modelmap.hubio import with_retries
from modelmap.ids import LOCAL_PREFIX, LocalPathError, Source, parse_model_id  # noqa: F401
from modelmap.schema import SCHEMA_VERSION, Edge, Graph, Node

log = logging.getLogger(__name__)


class UnsupportedArchitectureError(RuntimeError):
    pass


# model-id grammar lives in ids.py (torch-free, shared with the server)


def _list_files(src: Source, revision: str, token: str | None) -> list[str]:
    if src.local:
        if os.path.isfile(src.repo):
            return [os.path.basename(src.repo)]
        out = []
        for root, _, fs in os.walk(src.repo):
            rel = os.path.relpath(root, src.repo)
            for f in fs:
                out.append(f if rel == "." else f"{rel}/{f}")
        return sorted(out)
    return with_retries(lambda: HfApi(token=token).list_repo_files(src.repo, revision=revision))


def _weights_format(files: list[str]) -> str | None:
    low = [f.lower() for f in files]
    if any(f.endswith(".safetensors") for f in low):
        return "safetensors"
    if any(f.endswith(".gguf") for f in low):
        return "gguf"
    if any(f.endswith((".bin", ".pt", ".pth")) for f in low):
        return "pytorch"
    return None


_MOE_CLS = re.compile(r"(SparseMoe|Experts$|MoE$|MoeBlock$|MoEBlock$|MoeMLP$|MoELayer$|SparseMoeBlock)", re.I)


def extract_graph(
    model_id: str,
    revision: str = "main",
    token: str | None = None,
    trust_remote_code: bool = False,
    seq_len: int = trace.DEFAULT_SEQ_LEN,
    allow_local: bool = False,
) -> Graph:
    notes: list[str] = []
    plugins = load_plugins()
    if plugins:
        notes.append("plugins: " + ", ".join(plugins))
    src = parse_model_id(model_id, allow_local=allow_local)

    # The file listing is a Hub *API* call (rate-limited per IP, and a hosted
    # server shares its egress IP); config.json is a plain file fetch. So list
    # only when the id asks for it: a variant, a GGUF-looking repo name, or a
    # local path (free) — otherwise only if the config turns out unloadable.
    files: list[str] | None = None
    if src.local or src.variant or "gguf" in src.repo.lower():
        try:
            files = _list_files(src, revision, token)
        except Exception as e:
            if src.variant:
                raise
            log.warning("could not list files for %s: %s", src.repo, e)
            files = []
        variants = gguf.variants_of(files)
        if variants and (src.variant or _weights_format(files) != "safetensors"):
            return _extract_gguf(src, revision, token, files, variants, seq_len, notes)
        if src.variant:
            raise ValueError(f"'{src.repo}' has no GGUF files; ':{src.variant}' selects a GGUF variant")

    try:
        config = with_retries(
            lambda: AutoConfig.from_pretrained(
                src.repo, revision=revision, token=token, trust_remote_code=trust_remote_code
            )
        )
    except (ValueError, OSError) as e:
        # expected shapes: custom-code repos (execute arbitrary Python — refused
        # by default, §05), repos transformers can't parse at all (no
        # model_type, diffusers-style pipelines, brand-new architectures), and
        # local dirs without a config.json; all degrade to the weights view —
        # or to the GGUF path when that is what the repo holds
        if files is None:
            try:
                files = _list_files(src, revision, token)
            except Exception as le:
                # a repo that doesn't exist / isn't visible: surface the Hub's answer
                raise le if "not found" in str(le).lower() or "gated" in str(le).lower() or "401" in str(le) else e
        variants = gguf.variants_of(files)
        if variants:
            return _extract_gguf(src, revision, token, files, variants, seq_len, notes)
        if "trust_remote_code" in str(e):
            notes.append("repo requires trust_remote_code; refused — weights view only")
        else:
            notes.append(f"not a transformers-loadable config ({e}) — weights view only")
        return _weights_only(src, revision, token, notes, _weights_format(files))
    fmt = _weights_format(files) if files is not None else None

    # eager attention is plain matmul+softmax and always has meta kernels;
    # sdpa/flash backends may not
    config._attn_implementation = "eager"

    try:
        model = _instantiate_meta(config, trust_remote_code)
    except UnsupportedArchitectureError as e:
        notes.append(f"could not instantiate architecture: {e}")
        return _weights_only(src, revision, token, notes, fmt)

    graph = _graph_from_model(src, revision, model, config, seq_len, notes)
    graph.weights_format = fmt
    if _apply_dtypes(src, revision, token, config, graph.nodes, notes) and fmt is None:
        graph.weights_format = "safetensors"
    return graph


def _graph_from_model(src: Source, revision: str, model, config, seq_len: int, notes: list[str]) -> Graph:
    nodes = _build_nodes(model)
    steps, fidelity, trace_notes = trace.run_trace(model, config, seq_len=seq_len)
    notes.extend(trace_notes)
    nodes, repeats = collapse.collapse_repeats(nodes)
    edges = _build_edges(nodes, steps)
    archs = getattr(config, "architectures", None) or []
    return Graph(
        schema_version=SCHEMA_VERSION,
        model_id=src.model_id,
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


def _mixed_layers(headers) -> bool:
    """True when the same tensor role is stored at different quant types in
    different blocks (Q4_K_M keeps some ffn_down at Q6_K, e.g.)."""
    by_role: dict[str, set[str]] = defaultdict(set)
    for h in headers:
        for t in h.tensors:
            m = re.match(r"^blk\.\d+\.(.+)$", t.name)
            if m:
                by_role[m.group(1)].add(t.dtype)
    return any(len(v) > 1 for v in by_role.values())


def _weights_only(src: Source, revision: str, token: str | None, notes: list[str], fmt: str | None) -> Graph:
    if src.local:
        tensors = weights_view.local_safetensors(src.repo)
        if not tensors:
            hint = (
                " — this repo's config needs trust_remote_code; retry with --trust-remote-code "
                "(runs the repo's own Python)"
                if any("trust_remote_code" in n for n in notes)
                else ""
            )
            raise LocalPathError(f"{src.repo}: no config.json transformers can load and no safetensors files{hint}")
        return weights_view.weights_graph(src.model_id, revision=revision, notes=notes, tensors=tensors, weights_format=fmt)
    return weights_view.weights_graph(src.repo, revision=revision, token=token, notes=notes, weights_format=fmt)


def _extract_gguf(
    src: Source, revision: str, token: str | None, files: list[str],
    variants: dict[str, list[str]], seq_len: int, notes: list[str],
) -> Graph:
    label = gguf.choose_variant(variants, src.variant)
    shards = variants[label]
    if len(shards) > gguf.MAX_SHARDS:
        notes.append(f"variant {label} has {len(shards)} shards; reading the first {gguf.MAX_SHARDS}")
        shards = shards[: gguf.MAX_SHARDS]
    headers = []
    for f in shards:
        fetch = gguf.local_fetcher(os.path.join(src.repo, f) if os.path.isdir(src.repo) else src.repo) if src.local \
            else gguf.hub_fetcher(src.repo, f, revision, token)
        headers.append(with_retries(lambda fetch=fetch: gguf.read_header(fetch)))
    head = headers[0]
    extra = {"weights_format": "gguf", "variant": label, "variants": list(variants)}
    if len(shards) > 1:
        notes.append(f"{label}: {len(shards)} shards")
    try:
        config, cnotes = gguf.config_from_header(head)
        notes.extend(cnotes)
        config._attn_implementation = "eager"
        model = _instantiate_meta(config, False)
    except (gguf.GGUFError, UnsupportedArchitectureError) as e:
        notes.append(f"{e} — weights view from the GGUF tensor table")
        return weights_view.weights_graph(
            src.model_id, revision=revision, notes=notes, tensors=gguf.gguf_nodes(headers), **extra
        )
    graph = _graph_from_model(src, revision, model, config, seq_len, notes)
    graph.weights_format, graph.variant, graph.variants = "gguf", label, list(variants)
    # config-declared dtype first (f16/bf16 floor), then the real quant types
    declared = getattr(config, "dtype", None)
    for n in graph.nodes:
        if n.weight_shapes:
            n.dtype = str(declared).removeprefix("torch.") if declared else None
    gguf.apply_gguf_dtypes(graph.nodes, headers, notes)
    total = sum(t.numel * analytics.bytes_of(t.dtype, 2) for h in headers for t in h.tensors)
    notes.append(
        f"{label}: ≈{analytics.fmt_bytes(total)} of tensors in the file"
        + ("; quant types vary per layer — the map shows the first block's" if _mixed_layers(headers) else "")
    )
    # the document's config records what the header told us
    graph.config["gguf"] = {
        k: v for k, v in head.kv.items()
        if not k.startswith(("tokenizer.", "_")) and not isinstance(v, (list, dict)) and len(str(v)) < 200
    }
    return graph


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


_AUX = re.compile(r"(rotary|rope|pos_emb|position_embed|positional)", re.I)


def _is_aux(n: Node) -> bool:
    """A parameter-free side computation (rotary / positional phases) whose
    output feeds other modules but which the residual stream never passes
    through — drawn as a dashed input to its consumer, not a chain link."""
    leaf = n.id.rsplit(".", 1)[-1]
    return n.params == 0 and n.kind in ("module", "container") and bool(_AUX.search(n.cls + " " + leaf))


def _build_edges(nodes: list[Node], steps: list) -> list[Edge]:
    """Sibling chains in *execution* order (from the trace), with side
    computations attached as aux edges. Registration order is only the
    fallback for modules the trace never saw (or weights views)."""
    # first execution step of every module, propagated to its ancestors
    first: dict[str, int] = {}
    for st in steps:
        parts = st.node.split(".")
        for i in range(1, len(parts) + 1):
            a = ".".join(parts[:i])
            if a not in first:
                first[a] = st.step

    kids: dict[str, list[Node]] = defaultdict(list)
    for n in nodes:
        if n.parent is not None:
            kids[n.parent].append(n)
    edges: list[Edge] = []
    inf = float("inf")
    for ks in kids.values():
        ks.sort(key=lambda n: (first.get(n.id, inf), n.order))
        main = [n for n in ks if not _is_aux(n)]
        aux = [n for n in ks if _is_aux(n)]
        edges.extend(Edge(src=a.id, dst=b.id) for a, b in zip(main, main[1:]))
        for x in aux:
            # consumer: the first main sibling that runs after the side computation
            fx = first.get(x.id, inf)
            consumer = next((m for m in main if first.get(m.id, inf) > fx), None)
            if consumer is None and main:
                consumer = main[-1] if fx == inf else main[0]
            if consumer is not None:
                edges.append(Edge(src=x.id, dst=consumer.id, kind="aux"))
    return edges


_PACKED = ("qweight", "weight_packed", "qzeros", "weight.absmax")  # AWQ/GPTQ, compressed-tensors, bnb


def _quant_label(config) -> str | None:
    """A dtype label for quantized linear weights, from quantization_config."""
    q = getattr(config, "quantization_config", None)
    if q is None:
        return None
    q = q if isinstance(q, dict) else getattr(q, "to_dict", lambda: {})()
    method = str(q.get("quant_method") or "").lower()
    bits = q.get("bits") or q.get("weight_bits") or q.get("num_bits")
    if method in ("bitsandbytes", "bnb"):
        if q.get("load_in_4bit"):
            return str(q.get("bnb_4bit_quant_type") or "int4")
        if q.get("load_in_8bit"):
            return "int8"
    if method in ("fp8", "fbgemm_fp8", "finegrained_fp8"):
        return "f8_e4m3"
    if method == "mxfp4":
        return "mxfp4"
    if method == "compressed-tensors":
        groups = q.get("config_groups") or {}
        g0 = next(iter(groups.values()), {}) if isinstance(groups, dict) else {}
        w = (g0 or {}).get("weights") or {}
        nb, typ = w.get("num_bits"), str(w.get("type") or "int")
        if nb:
            return f"f8_e4m3" if typ == "float" and nb == 8 else f"int{nb}"
    if bits:
        return f"int{bits}"
    return None


def _apply_dtypes(src: Source, revision, token, config, nodes, notes) -> bool:
    """Meta params default to fp32; real dtypes come from config and headers.

    Per module, the `weight` tensor's dtype wins (scales / zero-points /
    absmax side tensors must not overwrite it); modules whose checkpoint
    tensors are packed quantized weights get the quantization_config label."""
    declared = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if declared is not None:
        d = str(declared).removeprefix("torch.")
        for n in nodes:
            if n.weight_shapes:
                n.dtype = d
    try:
        if src.local:
            tensors = weights_view.local_safetensors(src.repo)
        else:
            meta = with_retries(
                lambda: get_safetensors_metadata(src.repo, revision=revision, token=token)
            )
            tensors = [
                (tname, list(info.shape), str(info.dtype).lower())
                for fm in meta.files_metadata.values()
                for tname, info in fm.tensors.items()
            ]
    except Exception as e:
        notes.append(f"no safetensors metadata: {type(e).__name__}")
        return False
    qlabel = _quant_label(config)
    # checkpoints in a vendor layout (DeepSeek's "layers.N.ffn.experts.3.w1")
    # are renamed on load by transformers' conversion mapping; apply the same
    # renames so tensors find their modules
    renames = _checkpoint_renames(getattr(config, "model_type", None))
    if renames:
        tensors = [(_apply_renames(t, renames), shape, dtype) for t, shape, dtype in tensors]
    # a separate dtype for routed experts (DeepSeek-V4: fp4 experts, fp8 elsewhere)
    expert_label = getattr(config, "expert_dtype", None)
    expert_label = str(expert_label).lower() if isinstance(expert_label, str) else None
    # module → {leaf: (dtype, numel)}
    per_module: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    for tname, shape, dtype in tensors:
        mod, _, leaf = tname.rpartition(".")
        # "weight.absmax" style (bnb) nests one level deeper: fold into the module
        if leaf in ("absmax", "quant_map", "quant_state", "nested_absmax", "nested_quant_map") or leaf.startswith("quant_state"):
            mod, _, l2 = mod.rpartition(".")
            leaf = f"{l2}.{leaf}"
        numel = 1
        for d in shape:
            numel *= d
        per_module[mod or tname][leaf] = (dtype, numel)
    dtype_by_module: dict[str, str] = {}
    quantized = 0
    for mod, leaves in per_module.items():
        # the module's main tensor: `weight` if present, else the largest
        main_leaf = "weight" if "weight" in leaves else max(leaves, key=lambda k: leaves[k][1])
        main_dtype = leaves[main_leaf][0]
        packed = any(k in leaves for k in _PACKED) or main_dtype in ("i32", "u8", "i8", "f8_e4m3", "f8_e5m2")
        if qlabel and packed:
            dtype_by_module[mod] = expert_label if expert_label and re.search(r"(^|\.)experts(\.|$)", mod) and "shared" not in mod else qlabel
            quantized += 1
            continue
        dtype_by_module[mod] = main_dtype
    # checkpoint names may omit the base-model prefix (gpt2: "h.0.attn…" for
    # module "transformer.h.0.attn…"), and per-expert tensors
    # ("…experts.7.gate_proj") belong to a fused experts module — resolve each
    # checkpoint module to the nearest node that owns weights, then vote
    by_id = {n.id: n for n in nodes}
    owners = {n.id for n in nodes if n.weight_shapes}
    prefixes = {n.id.split(".", 1)[-1]: n.id for n in nodes if n.id}  # stripped → full

    def resolve(mod: str) -> str | None:
        for cand in (mod, prefixes.get(mod)):
            cur = cand
            while cur is not None:
                if cur in owners:
                    return cur
                cur = cur.rsplit(".", 1)[0] if "." in cur else None
        return None

    votes: dict[str, Counter] = defaultdict(Counter)
    for mod, d in dtype_by_module.items():
        target = resolve(mod)
        if target is not None:
            votes[target][d] += 1
    for nid, c in votes.items():
        by_id[nid].dtype = c.most_common(1)[0][0]
    if qlabel:
        qc = getattr(config, "quantization_config", None)
        method = str(qc.get("quant_method", "")) if isinstance(qc, dict) else ""
        hit = sum(1 for n in nodes if n.dtype and n.dtype in (qlabel, expert_label))
        if hit:
            notes.append(
                f"{hit} module types carry {qlabel} quantized weights"
                + (f"; routed experts {expert_label}" if expert_label and any(n.dtype == expert_label for n in nodes) else "")
                + (f" ({method})" if method else " (quantization_config)")
            )
        else:
            notes.append(
                f"quantization_config says {method or qlabel}, but the checkpoint's tensor names did not match "
                "the module tree — weights are shown at the config's declared dtype"
            )
    return bool(tensors)


def _checkpoint_renames(model_type: str | None) -> list[tuple[re.Pattern, str]]:
    """(pattern, replacement) pairs transformers applies to this model type's
    checkpoint keys on load (transformers ≥ 5 conversion mapping); [] otherwise."""
    if not model_type:
        return []
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    except ImportError:
        return []
    try:
        mapping = get_checkpoint_conversion_mapping(model_type) or []
    except Exception:
        return []
    out: list[tuple[re.Pattern, str]] = []
    for conv in mapping:
        srcs = getattr(conv, "source_patterns", None) or []
        tgts = getattr(conv, "target_patterns", None) or []
        if not srcs or not tgts:
            continue
        for src in srcs:
            try:
                out.append((re.compile(src), tgts[0]))
            except re.error:
                continue
    return out


def _apply_renames(name: str, renames: list[tuple[re.Pattern, str]]) -> str:
    for pat, rep in renames:
        if pat.search(name):
            name = pat.sub(rep, name, count=1)
    return name
