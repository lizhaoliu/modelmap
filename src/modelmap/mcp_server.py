"""MCP server: modelmap's knowledge as tools for coding agents (design doc §16).

    modelmap mcp                      # stdio, extracts locally (needs torch)
    modelmap mcp --remote https://modelmap.cc   # stdio, asks the hosted API

Tools: describe_model · estimate_cost · plan_serving · compare_models ·
list_modules · search_models · export_markdown. Every answer is derived from
the graph document (meta-device instantiation + traced forward; never the
weights), and every number comes with the assumptions it was computed under.

Claude Code:   claude mcp add modelmap -- uvx --from 'modelmap[mcp]' modelmap mcp
Cursor / others: command "uvx", args ["--from", "modelmap[mcp]", "modelmap", "mcp"]
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from typing import Any

from modelmap import __version__, cache
from modelmap.analytics import Assumptions, PlanRequest, WHATIF_DTYPES, module_rows, plan_serving, summarize
from modelmap.compare import align, diff_markdown
from modelmap.export import to_markdown
from modelmap.ids import is_local

log = logging.getLogger(__name__)

_REMOTE: str | None = None  # base URL of a modelmap server to delegate to


def _remote_get(path: str, params: dict[str, Any] | None = None) -> Any:
    import httpx

    assert _REMOTE is not None
    r = httpx.get(_REMOTE.rstrip("/") + path, params=params, timeout=300, follow_redirects=True)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"{_REMOTE} returned {r.status_code}: {detail}")
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text


def get_doc(model_id: str) -> dict:
    """The graph document for a model id: hosted API when --remote is set,
    otherwise the local disk cache or a fresh local extraction."""
    model_id = model_id.strip()
    if _REMOTE:
        return _remote_get(f"/api/graph/{model_id}")
    if not is_local(model_id):
        raw = cache.get_bytes(model_id, "main")
        if raw is not None:
            return json.loads(gzip.decompress(raw))
    from modelmap.extract import extract_graph

    doc = extract_graph(model_id, allow_local=True).to_json_dict()
    if not is_local(model_id):
        cache.put(model_id, "main", doc)
    return doc


def _assume(T: int, B: int, dtype: str) -> Assumptions:
    if dtype not in WHATIF_DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(WHATIF_DTYPES)}")
    return Assumptions(T=max(1, T), B=max(1, B), dtype=dtype)


# ------------------------------------------------------------------- tools


def describe_model(model_id: str, T: int = 4096, B: int = 1, dtype: str = "bf16") -> dict:
    """Architecture summary of a Hugging Face model: class, parameter count
    (and active parameters per token for MoE), layer stacks, the config
    essentials (hidden size, heads, KV heads, experts, context length…),
    checkpoint format / quantization, and compute / memory / KV-cache
    estimates at the given sequence length T, batch B and dtype.
    model_id is "owner/name", optionally ":Q4_K_M" to pick a GGUF variant."""
    doc = get_doc(model_id)
    s = summarize(doc, _assume(T, B, dtype))
    s["explore_url"] = f"https://modelmap.cc/m/{model_id}"
    return s


def estimate_cost(model_id: str, T: int = 4096, B: int = 1, dtype: str = "bf16") -> dict:
    """Compute (MACs per token and per forward), weight bytes at the stored
    dtypes, activation bytes, and KV-cache bytes per token and at T — for one
    model at sequence length T, batch B, activation dtype. Analytic estimates
    from traced tensor shapes: weight matmuls + the attention core; fused
    kernels, softmax and elementwise work are not counted."""
    doc = get_doc(model_id)
    s = summarize(doc, _assume(T, B, dtype))
    return {
        "model_id": s["model_id"], "params_total": s["params_total"], "active_params": s["active_params"],
        "assumptions": s["assumptions"], "cost": s["cost"], "notes": s["notes"],
    }


def plan_serving_tool(
    model_id: str, gpus: int = 1, gpu_memory_gb: float = 80, tp: int = 1, pp: int = 1,
    T: int = 4096, B: int = 1, dtype: str = "bf16", headroom: float = 0.1,
) -> dict:
    """Will this model fit on N GPUs? Tensor-/pipeline-parallel placement
    estimate: per-GPU weight, KV-cache and activation bytes for a tp × pp
    layout, which layer ranges land on which pipeline stage, whether every
    stage fits under (1 − headroom) × GPU memory, the activation bytes crossing
    each stage boundary per forward, and the KV-limited maximum context length
    at batch B. Use dtype for activations/KV; weights use their stored dtype."""
    if dtype not in WHATIF_DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(WHATIF_DTYPES)}")
    doc = get_doc(model_id)
    req = PlanRequest(gpus=gpus, gpu_memory_gb=gpu_memory_gb, tp=tp, pp=pp, T=T, B=B, dtype=dtype, headroom=headroom)
    return plan_serving(doc, req).to_dict()


def compare_models(a: str, b: str, format: str = "markdown") -> str | dict:
    """Module-by-module structural diff of two models (aligned by path, then
    by role): config changes, changed modules with their field-level diffs,
    and modules only in one of them. format: "markdown" (default) or "json"."""
    da, db = get_doc(a), get_doc(b)
    al = align(da, db)
    if format == "json":
        return {"a": a, "b": b, **al.to_dict(changed_only=True)}
    return diff_markdown(da, db, al)


def list_modules(
    model_id: str, filter: str = "", leaves_only: bool = False, max_rows: int = 200,
    T: int = 4096, B: int = 1, dtype: str = "bf16",
) -> list[dict]:
    """Per-module table: path, kind, class, repeats/multiplicity, parameters,
    stored dtype, weight shapes, traced input/output shapes, and cost columns
    (MACs per token, weight bytes, activation bytes, KV bytes per token).
    `filter` is a case-insensitive substring on the module path or class."""
    doc = get_doc(model_id)
    rows = module_rows(doc, _assume(T, B, dtype), leaves_only=leaves_only)
    if filter:
        f = filter.lower()
        rows = [r for r in rows if f in r["module"].lower() or f in r["class"].lower()]
    return rows[: max(1, min(max_rows, 5000))]


def search_models(query: str, limit: int = 10) -> list[dict]:
    """Search the Hugging Face Hub for model ids (sorted by downloads)."""
    if _REMOTE:
        return _remote_get("/api/search", {"q": query, "limit": limit})
    from huggingface_hub import HfApi

    return [
        {"id": m.id, "downloads": m.downloads, "likes": m.likes, "pipeline_tag": m.pipeline_tag}
        for m in HfApi().list_models(search=query, limit=max(1, min(limit, 50)), sort="downloads")
    ]


def export_markdown(model_id: str, T: int = 4096, B: int = 1, dtype: str = "bf16") -> str:
    """A Markdown card for the model: headline numbers, config table, layer
    stacks, cost estimates at T/B/dtype, and the largest weight matrices —
    ready to paste into a doc, issue or README."""
    return to_markdown(get_doc(model_id), _assume(T, B, dtype))


TOOLS = [
    ("describe_model", describe_model),
    ("estimate_cost", estimate_cost),
    ("plan_serving", plan_serving_tool),
    ("compare_models", compare_models),
    ("list_modules", list_modules),
    ("search_models", search_models),
    ("export_markdown", export_markdown),
]

INSTRUCTIONS = (
    "modelmap answers questions about Hugging Face model *architectures* without downloading weights: "
    "module trees, tensor shapes, parameter counts, MoE active parameters, quantized checkpoint sizes "
    "(GGUF variants via owner/name:Q4_K_M), compute / memory / KV-cache estimates, GPU placement plans, "
    "and structural diffs between two models. Numbers are analytic estimates — quote the assumptions "
    "(T, B, dtype) with them. Link users to https://modelmap.cc/m/<model_id> for the interactive map."
)


def build_server():
    """An MCP server instance with every tool registered (mcp ≥ 1.2 or 2.x)."""
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # mcp 2.x
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
        except ImportError as e:
            raise SystemExit(
                "the MCP server needs the 'mcp' package: pip install 'modelmap[mcp]'  (or: uvx --from 'modelmap[mcp]' modelmap mcp)"
            ) from e
    server = _Server(name="modelmap", instructions=INSTRUCTIONS, version=__version__)
    for name, fn in TOOLS:
        server.tool(name=name)(fn)
    return server


def run(remote: str | None = None, transport: str = "stdio", **kw) -> None:
    global _REMOTE
    _REMOTE = remote or os.environ.get("MODELMAP_REMOTE") or None
    if _REMOTE:
        log.info("delegating extraction to %s", _REMOTE)
    build_server().run(transport=transport, **kw)
