"""modelmap — interactive, animated architecture maps for Hugging Face models.

Python API (design doc §16) — everything the UI, CLI and MCP server use:

    from modelmap import extract_graph, summarize, plan_serving, align, render

    doc = extract_graph("Qwen/Qwen3-8B").to_json_dict()     # meta-device instantiate + traced forward
    summarize(doc, Assumptions(T=32768))["cost"]             # MACs / bytes / KV at T, B, dtype
    plan_serving(doc, PlanRequest(gpus=2, gpu_memory_gb=24, pp=2)).fits
    align(doc, extract_graph("Qwen/Qwen2.5-7B").to_json_dict()).counts
    render(doc, "md")                                        # csv | md | json | dot

Heavy imports are lazy: `import modelmap` stays torch-free (the server's
parent process depends on that); extract_graph pulls torch on first use.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = [
    "Assumptions", "PlanRequest", "align", "build_index", "compute_costs", "diff_markdown",
    "extract_graph", "module_rows", "plan_serving", "render", "summarize", "__version__",
]


def __getattr__(name: str):
    # lazy re-exports: keep `import modelmap` cheap and torch-free
    if name == "extract_graph":
        from modelmap.extract import extract_graph

        return extract_graph
    if name in ("Assumptions", "PlanRequest", "build_index", "compute_costs", "module_rows", "plan_serving", "summarize"):
        from modelmap import analytics

        return getattr(analytics, name)
    if name in ("align", "diff_markdown"):
        from modelmap import compare

        return getattr(compare, name)
    if name == "render":
        from modelmap.export import render

        return render
    raise AttributeError(name)
