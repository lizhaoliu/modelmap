"""MCP server: tools register with schemas and answer from graph documents."""
import asyncio
import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from modelmap import mcp_server  # noqa: E402

FIX = Path(__file__).parent.parent / "web" / "tests" / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads(gzip.decompress((FIX / f"{name}.graph.json.gz").read_bytes()))


@pytest.fixture
def server(monkeypatch):
    docs = {"Qwen/Qwen3-8B": _fixture("qwen3-8b"), "Qwen/Qwen2.5-7B": _fixture("qwen2.5-7b")}
    monkeypatch.setattr(mcp_server, "get_doc", lambda mid: docs[mid.strip()])
    return mcp_server.build_server()


def _text(result) -> str:
    # CallToolResult.content → text blocks (mcp 1.x and 2.x)
    content = getattr(result, "content", result)
    if isinstance(content, tuple):
        content = content[0]
    return "\n".join(getattr(c, "text", "") for c in content)


def test_tools_registered_with_schemas(server):
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"describe_model", "estimate_cost", "plan_serving", "compare_models", "list_modules", "search_models", "export_markdown"} <= names
    plan = next(t for t in tools if t.name == "plan_serving")
    schema = getattr(plan, "input_schema", None) or getattr(plan, "inputSchema", None)
    assert {"model_id", "gpus", "gpu_memory_gb", "tp", "pp", "T"} <= set(schema["properties"])
    assert "KV" in plan.description


def test_tool_calls(server):
    r = asyncio.run(server.call_tool("estimate_cost", {"model_id": "Qwen/Qwen3-8B", "T": 8192}))
    txt = _text(r)
    assert "kv_bytes_at_T" in txt and str(36 * 2 * 8 * 128 * 2 * 8192) in txt
    r = asyncio.run(server.call_tool("plan_serving", {"model_id": "Qwen/Qwen3-8B", "gpus": 2, "gpu_memory_gb": 24, "pp": 2}))
    assert '"fits": true' in _text(r).lower().replace(" ", " ") or '"fits":true' in _text(r).lower()
    r = asyncio.run(server.call_tool("compare_models", {"a": "Qwen/Qwen2.5-7B", "b": "Qwen/Qwen3-8B"}))
    assert "q_norm" in _text(r)
    r = asyncio.run(server.call_tool("list_modules", {"model_id": "Qwen/Qwen3-8B", "filter": "q_proj", "max_rows": 5}))
    assert "model.layers.0.self_attn.q_proj" in _text(r)
    r = asyncio.run(server.call_tool("export_markdown", {"model_id": "Qwen/Qwen3-8B"}))
    assert _text(r).startswith("# Qwen/Qwen3-8B")
    # bad assumptions surface as a tool error with the allowed values
    with pytest.raises(Exception, match="dtype must be one of"):
        asyncio.run(server.call_tool("describe_model", {"model_id": "Qwen/Qwen3-8B", "dtype": "q4"}))
