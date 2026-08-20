"""Server behavior: cache wire format, de-dup, back-pressure, rate limit, validation."""

import concurrent.futures
import gzip
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

import modelmap.server as server
from modelmap import cache


def _doc(model_id: str) -> dict:
    return {
        "schema_version": 1, "model_id": model_id, "revision": "main", "fidelity": "full",
        "architecture": "T", "params_total": 42, "config": {}, "nodes": [], "repeats": [],
        "edges": [], "trace": [], "notes": [],
    }


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELMAP_CACHE", str(tmp_path))
    calls = []
    lock = threading.Lock()

    def fake_extract(model_id, revision, token, allow_local=False, trust_remote_code=False):
        with lock:
            calls.append(model_id)
        time.sleep(0.15)  # long enough for concurrent requests to pile up
        return _doc(model_id)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    monkeypatch.setattr(server, "_extract_job", fake_extract)
    monkeypatch.setattr(server, "_get_pool", lambda: pool)
    monkeypatch.setattr(server, "_reset_pool", lambda: None)
    monkeypatch.setattr(server, "_limiter", server.RateLimiter(per_min=600, burst=100))
    server._inflight.clear()
    with TestClient(server.app) as c:
        c.calls = calls
        yield c
    pool.shutdown(wait=False)


def test_bad_model_id_rejected(client):
    assert client.get("/api/graph/a/b/c").status_code == 400
    assert client.get("/api/graph/bad id/x").status_code == 400
    assert client.get("/api/graph/gpt2").status_code == 200  # legacy top-level ids are valid


def test_extract_then_serve_pregzipped_with_etag(client):
    r = client.get("/api/graph/o/m", headers={"accept-encoding": "identity"})
    assert r.status_code == 200
    assert json.loads(r.content)["model_id"] == "o/m"
    assert cache.has("o/m", "main")
    etag = r.headers["etag"]
    assert r.headers["cache-control"].startswith("public")

    r2 = client.get("/api/graph/o/m", headers={"if-none-match": etag})
    assert r2.status_code == 304

    # cached path: raw gzip bytes, served with Content-Encoding when accepted
    raw = cache.get_bytes("o/m", "main")
    assert raw[:2] == b"\x1f\x8b"
    assert json.loads(gzip.decompress(raw))["model_id"] == "o/m"
    assert client.calls == ["o/m"]  # second request never re-extracted


def test_concurrent_requests_share_one_extraction(client):
    def hit():
        return client.get("/api/graph/dedup/x").status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        codes = list(ex.map(lambda _: hit(), range(6)))
    assert codes == [200] * 6
    assert client.calls.count("dedup/x") == 1


def test_backpressure_returns_429(client, monkeypatch):
    monkeypatch.setattr(server.settings, "max_inflight", 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(lambda i=i: client.get(f"/api/graph/bp/m{i}").status_code) for i in range(3)]
        codes = sorted(f.result() for f in futs)
    assert codes[0] == 200 and 429 in codes


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(server, "_limiter", server.RateLimiter(per_min=60, burst=2))
    assert client.get("/api/graph/rl/a").status_code == 200
    assert client.get("/api/graph/rl/b").status_code == 200
    r = client.get("/api/graph/rl/c")
    assert r.status_code == 429 and "retry-after" in r.headers
    # cached models are never rate limited
    assert client.get("/api/graph/rl/a").status_code == 200


def test_tokened_requests_bypass_cache_and_are_private(client):
    r = client.get("/api/graph/priv/m", headers={"x-hf-token": "hf_secret"})
    assert r.status_code == 200
    assert r.headers["cache-control"] == "private, no-store"
    assert not cache.has("priv/m", "main")


def test_gallery_and_health(client, monkeypatch):
    import modelmap.server as srv
    monkeypatch.setattr(srv, "trending", lambda: [{"id": "t/one", "pipeline_tag": "text-generation", "architecture": "X"}])
    g = client.get("/api/gallery").json()
    assert set(g) == {"trending", "classics"}
    assert {"id", "blurb", "cached", "summary"} <= set(g["classics"][0])
    assert g["trending"][0]["id"] == "t/one" and g["trending"][0]["cached"] is False
    h = client.get("/api/health").json()
    assert h["ok"] and "cache_entries" in h and "inflight" in h


def test_cache_paths_keep_model_and_revision(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELMAP_CACHE", str(tmp_path))
    cache.put("Qwen/Qwen3.8-27B", "main", _doc("Qwen/Qwen3.8-27B"))
    cache.put("Qwen/Qwen3.8-27B", "v2", _doc("Qwen/Qwen3.8-27B"))
    files = sorted(p.name for p in tmp_path.glob("*.json.gz"))
    assert len(files) == 2 and all(f.startswith("Qwen--Qwen3.8-27B.") for f in files)
    assert cache.has("Qwen/Qwen3.8-27B", "v2") and not cache.has("Qwen/Qwen3.8-27B", "other")


def test_broken_pool_is_rebuilt_on_submit(client, monkeypatch):
    """A worker that died (memory cap, native fault) must not 500 every later request."""
    from concurrent.futures.process import BrokenProcessPool

    class Broken:
        def submit(self, *a, **k):
            raise BrokenProcessPool("child died")

    good = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    state = {"pool": Broken(), "resets": 0}
    monkeypatch.setattr(server, "_get_pool", lambda: state["pool"])

    def reset():
        state["resets"] += 1
        state["pool"] = good

    monkeypatch.setattr(server, "_reset_pool", reset)
    assert client.get("/api/graph/broken/pool").status_code == 200
    assert state["resets"] == 1
    good.shutdown(wait=False)


# ---------------------------------------------------------------- public API (§16)

FIX = __import__("pathlib").Path(__file__).parent.parent / "web" / "tests" / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads(gzip.decompress((FIX / f"{name}.graph.json.gz").read_bytes()))


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Client whose extractor serves the real qwen fixtures for two ids."""
    monkeypatch.setenv("MODELMAP_CACHE", str(tmp_path))
    docs = {"Qwen/Qwen3-8B": _fixture("qwen3-8b"), "Qwen/Qwen2.5-7B": _fixture("qwen2.5-7b")}

    def fake_extract(model_id, revision, token, allow_local=False, trust_remote_code=False):
        if model_id in docs:
            return docs[model_id]
        if model_id.startswith("local:"):
            if not allow_local:
                raise ValueError("local paths are not enabled on this server")
            return _doc(model_id)
        return _doc(model_id)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    monkeypatch.setattr(server, "_extract_job", fake_extract)
    monkeypatch.setattr(server, "_get_pool", lambda: pool)
    monkeypatch.setattr(server, "_reset_pool", lambda: None)
    monkeypatch.setattr(server, "_limiter", server.RateLimiter(per_min=600, burst=100))
    server._inflight.clear()
    with TestClient(server.app) as c:
        yield c
    pool.shutdown(wait=False)


def test_summary_endpoint(api):
    r = api.get("/api/summary/Qwen/Qwen3-8B?T=8192")
    assert r.status_code == 200
    s = r.json()
    assert s["layers"] == 36 and s["params_total"] == 8190735360
    assert s["cost"]["kv_bytes_at_T"] == 36 * 2 * 8 * 128 * 2 * 8192
    assert s["urls"]["explore"] == "/m/Qwen/Qwen3-8B"
    assert api.get("/api/summary/Qwen/Qwen3-8B?dtype=q4").status_code == 400
    # CORS: any origin may read
    r = api.get("/api/summary/Qwen/Qwen3-8B", headers={"Origin": "https://example.org"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_export_endpoint_formats(api):
    r = api.get("/api/export/Qwen/Qwen3-8B?format=csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.text.splitlines()[0].startswith("module,kind,class") and "q_proj" in r.text
    r = api.get("/api/export/Qwen/Qwen3-8B?format=md&T=1024&download=1")
    assert r.status_code == 200 and r.text.startswith("# Qwen/Qwen3-8B")
    assert "attachment" in r.headers["content-disposition"] and "Qwen--Qwen3-8B.md" in r.headers["content-disposition"]
    r = api.get("/api/export/Qwen/Qwen3-8B?format=dot")
    assert r.text.startswith("digraph")
    r = api.get("/api/export/Qwen/Qwen3-8B?format=json")
    assert r.json()["model_id"] == "Qwen/Qwen3-8B"
    assert api.get("/api/export/Qwen/Qwen3-8B?format=xlsx").status_code == 422


def test_plan_endpoint(api):
    r = api.get("/api/plan/Qwen/Qwen3-8B?gpus=2&gpu_memory_gb=24&tp=1&pp=2&T=4096")
    assert r.status_code == 200
    p = r.json()
    assert p["fits"] is True and len(p["stages"]) == 2
    assert p["stages"][0]["layers"][0] == 0 and p["stages"][1]["layers"][1] == 35
    assert p["max_context_tokens"] > 4096
    assert api.get("/api/plan/Qwen/Qwen3-8B?dtype=nope").status_code == 400


def test_compare_endpoint(api):
    r = api.get("/api/compare?a=Qwen/Qwen2.5-7B&b=Qwen/Qwen3-8B")
    assert r.status_code == 200
    d = r.json()
    assert d["counts"]["removed"] == 0 and d["counts"]["added"] > 0
    assert any(c["field"] == "num_hidden_layers" and c["b"] == "36" for c in d["config_diff"])
    assert all(p["status"] != "same" for p in d["pairs"])
    r = api.get("/api/compare?a=Qwen/Qwen2.5-7B&b=Qwen/Qwen3-8B&format=md")
    assert r.headers["content-type"].startswith("text/markdown") and "q_norm" in r.text
    assert api.get("/api/compare?a=bad id&b=x").status_code == 400


def test_variant_ids_and_local_ids(api, monkeypatch):
    # owner/name:variant is a valid id (GGUF quant selector)
    assert api.get("/api/graph/Qwen/Qwen3-8B-GGUF:Q8_0").status_code == 200
    # local paths are refused unless enabled (403, not a Hub lookup)
    assert api.get("/api/graph/local:/tmp/x").status_code == 403
    monkeypatch.setattr(server.settings, "allow_local", True)
    import os, tempfile
    d = tempfile.mkdtemp()
    r = api.get(f"/api/graph/local:{d}")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "private, no-store"  # local docs are never cached
    assert api.get("/api/graph/local:/definitely/not/here").status_code == 404


def test_frame_policy_and_spa(api):
    # API responses refuse framing; the SPA (if built) may be embedded anywhere
    assert api.get("/api/health").headers.get("x-frame-options") == "SAMEORIGIN"
    r = api.get("/m/Qwen/Qwen3-8B?embed=1")
    if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
        assert "x-frame-options" not in r.headers


def test_hub_rate_limit_maps_to_503_with_retry_after(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELMAP_CACHE", str(tmp_path))

    def angry(model_id, revision, token, allow_local=False, trust_remote_code=False):
        raise RuntimeError("HfHubHTTPError: 429 Too Many Requests: you have reached your 'api' rate limit.\nRetry after 86 seconds (0/500 requests remaining)")

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(server, "_extract_job", angry)
    monkeypatch.setattr(server, "_get_pool", lambda: pool)
    monkeypatch.setattr(server, "_reset_pool", lambda: None)
    monkeypatch.setattr(server, "_limiter", server.RateLimiter(per_min=600, burst=100))
    server._inflight.clear()
    with TestClient(server.app) as c:
        r = c.get("/api/graph/some/model")
    assert r.status_code == 503 and r.headers["retry-after"] == "86" and "token" in r.json()["detail"]
    pool.shutdown(wait=False)


def test_train_endpoint_and_plan_throughput(api):
    r = api.get("/api/train/Qwen/Qwen3-8B?method=qlora&lora_rank=16&gpus=1&gpu_memory_gb=24&T=2048")
    assert r.status_code == 200
    d = r.json()
    assert d["fits"] is True and round(d["trainable_params"] / 1e6, 1) == 43.6
    assert d["max_microbatch"] >= 8
    r = api.get("/api/train/Qwen/Qwen3-8B?method=full&gpus=1&gpu_memory_gb=24&T=2048")
    assert r.json()["fits"] is False
    assert api.get("/api/train/Qwen/Qwen3-8B?method=galore").status_code == 422
    assert api.get("/api/train/Qwen/Qwen3-8B?gpu=GTX+9999").status_code == 400
    # serve plan with a named GPU adds throughput
    r = api.get("/api/plan/Qwen/Qwen3-8B?gpus=1&gpu_memory_gb=80&gpu=A100+80GB&T=4096")
    t = r.json()["throughput"]
    assert 65 < t["decode_tok_per_sec_b1"] < 80 and t["gpu"] == "A100 80GB"
    assert api.get("/api/plan/Qwen/Qwen3-8B?gpu=nope").status_code == 400
