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

    def fake_extract(model_id, revision, token):
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


def test_gallery_and_health(client):
    g = client.get("/api/gallery").json()
    assert {"id", "blurb", "cached", "summary"} <= set(g[0])
    h = client.get("/api/health").json()
    assert h["ok"] and "cache_entries" in h and "inflight" in h
