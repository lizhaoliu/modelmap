"""FastAPI service: graph API now, static SPA hosting in M2 (design doc §04)."""

from __future__ import annotations

import concurrent.futures
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from modelmap import __version__, cache
from modelmap.schema import SCHEMA_VERSION

log = logging.getLogger(__name__)

EXTRACTION_TIMEOUT_S = 180

# extraction runs in a worker process: isolates the API from native faults and
# gives a hard timeout on pathological repos (§10). Real sandboxing for the
# hosted deployment is an M4 requirement (§05 security note).
_pool: concurrent.futures.ProcessPoolExecutor | None = None


def _get_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    return _pool


def _reset_pool() -> None:
    # a timed-out worker can't be killed through the executor API; abandon the
    # pool (the stuck process dies on its own or at exit) and start fresh
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)


def _extract_job(model_id: str, revision: str, token: str | None) -> dict:
    from modelmap.extract import extract_graph

    return extract_graph(model_id, revision=revision, token=token).to_json_dict()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="modelmap", version=__version__, lifespan=_lifespan)


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "schema_version": SCHEMA_VERSION}


@app.get("/api/graph/{model_id:path}")
def graph(
    model_id: str,
    revision: str = "main",
    refresh: bool = False,
    x_hf_token: str | None = Header(default=None),
):
    # tokened (possibly private) repos bypass the shared cache in both directions
    if x_hf_token is None and not refresh:
        hit = cache.get(model_id, revision)
        if hit is not None:
            return hit

    future = _get_pool().submit(_extract_job, model_id, revision, x_hf_token)
    try:
        doc = future.result(timeout=EXTRACTION_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        _reset_pool()
        raise HTTPException(504, f"extraction exceeded {EXTRACTION_TIMEOUT_S}s")
    except Exception as e:
        name = type(e).__name__
        status = 404 if "NotFound" in name else 403 if "Gated" in name else 422
        raise HTTPException(status, f"could not extract '{model_id}': {name}: {e}")

    if x_hf_token is None:
        cache.put(model_id, revision, doc)
    return doc


@app.get("/api/search")
def search(q: str = Query(min_length=1), limit: int = Query(default=10, le=50)):
    from huggingface_hub import HfApi

    models = HfApi().list_models(search=q, limit=limit, sort="downloads")
    return [
        {"id": m.id, "downloads": m.downloads, "likes": m.likes, "pipeline_tag": m.pipeline_tag}
        for m in models
    ]


_web = Path(__file__).parent / "web"
if (_web / "index.html").exists():  # M2 build output
    app.mount("/", StaticFiles(directory=_web, html=True), name="spa")
