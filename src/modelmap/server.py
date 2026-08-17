"""FastAPI service: graph API + static SPA (design doc §04, hardened per §10/§05).

Request path for /api/graph:
  cache hit  → pre-gzipped bytes straight from disk (ETag / 304 aware)
  cache miss → rate-limit check → in-flight de-dup (N visitors, one extraction)
             → back-pressure (429 when the queue is full)
             → worker process with hard timeout + resource limits → cache → serve
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import multiprocessing
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from modelmap import __version__, cache
from modelmap.gallery import GALLERY
from modelmap.ratelimit import RateLimiter
from modelmap.schema import SCHEMA_VERSION
from modelmap.settings import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- worker pool

_pool: concurrent.futures.ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _make_pool() -> concurrent.futures.ProcessPoolExecutor:
    # spawn, not fork: forking the running uvicorn process (with live event-loop
    # threads) can deadlock the child nondeterministically
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=settings.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(settings.worker_mem_mb, settings.extraction_timeout_s),
    )


def _get_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _make_pool()
        return _pool


def _reset_pool() -> None:
    # a timed-out worker can't be killed through the executor API; abandon the
    # pool (the stuck process dies on its own or at exit) and start fresh
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = _make_pool()


def _worker_init(mem_mb: int, timeout_s: int) -> None:
    """Sandbox-lite for extraction processes: quiet, single-threaded, and
    hard-capped on address space and CPU time so a hostile repo can at worst
    kill its own worker. Platform-level isolation (container, no secrets,
    egress limited to the Hub) is documented in DEPLOY.md."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        import resource

        cap = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        cpu = timeout_s + 30
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except (ImportError, ValueError, OSError) as e:  # non-POSIX or disallowed
        log.warning("worker rlimits not applied: %s", e)


def _extract_job(model_id: str, revision: str, token: str | None) -> dict:
    from modelmap.extract import extract_graph

    return extract_graph(model_id, revision=revision, token=token).to_json_dict()


# ------------------------------------------------------- de-dup / back-pressure

# Hub ids: "owner/name", or a legacy top-level name like "gpt2"
_MODEL_ID = re.compile(r"[A-Za-z0-9][\w.\-]{0,95}(/[A-Za-z0-9][\w.\-]{0,95})?")

_inflight: dict[str, concurrent.futures.Future] = {}
_inflight_lock = threading.Lock()
_limiter = RateLimiter(settings.rate_per_min, settings.rate_burst)


def _client_key(request: Request) -> str:
    if settings.trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _submit_shared(key: str, model_id: str, revision: str, token: str | None):
    """One extraction per key: concurrent requests share the same future.
    Returns (future, is_owner)."""
    with _inflight_lock:
        fut = _inflight.get(key)
        if fut is not None:
            return fut, False
        if len(_inflight) >= settings.max_inflight:
            raise HTTPException(
                429,
                "the extractor is busy; try again in a few seconds",
                headers={"Retry-After": "5"},
            )
        fut = _get_pool().submit(_extract_job, model_id, revision, token)
        _inflight[key] = fut
        return fut, True


def _release(key: str) -> None:
    with _inflight_lock:
        _inflight.pop(key, None)


def _run_extraction(model_id: str, revision: str, token: str | None) -> bytes:
    """Blocking: returns the gzip wire bytes, or raises HTTPException with a
    friendly status. The request that owns the extraction is the only one
    that writes the cache; de-duplicated followers just encode the result."""
    key = f"{model_id}@{revision}"
    if token is not None:  # same token + repo still shares one extraction
        key += "#" + hashlib.sha1(token.encode()).hexdigest()[:12]
    fut, owner = _submit_shared(key, model_id, revision, token)
    try:
        doc = fut.result(timeout=settings.extraction_timeout_s)
        if owner and token is None:
            return cache.put(model_id, revision, doc)
        return cache.encode(doc)
    except concurrent.futures.TimeoutError:
        if owner:
            _reset_pool()
        raise HTTPException(504, f"extraction exceeded {settings.extraction_timeout_s}s")
    except HTTPException:
        raise
    except Exception as e:
        name = type(e).__name__
        if "Connect" in name or "Timeout" in name:
            raise HTTPException(
                502,
                f"the Hugging Face Hub was unreachable while extracting '{model_id}' "
                "— usually transient; try again",
            )
        if "BrokenProcessPool" in name or "MemoryError" in name:
            if owner:
                _reset_pool()
            raise HTTPException(
                422, f"extraction of '{model_id}' exceeded the worker's memory limit"
            )
        msg = str(e)
        low = msg.lower()
        if "gated" in low or ("401" in msg and "token" in low):
            raise HTTPException(
                403,
                f"'{model_id}' is a gated repo — accept its terms on huggingface.co and add "
                "your HF token (top bar → token) to view it",
            )
        if "NotFound" in name or "not a valid model identifier" in msg or "404" in msg:
            raise HTTPException(404, f"'{model_id}' was not found on the Hugging Face Hub")
        raise HTTPException(422, f"could not extract '{model_id}': {name}: {msg}")
    finally:
        if owner:
            _release(key)


# ------------------------------------------------------------------ warming


def warm(model_ids: list[str]) -> None:
    """Extract (and cache) each id in turn; skips ones already cached. Runs
    inside the server process on a thread, or from `modelmap warm`."""
    for mid in model_ids:
        if cache.has(mid, "main"):
            continue
        t0 = time.time()
        try:
            _run_extraction(mid, "main", None)
            log.info("warmed %s (%.1fs)", mid, time.time() - t0)
        except Exception as e:  # keep going; a bad entry shouldn't stop warming
            log.warning("warm failed for %s: %s", mid, e)


# ---------------------------------------------------------------------- app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if settings.warm_on_start:
        threading.Thread(
            target=warm, args=([g["id"] for g in GALLERY],), name="warm", daemon=True
        ).start()
    yield
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="modelmap", version=__version__, lifespan=_lifespan)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return resp


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "cache_entries": cache.count(),
        "workers": settings.workers,
        "inflight": len(_inflight),
    }


def _graph_response(raw: bytes, request: Request, cacheable: bool) -> Response:
    etag = '"' + hashlib.sha1(raw).hexdigest()[:20] + '"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    headers = {"ETag": etag, "Vary": "Accept-Encoding"}
    if cacheable:
        headers["Cache-Control"] = f"public, max-age={settings.cache_max_age_s}"
    else:
        headers["Cache-Control"] = "private, no-store"
    if "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
        return Response(raw, media_type="application/json", headers=headers)
    import gzip

    return Response(gzip.decompress(raw), media_type="application/json", headers=headers)


@app.get("/api/graph/{model_id:path}")
def graph(
    model_id: str,
    request: Request,
    revision: str = "main",
    refresh: bool = False,
    x_hf_token: str | None = Header(default=None),
):
    model_id = model_id.strip("/")
    if not _MODEL_ID.fullmatch(model_id):
        raise HTTPException(400, "model id must look like 'owner/name'")

    # tokened (possibly private) repos bypass the shared cache in both directions
    if x_hf_token is None and not refresh:
        raw = cache.get_bytes(model_id, revision)
        if raw is not None:
            return _graph_response(raw, request, cacheable=True)

    ok, retry = _limiter.allow(_client_key(request))
    if not ok:
        raise HTTPException(
            429, "too many extractions from this client; slow down",
            headers={"Retry-After": str(int(retry) + 1)},
        )

    raw = _run_extraction(model_id, revision, x_hf_token)
    return _graph_response(raw, request, cacheable=x_hf_token is None)


@app.get("/api/gallery")
def gallery():
    out = []
    for g in GALLERY:
        s = cache.summary(g["id"], "main") if cache.has(g["id"], "main") else None
        out.append({**g, "cached": s is not None, "summary": s})
    return out


@app.get("/api/search")
def search(
    request: Request, q: str = Query(min_length=1), limit: int = Query(default=10, le=50)
):
    ok, retry = _limiter.allow(_client_key(request))
    if not ok:
        raise HTTPException(429, "slow down", headers={"Retry-After": str(int(retry) + 1)})
    from huggingface_hub import HfApi

    models = HfApi().list_models(search=q, limit=limit, sort="downloads")
    return [
        {"id": m.id, "downloads": m.downloads, "likes": m.likes, "pipeline_tag": m.pipeline_tag}
        for m in models
    ]


_web = Path(__file__).parent / "web"
if (_web / "index.html").exists():  # SPA build output (web/ → npm run build)
    # catch-all so client routes like /m/Qwen/Qwen3-8B survive a refresh;
    # /api/* routes are declared above and win
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = (_web / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(_web):
            # Vite hashes asset filenames → immutable; fonts are stable too
            headers = (
                {"Cache-Control": "public, max-age=31536000, immutable"}
                if path.startswith(("assets/", "fonts/"))
                else {"Cache-Control": "no-cache"}
            )
            return FileResponse(candidate, headers=headers)
        return FileResponse(_web / "index.html", headers={"Cache-Control": "no-cache"})
