"""FastAPI service: graph API + static SPA (design doc §04, hardened per §10/§05).

Request path for /api/graph:
  cache hit  → pre-gzipped bytes straight from disk (ETag / 304 aware)
  cache miss → rate-limit check → in-flight de-dup (N visitors, one extraction)
             → back-pressure (429 when the queue is full)
             → worker process with hard timeout + resource limits → cache → serve

The public API (design doc §16; OpenAPI at /docs):
  GET /api/graph/{id}            the graph document (gzip)
  GET /api/summary/{id}          headline numbers + cost estimates (small JSON)
  GET /api/export/{id}?format=   csv | md | json | dot renderings
  GET /api/plan/{id}?gpus=…      serving placement estimate (§17)
  GET /api/compare?a=&b=         module-by-module diff (json | md)
  GET /api/gallery, /api/search  landing data, Hub search
All GET, CORS-enabled for any origin; ids accept owner/name[:variant].
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from modelmap import __version__, cache
from modelmap.analytics import Assumptions, PlanRequest, WHATIF_DTYPES, plan_serving, summarize
from modelmap.gallery import CLASSICS, trending, trending_ids
from modelmap.ids import is_local, parse_model_id
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
        initargs=(settings.worker_mem_mb,),
        max_tasks_per_child=64,  # recycle workers: bounds any slow memory growth
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
    with _pool_lock:
        _reset_pool_locked()


def _reset_pool_locked() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = _make_pool()


def _worker_init(mem_mb: int) -> None:
    """Sandbox-lite for extraction processes: quiet, single-threaded, and
    hard-capped on address space so a hostile repo can at worst kill its own
    worker; runaway *time* is bounded by the wall-clock timeout + pool reset
    (RLIMIT_CPU would be a per-process lifetime cap, wrong for a pool worker).
    Platform-level isolation is documented in DEPLOY.md."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # die with the parent: a SIGKILLed server must not leave 350 MB workers behind
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):  # non-Linux
        pass
    try:
        import resource

        cap = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except (ImportError, ValueError, OSError) as e:  # non-POSIX or disallowed
        log.warning("worker rlimits not applied: %s", e)


def _extract_job(model_id: str, revision: str, token: str | None, allow_local: bool = False) -> dict:
    from modelmap.extract import extract_graph

    return extract_graph(model_id, revision=revision, token=token, allow_local=allow_local).to_json_dict()


# ------------------------------------------------------- de-dup / back-pressure

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
        try:
            fut = _get_pool().submit(_extract_job, model_id, revision, token, settings.allow_local)
        except concurrent.futures.process.BrokenProcessPool:
            # a worker died (memory cap, native fault): rebuild once and retry
            log.warning("extraction pool was broken; rebuilding")
            _reset_pool()
            fut = _get_pool().submit(_extract_job, model_id, revision, token, settings.allow_local)
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
        # local checkpoints change under us: never cached on disk
        if owner and token is None and not is_local(model_id):
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
        if "429" in msg and "rate limit" in low:
            m = re.search(r"retry after (\d+)", low)
            wait = int(m.group(1)) if m else 60
            raise HTTPException(
                503,
                f"the Hugging Face Hub is rate-limiting this server's requests right now (shared IP); "
                f"try again in about {wait} s — or add your HF token (top bar → token), which carries its own quota",
                headers={"Retry-After": str(wait)},
            )
        if "NotFound" in name or "not a valid model identifier" in msg or "repository not found" in low or "404" in msg:
            # the Hub answers 401 "Repository Not Found" for unknown *and* private repos alike
            raise HTTPException(
                404,
                f"'{model_id}' was not found on the Hugging Face Hub — if it is private, add your HF token (top bar → token)",
            )
        if "gated" in low or ("401" in msg and "token" in low):
            raise HTTPException(
                403,
                f"'{model_id}' is a gated repo — accept its terms on huggingface.co and add "
                "your HF token (top bar → token) to view it",
            )
        if name in ("LocalPathError", "GGUFError"):
            raise HTTPException(422, msg)
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
            target=lambda: warm([g["id"] for g in CLASSICS] + trending_ids()), name="warm", daemon=True
        ).start()
    yield
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="modelmap",
    version=__version__,
    lifespan=_lifespan,
    description=(
        "Architecture graphs, cost estimates and diffs for Hugging Face models — "
        "no weights downloaded. Read-only; every endpoint is GET and CORS-open."
    ),
)
# the API is a public read-only data source: any origin may call it (§16)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["X-HF-Token", "If-None-Match"], expose_headers=["ETag", "Content-Disposition"], max_age=86400,
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # the SPA may be framed anywhere (?embed=1 in model cards, blogs, docs);
    # API responses are data and never need framing
    if request.url.path.startswith("/api/"):
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
        "allow_local": settings.allow_local,
    }


def _check_id(model_id: str) -> str:
    model_id = model_id.strip("/")
    try:
        parse_model_id(model_id, allow_local=settings.allow_local)
    except ValueError as e:  # LocalPathError is a ValueError
        raise HTTPException(400 if "must look like" in str(e) else 403 if "not enabled" in str(e) else 404, str(e))
    return model_id


def _assumptions(T: int | None, B: int | None, dtype: str | None) -> Assumptions:
    a = Assumptions()
    if T is not None:
        a.T = max(1, min(T, 1 << 22))
    if B is not None:
        a.B = max(1, min(B, 65536))
    if dtype:
        if dtype not in WHATIF_DTYPES:
            raise HTTPException(400, f"dtype must be one of {', '.join(WHATIF_DTYPES)}")
        a.dtype = dtype
    return a


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
    model_id = _check_id(model_id)
    raw, cacheable = _graph_bytes(model_id, revision, refresh, x_hf_token, request)
    return _graph_response(raw, request, cacheable=cacheable)


def _graph_bytes(model_id: str, revision: str, refresh: bool, token: str | None, request: Request) -> tuple[bytes, bool]:
    """Cached gzip document or a fresh extraction; (bytes, cacheable)."""
    # tokened (possibly private) repos bypass the shared cache in both directions
    if token is None and not refresh and not is_local(model_id):
        raw = cache.get_bytes(model_id, revision)
        if raw is not None:
            return raw, True
    ok, retry = _limiter.allow(_client_key(request))
    if not ok:
        raise HTTPException(
            429, "too many extractions from this client; slow down",
            headers={"Retry-After": str(int(retry) + 1)},
        )
    return _run_extraction(model_id, revision, token), token is None and not is_local(model_id)


def _load_doc(model_id: str, revision: str, token: str | None, request: Request) -> dict:
    import gzip
    import json

    raw, _ = _graph_bytes(model_id, revision, False, token, request)
    return json.loads(gzip.decompress(raw))


@app.get("/api/summary/{model_id:path}")
def summary(
    model_id: str,
    request: Request,
    revision: str = "main",
    T: int | None = Query(default=None, description="sequence length for the cost estimates"),
    B: int | None = Query(default=None, description="batch size"),
    dtype: str | None = Query(default=None, description="activation / what-if dtype: bf16 f16 f32 f8 int8 int4"),
    x_hf_token: str | None = Header(default=None),
):
    """Headline numbers for a model: params, active params, stacks, config
    essentials, and compute / memory / KV-cache estimates at T, B, dtype."""
    model_id = _check_id(model_id)
    doc = _load_doc(model_id, revision, x_hf_token, request)
    out = summarize(doc, _assumptions(T, B, dtype))
    out["urls"] = {
        "explore": f"/m/{model_id}",
        "graph": f"/api/graph/{model_id}",
        "export_csv": f"/api/export/{model_id}?format=csv",
        "export_md": f"/api/export/{model_id}?format=md",
    }
    return out


@app.get("/api/export/{model_id:path}")
def export_model(
    model_id: str,
    request: Request,
    format: str = Query(default="csv", pattern="^(csv|md|markdown|json|dot)$"),
    revision: str = "main",
    T: int | None = None,
    B: int | None = None,
    dtype: str | None = None,
    leaves_only: bool = Query(default=False, description="csv: drop container rows"),
    depth: int = Query(default=3, ge=1, le=8, description="dot: cluster depth"),
    download: bool = Query(default=False, description="send as an attachment"),
    x_hf_token: str | None = Header(default=None),
):
    """The model rendered for other tools: a per-module CSV (params, shapes,
    dtype, cost columns), a Markdown summary, the raw JSON document, or a
    Graphviz DOT file."""
    from modelmap.export import render

    model_id = _check_id(model_id)
    doc = _load_doc(model_id, revision, x_hf_token, request)
    text, media = render(doc, format, _assumptions(T, B, dtype), leaves_only=leaves_only, depth=depth, pretty=True)
    ext = {"csv": "csv", "md": "md", "markdown": "md", "json": "json", "dot": "dot"}[format]
    fname = model_id.replace("/", "--").replace(":", "_") + f".{ext}"
    headers = {"Content-Disposition": f"{'attachment' if download else 'inline'}; filename=\"{fname}\""}
    if x_hf_token is None and not is_local(model_id):
        headers["Cache-Control"] = f"public, max-age={settings.cache_max_age_s}"
    return PlainTextResponse(text, media_type=media + "; charset=utf-8", headers=headers)


@app.get("/api/plan/{model_id:path}")
def plan(
    model_id: str,
    request: Request,
    revision: str = "main",
    gpus: int = Query(default=1, ge=1, le=4096),
    gpu_memory_gb: float = Query(default=80, ge=0, le=100000),
    tp: int = Query(default=1, ge=1, le=4096, description="tensor-parallel degree"),
    pp: int = Query(default=1, ge=1, le=4096, description="pipeline-parallel degree"),
    T: int = Query(default=4096, ge=1, le=1 << 22),
    B: int = Query(default=1, ge=1, le=65536),
    dtype: str = Query(default="bf16"),
    headroom: float = Query(default=0.1, ge=0, le=0.9),
    x_hf_token: str | None = Header(default=None),
):
    """Serving placement estimate: per-GPU weights / KV / activation bytes for
    a TP × PP layout, which layers land on which stage, whether it fits, and
    the KV-limited maximum context at this batch (design doc §17)."""
    model_id = _check_id(model_id)
    if dtype not in WHATIF_DTYPES:
        raise HTTPException(400, f"dtype must be one of {', '.join(WHATIF_DTYPES)}")
    doc = _load_doc(model_id, revision, x_hf_token, request)
    req = PlanRequest(gpus=gpus, gpu_memory_gb=gpu_memory_gb, tp=tp, pp=pp, T=T, B=B, dtype=dtype, headroom=headroom)
    return plan_serving(doc, req).to_dict()


@app.get("/api/compare")
def compare_models(
    request: Request,
    a: str = Query(description="model id A"),
    b: str = Query(description="model id B"),
    format: str = Query(default="json", pattern="^(json|md|markdown)$"),
    changed_only: bool = Query(default=True, description="json: omit identical pairs"),
    x_hf_token: str | None = Header(default=None),
):
    """Align two module trees (by path, then role) and list what changed,
    was added, or removed — plus the config diff."""
    from modelmap.compare import align, diff_markdown

    a, b = _check_id(a), _check_id(b)
    da = _load_doc(a, "main", x_hf_token, request)
    db = _load_doc(b, "main", x_hf_token, request)
    al = align(da, db)
    if format != "json":
        return PlainTextResponse(diff_markdown(da, db, al), media_type="text/markdown; charset=utf-8")
    return {"a": a, "b": b, **al.to_dict(changed_only=changed_only)}


def _with_cache(entries: list[dict]) -> list[dict]:
    out = []
    for g in entries:
        s = cache.summary(g["id"], "main") if cache.has(g["id"], "main") else None
        out.append({**g, "cached": s is not None, "summary": s})
    return out


@app.get("/api/gallery")
def gallery():
    """Trending on the Hub right now (ungated transformers repos, deduped,
    refreshed hourly) plus the classics; each with its cache status."""
    return {"trending": _with_cache(trending()), "classics": _with_cache(CLASSICS)}


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
