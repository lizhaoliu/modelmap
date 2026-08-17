"""Disk cache for extracted graphs, keyed by (model_id, revision).

Each entry is the gzipped JSON document (served as-is with Content-Encoding:
gzip) plus a tiny sidecar summary used by the gallery without decompressing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def cache_dir() -> Path:
    root = os.environ.get("MODELMAP_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")), "modelmap"
    )
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stem(model_id: str, revision: str) -> Path:
    key = hashlib.sha1(f"{model_id}@{revision}".encode()).hexdigest()[:16]
    return cache_dir() / f"{model_id.replace('/', '--')}.{key}"


def _path(model_id: str, revision: str) -> Path:
    return _stem(model_id, revision).with_suffix(".json.gz")


def _meta_path(model_id: str, revision: str) -> Path:
    return _stem(model_id, revision).with_suffix(".meta.json")


def has(model_id: str, revision: str) -> bool:
    return _path(model_id, revision).exists()


def get_bytes(model_id: str, revision: str) -> bytes | None:
    """Raw gzip bytes of the cached document (the wire format)."""
    p = _path(model_id, revision)
    try:
        return p.read_bytes()
    except OSError:
        return None


def get(model_id: str, revision: str) -> dict[str, Any] | None:
    raw = get_bytes(model_id, revision)
    if raw is None:
        return None
    try:
        return json.loads(gzip.decompress(raw))
    except (OSError, json.JSONDecodeError):
        return None


def encode(doc: dict[str, Any]) -> bytes:
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode("utf-8"), mtime=0)


def summarize(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": doc.get("model_id"),
        "revision": doc.get("revision"),
        "architecture": doc.get("architecture"),
        "params_total": doc.get("params_total"),
        "fidelity": doc.get("fidelity"),
        "nodes": len(doc.get("nodes") or []),
        "trace_steps": len(doc.get("trace") or []),
        "cached_at": time.time(),
    }


def put(model_id: str, revision: str, doc: dict[str, Any]) -> bytes:
    """Store the document; returns the gzip bytes so the caller can serve them."""
    raw = encode(doc)
    _atomic_write(_path(model_id, revision), raw)
    _atomic_write(_meta_path(model_id, revision), json.dumps(summarize(doc)).encode())
    return raw


def _atomic_write(p: Path, data: bytes) -> None:
    # unique temp name per writer, then rename: concurrent writers can't
    # clobber each other's temp file, and readers never see a partial file
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(p)


def summary(model_id: str, revision: str) -> dict[str, Any] | None:
    """Sidecar summary; rebuilt from the document for entries written before sidecars."""
    mp = _meta_path(model_id, revision)
    try:
        return json.loads(mp.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    doc = get(model_id, revision)
    if doc is None:
        return None
    s = summarize(doc)
    try:
        mp.write_text(json.dumps(s))
    except OSError:
        pass
    return s


def count() -> int:
    return sum(1 for _ in cache_dir().glob("*.json.gz"))
