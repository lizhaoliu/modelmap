"""Disk cache for extracted graphs, keyed by (model_id, revision)."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_dir() -> Path:
    root = os.environ.get("MODELMAP_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")), "modelmap"
    )
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(model_id: str, revision: str) -> Path:
    key = hashlib.sha1(f"{model_id}@{revision}".encode()).hexdigest()[:16]
    return cache_dir() / f"{model_id.replace('/', '--')}.{key}.json.gz"


def get(model_id: str, revision: str) -> dict[str, Any] | None:
    p = _path(model_id, revision)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def put(model_id: str, revision: str, doc: dict[str, Any]) -> Path:
    p = _path(model_id, revision)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    tmp.replace(p)
    return p
