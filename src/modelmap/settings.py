"""Runtime configuration from environment variables (documented in DEPLOY.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    workers: int = field(default_factory=lambda: max(1, _int("MODELMAP_WORKERS", 2)))
    extraction_timeout_s: int = field(default_factory=lambda: _int("MODELMAP_TIMEOUT", 180))
    # simultaneous distinct extractions (queued + running) before 429
    max_inflight: int = field(default_factory=lambda: _int("MODELMAP_MAX_INFLIGHT", 8))
    # per-client budget for *uncached* extractions and Hub searches
    rate_per_min: int = field(default_factory=lambda: _int("MODELMAP_RATE_PER_MIN", 20))
    rate_burst: int = field(default_factory=lambda: _int("MODELMAP_RATE_BURST", 10))
    trust_proxy: bool = field(default_factory=lambda: _bool("MODELMAP_TRUST_PROXY", False))
    # worker sandbox: address-space cap and CPU-time cap for each extraction process
    worker_mem_mb: int = field(default_factory=lambda: _int("MODELMAP_WORKER_MEM_MB", 4096))
    warm_on_start: bool = field(default_factory=lambda: _bool("MODELMAP_WARM", False))
    cache_max_age_s: int = field(default_factory=lambda: _int("MODELMAP_CACHE_MAX_AGE", 300))
    # "local:/path" ids read checkpoints from this machine's disk — the CLI
    # turns it on for loopback serving; never set it on a public deployment
    allow_local: bool = field(default_factory=lambda: _bool("MODELMAP_ALLOW_LOCAL", False))
    # execute repos' own modeling Python during extraction (arbitrary code!) —
    # `modelmap serve --trust-remote-code` on your own machine only; the
    # hosted deployment must never set it
    trust_remote_code: bool = field(default_factory=lambda: _bool("MODELMAP_TRUST_REMOTE_CODE", False))


settings = Settings()
