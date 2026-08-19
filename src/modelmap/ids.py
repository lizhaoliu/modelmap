"""Model-id grammar (torch-free, shared by the server, CLI and extractor).

  owner/name            a Hub repo; legacy top-level names ("gpt2") work too
  owner/name:Q4_K_M     a GGUF variant inside the repo (any label returned in
                        the document's `variants`, case-insensitive)
  local:/path/to/ckpt   a local checkpoint dir / safetensors file / .gguf —
                        only where the caller allows it (CLI: yes; the hosted
                        server: no)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

LOCAL_PREFIX = "local:"
_HUB_ID = re.compile(
    r"^(?P<repo>[A-Za-z0-9][\w.\-]{0,95}(/[A-Za-z0-9][\w.\-]{0,95})?)(?::(?P<variant>[A-Za-z0-9][\w.\-]{0,63}))?$"
)


class LocalPathError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    """Where a model id points."""

    model_id: str  # as given (the cache key and the document's model_id)
    repo: str  # Hub repo id, or the absolute local path
    variant: str | None = None  # requested GGUF quant label
    local: bool = False


def is_local(model_id: str) -> bool:
    return model_id.strip().startswith(LOCAL_PREFIX)


def parse_model_id(model_id: str, *, allow_local: bool = False) -> Source:
    model_id = model_id.strip()
    if model_id.startswith(LOCAL_PREFIX):
        if not allow_local:
            raise LocalPathError("local paths are not enabled on this server")
        path = os.path.abspath(os.path.expanduser(model_id[len(LOCAL_PREFIX):]))
        if not os.path.exists(path):
            raise LocalPathError(f"no such file or directory: {path}")
        return Source(model_id=model_id, repo=path, local=True)
    m = _HUB_ID.match(model_id)
    if not m:
        raise ValueError("model id must look like 'owner/name' (optionally ':variant')")
    return Source(model_id=model_id, repo=m.group("repo"), variant=m.group("variant"))


def valid_hub_id(model_id: str) -> bool:
    return _HUB_ID.match(model_id.strip()) is not None
