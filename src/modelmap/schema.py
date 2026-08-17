"""Graph document schema — one JSON document per (model_id, revision).

See docs/design.html §06. The document is the single contract between the
Python extractor and the web client; bump SCHEMA_VERSION on breaking changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 2  # 2: node.attrs, MoE/norm classification fix (cache key includes this)

# node kinds drive color, icon, and collapse defaults in the client
KINDS = (
    "embedding", "attention", "mlp", "moe", "norm",
    "linear", "conv", "head", "container", "module",
)


@dataclass
class Node:
    id: str                 # qualified module path; "" is the root module
    kind: str
    cls: str                # python class name, e.g. "Qwen3Attention"
    parent: str | None      # None only for the root
    depth: int
    order: int              # registration index within the parent
    params: int             # parameters in this subtree
    dtype: str | None = None
    weight_shapes: dict[str, list[int]] | None = None  # own (non-recursive) params
    # per-module metadata from annotators (annotate.py): extra_repr attributes
    # like in_features / kernel_size / eps, and _src / _src_url for the class
    attrs: dict[str, Any] | None = None


@dataclass
class Repeat:
    parent: str             # container whose children collapsed
    representative: str     # only this child's subtree ships in `nodes`
    count: int
    signature: str          # structural hash proving the members identical
    members: list[str]      # collapsed child names, e.g. ["0", ..., "35"]


@dataclass
class Edge:
    src: str
    dst: str


@dataclass
class TraceStep:
    step: int
    node: str               # may reference a collapsed member; client maps via repeats
    inputs: list[list[int]]
    outputs: list[list[int]]


@dataclass
class Graph:
    schema_version: int
    model_id: str
    revision: str
    fidelity: str           # "full" | "structural" | "weights" (design §05 fallback ladder)
    architecture: str | None
    params_total: int
    config: dict[str, Any]
    nodes: list[Node]
    repeats: list[Repeat]
    edges: list[Edge]
    trace: list[TraceStep]
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
