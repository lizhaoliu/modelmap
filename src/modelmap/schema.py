"""Graph document schema — one JSON document per (model_id, revision).

See docs/design.html §06. The document is the single contract between the
Python extractor and the web client; bump SCHEMA_VERSION on breaking changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 5  # 5: only numbered siblings collapse; 4: variant / variants / weights_format; 3: execution-order + aux edges; 2: attrs

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
    # "flow": the main path between siblings in execution order
    # "aux": a side computation (rotary / position embeddings) feeding its consumer
    kind: str = "flow"


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
    # checkpoint flavour: "safetensors" | "gguf" | "pytorch" | None (config only)
    weights_format: str | None = None
    # GGUF repos hold several quantizations; `variant` is the one this document
    # describes ("Q4_K_M") and `variants` every label the repo offers
    variant: str | None = None
    variants: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
