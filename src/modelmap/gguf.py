"""GGUF checkpoints without downloading them (design doc §18).

A GGUF file starts with a self-describing header: key/value metadata (the
architecture hyper-parameters llama.cpp needs, which map onto a transformers
config) followed by every tensor's name, shape, quant type and offset. The
header is a few MB at most (the tokenizer vocabulary is the bulk), so HTTP
range reads give us everything modelmap needs — the module tree, and the
*real* bytes per weight of a Q4_K_M / Q8_0 / IQ2_XXS variant — while the
multi-GB tensor data stays on the Hub.

Pipeline for `owner/name[:variant]` whose repo holds *.gguf files:
  1. list files → group by quant label → pick the variant (default Q4_K_M)
  2. read the header of every shard of that variant (range requests)
  3. metadata → transformers config (transformers' own GGUF key mapping)
     → meta instantiate + traced forward, like a safetensors repo
  4. per-tensor quant types → per-module dtypes via the standard llama.cpp
     tensor names (blk.N.attn_q → model.layers.N.self_attn.q_proj)
  5. fallback: no config mapping → weights view straight from the tensor list
"""

from __future__ import annotations

import logging
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

MAGIC = b"GGUF"
INITIAL_FETCH = 2 * 1024 * 1024
MAX_HEADER = 256 * 1024 * 1024
MAX_SHARDS = 32

# ggml_type enum → name (llama.cpp ggml.h); names are lower-cased as dtypes
GGML_TYPES: dict[int, str] = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1",
    10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 15: "q8_k",
    16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s",
    22: "iq2_s", 23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 39: "mxfp4",
}

# general.file_type → llama.cpp's overall quant label
FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4",
}


class GGUFError(ValueError):
    pass


class _NeedMore(Exception):
    pass


@dataclass
class GGUFTensor:
    name: str
    shape: list[int]  # torch order (outermost first)
    dtype: str  # ggml type name, lower-case
    numel: int


@dataclass
class GGUFHeader:
    version: int
    kv: dict[str, Any]
    tensors: list[GGUFTensor]
    header_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def architecture(self) -> str | None:
        a = self.kv.get("general.architecture")
        return a if isinstance(a, str) else None

    @property
    def file_type(self) -> str | None:
        ft = self.kv.get("general.file_type")
        return FILE_TYPES.get(ft) if isinstance(ft, int) else None


# ------------------------------------------------------------ byte sources

Fetch = Callable[[int, int], bytes]  # (offset, length) → bytes (may be short at EOF)


def hub_fetcher(model_id: str, filename: str, revision: str, token: str | None) -> Fetch:
    import httpx
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(model_id, filename, revision=revision)
    base = build_hf_headers(token=token)
    client = httpx.Client(follow_redirects=True, timeout=60)

    def fetch(offset: int, length: int) -> bytes:
        h = {**base, "Range": f"bytes={offset}-{offset + length - 1}"}
        with client.stream("GET", url, headers=h) as r:
            if r.status_code == 416:
                return b""
            if r.status_code not in (200, 206):
                r.raise_for_status()
            return _read_window(r, offset, length)

    fetch.close = client.close  # type: ignore[attr-defined]
    return fetch


def _read_window(r, offset: int, length: int) -> bytes:
    """The requested byte window from a streamed response — never more.

    A server that ignores the Range header answers 200 with the WHOLE file;
    `r.content` on a 16 GB checkpoint would buffer all of it and kill the
    worker (seen in production when a repo turned gated mid-CDN-propagation).
    On 200 the body starts at byte 0, so skip to `offset` first; stop reading
    the moment the window is full."""
    skip = offset if r.status_code == 200 else 0
    out = bytearray()
    for chunk in r.iter_bytes():
        if skip:
            if len(chunk) <= skip:
                skip -= len(chunk)
                continue
            chunk = chunk[skip:]
            skip = 0
        out += chunk[: length - len(out)]
        if len(out) >= length:
            break
    return bytes(out)


def local_fetcher(path: str) -> Fetch:
    def fetch(offset: int, length: int) -> bytes:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    return fetch


# ----------------------------------------------------------------- parser

_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_T_STRING, _T_ARRAY = 8, 9


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise _NeedMore
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def scalar(self, t: int):
        fmt, size = _SCALAR[t]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.take(8))[0]
        if n > 64 * 1024 * 1024:
            raise GGUFError("implausible string length in GGUF header")
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, t: int, *, keep: bool = True):
        if t == _T_STRING:
            return self.string()
        if t == _T_ARRAY:
            et = struct.unpack("<I", self.take(4))[0]
            n = struct.unpack("<Q", self.take(8))[0]
            if n > 50_000_000:
                raise GGUFError("implausible array length in GGUF header")
            if not keep and et in _SCALAR:  # skip in one hop
                self.take(_SCALAR[et][1] * n)
                return n
            out = []
            for _ in range(n):
                v = self.value(et, keep=keep)
                if keep:
                    out.append(v)
            return out if keep else n
        if t in _SCALAR:
            return self.scalar(t)
        raise GGUFError(f"unknown GGUF value type {t}")


# big, uninteresting arrays (the vocabulary) are parsed but not retained
_SKIP_KEYS = re.compile(r"^tokenizer\.ggml\.(tokens|scores|token_type|merges)$")


def _parse(buf: bytes) -> GGUFHeader:
    r = _Reader(buf)
    if r.take(4) != MAGIC:
        raise GGUFError("not a GGUF file (bad magic)")
    version = struct.unpack("<I", r.take(4))[0]
    if version < 2:
        raise GGUFError(f"GGUF v{version} is not supported")
    n_tensors = struct.unpack("<Q", r.take(8))[0]
    n_kv = struct.unpack("<Q", r.take(8))[0]
    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        t = struct.unpack("<I", r.take(4))[0]
        keep = not _SKIP_KEYS.match(key)
        v = r.value(t, keep=keep)
        if keep:
            kv[key] = v
        elif key == "tokenizer.ggml.tokens":
            kv["_tokens_count"] = v  # the array length: the vocabulary size
    tensors: list[GGUFTensor] = []
    for _ in range(n_tensors):
        name = r.string()
        nd = struct.unpack("<I", r.take(4))[0]
        if nd > 8:
            raise GGUFError("implausible tensor rank in GGUF header")
        dims = [struct.unpack("<Q", r.take(8))[0] for _ in range(nd)]
        t = struct.unpack("<I", r.take(4))[0]
        r.take(8)  # offset
        shape = list(reversed(dims))  # ggml ne[0] is innermost
        numel = 1
        for d in dims:
            numel *= d
        tensors.append(GGUFTensor(name=name, shape=shape, dtype=GGML_TYPES.get(t, f"type{t}"), numel=numel))
    return GGUFHeader(version=version, kv=kv, tensors=tensors, header_bytes=r.pos)


def read_header(fetch: Fetch) -> GGUFHeader:
    """Parse a GGUF header through `fetch`, growing the window until the
    tensor table is complete."""
    size = INITIAL_FETCH
    buf = b""
    try:
        while True:
            chunk = fetch(len(buf), size - len(buf))
            buf += chunk
            try:
                return _parse(buf)
            except _NeedMore:
                if not chunk or len(buf) >= MAX_HEADER:
                    raise GGUFError("GGUF header is truncated or larger than the read limit")
                size = min(size * 2, MAX_HEADER)
    finally:
        close = getattr(fetch, "close", None)
        if close:
            close()


# --------------------------------------------------------------- variants

_SHARD = re.compile(r"-(\d{5})-of-(\d{5})$")
_QUANT = re.compile(
    r"(?i)(?:^|[-._ ])((?:UD-)?(?:IQ|Q|TQ)\d(?:_[A-Z0-9]{1,3})*|BF16|F16|F32|FP16|FP32|MXFP4)(?=[-._ ]|$)"
)


def _stem(filename: str) -> str:
    base = filename.rsplit("/", 1)[-1]
    base = base[: -len(".gguf")] if base.lower().endswith(".gguf") else base
    return _SHARD.sub("", base)


def variants_of(files: list[str]) -> dict[str, list[str]]:
    """Group a repo's *.gguf files into variants keyed by quant label
    ("Q4_K_M"); shards of one variant group together. Files whose name has
    no recognizable quant token are keyed by their distinguishing stem."""
    ggufs = sorted(f for f in files if f.lower().endswith(".gguf"))
    if not ggufs:
        return {}
    by_stem: dict[str, list[str]] = {}
    for f in ggufs:
        by_stem.setdefault(_stem(f), []).append(f)
    stems = list(by_stem)
    labels: dict[str, str] = {}
    for s in stems:
        m = _QUANT.search(s)
        labels[s] = m.group(1).upper() if m else ""
    # quant tokens must be unique among stems; otherwise fall back to the stem
    counts: dict[str, int] = {}
    for l in labels.values():
        counts[l] = counts.get(l, 0) + 1
    prefix = os.path.commonprefix(stems) if len(stems) > 1 else ""
    out: dict[str, list[str]] = {}
    for s in stems:
        l = labels[s]
        if not l or counts[l] > 1:
            l = s[len(prefix):].strip("-._ ") or s
        # still colliding (e.g. identical stems in sub-folders): disambiguate
        key = l
        i = 2
        while key in out:
            key = f"{l}#{i}"
            i += 1
        out[key] = sorted(by_stem[s], key=_shard_no)
    return out


def _shard_no(f: str) -> int:
    m = _SHARD.search(_stem_keep_shard(f))
    return int(m.group(1)) if m else 0


def _stem_keep_shard(f: str) -> str:
    base = f.rsplit("/", 1)[-1]
    return base[: -len(".gguf")] if base.lower().endswith(".gguf") else base


_PREFERRED = ["Q4_K_M", "Q4_K_S", "UD-Q4_K_XL", "Q5_K_M", "IQ4_XS", "Q4_0", "Q6_K", "Q8_0", "Q5_0", "BF16", "F16"]


def choose_variant(variants: dict[str, list[str]], requested: str | None) -> str:
    if requested:
        for k in variants:
            if k.lower() == requested.lower():
                return k
        for k in variants:  # substring match on the label or a file name
            if requested.lower() in k.lower() or any(requested.lower() in f.lower() for f in variants[k]):
                return k
        # the reverse: a pasted file name ("model-Q8_0") *contains* a label —
        # the longest match is the most specific ("BF16" beats "F16")
        cands = [k for k in variants if len(k) >= 2 and k.lower() in requested.lower()]
        if cands:
            return max(cands, key=len)
        raise GGUFError(f"no GGUF variant matches '{requested}'; available: {', '.join(variants)}")
    for p in _PREFERRED:
        if p in variants:
            return p
    return next(iter(variants))


# ------------------------------------------------------- config mapping

# gguf arch name → transformers model_type (transformers' reverse of its own hack)
_ARCH_TO_MODEL_TYPE = {
    "qwen2moe": "qwen2_moe", "qwen3moe": "qwen3_moe", "gemma3": "gemma3_text",
    "gemma4": "gemma4_text", "command-r": "cohere", "minimax-m2": "minimax_m2",
    "gpt-oss": "gpt_oss", "deepseek2": "deepseek_v3",
}


def config_from_header(h: GGUFHeader) -> tuple[Any, list[str]]:
    """Build a transformers config from GGUF metadata; raises GGUFError when
    the architecture has no known mapping."""
    from transformers import AutoConfig
    from transformers.integrations.ggml import GGUF_CONFIG_DEFAULTS_MAPPING, GGUF_CONFIG_MAPPING

    notes: list[str] = []
    arch = h.architecture
    if not arch:
        raise GGUFError("GGUF header has no general.architecture")
    model_type = _ARCH_TO_MODEL_TYPE.get(arch, arch)
    mapping = GGUF_CONFIG_MAPPING.get(model_type) or GGUF_CONFIG_MAPPING.get(arch)
    if mapping is None:
        raise GGUFError(f"no config mapping for GGUF architecture '{arch}'")
    kwargs: dict[str, Any] = dict(GGUF_CONFIG_DEFAULTS_MAPPING.get(model_type, {}))
    for suffix, cfg_key in mapping.items():
        if cfg_key is None:
            continue
        v = h.kv.get(f"{arch}.{suffix}")
        if v is not None:
            kwargs[cfg_key] = v
    # vocab: explicit key, else the embedding tensor's outer dim
    if "vocab_size" not in kwargs:
        emb = next((t for t in h.tensors if t.name == "token_embd.weight"), None)
        if emb:
            kwargs["vocab_size"] = emb.shape[0]
        elif isinstance(h.kv.get("_tokens_count"), int):
            kwargs["vocab_size"] = h.kv["_tokens_count"]
    # head_dim when only the rope dimension / key length is known
    if "head_dim" not in kwargs:
        kl = h.kv.get(f"{arch}.attention.key_length")
        if isinstance(kl, int):
            kwargs["head_dim"] = kl
    # tied embeddings: no separate output tensor
    kwargs["tie_word_embeddings"] = not any(t.name == "output.weight" for t in h.tensors)
    # instantiate under bf16: shapes don't care, and some MoE kernels assert
    # bf16 inputs even on meta tensors (the real quant types come from the
    # tensor table afterwards)
    kwargs["dtype"] = "bfloat16"
    try:
        config = AutoConfig.for_model(model_type, **kwargs)
    except Exception as e:  # unknown model_type in this transformers version
        raise GGUFError(f"transformers cannot build a '{model_type}' config: {e}") from e
    # a GGUF is a generative checkpoint: name the causal-LM class so the
    # module tree carries the lm_head (tied or not), like a Hub repo would
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

        lm = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type)
        if lm:
            config.architectures = [lm]
    except ImportError:
        pass
    name = h.kv.get("general.name") or h.kv.get("general.basename")
    notes.append(
        f"GGUF {arch}{f' ({name})' if name else ''}: config rebuilt from the file header"
        + (f"; file type {h.file_type}" if h.file_type else "")
    )
    return config, notes


# ------------------------------------------------------- tensor mapping

# llama.cpp standard tensor names → HF module paths (suffixes under model.layers.N)
_BLK = {
    "attn_norm": "input_layernorm", "attn_norm_2": "input_layernorm_2",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_qkv": "self_attn.qkv_proj",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_q_a": "self_attn.q_a_proj", "attn_q_b": "self_attn.q_b_proj", "attn_q_a_norm": "self_attn.q_a_layernorm",
    "attn_kv_a_mqa": "self_attn.kv_a_proj_with_mqa", "attn_kv_b": "self_attn.kv_b_proj",
    "attn_kv_a_norm": "self_attn.kv_a_layernorm",
    "ffn_norm": "post_attention_layernorm", "post_attention_norm": "post_attention_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm", "ffn_pre_norm": "pre_feedforward_layernorm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    "ffn_gate_inp": "mlp.gate", "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
    "ffn_gate_exps": "mlp.experts", "ffn_up_exps": "mlp.experts", "ffn_down_exps": "mlp.experts",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj", "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj", "exp_probs_b": "mlp.gate",
}
_TOP = {
    "token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head",
    "rope_freqs": "model.rotary_emb", "position_embd": "model.embed_positions",
}


def hf_module_for(name: str) -> str | None:
    """GGUF tensor name → HF module path (without the .weight/.bias leaf)."""
    base = name.rsplit(".", 1)[0] if name.endswith((".weight", ".bias")) else name
    m = re.match(r"^blk\.(\d+)\.(.+)$", base)
    if m:
        suffix = _BLK.get(m.group(2))
        return f"model.layers.{m.group(1)}.{suffix}" if suffix else None
    return _TOP.get(base)


def apply_gguf_dtypes(nodes, headers: list[GGUFHeader], notes: list[str]) -> None:
    """Per-module dtypes from the GGUF tensor table; modules the mapping
    cannot name keep the config-declared dtype."""
    by_module: dict[str, str] = {}
    unmapped = 0
    for h in headers:
        for t in h.tensors:
            mod = hf_module_for(t.name)
            if mod is None:
                unmapped += 1
                continue
            # a module with several tensors at different types (fused experts):
            # keep the lowest-precision one — it dominates the bytes
            prev = by_module.get(mod)
            by_module[mod] = t.dtype if prev is None else _coarser(prev, t.dtype)
    # checkpoint names may be relative to a base model prefix (e.g. a model
    # whose HF tree is "model.layers…" — or not); match by suffix
    by_id = {n.id: n for n in nodes}
    hits = 0
    for mod, d in by_module.items():
        n = by_id.get(mod)
        if n is None:
            cands = [x for x in nodes if x.id.endswith("." + mod) or x.id == mod.split(".", 1)[-1]]
            n = cands[0] if len(cands) == 1 else None
        if n is not None:
            n.dtype = d
            hits += 1
    if unmapped:
        notes.append(f"{unmapped} GGUF tensors have no standard HF name (dtypes kept from config)")
    if not hits:
        notes.append("GGUF tensor names did not match the module tree; quant dtypes not applied")


def _coarser(a: str, b: str) -> str:
    from modelmap.analytics import DTYPE_BYTES

    return a if DTYPE_BYTES.get(a, 2) <= DTYPE_BYTES.get(b, 2) else b


# ------------------------------------------------------------ weights view


def gguf_nodes(headers: list[GGUFHeader]):
    """Tensor list → (name, shape, dtype) triples for weights_view, with
    standard llama.cpp names rewritten to their HF paths so the fallback tree
    still classifies (q_proj → linear, self_attn → attention)."""
    out = []
    for h in headers:
        for t in h.tensors:
            leaf = t.name.rsplit(".", 1)[-1] if t.name.endswith((".weight", ".bias")) else "weight"
            mod = hf_module_for(t.name)
            out.append((f"{mod}.{leaf}" if mod else t.name, t.shape, t.dtype))
    return out
