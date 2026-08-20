"""modelmap CLI.

  modelmap [PATH]          serve + open the browser (the `uvx modelmap` experience);
                           with a local checkpoint PATH, open it directly
  modelmap serve           run the server (--host --port --open --warm --no-local)
  modelmap dump <id>       extract a model to JSON / CSV / Markdown / DOT
  modelmap cost <id>       headline numbers + compute / memory / KV estimates
  modelmap plan <id>       will it fit? TP × PP placement across N GPUs (+ speed with --gpu)
  modelmap train <id>      fine-tuning memory: full / LoRA / QLoRA, ZeRO, checkpointing
  modelmap diff <A> <B>    module-by-module structural diff
  modelmap warm [ids…]     pre-extract the gallery (or given ids) into the cache
  modelmap mcp             MCP server (stdio) for coding agents

Model ids: owner/name · owner/name:Q4_K_M (GGUF variant) · local:/path or a
plain existing path (local checkpoint dir, safetensors file or .gguf).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from modelmap import __version__
from modelmap.ids import LOCAL_PREFIX

_DTYPES = ("bf16", "f16", "f32", "f8", "int8", "int4")


def _norm_id(model_id: str) -> str:
    """A path that exists on disk is a local checkpoint; otherwise a Hub id."""
    if model_id.startswith(LOCAL_PREFIX):
        return model_id
    p = os.path.expanduser(model_id)
    if os.path.exists(p) and (model_id.startswith((".", "/", "~")) or os.path.isdir(p) or p.endswith((".gguf", ".safetensors"))):
        return LOCAL_PREFIX + os.path.abspath(p)
    return model_id


def _add_assumptions(p: argparse.ArgumentParser) -> None:
    p.add_argument("-T", "--seq", type=int, default=4096, help="sequence length for cost estimates (default 4096)")
    p.add_argument("-B", "--batch", type=int, default=1, help="batch size (default 1)")
    p.add_argument("--dtype", default="bf16", choices=_DTYPES, help="activation dtype (default bf16)")
    p.add_argument("--token", help="HF token for gated/private repos")
    p.add_argument("--refresh", action="store_true", help="ignore the disk cache and re-extract")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="modelmap",
        description="Interactive, animated architecture maps for Hugging Face models.",
    )
    parser.add_argument("--version", action="version", version=f"modelmap {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    d = sub.add_parser("dump", help="extract a model's graph to JSON / CSV / Markdown / DOT")
    d.add_argument("model_id", help='e.g. "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-GGUF:Q8_0", ./my-finetune')
    d.add_argument("-o", "--out", help="output path (default: <model>.graph.<ext>; '-' for stdout)")
    d.add_argument("-f", "--format", default="json", choices=("json", "csv", "md", "markdown", "dot"))
    d.add_argument("--revision", default="main")
    d.add_argument("--seq-len", type=int, default=7, help="dummy input sequence length for the trace")
    d.add_argument("--leaves-only", action="store_true", help="csv: drop container rows")
    d.add_argument("--depth", type=int, default=3, help="dot: cluster depth")
    d.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow the repo's custom Python to execute (dangerous; local use only)",
    )
    d.add_argument("--pretty", action="store_true", help="indent the output JSON")
    _add_assumptions(d)

    c = sub.add_parser("cost", help="headline numbers and cost estimates")
    c.add_argument("model_id")
    c.add_argument("--json", action="store_true", help="machine-readable output")
    _add_assumptions(c)

    pl = sub.add_parser("plan", help="TP × PP placement across GPUs: does it fit, max context")
    pl.add_argument("model_id")
    pl.add_argument("--gpus", type=int, default=1)
    pl.add_argument("--gpu-memory", type=float, default=80, help="GiB per GPU (default 80)")
    pl.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    pl.add_argument("--pp", type=int, default=1, help="pipeline-parallel degree")
    pl.add_argument("--headroom", type=float, default=0.1, help="fraction of memory kept free (default 0.1)")
    pl.add_argument("--gpu", help='GPU preset for a speed estimate, e.g. "A100 80GB" (see modelmap plan --list-gpus)')
    pl.add_argument("--list-gpus", action="store_true", help="print the GPU presets and exit")
    pl.add_argument("--json", action="store_true")
    _add_assumptions(pl)

    tr = sub.add_parser("train", help="fine-tuning memory: full / LoRA / QLoRA, optimizer, ZeRO")
    tr.add_argument("model_id")
    tr.add_argument("--method", default="lora", choices=("full", "lora", "qlora"))
    tr.add_argument("--optimizer", default="adamw", choices=("adamw", "adamw8bit"))
    tr.add_argument("--rank", type=int, default=16, help="LoRA rank (default 16)")
    tr.add_argument("--targets", default="attn-mlp", choices=("attention", "attn-mlp", "all-linear"))
    tr.add_argument("--gpus", type=int, default=1)
    tr.add_argument("--gpu-memory", type=float, default=80, help="GiB per GPU (default 80)")
    tr.add_argument("--sharding", default="none", choices=("none", "zero2", "zero3"))
    tr.add_argument("--no-checkpointing", action="store_true", help="keep every activation (no gradient checkpointing)")
    tr.add_argument("--no-flash", action="store_true", help="assume no flash attention (adds T² score memory)")
    tr.add_argument("--headroom", type=float, default=0.1)
    tr.add_argument("--gpu", help='GPU preset for a speed estimate, e.g. "H100 80GB"')
    tr.add_argument("--json", action="store_true")
    _add_assumptions(tr)

    df = sub.add_parser("diff", help="structural diff of two models")
    df.add_argument("a")
    df.add_argument("b")
    df.add_argument("-f", "--format", default="md", choices=("md", "json"))
    df.add_argument("--token")
    df.add_argument("--refresh", action="store_true")

    s = sub.add_parser("serve", help="run the server (API + web app)")
    s.add_argument("--host", default=os.environ.get("MODELMAP_HOST", "127.0.0.1"))
    # $PORT is the Cloud Run / Heroku convention; MODELMAP_PORT wins if both are set
    s.add_argument("--port", type=int, default=int(os.environ.get("MODELMAP_PORT") or os.environ.get("PORT") or "7860"))
    s.add_argument("--open", action="store_true", help="open the browser once the server is up")
    s.add_argument("--warm", action="store_true", help="pre-extract the gallery in the background")
    s.add_argument("--no-local", action="store_true", help="refuse local:/path ids even on loopback")
    s.add_argument("--allow-local", action="store_true", help="serve local:/path ids on a non-loopback host (trusted networks only)")
    s.add_argument(
        "--trust-remote-code", action="store_true",
        help="let extraction run repos' own modeling Python (arbitrary code — your own machine only)",
    )

    w = sub.add_parser("warm", help="pre-extract models into the cache")
    w.add_argument("model_ids", nargs="*", help="defaults to the landing gallery")

    m = sub.add_parser("mcp", help="MCP server over stdio for coding agents")
    m.add_argument("--remote", help="delegate to a hosted modelmap (e.g. https://modelmap.cc) instead of extracting locally")

    # bare `modelmap PATH`: a local checkpoint to open (not a subcommand name)
    argv = list(sys.argv[1:] if argv is None else argv)
    bare_path: str | None = None
    if argv and not argv[0].startswith("-") and argv[0] not in sub.choices:
        bare_path = argv.pop(0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.cmd is None:  # bare `modelmap [PATH]`: the uvx experience
        open_path = _norm_id(bare_path) if bare_path else None
        if bare_path and not (open_path or "").startswith(LOCAL_PREFIX):
            parser.error(f"{bare_path}: no such file or directory (use `modelmap serve` for the bare server)")
        args.cmd, args.host, args.port, args.open, args.warm, args.no_local, args.allow_local = (
            "serve", os.environ.get("MODELMAP_HOST", "127.0.0.1"),
            int(os.environ.get("MODELMAP_PORT") or os.environ.get("PORT") or "7860"), True, False, False, False,
        )
        args.trust_remote_code = False
        args.open_path = open_path
    else:
        args.open_path = None

    if args.cmd == "dump":
        _dump(args)
    elif args.cmd == "cost":
        _cost(args)
    elif args.cmd == "plan":
        _plan(args)
    elif args.cmd == "train":
        _train(args)
    elif args.cmd == "diff":
        _diff(args)
    elif args.cmd == "warm":
        from modelmap.gallery import CLASSIC_IDS, trending_ids
        from modelmap.server import warm
        from modelmap.zoo import ZOO_IDS

        # default: the classics, the curated zoo families, and whatever is
        # trending right now (so a container image bakes them all in)
        ids = args.model_ids or list(dict.fromkeys(CLASSIC_IDS + ZOO_IDS + trending_ids()))
        warm(ids)
    elif args.cmd == "mcp":
        from modelmap.mcp_server import run

        run(remote=args.remote)
    else:
        import uvicorn

        if args.warm:
            os.environ["MODELMAP_WARM"] = "1"
        loopback = args.host in ("127.0.0.1", "localhost", "::1")
        if (loopback and not args.no_local) or args.allow_local:
            os.environ["MODELMAP_ALLOW_LOCAL"] = "1"
        if getattr(args, "trust_remote_code", False):
            os.environ["MODELMAP_TRUST_REMOTE_CODE"] = "1"
            print(
                "WARNING: --trust-remote-code executes repos' own Python during extraction. "
                "Only extract repos you trust.", file=sys.stderr,
            )
        if args.open:
            target = f"http://{args.host}:{args.port}/"
            if args.open_path:
                target += f"m/{args.open_path}"
            _open_when_ready(f"http://{args.host}:{args.port}/", target)
        uvicorn.run("modelmap.server:app", host=args.host, port=args.port, log_level="info")


# ----------------------------------------------------------------- commands


def _load(model_id: str, *, token: str | None = None, refresh: bool = False, revision: str = "main",
          trust_remote_code: bool = False, seq_len: int = 7) -> dict:
    """Document from the disk cache or a fresh extraction (local paths allowed)."""
    import gzip

    from modelmap import cache
    from modelmap.ids import is_local

    model_id = _norm_id(model_id)
    if not refresh and token is None and not is_local(model_id):
        raw = cache.get_bytes(model_id, revision)
        if raw is not None:
            return json.loads(gzip.decompress(raw))
    from modelmap.extract import extract_graph

    g = extract_graph(
        model_id, revision=revision, token=token, trust_remote_code=trust_remote_code,
        seq_len=seq_len, allow_local=True,
    )
    doc = g.to_json_dict()
    if token is None and not is_local(model_id) and not trust_remote_code:
        cache.put(model_id, revision, doc)
    return doc


def _assumptions(args):
    from modelmap.analytics import Assumptions

    return Assumptions(T=args.seq, B=args.batch, dtype=args.dtype)


def _dump(args) -> None:
    from modelmap.export import render

    doc = _load(
        args.model_id, token=args.token, refresh=args.refresh, revision=args.revision,
        trust_remote_code=args.trust_remote_code, seq_len=args.seq_len,
    )
    fmt = args.format
    text, _ = render(doc, fmt, _assumptions(args), pretty=args.pretty, leaves_only=args.leaves_only, depth=args.depth)
    ext = {"json": "json", "csv": "csv", "md": "md", "markdown": "md", "dot": "dot"}[fmt]
    out = args.out or (_norm_id(args.model_id).split("/")[-1].split(":")[0].lower() + f".graph.{ext}")
    if out == "-":
        sys.stdout.write(text)
    else:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    print(
        f"{doc['model_id']}: fidelity={doc['fidelity']} nodes={len(doc['nodes'])} "
        f"repeats={len(doc['repeats'])} trace={len(doc['trace'])} "
        f"params={doc['params_total']:,}"
        + (f" variant={doc['variant']}" if doc.get("variant") else "")
        + (f" -> {out}" if out != "-" else ""),
        file=sys.stderr,
    )
    for note in doc.get("notes") or []:
        print(f"  note: {note}", file=sys.stderr)


def _cost(args) -> None:
    from modelmap.analytics import fmt_big, fmt_bytes, fmt_params, summarize

    doc = _load(args.model_id, token=args.token, refresh=args.refresh)
    s = summarize(doc, _assumptions(args))
    if args.json:
        print(json.dumps(s, indent=2))
        return
    c = s["cost"]
    a = s["assumptions"]
    print(f"{s['model_id']}  ·  {s['architecture'] or '?'}  ·  fidelity {s['fidelity']}" + (f"  ·  GGUF {s['variant']}" if s.get("variant") else ""))
    active = f"  ({fmt_params(s['active_params'])} active/token)" if s["active_params"] < 0.9 * (s["params_total"] or 1) else ""
    print(f"  params        {fmt_params(s['params_total'] or 0)}{active}")
    if s["repeat_stacks"]:
        print("  stacks        " + ", ".join(f"{st['parent']} ×{st['count']}" for st in s["repeat_stacks"]))
    cfg = s["config"]
    keys = [k for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size", "max_position_embeddings", "num_experts", "n_routed_experts", "num_experts_per_tok") if k in cfg]
    if keys:
        print("  config        " + "  ".join(f"{k}={cfg[k]}" for k in keys))
    print(f"  assumptions   T={a['T']:,}  B={a['B']}  dtype={a['dtype']}")
    print(f"  compute       {fmt_big(c['macs_per_token'], 'MAC')}/token  ·  {fmt_big(c['macs_per_forward'], 'MAC')} per forward")
    print(f"  weights       {fmt_bytes(c['weight_bytes'])} at stored dtypes")
    print(f"  activations   {fmt_bytes(c['activation_bytes'])} summed  ·  largest {fmt_bytes(c['largest_activation_bytes'])} ({c['largest_activation_node'] or '—'})")
    if c["kv_bytes_per_token"]:
        print(f"  kv cache      {fmt_bytes(c['kv_bytes_per_token'])}/token  ·  {fmt_bytes(c['kv_bytes_at_T'])} at T  ({c['kv_layers']} layers)")
    for n in s["notes"]:
        print(f"  note: {n}")


def _plan(args) -> None:
    from modelmap.analytics import GPU_SPECS, PlanRequest, estimate_throughput, fmt_bytes, plan_serving

    if args.list_gpus:
        for name, spec in GPU_SPECS.items():
            print(f"{name:<16} {spec['tflops']:>6.0f} TFLOPs bf16   {spec['bw']:>5.0f} GB/s")
        return
    doc = _load(args.model_id, token=args.token, refresh=args.refresh)
    req = PlanRequest(gpus=args.gpus, gpu_memory_gb=args.gpu_memory, tp=args.tp, pp=args.pp, T=args.seq, B=args.batch, dtype=args.dtype, headroom=args.headroom)
    p = plan_serving(doc, req)
    if args.json:
        print(json.dumps(p.to_dict(), indent=2))
        return
    r = p.request
    print(f"{doc['model_id']}  ·  {r.gpus}× {r.gpu_memory_gb:g} GB  ·  tp={r.tp} pp={r.pp}  ·  T={r.T:,} B={r.B} {r.dtype}  ·  headroom {r.headroom:.0%}")
    print(f"  {'FITS' if p.fits else 'DOES NOT FIT'}  ·  weights {fmt_bytes(p.weight_bytes)}  ·  KV {fmt_bytes(p.kv_bytes)} at T  ·  per-GPU capacity {fmt_bytes(p.per_gpu_capacity_bytes)}")
    print(f"  max context at B={r.B}: {p.max_context_tokens:,} tokens (KV-limited)")
    for st in p.stages:
        layers = f"layers {st.layers[0]}–{st.layers[1]}" if st.layers else "no layer stack"
        print(
            f"  stage {st.stage}  gpus {st.gpus[0]}–{st.gpus[-1]}  {layers} ({st.layer_count}) "
            f"weights {fmt_bytes(st.weight_bytes_per_gpu)}/gpu  kv {fmt_bytes(st.kv_bytes_per_gpu)}/gpu  "
            f"act {fmt_bytes(st.act_bytes_per_gpu)}/gpu  total {fmt_bytes(st.total_bytes_per_gpu)}  {'ok' if st.fits else 'OVER'}"
            + (f"  → next stage {fmt_bytes(st.boundary_bytes_out)}/forward" if st.boundary_bytes_out else "")
        )
    for n in p.notes:
        print(f"  note: {n}")
    if args.gpu:
        t = estimate_throughput(doc, args.gpu, tp=r.tp, T=r.T, B=r.B, dtype=r.dtype)
        if t is None:
            print(f"  unknown GPU preset '{args.gpu}' — see modelmap plan --list-gpus")
        else:
            print(
                f"  speed on {t.gpu} × {r.tp}: prefill ≈ {t.prefill_tok_per_sec:,.0f} tok/s · "
                f"decode ≈ {t.decode_tok_per_sec_b1:,.0f} tok/s at B=1"
                + (f" · ≈ {t.decode_tok_per_sec_at_b:,.0f} tok/s total at B={t.batch}" if t.batch > 1 else "")
            )
            for n in t.notes:
                print(f"  note: {n}")


def _train(args) -> None:
    from modelmap.analytics import TrainRequest, fmt_bytes, fmt_params, plan_training

    doc = _load(args.model_id, token=args.token, refresh=args.refresh)
    req = TrainRequest(
        method=args.method, optimizer=args.optimizer, lora_rank=args.rank, lora_targets=args.targets,
        gpus=args.gpus, gpu_memory_gb=args.gpu_memory, sharding=args.sharding, T=args.seq, B=args.batch,
        grad_checkpoint=not args.no_checkpointing, flash_attention=not args.no_flash,
        headroom=args.headroom, gpu=args.gpu,
    )
    p = plan_training(doc, req)
    if args.json:
        print(json.dumps(p.to_dict(), indent=2))
        return
    r = p.request
    label = r.method + (f" r={r.lora_rank} ({r.lora_targets})" if r.method != "full" else "")
    print(f"{doc['model_id']}  ·  {label}  ·  {r.gpus}× {r.gpu_memory_gb:g} GB  ·  {r.sharding}  ·  T={r.T:,} B={r.B}/gpu  ·  {r.optimizer}")
    print(f"  {'FITS' if p.fits else 'DOES NOT FIT'}  ·  trainable {fmt_params(p.trainable_params)} of {fmt_params(p.total_params)}  ·  capacity {fmt_bytes(p.per_gpu_capacity_bytes)}/gpu")
    print(f"  per GPU:  weights {fmt_bytes(p.weight_bytes_per_gpu)}  ·  grads {fmt_bytes(p.grad_bytes_per_gpu)}  ·  optimizer {fmt_bytes(p.optimizer_bytes_per_gpu)}  ·  activations {fmt_bytes(p.activation_bytes_per_gpu)}  ·  total {fmt_bytes(p.total_bytes_per_gpu)}")
    if p.max_microbatch:
        print(f"  largest micro-batch at T={r.T:,}: {p.max_microbatch}/gpu")
    if p.train_tokens_per_sec:
        print(f"  speed ≈ {p.train_tokens_per_sec:,.0f} training tok/s across {r.gpus} GPUs")
    for n in p.notes:
        print(f"  note: {n}")


def _diff(args) -> None:
    from modelmap.compare import align, diff_markdown

    da = _load(args.a, token=args.token, refresh=args.refresh)
    db = _load(args.b, token=args.token, refresh=args.refresh)
    al = align(da, db)
    if args.format == "json":
        print(json.dumps({"a": da["model_id"], "b": db["model_id"], **al.to_dict()}, indent=2))
    else:
        print(diff_markdown(da, db, al))


def _open_when_ready(health_base: str, url: str) -> None:
    import threading
    import time
    import urllib.request
    import webbrowser

    def go():
        for _ in range(100):
            try:
                urllib.request.urlopen(health_base.rstrip("/") + "/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        webbrowser.open(url)

    threading.Thread(target=go, daemon=True).start()


if __name__ == "__main__":
    main()
