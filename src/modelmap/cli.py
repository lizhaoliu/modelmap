"""modelmap CLI.

  modelmap                 serve + open the browser (the `uvx modelmap` experience)
  modelmap serve           run the server (options: --host --port --open --warm)
  modelmap dump <id>       extract a model's graph to JSON
  modelmap warm [ids…]     pre-extract the gallery (or given ids) into the cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from modelmap import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="modelmap",
        description="Interactive, animated architecture maps for Hugging Face models.",
    )
    parser.add_argument("--version", action="version", version=f"modelmap {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    d = sub.add_parser("dump", help="extract a model's graph to JSON")
    d.add_argument("model_id", help='e.g. "Qwen/Qwen3-8B"')
    d.add_argument("-o", "--out", help="output path (default: <model>.graph.json)")
    d.add_argument("--revision", default="main")
    d.add_argument("--token", help="HF token for gated/private repos")
    d.add_argument("--seq-len", type=int, default=7, help="dummy input sequence length")
    d.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow the repo's custom Python to execute (dangerous; local use only)",
    )
    d.add_argument("--pretty", action="store_true", help="indent the output JSON")

    s = sub.add_parser("serve", help="run the server (API + web app)")
    s.add_argument("--host", default=os.environ.get("MODELMAP_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("MODELMAP_PORT", "7860")))
    s.add_argument("--open", action="store_true", help="open the browser once the server is up")
    s.add_argument("--warm", action="store_true", help="pre-extract the gallery in the background")

    w = sub.add_parser("warm", help="pre-extract models into the cache")
    w.add_argument("model_ids", nargs="*", help="defaults to the landing gallery")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.cmd is None:  # bare `modelmap`: the uvx experience
        args.cmd, args.host, args.port, args.open, args.warm = (
            "serve", os.environ.get("MODELMAP_HOST", "127.0.0.1"),
            int(os.environ.get("MODELMAP_PORT", "7860")), True, False,
        )

    if args.cmd == "dump":
        from modelmap.extract import extract_graph

        g = extract_graph(
            args.model_id,
            revision=args.revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            seq_len=args.seq_len,
        )
        out = args.out or args.model_id.split("/")[-1].lower() + ".graph.json"
        with open(out, "w", encoding="utf-8") as f:
            if args.pretty:
                json.dump(g.to_json_dict(), f, indent=2)
            else:
                json.dump(g.to_json_dict(), f, separators=(",", ":"))
        print(
            f"{args.model_id}: fidelity={g.fidelity} nodes={len(g.nodes)} "
            f"repeats={len(g.repeats)} trace={len(g.trace)} "
            f"params={g.params_total:,} -> {out}",
            file=sys.stderr,
        )
        for note in g.notes:
            print(f"  note: {note}", file=sys.stderr)
    elif args.cmd == "warm":
        from modelmap.gallery import GALLERY_IDS
        from modelmap.server import warm

        warm(args.model_ids or GALLERY_IDS)
    else:
        import uvicorn

        if args.warm:
            os.environ["MODELMAP_WARM"] = "1"
        if args.open:
            _open_when_ready(f"http://{args.host}:{args.port}/")
        uvicorn.run("modelmap.server:app", host=args.host, port=args.port, log_level="info")


def _open_when_ready(url: str) -> None:
    import threading
    import time
    import urllib.request
    import webbrowser

    def go():
        for _ in range(100):
            try:
                urllib.request.urlopen(url.rstrip("/") + "/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        webbrowser.open(url)

    threading.Thread(target=go, daemon=True).start()


if __name__ == "__main__":
    main()
