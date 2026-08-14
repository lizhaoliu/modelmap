"""modelmap CLI: `modelmap dump <id>` and `modelmap serve`."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="modelmap",
        description="Interactive, animated architecture maps for Hugging Face models.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

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

    s = sub.add_parser("serve", help="run the API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

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
    else:
        import uvicorn

        uvicorn.run("modelmap.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
