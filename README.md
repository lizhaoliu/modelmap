# modelmap

Paste a Hugging Face model id. Get a living map of the network — explorable down to
every projection, animated so you can watch a token flow from embedding to logits.

**No weights are ever downloaded.** modelmap fetches `config.json` (~2 KB), instantiates
the model on PyTorch's meta device, runs a hooked fake forward pass to capture real
execution order and tensor shapes, and serves the result as a compact hierarchical graph.

Full design: [docs/design.html](docs/design.html) (v1.0, approved 2026-08-14).

## Status

- [x] Design doc
- [ ] **M1 — extractor + CLI + API** *(in progress)*
- [ ] M2 — Explore mode (interactive graph UI)
- [ ] M3 — Flow mode (animated forward pass)
- [ ] M4 — micro-views, gallery, hosted deployment

## Quickstart (M1)

```bash
uv sync

# dump a graph JSON for any HF model id
uv run modelmap dump Qwen/Qwen3-8B -o qwen3-8b.graph.json

# serve the API (the SPA lands in M2)
uv run modelmap serve --port 7860
# → GET http://127.0.0.1:7860/api/graph/Qwen/Qwen3-8B
```

Repos that require `trust_remote_code` are refused by default (they execute arbitrary
Python); pass `--trust-remote-code` only for repos you trust, ideally locally.
