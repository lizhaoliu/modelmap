# modelmap

Paste a Hugging Face model id. Get a living map of the network — explorable down to
every projection, animated so you can watch a token flow from embedding to logits.

**No weights are ever downloaded.** modelmap fetches `config.json` (~2 KB), instantiates
the model on PyTorch's meta device, runs a hooked fake forward pass to capture real
execution order and tensor shapes, and serves the result as a compact hierarchical graph.

Full design: [docs/design.html](docs/design.html) (v1.0, approved 2026-08-14).

## Status

- [x] Design doc
- [x] M1 — extractor + CLI + API
- [x] M2 — Explore mode (interactive graph UI)
- [x] M3 — Flow mode (animated forward pass with captions)
- [ ] **M4 — micro-views, gallery, hosted deployment** *(next)*

M4 also folds in the hosted-hardening list (measured 2026-08-14): lazy torch import in
the server parent (−250 MB), worker-count env knob, in-flight extraction de-dup,
pre-gzipped cached responses, queue back-pressure (429), and the §05 extraction sandbox.

## Quickstart

```bash
uv sync
(cd web && npm install && npm run build)   # builds the SPA into src/modelmap/web

uv run modelmap serve --port 7860
# → open http://127.0.0.1:7860 · search a model or open /m/Qwen/Qwen3-8B

# or dump a graph JSON directly
uv run modelmap dump Qwen/Qwen3-8B -o qwen3-8b.graph.json
```

### Frontend development

```bash
uv run modelmap serve --port 7860   # API
cd web && npm run dev               # Vite dev server on :5173, proxies /api
```

Repos that require `trust_remote_code` are refused by default (they execute arbitrary
Python); pass `--trust-remote-code` only for repos you trust, ideally locally.
