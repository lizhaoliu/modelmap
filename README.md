# modelmap

Paste a Hugging Face model id. Get a living map of the network — explorable down to
every projection, animated so you can watch a token flow from embedding to logits.

**No weights are ever downloaded.** modelmap fetches `config.json` (~2 KB), instantiates
the model on PyTorch's meta device, runs a hooked fake forward pass to capture real
execution order and tensor shapes, and serves the result as a compact hierarchical graph.
An 8B or 671B model costs the same few seconds; the graph ships as ~10 KB gzipped.

**Live demo:** https://modelmap.cc (free-tier Cloud Run: cached models are instant; the first uncached model on a cold instance takes ~30 s)

Design doc: [docs/design.html](docs/design.html) · API (REST · Python · CLI · MCP): [docs/API.md](docs/API.md) · Deployment: [DEPLOY.md](DEPLOY.md) · Extending: [EXTENDING.md](EXTENDING.md)

## Run it

```bash
# one-off, nothing to install (pins the CPU torch wheel instead of the 2.5 GB CUDA one)
uvx --index https://download.pytorch.org/whl/cpu modelmap        # opens http://127.0.0.1:7860

# from a checkout
uv sync && (cd web && npm install && npm run build)
uv run modelmap                       # serve + open browser
uv run modelmap ./my-finetune         # …opening a local checkpoint (dir, .safetensors or .gguf)
uv run modelmap serve --warm          # pre-extract the gallery in the background
uv run modelmap dump Qwen/Qwen3-8B -f md          # graph as JSON / CSV / Markdown / DOT
uv run modelmap cost Qwen/Qwen3-8B-GGUF:Q4_K_M    # headline numbers for a quantized variant
uv run modelmap plan Qwen/Qwen3-8B --gpus 2 --gpu-memory 24 --pp 2 -T 32768   # does it fit?
uv run modelmap diff Qwen/Qwen2.5-7B Qwen/Qwen3-8B
uv run modelmap mcp                   # MCP server for Claude Code / Cursor (pip install 'modelmap[mcp]')

# container (non-root, read-only rootfs, capabilities dropped)
docker compose up --build
```

`uvx modelmap` needs the package on PyPI (`uv publish`) or a wheel from a GitHub release
(`uvx --from <wheel-url> …`); CI builds and attaches one per release.

## What you get

- **Explore mode** — a zoomable, collapsible map of the module tree. Repeated layers collapse
  into stacks (`decoder block ×36`, `experts ×128`) so a 235B MoE opens as ~10 nodes — including
  interleaved designs (DeepSeek-V4's alternating attention layers collapse into `×43 — 2 repeated designs`). Click any
  node for class, params (absolute + share), dtype, weight shapes with inline dim labels
  (`[151936 vocab × 4096 hidden]`), traced I/O shapes (`[1 batch × 7 seq × 4096 hidden]`), and a
  parameter treemap of its children, the module's own attributes (`in_features`, `eps`, kernel…)
  and a link to its defining source line. A **cost lens** (`params · compute · memory · kv`) re-encodes
  the map with analytic MACs, activation and KV-cache bytes and active-params-per-token for MoE, all
  re-derived live from a what-if bar (sequence length, batch, dtype). Breadcrumb trail, semantic colors, light/dark, shareable
  URLs that reproduce the exact view, keyboard-first (`?` lists shortcuts).
- **Flow mode** — replays the traced forward pass: an amber pulse travels the graph in real
  execution order while a HUD narrates each step with true shapes and a plain-language caption.
  Repeat stacks compress with a `layer 12 / 36` counter (~12 s for a full 8B replay). Expand a
  block and the pulse walks its internals. **Micro-views** show a beat's inner choreography —
  a block's norm → attention → ⊕ → norm → MLP → ⊕, attention's Q/K/V → scores → softmax → merge,
  a gated MLP, an MoE router — all filled from the model's own config and trace.
- **Compare** — `/compare/A...B` (or “compare…” in the top bar): two models aligned by path, then by
  role, with linked pan/zoom/expansion/selection; unchanged modules dim, changed ones outline amber,
  additions/removals get `+`/`−`; a two-column diff inspector and a summary strip (params, layers,
  heads, KV heads, hidden, ffn, vocab, context, compute/token, KV/token). Base vs fine-tune reports
  no structural change; Qwen2.5-7B vs Qwen3-8B calls out q/k norms, dropped biases, ffn, layers.
- **Take it with you** — `export ▾` saves a PNG/SVG of the current view, a Markdown summary, a CSV module
  table (params, shapes, dtype, cost columns), the JSON document or Graphviz DOT; copies a link to the exact
  view, an `<iframe>` embed (`?embed=1` is a chrome-less mode for model cards and blogs) or the API URL.
  Every node has a *copy link*. The same data is a REST API (`/api/summary`, `/api/export`, `/api/plan`,
  `/api/compare`, OpenAPI at `/docs`), a Python API, a CLI (`dump · cost · plan · diff`) and an
  **MCP server** so a coding agent can ask "will Qwen3-32B fit on 2×A100 at 32k?" — see [docs/API.md](docs/API.md).
  A Chrome extension / userscript adds a *view in modelmap* button to Hugging Face pages ([extensions/](extensions/)).
- **Serving planner** — `fit?` in the top bar: pick GPUs × memory, tensor / pipeline parallel degree and
  headroom; see per-stage weights / KV / activation bytes, which layers land where, the activation bytes crossing
  each pipeline boundary, whether it fits, and the KV-limited maximum context at the chosen batch.
- **Quantized checkpoints** — `owner/name:Q4_K_M` opens a GGUF variant (a picker lists the repo's quants): the
  config is rebuilt from the GGUF header via range reads, the module tree and trace are real, and every module
  shows its actual quant type and bits per weight, so the Cost lens reports true on-disk bytes. FP8 / AWQ / GPTQ /
  bitsandbytes repos report their quantized dtypes too.
- **Local checkpoints** — `modelmap ./ckpt` (or `local:/path` in the local UI): your own fine-tunes, merges and
  GGUFs, never leaving your machine. Compare one against its base to see what changed.
- **Any public repo.** Text LLMs (dense and MoE), encoder-decoders (T5/BART), speech (Whisper),
  BERT, ViT, vision-language and audio-language (the encoder tower and
  its projector are traced too — on the meta device when possible, otherwise via a depth-1 CPU
  twin, so image → patches → blocks → image tokens → LLM replays end to end), and a fallback
  ladder for the rest: architectures transformers can't instantiate, or repos requiring
  `trust_remote_code` (refused by default — it executes arbitrary Python), get a structural
  **weights view** rebuilt from safetensors headers alone. Gated repos work once you add a
  token (top bar → token; stored in your browser only, never cached server-side).

## Status

- [x] Design doc
- [x] M1 — extractor + CLI + API
- [x] M2 — Explore mode
- [x] M3 — Flow mode with captions
- [x] M4 — micro-views, gallery, treemap, `uvx` packaging, hardened deploy-ready container
- [x] M5 — Cost lens: compute / memory / KV analytics with a what-if bar (client-side, from traced shapes)
- [x] M6 — Compare: two models side by side, aligned by path then role, differences first
- [x] M7 — Export & integrate: PNG/SVG/CSV/Markdown/JSON/DOT, embed mode, node links, REST + Python + CLI + MCP
- [x] M8 — GGUF / quantized variants, local checkpoints, serving planner, interleaved repeat stacks
- [x] Deployed on Google Cloud Run (free tier) at https://modelmap.cc; see DEPLOY.md

## Development

```bash
uv run modelmap serve --port 7860     # API + built SPA
cd web && npm run dev                 # Vite dev server on :5173, proxies /api

uv run pytest                         # unit tests (collapse, hub retries, server + API, analytics, compare, export, MCP)
(cd web && npm test)                  # vitest: cost lens, planner, compare alignment on real graph fixtures
uv run python tests/e2e/test_explore.py   # browser acceptance suites (need a running server
uv run python tests/e2e/test_flow.py      # on :7860, playwright, and a Chromium build)
uv run python tests/e2e/test_m7.py        # export / embed / planner / GGUF / local
```

Layout: `src/modelmap/` (extractor, server, CLI) · `web/` (React + React Flow + elkjs SPA,
built into `src/modelmap/web/` and shipped in the wheel) · `tests/` · `docs/design.html`.

## License

MIT — see [LICENSE](LICENSE).
