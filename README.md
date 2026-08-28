# modelmap

Paste a Hugging Face model id. Get a living map of the network — explorable down to
every projection, animated so you can watch a token flow from embedding to logits — and
the answer to the question you re-ask every week: **will it fit on my GPU?**

**No weights are ever downloaded.** modelmap fetches `config.json` (~2 KB), instantiates
the model on PyTorch's meta device, runs a hooked fake forward pass to capture real
execution order and tensor shapes, and serves the result as a compact hierarchical graph.
An 8B or 671B model costs the same few seconds; the graph ships as ~10 KB gzipped.

**Live demo:** https://modelmap.cc (free-tier Cloud Run: cached models are instant; the first uncached model on a cold instance takes ~30 s).
Try **⚡ live**: open [TinyLLama-v0](https://modelmap.cc/m/Maykeye/TinyLLama-v0), press ⚡ live, type a prompt — real
next-token probabilities, per-head attention and a logit lens, computed in *your* browser (9 MB download; the server still never touches weights).

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

`uvx modelmap` needs the package on PyPI (`uv publish`); until then install from GitHub:
`uvx --index https://download.pytorch.org/whl/cpu --from modelmap@git+https://github.com/lizhaoliu/modelmap modelmap`
(add `[mcp]` → `modelmap[mcp]@git+…` for the MCP server). CI also attaches a wheel to each GitHub release.

## What you get

- **Explore mode** — a zoomable, collapsible map of the module tree. Repeated layers collapse
  into stacks (`decoder block ×36`, `experts ×128`) so a 235B MoE opens as ~10 nodes — including
  interleaved designs (DeepSeek-V4's alternating attention layers collapse into `×43 — 2 repeated designs`). Click any
  node for class, params (absolute + share), dtype, weight shapes with inline dim labels
  (`[151936 vocab × 4096 hidden]`), traced I/O shapes (`[1 batch × 7 seq × 4096 hidden]`), and a
  parameter treemap of its children, the module's own attributes (`in_features`, `eps`, kernel…)
  and a link to its defining source line. A **cost lens** (`params · compute · memory · kv · vram`) re-encodes
  the map with analytic MACs, activation and KV-cache bytes and active-params-per-token for MoE, all
  re-derived live from a what-if bar (sequence length, batch, dtype). Breadcrumb trail, semantic colors, light/dark, shareable
  URLs that reproduce the exact view, keyboard-first (`?` lists shortcuts).
- **Flow mode** — the animation is the default impression: the landing page opens on a running mini-replay,
  a first visit to any model starts its own replay (once; never under reduced motion), edges drift while it runs,
  edge thickness encodes the tensor flowing through, and the camera follows the pulse (pan to take over). It replays the traced forward pass: an amber pulse travels the graph in real
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
- **⚡ Live mode** — for small models (llama-family / gpt2, single-file safetensors ≤ 700 MB) the browser downloads
  the weights and runs *real inference* on your CPU in a Web Worker — a hand-written TS engine (RMSNorm/rotary/GQA/SwiGLU
  and the GPT-2 stack, KV cache, two BPE tokenizer families read straight from tokenizer.json). Type a prompt: true
  next-token probabilities, a per-head **attention heatmap** (layer slider, head picker, or click an attention block
  on the map), a **logit lens** showing the prediction sharpen layer by layer, and streamed generation that ripples
  through the map. TinyLLama-v0 (9 MB, ~4 ms/token) is the instant default; SmolLM2-135M and distilgpt2 also run.
  The engine is pinned to real `transformers`/`tokenizers` outputs by fixture tests (logits 2e-4, attention 2e-5,
  greedy token-for-token, byte-identical tokenization).
- **Take it with you** — `export ▾` saves a PNG/SVG of the current view, a Markdown summary, a CSV module
  table (params, shapes, dtype, cost columns), the JSON document or Graphviz DOT; copies a link to the exact
  view, an `<iframe>` embed (`?embed=1` is a chrome-less mode for model cards and blogs) or the API URL.
  Every node has a *copy link*. The same data is a REST API (`/api/summary`, `/api/export`, `/api/plan`,
  `/api/compare`, OpenAPI at `/docs`), a Python API, a CLI (`dump · cost · plan · diff`) and an
  **MCP server** so a coding agent can ask "will Qwen3-32B fit on 2×A100 at 32k?" — see [docs/API.md](docs/API.md).
  A Chrome extension / userscript adds a *view in modelmap* button to Hugging Face pages ([extensions/](extensions/)).
- **Serving + fine-tuning planner** — `fit?` in the top bar has two tabs. *Serve*: GPUs × memory, TP/PP
  split, per-stage weights / KV / activation bytes, fits?, KV-limited max context — plus **roofline speed
  estimates** per GPU preset (prefill tok/s compute-bound, decode tok/s bandwidth-bound, MoE streams active
  experts only). *Fine-tune*: full / **LoRA / QLoRA** with rank and target choices, AdamW vs 8-bit, ZeRO-2/3
  sharding, gradient-checkpointing and flash-attention toggles → per-GPU weights / grads / optimizer /
  activations, largest micro-batch, training tok/s. Same numbers from `modelmap train`, `modelmap plan --gpu`,
  `/api/train`, and the MCP `plan_finetune` tool (Python + TS twins pinned to the same fixtures).
- **Paste what you have** — a full `huggingface.co/…` URL (file-tree links included), an ollama-style
  `hf.co/owner/name:Q4_K_M` id, even an id that picked up typographic dashes in a doc: everything normalizes to
  the canonical `owner/name[:variant]` before loading. Gated repos answer with the actual fix (accept the license,
  add your token) instead of a header error.
- **Quantized checkpoints** — `owner/name:Q4_K_M` opens a GGUF variant (a picker lists the repo's quants): the
  config is rebuilt from the GGUF header via range reads, the module tree and trace are real, and every module
  shows its actual quant type and bits per weight, so the Cost lens reports true on-disk bytes. FP8 / AWQ / GPTQ /
  bitsandbytes repos report their quantized dtypes too.
- **Local checkpoints** — `modelmap ./ckpt` (or `local:/path` in the local UI): your own fine-tunes, merges and
  GGUFs, never leaving your machine. Compare one against its base to see what changed.
- **Custom-code repos** (Kimi-K3 and friends) — the hosted server refuses `trust_remote_code` by design, but you
  can run `modelmap dump <id> --trust-remote-code` on your machine and **drop the `.graph.json` anywhere on the
  site**: it renders fully client-side (flow, lenses, exports) and nothing is uploaded. `modelmap serve
  --trust-remote-code` opts your own server in.
- **The architecture zoo** — [`/models`](https://modelmap.cc/models): every mapped model as a filterable table of
  derived structural tags (`moe 8/128`, `mla`, `gqa 4×`, `vlm`, `ctx 1M`…); [`/arch/qwen`](https://modelmap.cc/arch/qwen)
  and seven more family pages whose lineage arrows are *live structural diffs* — the Qwen2→Qwen3 q/k-norm story
  told by the data, not by prose. Also `/api/models` and `/api/families`.
- **Live interpretability** — in ⚡ live mode, **silence any attention head** and watch the next-token
  probabilities shift (▲/▼ deltas vs baseline; restore is bit-exact), per-head pattern tags (`prev-token`,
  `sink`, `broad`), and qwen2/qwen3 checkpoints join llama/gpt2 in the in-browser engine.
- **The planner, on the map** — the `vram` lens paints serving memory onto the graph: every module shows its
  weights plus the KV cache its attention holds at the chosen context and batch, and a strip above the map holds
  the knobs — context slider, batch, **weight precision** (as stored / bf16 / fp8 / int8 / int4), GPU preset, headroom —
  with the verdict: weights + KV + activations stacked against the card (`fits on 1× RTX 4090 · up to 45k tokens`,
  `needs 2× (tensor-parallel)`). Drag context to 128k and watch the cache outweigh the model. The landing page
  opens with *will it fit on &lt;GPU&gt;?* and every gallery card has a `fit?` button. The precision toggle is the same
  `--weights int4` on the CLI, `weights=int4` on `/api/plan` and `/api/summary`, and `weights` on the MCP tools.
- **Takeaways** — compare pages, the zoo's lineage arrows, `modelmap diff`, `/api/compare` and the MCP tool all
  lead with derived, quantified sentences instead of leaving the interpretation to you: *“Attention: GQA (8 KV heads
  for 32 query heads) vs GQA (4 for 28) → KV cache per token 144 KB vs 56 KB (2.6× larger)”*, *“B is a mixture of
  experts: 8 of 128 experts run per token, so 3.35B of its 30.5B parameters are active”*, MLP shape (gated SwiGLU
  vs 2-matrix GELU), positions (RoPE vs learned vs relative bias, base and scaling), norms, q/k norm, biases, tying,
  context, vocab. Every model also gets a one-line **recipe** (`MoE 8/256 +1 shared · MLA · gated SiLU MLP · RoPE · RMSNorm`).
- **Links that unfurl** — every page carries server-rendered `og:`/`twitter:` tags and a 1200×630 **social card**
  drawn from the graph itself (`/og/m/<id>.png`: headline numbers, structural tags and a miniature of the
  architecture; compare cards carry the takeaways; family cards the lineage), so a shared link shows the model, not
  a bare domain. `/badge/<id>.svg` is a README badge (`modelmap | 8.19B · GQA 4× · 36 layers`) — *copy README badge*
  is in the export menu.
- **Find anything** — press `/` on a map to search modules by path or class (`k_norm`, `layers.17`, `RMSNorm`);
  Enter opens every ancestor and frames the module. Deep links (`?sel=model.layers.0.mlp.down_proj`) do the same.
  Double-click opens a container (and a leaf now tells you it is one).
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
- [x] M9 — alive by default: landing hero replay, first-visit autoplay, drifting edges, camera follow
- [x] M10 — ⚡ live mode: real in-browser inference (llama + gpt2), attention heatmaps, logit lens, streamed generation
- [x] M11 — coverage: drag-and-drop graph files for trust_remote_code models, `serve --trust-remote-code`
- [x] M12 — fine-tuning planner (LoRA/QLoRA/full, ZeRO) + roofline throughput estimates per GPU
- [x] M13 — architecture zoo: /models catalog with structural tags, /arch family pages with live lineage diffs
- [x] M14 — live interpretability: head ablation with Δ display, head pattern tags, qwen2/qwen3 engines
- [x] M15 — social cards + per-URL meta tags, README badge
- [x] M16 — the planner on the map: vram lens, weight-precision toggle (UI · CLI · API · MCP), "will it fit on <GPU>?" landing
- [x] M17 — takeaways: derived architecture insights on compare, zoo lineages, diff, API and MCP; per-model recipe
- [x] M18 — node finder (`/`), revealing deep links, double-click-to-open fixed, VLM cost fix (text_config)
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
uv run python tests/e2e/test_m9.py        # hero replay, autoplay, camera follow
uv run python tests/e2e/test_live.py      # ⚡ live: downloads TinyLLama-v0 and runs it (+ head ablation)
uv run python tests/e2e/test_zoo.py       # catalog, family pages, dropped graph files
```

Layout: `src/modelmap/` (extractor, server, CLI) · `web/` (React + React Flow + elkjs SPA,
built into `src/modelmap/web/` and shipped in the wheel) · `tests/` · `docs/design.html`.

## License

MIT — see [LICENSE](LICENSE).
