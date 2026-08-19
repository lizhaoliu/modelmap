# modelmap API — REST · Python · CLI · MCP

Everything modelmap knows about a model comes from one JSON document (the *graph*): the
module tree from a meta-device instantiation, execution order and tensor shapes from a traced
fake forward pass, dtypes from checkpoint headers, and analytics derived from all three.
Every surface below reads that document; no surface ever downloads weights.

Interactive reference for the hosted API: **https://modelmap.cc/docs** (OpenAPI).

## Model ids

| form | meaning |
|---|---|
| `owner/name` | a Hub repo (`config.json` + safetensors / pytorch bins); legacy top-level names like `gpt2` work |
| `owner/name:Q4_K_M` | a **GGUF** variant in the repo — config rebuilt from the GGUF header, real quant dtype per module (`Q4_K · 4.5 bpw`). Any label listed in the document's `variants` (case-insensitive); default `Q4_K_M` when present |
| `local:/path/to/ckpt` | a local checkpoint directory, a `.safetensors` file or a `.gguf` — **only** where local paths are enabled (the CLI always; `modelmap serve` on loopback; never on the hosted site) |

## REST

All endpoints are `GET`, CORS-open to any origin, and return JSON unless noted.
Gated / private repos: send the `X-HF-Token: hf_…` header (such responses are never cached server-side).

| endpoint | returns |
|---|---|
| `/api/graph/{id}[?revision=main]` | the graph document (gzip, ETag / 304) |
| `/api/summary/{id}[?T=4096&B=1&dtype=bf16]` | headline numbers: params, active params, stacks, config essentials, checkpoint format / variant, and `cost` (MACs per token and forward, weight / activation / KV bytes at T, B, dtype) |
| `/api/export/{id}?format=csv\|md\|json\|dot[&T&B&dtype&leaves_only&depth&download=1]` | the model rendered for other tools (text) |
| `/api/plan/{id}?gpus=2&gpu_memory_gb=24&tp=1&pp=2[&T&B&dtype&headroom=0.1]` | serving placement: per-stage / per-GPU weights, KV, activation bytes, fits?, layer ranges, boundary traffic, KV-limited max context |
| `/api/compare?a={id}&b={id}[&format=json\|md&changed_only=1]` | module-by-module diff (aligned by path, then role) + config diff |
| `/api/gallery` · `/api/search?q=` · `/api/health` | landing data, Hub search, liveness |

Cost numbers are analytic estimates: weight-matmul + attention-core MACs from traced shapes; bytes = shapes × dtype.
`dtype` sets activations/KV and weights whose stored dtype is unknown — stored dtypes (bf16, f8, int4, Q4_K…) are always used for weight bytes.

```bash
curl -s "https://modelmap.cc/api/summary/Qwen/Qwen3-8B?T=32768" | jq .cost
curl -sL "https://modelmap.cc/api/export/Qwen/Qwen3-8B?format=csv" -o qwen3-8b.csv
curl -s "https://modelmap.cc/api/plan/Qwen/Qwen3-235B-A22B?gpus=8&gpu_memory_gb=80&tp=8&T=32768" | jq '.fits, .max_context_tokens'
curl -s "https://modelmap.cc/api/compare?a=Qwen/Qwen2.5-7B&b=Qwen/Qwen3-8B&format=md"
```

## CLI

```bash
uvx --index https://download.pytorch.org/whl/cpu modelmap              # serve + open the browser
modelmap ./my-finetune                                                   # open a local checkpoint
modelmap dump Qwen/Qwen3-8B -f md -o -                                   # json | csv | md | dot (to stdout with -o -)
modelmap dump Qwen/Qwen3-8B-GGUF:Q8_0 -f csv --leaves-only
modelmap cost Qwen/Qwen3-235B-A22B -T 32768 -B 4 --dtype f8              # headline numbers + cost table (--json)
modelmap plan Qwen/Qwen3-8B --gpus 2 --gpu-memory 24 --pp 2 -T 32768     # fits? stages, max context (--json)
modelmap diff Qwen/Qwen2.5-7B Qwen/Qwen3-8B                              # markdown diff (-f json)
modelmap serve --host 0.0.0.0 --no-local                                 # the hosted configuration
```

## Python

```python
from modelmap import extract_graph, summarize, plan_serving, align, render, Assumptions, PlanRequest

doc = extract_graph("Qwen/Qwen3-8B").to_json_dict()          # or "Qwen/Qwen3-8B-GGUF:Q4_K_M", or allow_local=True + "local:/ckpt"
s = summarize(doc, Assumptions(T=32768, B=1, dtype="bf16"))   # dict: params, active_params, config, cost{...}
plan = plan_serving(doc, PlanRequest(gpus=2, gpu_memory_gb=24, pp=2, T=32768))
plan.fits, plan.max_context_tokens, [st.layers for st in plan.stages]
al = align(doc, extract_graph("Qwen/Qwen2.5-7B").to_json_dict())
al.counts, al.config_diff
text, media_type = render(doc, "md")                           # csv | md | json | dot
```

`import modelmap` is torch-free; `extract_graph` imports torch on first use. Graph documents are plain dicts
(see `src/modelmap/schema.py`, `SCHEMA_VERSION`), so anything you cache or `dump` can be fed back into the analytics.

## MCP server (for coding agents)

```bash
claude mcp add modelmap -- uvx --from 'modelmap[mcp]' modelmap mcp                     # local extraction
claude mcp add modelmap -- uvx --from 'modelmap[mcp]' modelmap mcp --remote https://modelmap.cc   # ask the hosted API
```

Cursor / Windsurf / others: command `uvx`, args `["--from", "modelmap[mcp]", "modelmap", "mcp"]`. Transport: stdio.

Tools: `describe_model`, `estimate_cost`, `plan_serving`, `compare_models`, `list_modules`, `search_models`,
`export_markdown` — each takes model ids in the grammar above and the same `T / B / dtype` assumptions, and answers
with the same numbers the UI shows. Ask your agent *"will Qwen3-32B fit on 2×A100 80GB at 32k context?"* and it has real numbers.

## Embedding

`https://modelmap.cc/m/<id>?embed=1` is the chrome-less view (no top bar / inspector, an attribution badge
that opens the full page). Every URL parameter of the full view applies — `sel=` selection, `lens=`, `T=`, `mode=flow`.
The export menu on any model page copies the `<iframe>` snippet for the current view.

The `extensions/` folder holds a Chrome extension and a userscript that add a **view in modelmap ↗** button to
Hugging Face model pages.
