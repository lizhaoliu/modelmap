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
| `/api/summary/{id}[?T=4096&B=1&dtype=bf16&weights=int4]` | headline numbers: params, active params, stacks, config essentials, `recipe` (`["dense", "GQA 4×", "gated SiLU MLP", "RoPE", "RMSNorm + q/k norm"]`), checkpoint format / variant, and `cost` (MACs per token and forward, weight / activation / KV bytes at T, B, dtype) |
| `/api/export/{id}?format=csv\|md\|json\|dot[&T&B&dtype&leaves_only&depth&download=1]` | the model rendered for other tools (text) |
| `/api/plan/{id}?gpus=2&gpu_memory_gb=24&tp=1&pp=2[&T&B&dtype&weights=int4&headroom&gpu=A100+80GB]` | serving placement: per-stage / per-GPU weights, KV, activation bytes, fits?, layer ranges, boundary traffic, KV-limited max context; a `gpu` preset name adds a roofline `throughput` block (prefill / decode tok/s) |
| `/api/train/{id}?method=lora\|qlora\|full[&lora_rank&lora_targets&optimizer&gpus&gpu_memory_gb&sharding=none\|zero2\|zero3&T&B&grad_checkpoint&flash_attention&gpu]` | fine-tuning memory: trainable params, per-GPU weights / grads / optimizer / activations, fits?, largest micro-batch, optional training tok/s |
| `/api/models` · `/api/families` | the architecture zoo: structural facts, derived tags and the recipe per cached graph; curated family lineages |
| `/api/compare?a={id}&b={id}[&format=json\|md&changed_only=1]` | `insights` (derived takeaways: `[{topic, text, a, b}]` — attention scheme → KV ratio, MoE routing, MLP shape, positions, norms, context, vocab…), then the module-by-module diff (aligned by path, then role) + config diff; markdown leads with a *Takeaways* section |
| `/og/m/{id}.png` · `/og/compare.png?a&b` · `/og/arch/{family}.png` · `/og/default.png` | 1200×630 social cards drawn from the graph (cache-only: an unmapped model gets the generic card until someone opens it); every HTML page carries matching `og:`/`twitter:` tags |
| `/badge/{id}.svg` | README badge: `modelmap \| 8.19B · GQA 4× · 36 layers` |
| `/api/gallery` · `/api/search?q=` · `/api/health` | landing data, Hub search, liveness |

Cost numbers are analytic estimates: weight-matmul + attention-core MACs from traced shapes; bytes = shapes × dtype.
`dtype` sets activations/KV and weights whose stored dtype is unknown — stored dtypes (bf16, f8, int4, Q4_K…) are used for weight
bytes unless `weights=` names a precision to plan for (`bf16 f16 f32 f8 int8 int4`): then every weight tensor is re-priced at it
("what if I serve this bf16 checkpoint at int4?"); activations and KV keep `dtype`.

```bash
curl -s "https://modelmap.cc/api/summary/Qwen/Qwen3-8B?T=32768" | jq .cost
curl -sL "https://modelmap.cc/api/export/Qwen/Qwen3-8B?format=csv" -o qwen3-8b.csv
curl -s "https://modelmap.cc/api/plan/Qwen/Qwen3-235B-A22B?gpus=8&gpu_memory_gb=80&tp=8&T=32768" | jq '.fits, .max_context_tokens'
curl -s "https://modelmap.cc/api/plan/Qwen/Qwen3-8B?gpu_memory_gb=16&weights=int4&gpu=T4+16GB" | jq '.fits, .throughput.decode_tok_per_sec_b1'
curl -s "https://modelmap.cc/api/compare?a=Qwen/Qwen2.5-7B&b=Qwen/Qwen3-8B" | jq '.insights[].text'
```

## CLI

```bash
uvx --index https://download.pytorch.org/whl/cpu modelmap              # serve + open the browser
#  (not on PyPI yet: add  --from modelmap@git+https://github.com/lizhaoliu/modelmap)
modelmap ./my-finetune                                                   # open a local checkpoint
modelmap dump Qwen/Qwen3-8B -f md -o -                                   # json | csv | md | dot (to stdout with -o -)
modelmap dump Qwen/Qwen3-8B-GGUF:Q8_0 -f csv --leaves-only
modelmap cost Qwen/Qwen3-235B-A22B -T 32768 -B 4 --dtype f8              # headline numbers + cost table (--json)
modelmap plan Qwen/Qwen3-8B --gpus 2 --gpu-memory 24 --pp 2 -T 32768     # fits? stages, max context (--json)
modelmap plan Qwen/Qwen3-8B --gpu "A100 80GB"                            # + prefill/decode tok/s (--list-gpus for presets)
modelmap plan Qwen/Qwen3-8B --gpu-memory 16 --weights int4               # what if I serve it quantized? (cost/dump take --weights too)
modelmap train Qwen/Qwen3-8B --method qlora --rank 16 --gpus 1 --gpu-memory 24   # fine-tuning memory + largest micro-batch
modelmap diff Qwen/Qwen2.5-7B Qwen/Qwen3-8B                              # takeaways + markdown diff (-f json adds "insights")
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
# until the package is on PyPI, install straight from GitHub (the CPU torch index keeps it small):
MM="uvx --index https://download.pytorch.org/whl/cpu --from modelmap[mcp]@git+https://github.com/lizhaoliu/modelmap modelmap"
claude mcp add modelmap -- $MM mcp                                # local extraction (needs torch, ~1 min first run)
claude mcp add modelmap -- $MM mcp --remote https://modelmap.cc  # delegate to the hosted API
```

Cursor / Windsurf / others: command `uvx`, args `["--index", "https://download.pytorch.org/whl/cpu", "--from",
"modelmap[mcp]@git+https://github.com/lizhaoliu/modelmap", "modelmap", "mcp", "--remote", "https://modelmap.cc"]`. Transport: stdio.
Once published to PyPI the `--from` becomes just `modelmap[mcp]`.

Tools: `describe_model` (includes the `recipe`), `estimate_cost`, `plan_serving` (pass `gpu: "A100 80GB"` for tok/s
estimates, `weights: "int4"` to plan a quantized deployment), `plan_finetune` (full/LoRA/QLoRA memory + largest
micro-batch), `compare_models` (leads with derived takeaways), `list_modules`, `search_models`, `export_markdown` —
each takes model ids in the grammar above and the same `T / B / dtype` assumptions, and answers with the same
numbers the UI shows. Ask your agent *"will Qwen3-32B fit on 2×A100 80GB at 32k context?"* and it has real numbers.

## Embedding

`https://modelmap.cc/m/<id>?embed=1` is the chrome-less view (no top bar / inspector, an attribution badge
that opens the full page). Every URL parameter of the full view applies — `sel=` selection, `lens=`, `T=`, `mode=flow`.
The export menu on any model page copies the `<iframe>` snippet for the current view, and a README badge:

```markdown
[![modelmap: Qwen/Qwen3-8B](https://modelmap.cc/badge/Qwen/Qwen3-8B.svg)](https://modelmap.cc/m/Qwen/Qwen3-8B)
```

Shared links unfurl: every page serves `og:`/`twitter:` tags and a social card rendered from the graph
(`/og/m/<id>.png`). Self-hosters set `MODELMAP_PUBLIC_URL=https://maps.example.com` so the absolute `og:url` /
`og:image` and the sitemap point at their origin.

The `extensions/` folder holds a Chrome extension and a userscript that add a **view in modelmap ↗** button to
Hugging Face model pages.
