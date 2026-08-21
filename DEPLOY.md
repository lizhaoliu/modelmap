# Deploying modelmap

modelmap is one container: a FastAPI app that serves the built web app and the
graph API, plus short-lived extraction worker processes. It needs **no GPU, no
model weights, and no secrets** — the only outbound traffic is Hub metadata.

## Quick local production run

```bash
docker compose up --build        # http://127.0.0.1:7860
```

`compose.yaml` runs the image the way it should run anywhere: non-root,
read-only root filesystem, all capabilities dropped, memory + pid limits, and a
single named volume at `/data` for the graph cache and HF metadata cache.

## Sizing (measured)

| | |
|---|---|
| Server parent process | ~50 MB RSS (torch-free) |
| Each extraction worker | ~350–450 MB RSS (torch + transformers); vision-language models add a transient ~100–300 MB while a depth-1 twin of the vision tower runs on the CPU |
| Uncached extraction | 1–5 s typical, ~15 s for 100+-shard repos (Hub I/O bound) |
| Cached hit | pre-gzipped bytes from disk; ~1 ms of CPU, ETag/304 aware |
| Cache footprint | ~5–25 KB per model; ~1 MB per 100 models |
| Image size | ~2.2 GB (CPU torch dominates) |

Rule of thumb: **1 GB RAM per worker + 0.5 GB headroom.** A 1 GB box runs
`MODELMAP_WORKERS=1` comfortably; the default is 2 for local use.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MODELMAP_HOST` / `MODELMAP_PORT` | `127.0.0.1` / `7860` | bind address (image sets host `0.0.0.0`) |
| `MODELMAP_WORKERS` | `2` | extraction worker processes (image default `1`) |
| `MODELMAP_TIMEOUT` | `180` | seconds per extraction before the worker is abandoned (504) |
| `MODELMAP_MAX_INFLIGHT` | `8` | distinct extractions queued+running before `429 Retry-After` |
| `MODELMAP_RATE_PER_MIN` / `MODELMAP_RATE_BURST` | `20` / `10` | per-client budget for *uncached* extractions and Hub searches (cached hits are never limited) |
| `MODELMAP_TRUST_PROXY` | `0` | set `1` behind a reverse proxy so `X-Forwarded-For` identifies clients |
| `MODELMAP_WORKER_MEM_MB` | `4096` | `RLIMIT_AS` (virtual) for each worker; a hostile repo can at worst kill its own worker, and the pool rebuilds itself. Workers are recycled every 64 tasks. Measured peaks: ~0.7 GB (text), ~2.2 GB (27B VLM twin) |
| `MODELMAP_WARM` | `0` | pre-extract the landing gallery in a background thread on startup |
| `MODELMAP_CACHE` | `~/.cache/modelmap` (image: `/data/graphs`) | graph cache directory |
| `MODELMAP_CACHE_MAX_AGE` | `300` | `Cache-Control: max-age` for cached graphs (a `revision=main` repo can move) |
| `HF_HOME` | (image: `/data/hf`) | where `config.json` metadata is cached |
| `HF_TOKEN` | unset | **do not set on a public deployment** — visitors send their own token per request (`X-HF-Token`), which bypasses the shared cache in both directions |
| `MODELMAP_ALLOW_LOCAL` | `0` | accept `local:/path` ids (read checkpoints from the server's disk). `modelmap serve` on loopback turns it on automatically (`--no-local` to refuse); **never set it on a public deployment** — it would let visitors map any readable file on the host |
| `MODELMAP_TRUST_REMOTE_CODE` | `0` | let extraction execute repos' own modeling Python (`serve --trust-remote-code`). Arbitrary code execution by construction — **never on a public deployment**; the drag-and-drop graph viewer is the hosted answer for such repos |
| `MODELMAP_PUBLIC_URL` | `https://modelmap.cc` | absolute origin used in `og:url` / `og:image` tags, the README badge links and the sitemap — set it to your own host when self-hosting |

## Public API surface

Everything is `GET`, read-only, CORS-open: `/api/graph`, `/api/summary`, `/api/export`, `/api/plan`, `/api/compare`,
`/api/gallery`, `/api/search`, `/api/health`, plus OpenAPI at `/docs`. The SPA may be framed anywhere (`?embed=1`
embeds); API responses send `X-Frame-Options: SAMEORIGIN`. Export/summary/plan/compare endpoints reuse the graph
cache and the same rate limiting as `/api/graph` for uncached ids (see [docs/API.md](docs/API.md)).

## Security model (design doc §05)

Extraction handles **attacker-chosen input** (any public repo id), so:

- Repos requiring `trust_remote_code` are refused server-side, unconditionally; they get the
  structural weights view built from safetensors headers alone. Only the CLI (`modelmap dump
  --trust-remote-code`) can opt in, on a machine you control.
- Each extraction runs in a **spawned worker process** with `RLIMIT_AS`, one thread, and a
  hard wall-clock timeout; a timed-out or crashed worker is abandoned and the pool rebuilt
  (there is deliberately no `RLIMIT_CPU`: it is a per-process lifetime cap, wrong for a pool
  worker that serves many tasks). Faults never reach the API process.
- The container runs **non-root, read-only rootfs, `cap_drop: ALL`, `no-new-privileges`**,
  with the only writable path a data volume. Nothing in the image is a secret.
- Egress needed: `huggingface.co` and its CDN (`*.hf.co`) only. If your platform supports
  egress policies, allow just those hosts.
- Per-client rate limiting and global back-pressure bound the cost of abuse; put a real
  limiter/WAF at the edge for anything internet-facing at scale.
- Search proxies the Hub's public model search; it is rate-limited like extraction.

What this is *not*: a syscall sandbox. The trust boundary is "transformers' own model
classes instantiated from a config dict"; if you need stronger isolation, run the worker
tier under gVisor/Firecracker at the platform level.

## Reverse proxy notes

- Terminate TLS at the proxy; forward to `:7860`.
- Set `MODELMAP_TRUST_PROXY=1` and make sure the proxy sets `X-Forwarded-For`.
- Pass `Accept-Encoding` through unchanged — cached graphs are served pre-gzipped and the
  server negotiates on it.
- `/assets/*` and `/fonts/*` are immutable (hashed / stable); `index.html` is `no-cache`.

## Google Cloud Run (deployed 2026-08-17, free tier; v0.2.0 M7/M8 redeployed 2026-08-19 with the same recipe — build with `gcloud builds submit --tag …/modelmap:latest .` then `gcloud run deploy modelmap --image …`)

What was run, in a fresh project linked to a billing account (the free tier still needs one):

```bash
gcloud projects create modelmap-XXXXXX && gcloud billing projects link modelmap-XXXXXX --billing-account=…
gcloud config set project modelmap-XXXXXX
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create cloud-run-source-deploy --repository-format=docker --location=us-central1
# new projects: Cloud Build runs as the compute default SA, which starts with no permissions
SA="$(gcloud projects describe modelmap-XXXXXX --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
for r in roles/cloudbuild.builds.builder roles/run.builder roles/artifactregistry.writer roles/storage.objectViewer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding modelmap-XXXXXX --member="serviceAccount:$SA" --role="$r"; done
gcloud run deploy modelmap --source . --region us-central1 --allow-unauthenticated \
  --memory 1Gi --cpu 1 --concurrency 40 --timeout 300 --min-instances 0 --max-instances 2 --port 7860 \
  --set-env-vars MODELMAP_WORKERS=1,MODELMAP_TRUST_PROXY=1,MODELMAP_MAX_INFLIGHT=4
```

Measured there (1 vCPU, request-based CPU, `min-instances 0`):

| | |
|---|---|
| Cached graph (gallery baked into the image) | ~120 ms end to end |
| Landing + gallery interactive, first visit | ~5 s (2 MB app download; immutable-cached after) |
| Gallery card → rendered graph | ~0.8 s |
| **First** uncached extraction on a fresh instance | **30–40 s** — the worker's first `import torch` streams ~1 GB of image layers (Cloud Run lazy-loads the image) |
| Subsequent uncached extractions on that instance | 4.5–7.5 s (~2× local; 1 slow vCPU) |
| VLM (shallow twin) on a warm instance | ~7 s |

Custom domain: `modelmap.cc` is mapped with `gcloud beta run domain-mappings create --service modelmap
--domain modelmap.cc --region us-central1` (after a one-time Search Console verification of the domain
for the deploying Google account); DNS at the registrar carries Google's four A / four AAAA records
for `@` and `CNAME www → ghs.googlehosted.com`. Google issues and renews the TLS certificate.

The image bake (`modelmap warm` at build time) covers the classics *and* whatever is trending on the Hub at build time; the landing page refreshes the trending list hourly at runtime, and newer entries extract on first visit.

Notes: instances keep their cache while alive (`cache_entries` grows), but every new instance
starts with only the baked gallery. Runtime warming (`MODELMAP_WARM`) does not help here — CPU is
throttled between requests — which is why the Dockerfile bakes the warmed gallery in. To remove
the first-extraction penalty you'd pay for `--min-instances 1 --no-cpu-throttling` (~$10–15/mo);
for a free trial, accept it. `--max-instances 2` caps any surprise spend. Tear down with
`gcloud run services delete modelmap --region us-central1` or delete the project.

## Other platforms

Any host that runs a container works. Notes per popular target:

- **Fly.io / Railway / Render**: `Dockerfile` as-is; attach a 1 GB volume at `/data`;
  set `MODELMAP_WARM=1`; 1 GB RAM machine → `MODELMAP_WORKERS=1`.
- **Hugging Face Space (Docker)**: fits the theme; port 7860 already; `scripts/deploy_space.py`
  creates and updates it. Docker Spaces now require a PRO subscription ($9/mo) — the free tier
  only hosts static Spaces.
- **VPS**: `docker compose up -d` behind Caddy/nginx; Caddy gives TLS for free.

## Publishing the CLI

```bash
uv build            # dist/*.whl includes the built SPA (run `cd web && npm run build` first)
uv publish          # once you want it on PyPI; then:
uvx --index https://download.pytorch.org/whl/cpu modelmap
```
The `--index` flag pins the CPU torch wheel (~200 MB) instead of the CUDA build (~2.5 GB).
CI builds the wheel on every push and attaches it to GitHub releases.
