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
| `MODELMAP_WORKER_MEM_MB` | `4096` | `RLIMIT_AS` for each worker; a hostile repo can at worst kill its own worker |
| `MODELMAP_WARM` | `0` | pre-extract the landing gallery in a background thread on startup |
| `MODELMAP_CACHE` | `~/.cache/modelmap` (image: `/data/graphs`) | graph cache directory |
| `MODELMAP_CACHE_MAX_AGE` | `300` | `Cache-Control: max-age` for cached graphs (a `revision=main` repo can move) |
| `HF_HOME` | (image: `/data/hf`) | where `config.json` metadata is cached |
| `HF_TOKEN` | unset | **do not set on a public deployment** — visitors send their own token per request (`X-HF-Token`), which bypasses the shared cache in both directions |

## Security model (design doc §05)

Extraction handles **attacker-chosen input** (any public repo id), so:

- Repos requiring `trust_remote_code` are refused server-side, unconditionally; they get the
  structural weights view built from safetensors headers alone. Only the CLI (`modelmap dump
  --trust-remote-code`) can opt in, on a machine you control.
- Each extraction runs in a **spawned worker process** with `RLIMIT_AS`, `RLIMIT_CPU`, one
  thread, and a hard wall-clock timeout; a timed-out or crashed worker is abandoned and the
  pool rebuilt. Faults never reach the API process.
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

## Platform sketches (nothing is deployed yet)

Any host that runs a container works. Notes per popular target:

- **Fly.io / Railway / Render**: `Dockerfile` as-is; attach a 1 GB volume at `/data`;
  set `MODELMAP_WARM=1`; 1 GB RAM machine → `MODELMAP_WORKERS=1`.
- **Hugging Face Space (Docker)**: fits the theme; the Space port is 7860 already; expect
  cold starts — warming helps a lot; keep `MODELMAP_WORKERS=1` on the free tier.
- **VPS**: `docker compose up -d` behind Caddy/nginx; Caddy gives TLS for free.

## Publishing the CLI

```bash
uv build            # dist/*.whl includes the built SPA (run `cd web && npm run build` first)
uv publish          # once you want it on PyPI; then:
uvx --index https://download.pytorch.org/whl/cpu modelmap
```
The `--index` flag pins the CPU torch wheel (~200 MB) instead of the CUDA build (~2.5 GB).
CI builds the wheel on every push and attaches it to GitHub releases.
