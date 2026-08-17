# modelmap — deploy-ready image (design doc §04/§05, hardening per §10).
#
#   docker build -t modelmap .
#   docker run --rm -p 7860:7860 -v modelmap-cache:/data modelmap
#
# Multi-stage: build the SPA with Node, then a slim Python runtime with the
# CPU-only torch wheel (extraction never touches weights or a GPU).

# ---------- stage 1: web ----------
FROM node:22-alpine AS web
WORKDIR /build/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build          # → /build/src/modelmap/web (vite outDir is ../src/modelmap/web)

# ---------- stage 2: python ----------
FROM python:3.12-slim-bookworm AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # everything writable lives under /data (mount a volume there)
    MODELMAP_CACHE=/data/graphs \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    # conservative defaults for a small box; override at run time
    MODELMAP_WORKERS=1 \
    MODELMAP_HOST=0.0.0.0 \
    MODELMAP_PORT=7860

WORKDIR /app
# dependency layer first (cached across source changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
# source + built SPA
COPY src/ ./src/
COPY --from=web /build/src/modelmap/web ./src/modelmap/web
RUN uv sync --frozen --no-dev

# non-root, writable data dir only
RUN useradd -r -u 10001 -d /data modelmap && mkdir -p /data && chown -R modelmap:modelmap /data
USER modelmap
VOLUME ["/data"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=2).status==200 else 1)"

ENTRYPOINT ["/app/.venv/bin/modelmap"]
CMD ["serve"]
