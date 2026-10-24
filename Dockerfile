# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra azure

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra azure

# Bake the embedding model into the image: no download on cold start, works without egress.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"


FROM python:3.12-slim
RUN useradd --create-home --uid 10001 copilot
WORKDIR /app
COPY --from=build --chown=copilot /app /app
ENV PATH=/app/.venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/app/models \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1
USER copilot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "hireflow_copilot.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
