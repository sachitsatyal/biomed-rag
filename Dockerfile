# biomed-rag web app: API + web page + dense index + embedding model in one image.
# Build:  docker build -t biomed-rag .
# Run:    docker run --rm -p 8000:8000 --env-file .env biomed-rag

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Dependencies only. This layer is cached and rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) Bake the embedding model into the image, so the container never downloads at startup.
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
S('BAAI/bge-small-en-v1.5')"

# 3) Our own code (changes most often, so it comes last).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# 4) The prebuilt FAISS index (built locally with scripts/build_dense_index.py).
COPY data/indexes/dense_bge-small_abstract ./data/indexes/dense_bge-small_abstract

# 5) Run as a normal user, not root (safer). Offline: everything is already inside.
RUN useradd --create-home --uid 1000 app
USER app
ENV HF_HUB_OFFLINE=1 \
    PORT=8000
EXPOSE 8000

# Cloud platforms set $PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn biomed_rag.serving.app:app --host 0.0.0.0 --port ${PORT}"]
