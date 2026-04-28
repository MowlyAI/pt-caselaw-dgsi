# -------- Stage 1: build (deps + optional data concatenation) --------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

# Optional: concatenate split JSONL shards (each <100MB) back into single files.
# Any directory matching `data/**/chunk_*.jsonl` will be merged into `<dir>.jsonl`.
# Safe no-op when no chunks exist (the API reads from Supabase at runtime).
COPY scripts ./scripts
COPY data ./data
RUN python scripts/concat_chunks.py data || true


# -------- Stage 2: runtime (slim image for Render) --------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY --from=builder /install /usr/local

COPY api ./api
COPY scraper ./scraper
COPY embedder ./embedder
COPY extractor ./extractor
COPY scripts ./scripts
# Concatenated data is copied from the builder stage (if present)
COPY --from=builder /build/data ./data

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render sets $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
