# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    # Keep model downloads on a mounted volume so they survive a rebuild.
    HF_HOME=/app/.cache/huggingface \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Build tooling + curl for healthchecks. Playwright's own OS deps come later
# via `playwright install --with-deps`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only torch build first. The default PyPI wheel bundles CUDA
# and is ~2.5 GB; the CPU wheel is ~200 MB and is all we need for embeddings.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# crawl4ai drives a real browser, so the browser has to exist in the image.
RUN playwright install --with-deps chromium

COPY . .

# Runtime dirs (also mounted as volumes in docker-compose).
RUN mkdir -p /app/sitecontent /app/chroma_store /app/sessions /app/output /app/.cache/huggingface

EXPOSE 5003 5005 8501 8502

CMD ["python", "-c", "print('Specify a command via docker-compose')"]
