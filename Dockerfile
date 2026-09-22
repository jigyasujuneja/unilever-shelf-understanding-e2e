# =============================================================================
# Google Cloud Run Production Container — Unilever Shelf Understanding Benchmark
# Hosts both the REST Benchmark API (/api/run-live, /api/benchmark-data) and
# the Executive Interactive Studio on $PORT (default 8080).
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GCP_PROJECT_ID="unilever-shelf-understanding"

WORKDIR /app

# Install system dependencies for Pillow / image cropping
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY ui/ ./ui/
COPY code_samples/ ./code_samples/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8080

# Start the Cloud Run HTTP server (serves UI + /api/run-live + OTel Cloud Logging export)
CMD ["python", "ui/server.py"]
