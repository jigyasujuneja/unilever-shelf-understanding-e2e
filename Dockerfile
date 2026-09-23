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

# The dashboard reads pre-generated artifacts from these paths at request time
# (ui/server.py load_dashboard_payload -> reports/, and GET /shelf-image.png).
# They were previously absent from the image, so the deployed UI rendered a blank
# page with no error -- the frontend swallows the 404 on the shelf image and
# treats an empty reports/ as "no runs yet".
#
# NOTE: this couples the image to generated output that is checked into the repo.
# The better end state is to hydrate reports/ from GCS at boot
# (reporting.gcs_reports_prefix) and serve an explicit "no reports" state
# otherwise; see AUDIT.md Stage 2. Do not untrack reports/ from VCS without
# making that change first, or this COPY will fail the build.
COPY reports/ ./reports/
COPY shelf-image.png ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER 10001

EXPOSE 8080

# Start the Cloud Run HTTP server (serves UI + /api/run-live + OTel Cloud Logging export)
CMD ["python", "ui/server.py"]
