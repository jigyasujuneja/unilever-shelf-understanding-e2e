#!/usr/bin/env bash
# Production Cloud Build + Cloud Run Jobs & Control Plane Service Deployment Pipeline
# Usage: ./scripts/deploy_cloud_run.sh <GCP_PROJECT_ID> [REGION] [MODE: job|service]
set -euo pipefail

PROJECT_ID="${1:-${SHELF_BENCH_PROJECT:-jjuneja-fde-sandbox}}"
REGION="${2:-${SHELF_BENCH_REGION:-us-central1}}"
MODE="${3:-job}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CLI=(PYTHONPATH=src python3 -m cli)
if [[ -x ".venv/bin/shelf-bench" ]]; then
  CLI=(.venv/bin/shelf-bench)
fi

if [[ "${MODE}" == "service" ]]; then
  echo "Deploying Always-On Perfect Store Control Plane Service to Cloud Run (${PROJECT_ID} / ${REGION}) ..."
  "${CLI[@]}" --project "${PROJECT_ID}" --region "${REGION}" cloud-service --port 8080
else
  echo "Executing EPIC Benchmark Pipeline on Cloud Run Jobs (${PROJECT_ID} / ${REGION}) ..."
  "${CLI[@]}" --project "${PROJECT_ID}" --region "${REGION}" cloud-run \
    -a hul_8stage_gemini38_hybrid \
    -m gemini-3.8-flash \
    --split test \
    --limit 5 \
    --workers 5
fi
