#!/usr/bin/env bash
# Turnkey GCP API, UBLA Bucket, Resource IAM, and Dataset Provisioning Script
# Usage: ./scripts/bootstrap_project.sh <GCP_PROJECT_ID> [REGION]
set -euo pipefail

PROJECT_ID="${1:-${SHELF_BENCH_PROJECT:-jjuneja-fde-sandbox}}"
REGION="${2:-${SHELF_BENCH_REGION:-us-central1}}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[1/2] Provisioning Argolis infrastructure and syncing datasets in project=${PROJECT_ID} (${REGION}) ..."
if [[ -x ".venv/bin/shelf-bench" ]]; then
  .venv/bin/shelf-bench --project "${PROJECT_ID}" --region "${REGION}" bootstrap
else
  PYTHONPATH=src python3 -m cli --project "${PROJECT_ID}" --region "${REGION}" bootstrap
fi

echo "[2/2] Bootstrap complete for ${PROJECT_ID}."
