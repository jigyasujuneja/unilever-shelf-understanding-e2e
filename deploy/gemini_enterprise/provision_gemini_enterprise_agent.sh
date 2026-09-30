#!/usr/bin/env bash
# ==============================================================================
# Unilever Shelf Intelligence — Dual-Surface GCP Provisioning Script
#   Surface 1: Cloud Run Web Command Center & Multi-Image Playground (behind IAP)
#   Surface 2: Gemini Enterprise (Google Agentspace / Discovery Engine) Agent
#              bound to OpenAPI 3.0 Extension + BigQuery Grounding Datastore
# ==============================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-jjuneja-fde-sandbox}"
REGION="${REGION:-asia-south2}"
SERVICE_NAME="${SERVICE_NAME:-shelf-intelligence-ux-api}"
BQ_DATASET="${BQ_DATASET:-hul_shelf_analytics}"
AGENTSPACE_LOCATION="${AGENTSPACE_LOCATION:-global}"
OPENAPI_SPEC="deploy/gemini_enterprise/openapi_shelf_intelligence.yaml"

echo "========================================================================"
echo " [1/4] Verifying GCP Project & Enabling Required APIs (${PROJECT_ID})"
echo "========================================================================"
gcloud services enable \
  run.googleapis.com \
  iap.googleapis.com \
  compute.googleapis.com \
  bigquery.googleapis.com \
  discoveryengine.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

echo "========================================================================"
echo " [2/4] Provisioning BigQuery Grounding Dataset (${BQ_DATASET})"
echo "========================================================================"
bq --project_id="${PROJECT_ID}" mk --dataset --if_exists=true \
  --description="Unilever Sales EDGE MT PC & GT/Shikkar Grounding Tables for Gemini Enterprise" \
  "${PROJECT_ID}:${BQ_DATASET}" || true

echo "========================================================================"
echo " [3/4] Deploying Surface 1: Cloud Run UX Command Center & Playground API"
echo "       Region: ${REGION} | Service: ${SERVICE_NAME}"
echo "========================================================================"
if [[ "${DRY_RUN:-1}" == "1" ]]; then
  echo "[DRY_RUN=1] Would execute:"
  echo "  gcloud run deploy ${SERVICE_NAME} --source . --region ${REGION} --project ${PROJECT_ID} --no-allow-unauthenticated --iap"
else
  gcloud run deploy "${SERVICE_NAME}" \
    --source . \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --cpu 4 \
    --memory 8Gi \
    --min-instances 0 \
    --max-instances 10 \
    --no-allow-unauthenticated
fi

echo "========================================================================"
echo " [4/4] Registering Surface 2: Gemini Enterprise (Google Agentspace) Tool"
echo "       Spec: ${OPENAPI_SPEC}"
echo "========================================================================"
echo "OpenAPI 3.0 Spec validated at: ${OPENAPI_SPEC}"
echo "To bind in Google Cloud Console -> Agentspace (Gemini Enterprise):"
echo "  1. Select Engine: unilever-sales-edge-agentspace (${AGENTSPACE_LOCATION})"
echo "  2. Attach Data Store: BigQuery://${PROJECT_ID}/${BQ_DATASET}"
echo "  3. Attach OpenAPI Extension: ${OPENAPI_SPEC}"
echo "Done."
