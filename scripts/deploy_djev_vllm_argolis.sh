#!/usr/bin/env bash
# ==============================================================================
# Deploy `DiffusionGemma-26B-A4B-it` (`dJev` / `vLLM PR #57250`) on Argolis GCP
# ==============================================================================
# Verified against merged upstream PR: https://github.com/vllm-project/vllm/pull/57250
# and reference server: https://github.com/mmastrac/djev
#
# Supported Argolis GPU Hardware Profiles (No A100 Quota Required!):
#   Profile A (Single L4 24GB - Cloud Run GPU or g2-standard-8):
#     Model: nvidia/diffusiongemma-26B-A4B-it-NVFP4 (~13.5 GB VRAM)
#   Profile B (Dual L4 48GB - GCE g2-standard-24, TP=2):
#     Model: RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic (~26 GB VRAM)
#   Profile C (Quad L4 96GB - GCE g2-standard-48, TP=4 or 1x A100 80GB):
#     Model: google/diffusiongemma-26B-A4B-it (Full BF16, ~52 GB VRAM)
# ==============================================================================
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-jjuneja-fde-sandbox}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
ZONE="${GOOGLE_CLOUD_ZONE:-us-central1-a}"
PROFILE="${DJEV_GPU_PROFILE:-single-l4-fp4}" # single-l4-fp4 | dual-l4-fp8 | full-bf16

echo "[1/4] Configuring Argolis project: ${PROJECT_ID} (${ZONE}) for profile: ${PROFILE}"

if [[ "${PROFILE}" == "single-l4-fp4" ]]; then
  MACHINE_TYPE="g2-standard-8"
  GPU_COUNT=1
  MODEL_ID="nvidia/diffusiongemma-26B-A4B-it-NVFP4"
  TP_SIZE=1
elif [[ "${PROFILE}" == "dual-l4-fp8" ]]; then
  MACHINE_TYPE="g2-standard-24"
  GPU_COUNT=2
  MODEL_ID="RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
  TP_SIZE=2
else
  MACHINE_TYPE="g2-standard-48"
  GPU_COUNT=4
  MODEL_ID="google/diffusiongemma-26B-A4B-it"
  TP_SIZE=4
fi

cat <<EOF > /tmp/djev_vllm_startup.sh
#!/usr/bin/env bash
set -euo pipefail
apt-get update && apt-get install -y python3-pip git curl
pip3 install --upgrade uv
uv venv /opt/djev-venv
source /opt/djev-venv/bin/activate

# Install vLLM nightly/main containing merged PR #57250 + #57589 (multimodal DiffusionGemma)
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
git clone https://github.com/mmastrac/djev.git /opt/djev

# 1. Start upstream vLLM server with PR #57250 structured diffusion canvas config (Port 8000)
nohup vllm serve "${MODEL_ID}" \\
  --tensor-parallel-size "${TP_SIZE}" \\
  --diffusion-config '{"canvas_length": 64}' \\
  --max-logprobs 32 \\
  --enable-prefix-caching \\
  --port 8000 > /var/log/vllm_diffusiongemma.log 2>&1 &

# Wait for vLLM port 8000 readiness
for i in {1..90}; do
  if curl -s http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 5
done

# 2. Start PR #57250 / mmastrac/djev structured_server.py (/v1/systemone & /v1/chat/completions on Port 8011)
nohup python3 /opt/djev/structured_server.py \\
  --upstream http://127.0.0.1:8000 \\
  --tokenizer "${MODEL_ID}" \\
  --canvas 64 \\
  --port 8011 > /var/log/djev_structured_server.log 2>&1 &
EOF

echo "[2/4] Provisioning GCE L4 GPU VM (${MACHINE_TYPE}, ${GPU_COUNT}x nvidia-l4) in ${PROJECT_ID}..."
gcloud compute instances create djev-diffusiongemma-server \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=nvidia-l4,count=${GPU_COUNT}" \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata-from-file=startup-script=/tmp/djev_vllm_startup.sh \
  --tags=http-server

EXTERNAL_IP=$(gcloud compute instances describe djev-diffusiongemma-server \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "[3/4] DiffusionGemma-Jev (vLLM PR #57250) server launching at: http://${EXTERNAL_IP}:8011"
echo "[4/4] Export DJEV_ENDPOINT_URL to wire Stage 4.5 / 5a into live vLLM PR #57250:"
echo "      export DJEV_ENDPOINT_URL=\"http://${EXTERNAL_IP}:8011/v1/systemone\""
