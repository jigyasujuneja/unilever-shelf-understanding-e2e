# Unilever Perfect Store: CLEF Contrastive Contact-Sheet Cascade (`SPEC v0.2`) — Partner Quick-Start Guide

Welcome to the **CLEF Contrastive Contact-Sheet Cascade API (`SPEC v0.2`)**. This guide provides copy-pasteable examples in `curl` and Python to identify products in **individual crop images** or detect and recognize all products in **full retail shelf photos**.

---

## 1. Overview & Deployed Cloud Run Endpoints

* **Architecture**: **CLEF Contrastive Contact-Sheet Cascade** (`clef_contrastive_metadata` + `distilled_yolo_onnx` + `gemini-3.5-flash-lite`) — #1 Production Market Share & SKU Recognition Pipeline
* **Vision Model**: `gemini-3.5-flash-lite` (`tier: priority`)
* **Vector Embeddings**: `gemini-embedding-2` (`768-D` MRL / `3,072-D` full, zero-retraining catalog)
* **Shelf Detector**: `distilled_yolo_onnx` (CPU ONNX Stage 1 detector in ~40ms)
* **Master Catalog**: **2,659 SKUs** from `hul_golden` (Unilever + Competitor brands)
* **v0.2 Calibration & Abstention Contract**:
  * **Confidence Calibration**: Lidstone open-set smoothing + dustbin/cosine scaling in `[0.02, 0.985]` (never saturates to `1.0`).
  * **Sibling-Variant Margin Guard**: Enforces $\cos_{\text{top1}} - \cos_{\text{top2}} \ge 0.035$ when top candidates share the same brand before bypassing Stage 2B.
  * **Layer-5 `pack_size` Verifier**: Deterministically verifies extracted net-weight text against same-brand sister SKUs.
  * **Unified Abstention Threshold (`0.78`)**: Every response reports `abstain_recommended`, `abstained`, `abstention_reason`, and `conf_threshold` (default `0.78`). Pass `"allow_abstain": true` to emit `sku_id: null` when confidence falls below `conf_threshold`.

### Service Endpoint

Deployed in `us-central1` on Google Cloud Run (`ul-fs-s-sandbx-16-prj`), protected by GCP Project IAM (`roles/run.invoker` / `roles/owner`):

| Service | Base URL | Routes |
| :--- | :--- | :--- |
| **`clef-contrastive-api`** | `https://clef-contrastive-api-jbn2qgbina-uc.a.run.app` | `POST /api/predict/crop`<br>`POST /api/predict/shelf`<br>`GET /api/predict/health` |

---

## 2. Authentication

Before calling the endpoints, obtain a Google Cloud Identity Token using `gcloud`:

```bash
export TOKEN=$(gcloud auth print-identity-token)
export BASE_URL="https://clef-contrastive-api-jbn2qgbina-uc.a.run.app"
```

Verify service health:
```bash
curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/predict/health"
```

Expected health response:
```json
{
  "status": "healthy",
  "spec_version": "v0.2",
  "pipeline": "clef_contrastive_contact_sheet",
  "approach": "clef_contrastive_metadata",
  "model": "gemini-3.5-flash-lite",
  "tier": "priority",
  "embedding_model": "gemini-embedding-2",
  "shelf_detector": "distilled_yolo_onnx",
  "dataset": "hul_golden",
  "catalog_classes": 2659,
  "default_conf_threshold": 0.78,
  "sibling_margin_guard": 0.035,
  "layer5_pack_size_verifier": true,
  "confidence_range": [0.02, 0.985],
  "is_ready": true,
  "routes": {
    "crop": "/api/predict/crop",
    "shelf": "/api/predict/shelf",
    "health": "/api/predict/health"
  }
}
```

---

## 3. Entrypoint 1: Individual Crop Image (`POST /api/predict/crop`)

Use this endpoint when you already have cropped product images. It runs **Stage 2A (CLM Fast Gate)** for instant matching (~68% 0-VLM early exit) and escalates ambiguous packs to **Stage 2B (CLEF Zoom-Strip Contact Sheet + Layer-5 Verifier)**.

### A. Via cURL (Using GCS URI)
```bash
curl -X POST "$BASE_URL/api/predict/crop" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_uri": "gs://marketshare-datasets/derived_manifests/hul_golden/images/Dove_Daily_Shine_Sample.jpg",
    "model": "gemini-3.5-flash-lite",
    "tier": "priority",
    "allow_abstain": false,
    "conf_threshold": 0.78
  }'
```

### B. 10-Line Python Snippet (Using Local File or GCS URI)
```python
import base64, requests, subprocess

token = subprocess.check_output(["gcloud", "auth", "print-identity-token"], text=True).strip()
url = "https://clef-contrastive-api-jbn2qgbina-uc.a.run.app/api/predict/crop"

# Option 1: Send GCS URI (set allow_abstain=True to abstain below 0.78 confidence)
payload = {
    "image_uri": "gs://marketshare-datasets/derived_manifests/hul_golden/images/sample.jpg",
    "allow_abstain": True,
    "conf_threshold": 0.78,
}

# Option 2: Send local image via Base64
# with open("my_crop.jpg", "rb") as f:
#     payload = {"image_b64": base64.b64encode(f.read()).decode("ascii"), "allow_abstain": True}

resp = requests.post(url, json=payload, headers={"Authorization": f"Bearer {token}"})
print(resp.json())
```

### Sample JSON Output:
```json
{
  "pipeline": "clef_contrastive_contact_sheet",
  "approach": "clef_contrastive_metadata",
  "model": "gemini-3.5-flash-lite",
  "tier": "priority",
  "mode": "crop",
  "image_uri": "gs://marketshare-datasets/.../sample.jpg",
  "width": 180,
  "height": 360,
  "sku_id": 42,
  "prediction": {
    "sku_id": 42,
    "is_hul": true,
    "category": "Hair_Care",
    "packaging_type": "Bottle",
    "brand": "Dove",
    "base_variant": "Dove__Daily_Shine",
    "product": "Dove Daily Shine Shampoo",
    "pack_size": "180 ml"
  },
  "stage": "stage2a_fast_gate",
  "guard_mode": "stage2a_fast_gate",
  "confidence": 0.942,
  "sku_confidence": 0.912,
  "brand_confidence": 0.948,
  "conf_threshold": 0.78,
  "abstain_recommended": false,
  "abstained": false,
  "abstention_reason": null,
  "top1_cosine": 0.887,
  "embedding_output": {
    "model": "gemini-embedding-2-preview",
    "fast_gate_passed": true,
    "escalation_reason": null,
    "top1_sku_id": 42,
    "top1_sku_name": "Dove__Daily_Shine_Shampoo",
    "top1_cosine": 0.887,
    "top2_cosine": 0.761,
    "cosine_margin": 0.126,
    "required_margin": 0.009,
    "z_margin": 1.842,
    "lid": 8.91,
    "sku_confidence": 0.912,
    "brand_confidence": 0.948
  },
  "gemini_output": {
    "called": false,
    "model": null,
    "panel_choice": null,
    "sku_id": null,
    "sku_name": null,
    "agreed_with_embedding_top1": null
  },
  "shortlist_top5": [
    {"sku_id": 42, "cosine": 0.887},
    {"sku_id": 99, "cosine": 0.761}
  ],
  "latency_s": 0.125
}
```

---

## 4. Entrypoint 2: Full Shelf Image (`POST /api/predict/shelf`)

Use this endpoint for full uncropped store photos (General Trade Kirana or Modern Trade). It automatically executes:
1. **Stage 1 CPU ONNX Detector**: Locates all product facings and filters out promotional bays / headers.
2. **Stage 2 Recognition Cascade**: Recognizes every product pack, computes Share of Shelf (SOS), and aggregates brand / category counts.

### A. Via cURL (Using GCS URI)
```bash
curl -X POST "$BASE_URL/api/predict/shelf" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_uri": "gs://marketshare-datasets/derived_manifests/gtmerch_golden/images/Store_Shelf_Sample.jpg",
    "model": "gemini-3.5-flash-lite",
    "tier": "priority"
  }'
```

---

## 5. CLI Usage & Cloud Run Execution (From Repo Root)

```bash
# Option A: Activate the virtual environment once in your terminal
source .venv/bin/activate

# Option B: Or call .venv/bin/shelf-bench directly:
# Predict a single crop (executed 100% on Cloud Run without downloading images)
.venv/bin/shelf-bench predict-crop gs://marketshare-datasets/.../crop.jpg --cloud-run

# Predict a full shelf (executed 100% on Cloud Run without downloading images)
.venv/bin/shelf-bench predict-shelf gs://marketshare-datasets/.../shelf.jpg --cloud-run

# Run the 5-crop + 5-shelf verification smoke test on Cloud Run
.venv/bin/shelf-bench smoke-test-clef --crops 5 --shelves 5 --cloud-run
```

---

## 6. Running the Full Benchmark Suite

To reproduce the 5,000-sample `hul_golden` benchmark evaluation (`test_5k`) for `clef_contrastive_metadata`:

```bash
.venv/bin/shelf-bench run \
  -a clef_contrastive_metadata \
  -m gemini-3.5-flash-lite \
  -t priority \
  -d hul_golden \
  --split test_5k \
  --limit 5000 \
  --workers 16
```
*(Pass `--limit 50` for a quick 50-sample evaluation, or replace `run` with `cloud-run` to execute as a Cloud Run Job.)*
