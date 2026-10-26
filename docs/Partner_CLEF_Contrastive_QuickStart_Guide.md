# Unilever Perfect Store: CLEF Contrastive Contact-Sheet Cascade — Partner Quick-Start Guide

Welcome to the **CLEF Contrastive Contact-Sheet Cascade API**. This guide provides copy-pasteable examples in `curl` and Python to identify products in **individual crop images** or detect and recognize all products in **full retail shelf photos**.

---

## 1. Overview & Deployed Cloud Run Endpoints

* **Architecture**: **CLEF Contrastive Contact-Sheet Cascade** (`clef_contrastive_metadata` + `distilled_yolo_onnx` + `gemini-3.5-flash-lite`) — #1 Production Market Share & SKU Recognition Pipeline
* **Vision Model**: `gemini-3.5-flash-lite` (`tier: priority`)
* **Vector Embeddings**: `gemini-embedding-2` (`3,072-D`, zero-retraining catalog)
* **Shelf Detector**: `distilled_yolo_onnx` (CPU ONNX Stage 1 detector in ~40ms)
* **Master Catalog**: **2,659 SKUs** from `hul_golden` (Unilever + Competitor brands)

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
  "pipeline": "clef_contrastive_contact_sheet",
  "approach": "clef_contrastive_metadata",
  "model": "gemini-3.5-flash-lite",
  "tier": "priority",
  "embedding_model": "gemini-embedding-2",
  "shelf_detector": "distilled_yolo_onnx",
  "dataset": "hul_golden",
  "catalog_classes": 2659,
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

Use this endpoint when you already have cropped product images. It runs **Stage 2A (CLM Fast Gate)** for instant matching (~68% 0-VLM early exit) and escalates ambiguous packs to **Stage 2B (CLEF Zoom-Strip Contact Sheet)**.

### A. Via cURL (Using GCS URI)
```bash
curl -X POST "$BASE_URL/api/predict/crop" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_uri": "gs://marketshare-datasets/derived_manifests/hul_golden/images/Dove_Daily_Shine_Sample.jpg",
    "model": "gemini-3.5-flash-lite",
    "tier": "priority"
  }'
```

### B. 10-Line Python Snippet (Using Local File or GCS URI)
```python
import base64, requests, subprocess

token = subprocess.check_output(["gcloud", "auth", "print-identity-token"], text=True).strip()
url = "https://clef-contrastive-api-jbn2qgbina-uc.a.run.app/api/predict/crop"

# Option 1: Send GCS URI
payload = {"image_uri": "gs://marketshare-datasets/derived_manifests/hul_golden/images/sample.jpg"}

# Option 2: Send local image via Base64
# with open("my_crop.jpg", "rb") as f:
#     payload = {"image_b64": base64.b64encode(f.read()).decode("ascii")}

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
  "confidence": 0.942,
  "top1_cosine": 0.887,
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

### B. 10-Line Python Snippet (Using Local File or GCS URI)
```python
import base64, requests, subprocess

token = subprocess.check_output(["gcloud", "auth", "print-identity-token"], text=True).strip()
url = "https://clef-contrastive-api-jbn2qgbina-uc.a.run.app/api/predict/shelf"

# Option 1: Send GCS URI
payload = {"image_uri": "gs://marketshare-datasets/derived_manifests/gtmerch_golden/images/shelf.jpg"}

# Option 2: Send local image via Base64
# with open("my_shelf.jpg", "rb") as f:
#     payload = {"image_b64": base64.b64encode(f.read()).decode("ascii")}

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
  "mode": "shelf",
  "image_uri": "gs://.../shelf.jpg",
  "width": 1920,
  "height": 1080,
  "total_facings": 28,
  "hul_facings": 19,
  "competitor_facings": 9,
  "hul_facing_sos": 0.6786,
  "hul_area_sos": 0.7120,
  "stage2a_fast_gate_count": 18,
  "stage2b_vlm_count": 10,
  "brand_counts": {
    "Dove": 12,
    "Sunsilk": 7,
    "Pantene": 5,
    "Head & Shoulders": 4
  },
  "category_counts": {
    "Hair_Care": 28
  },
  "detections": [
    {
      "box_index": 0,
      "box": [112.0, 145.0, 198.0, 340.0],
      "sku_id": 42,
      "prediction": {
        "sku_id": 42,
        "is_hul": true,
        "brand": "Dove",
        "category": "Hair_Care",
        "packaging_type": "Bottle",
        "base_variant": "Dove__Daily_Shine",
        "pack_size": "180 ml"
      },
      "stage": "stage2a_fast_gate",
      "confidence": 0.94
    }
  ],
  "latency_s": 2.14
}
```

---

## 5. CLI Usage & Cloud Run Execution (From Repo Root)

If you have cloned the repository, `shelf-bench` is installed inside `.venv/bin/shelf-bench` (run `make venv` first if `.venv` does not exist yet). Either prefix commands with `.venv/bin/shelf-bench` or activate the virtual environment (`source .venv/bin/activate`):

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
