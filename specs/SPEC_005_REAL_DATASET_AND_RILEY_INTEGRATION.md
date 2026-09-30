# SPEC-005: Real Open-Source Dataset Ingestion (`SKU-110k` + `Smart-Retail-Shelf-Auditing`) & Riley Migration Integration

## 1. System Intent & Ground-Truth Mandate
- **No Toy 17-Facing Slices:** Replace the 10-image synthetic slice and Riley's 17-facing single-image sample with **25 real, high-resolution open-source retail shelf images** (`20` from official `SKU-110k` CVPR19 `finedet/sku110k` validation split + `5` from `adnankhan-11/smart-retail-shelf-auditing-v1` validation split) containing **3,600+ real human-annotated product bounding boxes** (`~145 SKUs/image`).
- **Two Strict Splits (`Public` vs. `Private Holdout`):**
  - `public`: 13 real shelf images (~1,850 human-annotated product bounding boxes).
  - `private_holdout`: 12 real shelf images (~1,800 human-annotated product bounding boxes).
- **End-to-End Integration of Riley's Production Modules with `Track C` & `Track D`:**
  1. **Unilever 7-Dimension Taxonomy & Rule-Derived Size Buckets (`src/shelf_e2e/taxonomy.py` + `configs/unilever_taxonomy.json`)**
  2. **2nd-Row "Depth Ghost" Suppression (`deduplicate_depth_stacked_facings`) & Shelf Row Clustering (`src/shelf_e2e/geometry.py`)**
  3. **Margin-Gated Stage 1–3 Routing inside `Track C` (Tiered Hybrid) & `Track D` (`Jev` + `GeminiDiffusion-as-Jev`)**
  4. **"Run Now, Score Later" Deferred Ground-Truth Re-Scoring (`src/shelf_e2e/scoring.py`)**
  5. **5-Bucket Enterprise GCP Billing (Reserved GSU Provisioned Throughput vs. PAYG + Cloud Run L4 GPU + BigQuery Reconciliation SQL) (`src/shelf_e2e/pricing.py`)**
  6. **W3C OpenTelemetry (`otel_logs.jsonl`) & Vertex AI Supervised Fine-Tuning (SFT) JSONL Exporter (`src/shelf_e2e/telemetry_and_sft.py`)**

---

## 2. Real Open-Source Dataset Architecture (`data/sku110k/`)
1. **Source Provenance:**
   - `finedet/sku110k` (official `SKU110K_CVPR19` validation split, `2336x4160`, `2448x3264`, `3024x3024` px).
   - `adnankhan-11/smart-retail-shelf-auditing-v1` (`valid` split, `2448x3264`, `1920x2560` px).
2. **Coordinate Normalization & Depth-Row Tagging:**
   - Every human-annotated COCO `[x, y, w, h]` pixel bounding box is converted to normalized `[ymin, xmin, ymax, xmax]` in `[0, 1000]`.
   - Shelf rows (`top`, `upper_middle`, `middle`, `lower_middle`, `bottom`) are assigned via vertical y-centroid clustering (`cluster_boxes_into_shelf_rows`).
   - Boxes with high 1D horizontal overlap ($\ge 0.55$) and recessed top edges (`ymin` higher up the shelf with smaller area) are flagged as `is_back_row_depth_ghost` in the ground truth so `deduplicate_depth_stacked_facings` is quantitatively benchmarked on real shelf geometry.

---

## 3. Track C & Track D Integration with Depth-Ghost NMS & Margin Gating
- **Stage 1 (Class-Agnostic Detection + Depth-Ghost NMS):**
  - Detects all candidate product boxes on the full 4K shelf image and runs `deduplicate_depth_stacked_facings(x_overlap_threshold=0.55)` to eliminate 2nd-row depth ghosts before cropping or embedding.
- **Stage 2 (Physical Crop Extraction & `Jev` / `multimodalembedding@001` Vector Projection):**
  - Extracts pixel features from the downloaded JPEG/PNG shelf image crop (`[ymin, xmin, ymax, xmax]`) and projects into L2-normalized embedding space.
- **Stage 3 (Margin-Gated Cosine ANN + `GeminiDiffusion-as-Jev` / VLM Escalation):**
  - Computes Top-1 and Top-2 cosine similarity against the Unilever 7-Dimension Catalog (`configs/unilever_taxonomy.json` + `configs/mock_rpc_catalog.json`).
  - **Margin Gate:** Let $s_1 = \text{Top-1 cosine}$, $s_2 = \text{Top-2 cosine}$, and $\Delta = s_1 - s_2$:
    - **Fast Path ($s_1 \ge 0.85$ and $\Delta \ge 0.06$):** Resolved immediately in Tier 1/2 (`$0.0001` per image).
    - **Ambiguous / Glare Band ($0.62 \le s_1 < 0.85$ or $\Delta < 0.06$):**
      - In **`Track C`**: Escalates ambiguous crops to Tier 3 (`gemini-2.5-flash` / `gemini-3.8-flash` micro-crop grid VLM).
      - In **`Track D`**: Routes ambiguous crops through **`GeminiDiffusion-as-Jev`** (`OfflineGeminiDiffusionCropRefiner` / distilled diffusion latent denoiser) to boost SNR before re-querying Vector Search, only falling back to VLM when $s_1 < 0.62$.
  - **Rule-Derived Size & Taxonomy Resolution:** Applies `resolve_rule_derived_size_bucket()` and `normalize_brand_and_hul_flag()` from `src/shelf_e2e/taxonomy.py`.
