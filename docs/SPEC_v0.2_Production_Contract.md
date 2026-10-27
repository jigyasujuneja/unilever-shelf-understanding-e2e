# Unilever Shelf Understanding v0.2: Formal Production Specification & Contract

**Project:** `ul-fs-s-sandbx-16-prj` (`us-central1`)  
**Canonical Data & Results Bucket:** `gs://ul-fs-s-sandbx-16-prj-shelf-images`  
**Pipeline Version:** `v0.2`


## 1. Upstream & Downstream Production Contract (`PR05` -> `PR19` -> `PR06-PR08` -> `PR09-PR11`)

1. **Ingress (`PR05` Backend -> `PR19` Cloud Pub/Sub):**
   - Store photos uploaded by field reps (`Sales EDGE`, `Shikhar`, `MTPC`) are written to GCS and published to `PR19` Cloud Pub/Sub with store metadata (`outlet_id`, `session_id`, `channel`, `template_code`).
2. **Inference Workers (`PR06-PR08` Cloud Run / Vertex AI):**
   - **Option 1 (`option_1_fast_clm`, Kirana / Volume Spillover):** Stage 0 Dual-Buffer Photometry (`Shades-of-Gray` + `L* CLAHE`) + Stage 1 960px ONNX Detection (`Gaussian Soft-NMS`) + Spatial Neighbor Medoid Grouping + Stage 2A 2,659-SKU Vector Catalog Match (`CLM` zero-generative-VLM, `₹0.17/shelf`, `0.95s` latency).
   - **Option 2 (`option_2_hybrid_clef`, Primary Production Default):** Stage 0 + Stage 1 + 60% Stage 2A Vector Fast-Gate + 40% 8-Crop Batched Zoom Sheet (`CLEF` with Bottom-30% Weight Strip, Self-Supervised Saliency-Diff Strip, and Top-22% Crown Strip) + Layer-5 Deterministic Pack-Size Sister-SKU Verifier (`₹0.21-₹0.31/shelf`, `2.08s-2.29s` latency).
   - **Option 3 (`option_3_custom_sft`, Custom LoRA Endpoint):** Vertex AI SFT Gemini 2.5 Flash LoRA endpoint (`projects/536522937000/locations/us-central1/endpoints/3291944946497421312`).
   - **Merchandising Bay-Scoped E2E (`merchandising_bay_e2e`):** Dual-Class Bay/Pack Detection + Stage 1.5 Pre-Embedding Class-0 Bay Gate (`filter_preds_to_asset_bays`, pruning `-61%` of background store shelf crops before embedding) + Stage 6 Planogram Purity & OOS Compliance.
3. **Level-4 Total Contract Payout Quarantine Gate (`PT180S` SLA):**
   - Any store photo exceeding the `PT180S` processing window or encountering upstream transient exhaustion MUST emit `contract_status = "RETRY_PENDING"`, `awp_presence = "PENDING"`, and `earning_override = None` to `PR19_pubsub_dlq`.
   - Workers MUST NEVER emit `Earning = '0'` on a timeout or infrastructure error, preventing false retailer payout denials in `PR09-PR11`.


## 2. Canonical 5-Level Taxonomy & Schema Invariants

Every prediction and ground-truth label across all 10 canonical datasets (`hul_golden`, `hul_brands`, `hul_variants`, `hul_packs`, `hul_promos`, `hul_sessions`, `gt_hul_sessions`, `hul_talkers_golden`, `gtmerch_golden`, `ms_json_golden`, `mt_json_golden`) adheres to the 5-level hierarchy:

- **L1 (`is_hul`):** Boolean Unilever vs. Competitor ownership (`True` / `False`).
- **L2 (`category`, `packaging_type`):** 14 canonical FMCG categories (`CANONICAL_CATEGORIES`) and 16 canonical physical container forms (`CANONICAL_PACKAGING_TYPES`). Zero `"Unknown"` or `"Unbranded"` strings permitted.
- **L3 (`brand`):** 934 canonical brand families (`HUL_BRAND_REGISTRY` + verified competitor brands).
- **L4 (`base_variant`):** Canonical variant family stripped of pack-size and container suffixes.
- **L5 (`product` / `sku_id`, `promo_type`):** Exact 2,659-SKU catalog identity (`1..2659` in `data/hul_golden/rpc.json`, verified via `catalog_sha256`).


## 3. Zero-Leakage, Zero-Fallback, and Zero-Error Production Gates (15 Hard QC Gates)

All runs and datasets are continuously audited by `run_12_gate_qc()` in [`src/utils/qc.py`](../src/utils/qc.py) (15 automated gates):

1. `catalog_label_parity`: 100% of split labels match canonical catalog metadata.
2. `zero_burned_in_overlays`: 0 green/cyan OpenCV debug rectangles in evaluation images.
3. `split_hash_disjointness`: 0 URI overlap across `train` (`gallery`), `val`, and `test`.
4. `stratified_50_50_balance`: 50/50 HUL vs. Non-HUL balance on `hul_golden`.
5. `zero_unknown_taxonomy`: 0 `"Unknown"` categories or packaging types across all manifests.
6. `shortlist_oracle_deleakage`: 0 ground-truth oracle injection into candidate shortlists.
7. `gtmerch_brand_and_bay_integrity`: Validated display bay (`asset_boxes`) and brand labels on `gtmerch_golden`.
8. `results_taxonomy_parity`: All `results/*/summary.json` and `images.jsonl` match canonical taxonomy.
9. `five_attribute_crop_coverage`: All 5 attributes evaluated on every crop run.
10. `closed_set_non_abstention`: Closed-set tasks never abstain (`choice=0` resolved via Tier-1/Tier-2 guardrails).
11. `e2e_dual_lens_eval_parity`: End-to-End runs report both Full-Catalog (`2,659` SKUs) and Annotated-Row Box Metrics.
12. `detector_dedup_iou_audit`: Stage 1 NMS suppresses duplicate overlapping boxes (`IoU > 0.55`).
13. `outlet_and_md5_split_deleakage`: Zero store `outlet_code` or image MD5 overlap across `train`/`val`/`test`.
14. `dual_buffer_photometry_integrity`: Stage 0 preserves linear RGB (`color_image`) for Stage 2 CIELAB while routing `L* CLAHE` (`detector_image`) to Stage 1 detection.
15. `zero_synthetic_fallbacks_and_zero_official_errors`: Zero synthetic/dummy fallbacks in [`src/utils/server.py`](../src/utils/server.py) or [`src/utils/hul_builder.py`](../src/utils/hul_builder.py), zero dead 100%-errored runs, and strictly `errors == 0` required for official leaderboard ranking in [`src/runner.py`](../src/runner.py).
