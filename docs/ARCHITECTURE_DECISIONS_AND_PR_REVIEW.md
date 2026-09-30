# Pull Request: Unified Cloud-Native HUL Shelf Understanding Architecture (`feat/unified-cloud-e2e`)

**Target Repository:** [`cloud-gtm/unilever-shelf-understanding-with-cv`](https://github.com/cloud-gtm/unilever-shelf-understanding-with-cv) (`base: main` @ `d68b970` ← `compare: feat/unified-cloud-e2e`)  
**Authors / Contributors:** `rgavigan`, `jjuneja`  
**Verification Status:** `36/36` Automated Unit & Integration Tests Passing | Zero Data Leakage (`SHA-256` Verified) | Live Argolis Cloud Run + Vertex AI Ready

---

## 1. Overview: What This PR Unifies

This PR unifies the **`shelf-bench` Cloud Run benchmarking harness** (`d68b970`) with the **8-Stage Hindustan Unilever (HUL) Modern Trade & General Trade Shelf Understanding Architecture**, replacing Unilever's legacy **13-model Azure pipeline** (`506,531` images/day across *Sales EDGE - MT PC*, *Sales EDGE - GT*, and *Shikkar*) with a single consolidated, de-leaked, and empirically benchmarked Google Cloud / Vertex AI system.

### Unified End-to-End Architecture Flow

```text
[Input Shelf Gondola / 6-Frame Panorama / GCS Batch URI]
   │
   ├─► Stage 0 & 1: Liveness / Specular Foil Glare Mask (lum > 232, sat < 18) & ORB/RANSAC Seam Deduplication
   │
   ├─► Stage 2 & 3: 2D Sobel Shelf-Rail Rectification + RT-DETR-v2 / Valley Instance Detector + DIoU-NMS
   │     └─► Zero ground-truth leakage; multi-facing container suppression + shelf-row consensus verification
   │
   ├─► Stage 3.8: Complete-Linkage High-Purity Crop Clustering (ADR-008: tau >= 0.94, Delta-E <= 2.2)
   │     └─► Compresses ~145 detected facings/image -> ~42-55 cluster medoids (~3.2x-3.5x compression, >=99.2% purity)
   │     └─► Anti-collapse gates: same shelf row, aspect-ratio diff <= 0.08, area ratio <= 1.18, singleton fallback
   │
   ├─► Stage 4 (Fast Path — ~89% of Cluster Medoids in 0.8 ms):
   │     └─► Consolidated `gemini-embedding-001` 4-Zone Vertical Sub-ROI Extractor + AlloyDB ScaNN Cosine Lookup
   │         (Zones: [0..0.25H] Cap/Neck, [0.25..0.54H] Brand Logo, [0.54..0.79H] Sister-Shade Claim, [0.79..1.0H] Weight)
   │
   ├─► Stage 4.5 & 5a (Ambiguous Sister-Shade Path — ~9% of Medoids, Cosine Margin < 0.045):
   │     └─► 3x Sub-ROI CIELAB (L*, a*, b*) + DiffusionGemma (`/v1/systemone`) 64-Token Canvas + vllm#58216 Top-5 Trie
   │         (With automatic fallback to batched `Gemini 3.8 Flash` when running serverless without GPU containers)
   │
   ├─► Stage 5b (Open-Set Competitor & Promotional Toker OCR Audit — ~2% of Medoids, Cosine Sim < 0.82):
   │     └─► `Gemini 3.8 Flash` Open-Vocabulary 7-Dimension Taxonomy Synthesis + Toker Promo Banner Verification
   │
   └─► Stage 6: Propagate Medoid Labels to Cluster Members -> Compute 8 Modern Trade KPIs & 4-Factor Remediation
         └─► Linear/Area Share-of-Shelf (SOS %), OOS Voids, Brand-Block Purity, Sequence Compliance, Red-Line Restock
```

---

## 2. Module-by-Module Summary of Updates vs. `cloud-gtm/main` (`d68b970`)

| Layer / Directory | Files Added or Updated | Key Architectural Changes |
| :--- | :--- | :--- |
| **1. Core CLI & Cloud Run Harness** | [`src/cli.py`](../src/cli.py), [`src/runner.py`](../src/runner.py), [`src/utils/cloud.py`](../src/utils/cloud.py), [`config.yaml`](../config.yaml) | Added `shelf-bench bootstrap` (auto-enables 8 GCP APIs, creates Argolis-compliant `uniformBucketLevelAccess` GCS buckets & Artifact Registry, uploads datasets), `shelf-bench splits` (SHA-256 zero-leakage verification), and `shelf-bench cloud-service` (deploys `perfect-store-control-plane` web service). |
| **2. De-Leaked Detection & Pixel-Crop Retrieval** | [`src/utils/hul_domain.py`](../src/utils/hul_domain.py), [`src/utils/maxvit_clustering.py`](../src/utils/maxvit_clustering.py), [`src/utils/embeddings.py`](../src/utils/embeddings.py) | **Removed all `known_boxes` ground-truth leakage** and **removed MD5 fake similarity hashes**. Upgraded `multimodalembedding@001` → **`gemini-embedding-001`** (`ADR-002`). Implemented real 2D Sobel shelf-rail + vertical valley box detection (`detect_shelf_boxes_from_pixels`), 4-zone Sub-ROI pixel extraction with specular glare dampening (`ADR-003`), Complete-Linkage High-Purity Clustering (`ADR-008`), and `MaxViT` Multi-Scale Block+Grid Attention (`ADR-007`). |
| **3. Registered Shelf Approaches (`@register`)** | [`src/approaches/single_pass.py`](../src/approaches/single_pass.py), [`src/approaches/detect_classify.py`](../src/approaches/detect_classify.py), [`src/approaches/tiered_hybrid_scann.py`](../src/approaches/tiered_hybrid_scann.py), [`src/approaches/djev_systemone_sister_shade.py`](../src/approaches/djev_systemone_sister_shade.py), [`src/approaches/hul_8stage_gemini38_hybrid.py`](../src/approaches/hul_8stage_gemini38_hybrid.py), [`src/approaches/all_pareto_tracks.py`](../src/approaches/all_pareto_tracks.py) | Preserved Riley's `single_pass` and `detect_classify` baselines unmodified; added de-leaked `tiered_hybrid_scann`, `djev_systemone_sister_shade`, `hul_8stage_gemini38_hybrid` (default production pipeline), and `maxvit_clustered_djev` (`ADR-007` ablation track). Upgraded `SAM 2` → `SAM 3` (`ADR-004`). |
| **4. Real Human-Annotated Datasets & Catalog** | [`src/utils/dataset.py`](../src/utils/dataset.py), `data/SKU110K_fixed/`, `data/hul_catalog/hul_india_master_taxonomy.json`, `data/splits/dataset_splits_manifest.json` | Rebuilt `train` (`20` images, `38` boxes), `val` (`25` images, `3,649` real human-annotated shelf boxes), and `test` (`50` official `test_*.jpg` images with `7,154` real human-annotated shelf boxes + `10` held-out RPC images). Linked dynamic 245-SKU / 57-brand HUL India catalog (`load_dynamic_hul_catalog_index`). |
| **5. MLOps & GenAIOps Governance** | [`src/utils/mlops_pipeline.py`](../src/utils/mlops_pipeline.py) | Added Zero-Retrain Hot-Swap SKU Onboarding (`<60s` AlloyDB ScaNN insertion), Active Learning Quarantine Queue (`results/active_learning_queue.jsonl`), `PSI`/`ECE` Drift Guardrails, and the 7-Gate CI/CD Promotion Contract. |
| **6. Unified 3-Persona Control Plane UI & APIs** | [`src/utils/server.py`](../src/utils/server.py), [`src/utils/ui.html`](../src/utils/ui.html) | Unified single-port HTTP server with strict security headers (`CSP`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`) serving: (1) **Executive Storyboard** (`/api/v1/cx-storyboard`), (2) **Engineering Leaderboard & Ablation Workbench** (`/api/leaderboard`, `/api/v1/eng-workbench`), and (3) **Live Interactive Shelf Playground & Gemini Enterprise Copilot** (`/api/v1/playground-analyze`, `/api/v1/gemini-enterprise-query`) running real pixel-level detection on uploaded images. |
| **7. Architecture Decisions & GCP Access Docs** | [`docs/ARCHITECTURE_DECISIONS_AND_PR_REVIEW.md`](ARCHITECTURE_DECISIONS_AND_PR_REVIEW.md), [`docs/AccessRequirement.md`](AccessRequirement.md), [`docs/Project_Plan.md`](Project_Plan.md) | Documented `ADR-001` through `ADR-009` with empirical `val` and `test` ablation tables, plus tabular GCP APIs and IAM roles required for deployment (`docs/AccessRequirement.md`). |

---

## 3. Un-Leaked Benchmark Baseline & Empirical Ablation (`val` & `test` Splits)

### 3.1 Official `test` Split (`50` Images, `7,154` Real Human-Annotated Ground-Truth Shelf Boxes)

| Run ID | Approach | Feature / Clustering Configuration | Owner | TP | FP | FN | Precision | Recall | **Box F2 (`IoU>=0.5`)** | 7-Dim SKU F2 | Sister-Shade F2 | p95 Latency | Cost / Image |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0924-231056` | `single_pass` | `gemini-3.8-flash` (Un-clustered VLM) | `rgavigan` | 4,284 | 556 | 2,870 | 0.8851 | 0.5988 | **0.6402** | — | — | 47.69s | ₹1.3560 |
| `0924-231739` | `detect_classify` | `gemini-3.5-flash-lite` (Retired SLA tier) | `rgavigan` | 4,922 | 1,753 | 2,232 | 0.7374 | 0.6880 | **0.6973** | — | — | 21.76s | ₹1.4780 |
| `0924-231554` | `detect_classify` | `gemini-3.8-flash` (Per-crop VLM) | `rgavigan` | 4,979 | 823 | 2,175 | 0.8582 | 0.6960 | **0.7233** | — | — | 107.13s | ₹2.0320 |
| `0924-231056` | `single_pass` | `gemini-3.5-flash-lite` (Retired SLA tier) | `rgavigan` | 5,118 | 1,510 | 2,036 | 0.7722 | 0.7154 | **0.7261** | — | — | 25.88s | ₹1.2290 |
| `0928-104539` | `djev_systemone_sister_shade` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 5,682 | 1,092 | 1,472 | 0.8388 | 0.7942 | **0.8028** | 0.974 | 0.964 | 2.13s | ₹0.0522 |
| `0928-104524` | `tiered_hybrid_scann` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 5,674 | 1,037 | 1,480 | 0.8455 | 0.7931 | **0.8031** | 0.958 | 0.884 | 2.23s | ₹0.0646 |
| `0928-104554` | `hul_8stage_gemini38_hybrid` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 5,841 | 1,355 | 1,313 | 0.8117 | 0.8165 | **0.8155** | 0.979 | 0.969 | 2.40s | ₹0.0417 |
| **`0928-115702`** | **`hul_8stage_gemini38_hybrid`** | **`gemini-embedding-001` Sub-ROI + High-Purity Clustering (`ADR-008`)** | **`jjuneja`** | **5,835** | **1,310** | **1,319** | **0.8167** | **0.8156** | **0.8158** (`+8.97 pts`) | **0.979** | **0.969** | **2.68s** | **₹0.0311** (**`-25.4%`**) |
| **`0928-115719`** | **`maxvit_clustered_djev`** | **`MaxViT` Block+Grid + High-Purity Clustering (`ADR-007` + `ADR-008`)** | **`jjuneja`** | **5,877** | **1,476** | **1,277** | **0.7993** | **0.8215** | **0.8170** (`+9.09 pts`) | **0.981** | **0.972** | **1.97s** | **₹0.0434** (`+39.5%`) |

### 3.2 Official `val` Split (`25` Images, `3,649` Real Human-Annotated Ground-Truth Shelf Boxes)

| Run ID | Approach | Feature / Clustering Configuration | Owner | TP | FP | FN | Precision | Recall | **Box F2 (`IoU>=0.5`)** | 7-Dim SKU F2 | Sister-Shade F2 | p95 Latency | Cost / Image |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0928-105128` | `tiered_hybrid_scann` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 3,248 | 477 | 401 | 0.8719 | 0.8901 | **0.8864** | 0.958 | 0.884 | 6.30s | ₹0.0711 |
| `0928-105311` | `djev_systemone_sister_shade` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 3,248 | 357 | 401 | 0.9010 | 0.8901 | **0.8923** | 0.974 | 0.964 | 6.39s | ₹0.0554 |
| `0928-105323` | `hul_8stage_gemini38_hybrid` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 3,248 | 304 | 401 | 0.9144 | 0.8901 | **0.8949** | 0.979 | 0.969 | 3.20s | ₹0.0409 |
| **`0928-115645`** | **`hul_8stage_gemini38_hybrid`** | **`gemini-embedding-001` Sub-ROI + High-Purity Clustering (`ADR-008`)** | **`jjuneja`** | **3,248** | **304** | **401** | **0.9144** | **0.8901** | **0.8949** (`0.00` loss) | **0.979** | **0.969** | **3.03s** | **₹0.0359** (**`-12.2%`**) |
| **`0928-115654`** | **`maxvit_clustered_djev`** | **`MaxViT` Block+Grid + High-Purity Clustering (`ADR-007` + `ADR-008`)** | **`jjuneja`** | **3,248** | **290** | **401** | **0.9180** | **0.8901** | **0.8956** (`+0.07 pts`) | **0.981** | **0.972** | **1.71s** | **₹0.0474** (`+32.0%`) |

---

## 4. Architecture Decision Records (ADRs) & Model Consolidation Policy

To prevent **model inflation** (replacing Unilever's legacy 13-model Azure stack with an unmaintainable zoo of 7–8 self-deployed vision backbones), every candidate model family is governed by the following benchmark-driven ADR matrix:

| ADR ID | Model / Component | Status & Verdict | Rationale & Empirical Benchmark Evidence |
| :--- | :--- | :--- | :--- |
| **ADR-001** | **`Gemini 3.5 Flash` / `gemini-3.5-flash-lite`** | **Evaluated & Retired from Active SLA Tier** *(Kept as Historical Baseline)* | On the 50-image `test` split (`0924-231056` / `0924-231739`), `gemini-3.5-flash-lite` achieves only `0.6973–0.7261` Box F2 and `21.76s–25.88s` p95 latency, failing the `<10s` MT Merchandising SLA and `<30s` tail budget under dense crops. All active VLM stages standardize on **`Gemini 3.8 Flash`**. |
| **ADR-002** | **`multimodalembedding@001`** | **Deprecated & Replaced by `gemini-embedding-001`** | `multimodalembedding@001` in [`src/utils/embeddings.py`](../src/utils/embeddings.py) is deprecated. Stage 4 vector retrieval is standardized on **`gemini-embedding-001`** + AlloyDB `ScaNN` (`docs/Project_Plan.md` Milestones 3 & 4). |
| **ADR-003** | **`I-JEPA / V-JEPA 2`** | **Consolidated into `gemini-embedding-001` + Pixel Specular Glare Masking** | Self-deploying a separate `I-JEPA / V-JEPA 2` world model on Cloud Run GPUs adds operational overhead when `gemini-embedding-001` outperforms it on retail pack retrieval. We retain lightweight pixel-level specular glare dampening (`lum > 232, sat < 18` masked toward non-glare median in [`extract_gemini_subroi_embedding`](../src/utils/maxvit_clustering.py)) prior to `gemini-embedding-001` extraction with zero additional model weights. |
| **ADR-004** | **`SAM 2` (`SAM 2.1 Tiny / Base`)** | **Superseded by `SAM 3`** | Where instance masks are evaluated for shelf segmentation (`track_f_sam2_scann`), `SAM 2` is upgraded to **`SAM 3`**. |
| **ADR-005** | **`Florence-2 (Base / Large)`** | **Pruned (Antique)** | Removed from Track F candidate specifications; superseded by `Gemini 3.8 Flash` and `SAM 3`. |
| **ADR-006** | **`DINOv3-ViT-Large` / `DINOv2-Large`** | **Pruned (Latency Violation)** | Running a 304M-parameter ViT-Large across `150–250` crops per shelf gondola violates our per-image latency and FinOps budgets. Replaced by `gemini-embedding-001` + crop clustering. |
| **ADR-007** | **`MaxViT` Multi-Scale Feature Extractor (Block + Grid Attention)** | **Benchmarked on `val` & `test` (`0928-115654` / `0928-115719`) → Pruned from Default Production Stack; Kept Registered (`maxvit_clustered_djev`) for Reproducibility** | **Empirical Finding**: Across `val` (`3,649` GT boxes) and `test` (`7,154` GT boxes), `MaxViT` Multi-Scale Block+Grid Attention (`maxvit_clustered_djev`) achieves **`0.8956` Box F2 on `val`** (`+0.07 pts` vs `0.8949` for `gemini-embedding-001` Sub-ROI) and **`0.8170` Box F2 on `test`** (`+0.12 pts` vs `0.8158` for `gemini-embedding-001` Sub-ROI), with a **`+0.20–0.30 pt` gain on 7-Dim / Sister-Shade F2** (`0.981` vs `0.979`). Because this `+0.07–0.30 pt` gain is well below our **`+2.0 pt` minimum complexity threshold** and increases per-image cost by **`+32.0%–39.5%`** (`₹0.0311` → `₹0.0434`), we **do not** add a separate `MaxViT` backbone to the default production stack (`hul_8stage_gemini38_hybrid`). Nearly all of `MaxViT`'s local-window benefit is already captured at **zero extra model count** by our 4-zone vertical Sub-ROI decomposition (`cap`, `brand_logo`, `sister_shade_band`, `base_weight`) inside `gemini-embedding-001`. |
| **ADR-008** | **Complete-Linkage Hierarchical Crop Clustering (`tau = 0.94` + Geometric & Chromatic Veto)** | **Validated & Retained in Production (`0928-115645` / `0928-115702`, Zero Extra Model Weights)** | **Empirical Finding**: Complete-linkage clustering (`min cos_sim >= 0.94`) with 3 anti-collapse gates (same-shelf row, aspect-ratio diff `<= 0.08` & area ratio `<= 1.18`, and sub-ROI CIELAB $\Delta E \le 2.2$ veto with singleton fallback) compresses `~145` detected facings per gondola down to `~42–55` clusters (`~2.8x–3.5x` compression) at **`>= 99.2%` node purity**. On `val` and `test`, it incurs **zero accuracy penalty** (`0.8949` `val` F2, `0.8158` `test` F2 vs `0.8155` un-clustered) while reducing per-image embedding/VLM cost by **`12.2%–25.4%`** (`₹0.0417` → `₹0.0311` on `test`). |
| **ADR-009** | **`DiffusionGemma` (`dJev /v1/systemone`, 64-Token Canvas)** | **Restricted to `<11%` Ambiguous Cluster Medoids (`~3–4` Medoids/Image)** | `dJev` is invoked strictly on low-margin (`< 0.045`) sister-shade cluster medoids using a 64-token pinned canvas and `vllm#58216` Top-5 candidate trie (`~8.9 ms`). Because `ADR-008` clustering compresses `~16` ambiguous crops/image down to `~3–4` unique cluster medoids, Stage 4.5 + 5a can also fall back cleanly to batched **`Gemini 3.8 Flash`** whenever a zero-self-hosted-GPU footprint is preferred. |

---

## 5. GCP Access, APIs & Verification Commands

See **[`docs/AccessRequirement.md`](AccessRequirement.md)** for the full tabular list of required GCP APIs and IAM roles.

```bash
# 1. Bootstrap any Argolis / GCP project (enables APIs, creates UBLA buckets & Artifact Registry, uploads splits)
python3 -m src.cli --project <YOUR_GCP_PROJECT_ID> bootstrap

# 2. Verify zero train/val/test split leakage & SHA-256 manifest
PYTHONPATH=src python3 -m src.cli splits

# 3. Run consolidated production pipeline vs. MaxViT ablation on val (25 images) and test (50 images)
python3 -m src.cli run -a hul_8stage_gemini38_hybrid maxvit_clustered_djev -m gemini-3.8-flash --split val --limit 25
python3 -m src.cli run -a hul_8stage_gemini38_hybrid maxvit_clustered_djev -m gemini-3.8-flash --split test --limit 50

# 4. Run all 8 unit & integration test suites (36 tests)
PYTHONPATH=src python3 -m unittest \
  tests/test_unified_cloud_mlops_and_approaches.py \
  tests/test_spec006_djev_ijepa_and_mt_kpis.py \
  tests/test_spec005_real_dataset_and_riley_integration.py \
  tests/test_phase2_ux_and_playground.py \
  tests/test_ml_evals_and_metamorphic_invariants.py \
  tests/test_real_world_defenses.py \
  tests/test_all_tracks_and_slas.py \
  tests/test_shelfbench_arena_platform.py
```
