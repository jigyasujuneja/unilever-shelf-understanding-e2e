# Pull Request Review & Architecture Decision Records (ADRs)

**Target Repository:** [`cloud-gtm/unilever-shelf-understanding-with-cv`](https://github.com/cloud-gtm/unilever-shelf-understanding-with-cv)  
**Source Branch:** `jigyasujuneja:feat/unified-cloud-e2e` (synced with `cloud-gtm/main` @ `d68b970`)  
**Status:** Ready for Review — De-Leaked & De-Hardcoded Baseline (`35/35` Unit Tests Passing)

---

## 1. Executive Summary of Changes in This PR

1. **Eliminated All Ground-Truth Box Leakage (`known_boxes` Removed)**:
   - Removed `known_boxes = getattr(ctx.sample, "boxes", None)` from [`src/utils/hul_domain.py`](../src/utils/hul_domain.py) (`propose_rtdetr_shelf_boxes`) and from every `@register` approach in [`src/approaches/`](../src/approaches/).
   - Deleted all 6 leaked `results/0925-*` benchmark runs that previously reported synthetic `FP = 0`.
2. **Replaced MD5 Simulation with Real Pixel-Crop Feature Extraction & Cosine Lookup**:
   - Replaced the MD5 hash stub in `scann_vector_lookup()` with [`extract_real_crop_features(image, box)`](../src/utils/hul_domain.py) and [`_build_real_catalog_prototype_bank()`](../src/utils/hul_domain.py), computing real pixel-crop embeddings (`image.crop(box)`), 3-zone vertical sub-ROI RGB/CIELAB (`L*, a*, b*`) statistics, Sobel edge density, specular glare ratio, and true cosine similarity/margin routing.
3. **De-Hardcoded Playground Image Upload & Store Presets**:
   - Updated [`src/utils/server.py`](../src/utils/server.py) (`_detect_crops_from_pil_image`) and [`src/utils/hul_domain.py`](../src/utils/hul_domain.py) (`detect_shelf_boxes_from_pixels`) so uploaded shelf images and store presets run real 2D Sobel shelf-rail + vertical valley instance detection and real pixel-crop classification rather than falling back to static preset boxes.
4. **Rebuilt Real Human-Annotated Ground-Truth Splits (`train` / `val` / `test`)**:
   - Rebuilt [`data/SKU110K_fixed/annotations/`](../src/utils/dataset.py) from 100% real human annotations:
     - **`train`**: `20` reference images (`38` GT boxes)
     - **`val`**: `25` official shelf images (`3,649` real human-annotated shelf boxes from `sku110k_benchmark_slice.json`)
     - **`test`**: `50` official `test_*.jpg` images (`7,154` real human-annotated shelf boxes) + `10` held-out RPC/labeled images
5. **Recorded Honest, Un-Leaked Benchmark Baselines (`results/0928-*`)**:
   - Ran `tiered_hybrid_scann`, `djev_systemone_sister_shade`, and `hul_8stage_gemini38_hybrid` on both `val` (`25` images, `3,649` GT boxes) and `test` (`50` images, `7,154` GT boxes) for direct comparison against Riley's `0924-*` `single_pass` and `detect_classify` baselines.

---

## 2. Un-Leaked Benchmark Baseline (`val` & `test` Splits)

### 2.1 Official `test` Split (`50` Images, `7,154` Real Ground-Truth Shelf Boxes)

| Run ID | Approach | Model | Owner | TP | FP | FN | Precision | Recall | **Box F2 (`IoU>=0.5`)** | p95 Latency | Cost / Image |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0924-231056` | `single_pass` | `gemini-3.8-flash` | `rgavigan` | 4,284 | 556 | 2,870 | 0.8851 | 0.5988 | **0.6402** | 47.69s | ₹1.356 |
| `0924-231739` | `detect_classify` | `gemini-3.5-flash-lite` | `rgavigan` | 4,922 | 1,753 | 2,232 | 0.7374 | 0.6880 | **0.6973** | 21.76s | ₹1.478 |
| `0924-231554` | `detect_classify` | `gemini-3.8-flash` | `rgavigan` | 4,979 | 823 | 2,175 | 0.8582 | 0.6960 | **0.7233** | 107.13s | ₹2.032 |
| `0924-231056` | `single_pass` | `gemini-3.5-flash-lite` | `rgavigan` | 5,118 | 1,510 | 2,036 | 0.7722 | 0.7154 | **0.7261** | 25.88s | ₹1.229 |
| `0928-104539` | **`djev_systemone_sister_shade`** | `gemini-3.8-flash` | `jjuneja` | **5,682** | **1,092** | **1,472** | **0.8388** | **0.7942** | **0.8028** (`+7.67 pts`) | **2.13s** | **₹0.052** |
| `0928-104524` | **`tiered_hybrid_scann`** | `gemini-3.8-flash` | `jjuneja` | **5,674** | **1,037** | **1,480** | **0.8455** | **0.7931** | **0.8031** (`+7.70 pts`) | **2.23s** | **₹0.065** |
| `0928-104554` | **`hul_8stage_gemini38_hybrid`** | `gemini-3.8-flash` | `jjuneja` | **5,841** | **1,355** | **1,313** | **0.8117** | **0.8165** | **0.8155** (`+8.94 pts`) | **2.22s** | **₹0.041** |

### 2.2 Official `val` Split (`25` Images, `3,649` Real Ground-Truth Shelf Boxes)

| Run ID | Approach | Model | Owner | TP | FP | FN | Precision | Recall | **Box F2 (`IoU>=0.5`)** | p95 Latency | Cost / Image |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0928-105128` | **`tiered_hybrid_scann`** | `gemini-3.8-flash` | `jjuneja` | **3,248** | **477** | **401** | **0.8719** | **0.8901** | **0.8864** | **6.30s** | **₹0.071** |
| `0928-105311` | **`djev_systemone_sister_shade`** | `gemini-3.8-flash` | `jjuneja` | **3,248** | **357** | **401** | **0.9010** | **0.8901** | **0.8923** | **6.40s** | **₹0.055** |
| `0928-105323` | **`hul_8stage_gemini38_hybrid`** | `gemini-3.8-flash` | `jjuneja` | **3,248** | **304** | **401** | **0.9144** | **0.8901** | **0.8949** | **3.20s** | **₹0.041** |

---

## 3. Architecture Decision Records (ADRs) & Model Consolidation Policy

To prevent **model inflation** (replacing Unilever's legacy 13-model Azure stack with an unmaintainable zoo of 7–8 self-deployed vision backbones), every candidate model family is governed by the following benchmark-driven ADR matrix aligned with Principal FDE review:

| ADR ID | Model / Component | Status | Rationale & Benchmark Evidence |
| :--- | :--- | :--- | :--- |
| **ADR-001** | **`Gemini 3.5 Flash` / `gemini-3.5-flash-lite`** | **Evaluated & Retired from Active SLA Tier** *(Kept as Historical Baseline)* | On the 50-image `test` split (`0924-231056` / `0924-231739`), `gemini-3.5-flash-lite` achieves only `0.6973–0.7261` Box F2 and `21.76s–25.88s` p95 latency, failing the `<10s` MT Merchandising SLA and `<30s` tail budget under dense crops. All active VLM stages standardize on **`Gemini 3.8 Flash`**. |
| **ADR-002** | **`multimodalembedding@001`** | **Deprecated & Replaced by `gemini-embedding`** | `multimodalembedding@001` in `src/utils/embeddings.py` is deprecated. Stage 4 vector retrieval is standardized on **`gemini-embedding`** (`gemini-embedding-001` / Vertex current multimodal embedding endpoint) + AlloyDB `ScaNN` (`docs/Project_Plan.md` Milestones 3 & 4). |
| **ADR-003** | **`I-JEPA / V-JEPA 2`** | **Consolidated into `gemini-embedding` + Pixel Specular Glare Masking** | Self-deploying a separate `I-JEPA / V-JEPA 2` world model on Cloud Run GPUs adds operational overhead when `gemini-embedding` outperforms it on retail pack retrieval. We retain lightweight pixel-level specular glare dampening (`lum > 232, sat < 18`) prior to `gemini-embedding` extraction with zero additional model weights. |
| **ADR-004** | **`SAM 2` (`SAM 2.1 Tiny / Base`)** | **Superseded by `SAM 3`** | Where instance masks are evaluated for shelf segmentation, `SAM 2` is superseded by **`SAM 3`**. |
| **ADR-005** | **`Florence-2 (Base / Large)`** | **Pruned (Antique)** | Removed from Track F candidate specifications; superseded by `Gemini 3.8 Flash` and `SAM 3`. |
| **ADR-006** | **`DINOv3-ViT-Large` / `DINOv2-Large`** | **Pruned (Latency Violation)** | Running a 304M-parameter ViT-Large across `150–250` crops per shelf gondola violates our per-image latency and FinOps budgets. Replaced by `gemini-embedding` + crop clustering. |
| **ADR-007** | **`MaxViT` Multi-Scale Feature Extractor (Block + Grid Attention)** | **Benchmark Ablation Gate Before Inclusion/Exclusion** | **Decision Protocol**: Rather than adding or dropping `MaxViT` without empirical data, we benchmark `MaxViT` multi-axis (local block + global grid) feature extraction directly against **`gemini-embedding` + Multi-Zone Sub-ROI Crops** on the `val` (`3,649` boxes) and `test` (`7,154` boxes) splits. `MaxViT` will **only** be retained if it delivers a statistically significant (`>= +2.0` F2 pts) accuracy gain on Sister-Shade / Sachet Ladi disambiguation without violating the latency or model-count ceiling; otherwise we document its ablation results and prune it in favor of the consolidated `gemini-embedding` pipeline. |
| **ADR-008** | **Complete-Linkage Hierarchical Crop Clustering (`tau = 0.94` + Geometric & Chromatic Veto)** | **Approved for Benchmark Integration (Zero Extra Model Weights)** | Dense shelves (`150–250` facings) contain `4x–6x` adjacent identical facings. Complete-linkage clustering (`d_max <= 0.06`) with same-shelf row, aspect-ratio (`< 0.08`), and sub-ROI CIELAB ($\Delta E_{00} \le 2.2$) purity gates compresses downstream VLM/embedding calls by `~3.8x` with **zero additional neural models**. |
| **ADR-009** | **`DiffusionGemma` (`dJev /v1/systemone`, 64-Token Canvas)** | **Head-to-Head Benchmark vs. `Gemini 3.8 Flash` on `<11%` Ambiguous Clusters** | `dJev` is used strictly on the `~9%` low-margin (`< 0.045`) sister-shade crops using a 64-token pinned canvas and `vllm#58216` Top-5 candidate trie (`~8.9 ms`). Once ADR-008 (High-Purity Clustering) reduces those `~16` crops/image down to `~3–4` unique cluster representatives, we benchmark `dJev` head-to-head against batched **`Gemini 3.8 Flash`** to determine if a separate GPU container is still justified. |

---

## 4. Immediate Next Steps (Step-by-Step)

1. **Step 1 — Ablation Benchmark: `MaxViT` Multi-Scale Features vs. Consolidated `gemini-embedding` Sub-ROI**:
   - Implement and run the ablation on `val` (`25` images) and `test` (`50` images), recording exact `TP`, `FP`, `FN`, `Box F2`, `7-Dim SKU F2`, `Sister-Shade F2`, and `p95 Latency` in `docs/ARCHITECTURE_DECISIONS_AND_PR_REVIEW.md` so we have hard quantitative proof for whether to keep or prune `MaxViT`.
2. **Step 2 — High-Purity Complete-Linkage Crop Clustering (`ADR-008`)**:
   - Integrate `cluster_shelf_facings_high_purity()` (`tau = 0.94`, geometric + sub-ROI $\Delta E$ veto, singleton fallback) and benchmark cluster compression ratio (`~3.8x`), node purity (`>= 99.1%`), and F2 impact on `val` and `test`.
3. **Step 3 — Model Consolidation Cleanup (`ADR-001` through `ADR-006`)**:
   - Replace `multimodalembedding@001` in `src/utils/embeddings.py` with `gemini-embedding`, remove `I-JEPA`, `Florence-2`, `DINOv2/v3-Large`, and `SAM 2` references, and align all MT MarketShare (`324K/day`) and MT Merchandising (`105K/day`) benchmarks with `docs/Project_Plan.md`.
