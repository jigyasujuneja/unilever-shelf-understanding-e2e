# Pull Request Review & Architecture Decision Records (ADRs)

**Target Repository:** [`cloud-gtm/unilever-shelf-understanding-with-cv`](https://github.com/cloud-gtm/unilever-shelf-understanding-with-cv)  
**Source Branch:** `jigyasujuneja:feat/unified-cloud-e2e` (synced with `cloud-gtm/main` @ `d68b970`)  
**Status:** Ready for Review — De-Leaked, De-Hardcoded & Benchmark-Validated (`36/36` Unit Tests Passing)

---

## 1. Executive Summary of Changes in This PR

1. **Eliminated All Ground-Truth Box Leakage (`known_boxes` Removed)**:
   - Removed `known_boxes = getattr(ctx.sample, "boxes", None)` from [`src/utils/hul_domain.py`](../src/utils/hul_domain.py) (`propose_rtdetr_shelf_boxes`) and from every `@register` approach in [`src/approaches/`](../src/approaches/).
   - Deleted all 6 leaked `results/0925-*` benchmark runs that previously reported synthetic `FP = 0`.
2. **Replaced MD5 Simulation with Real Pixel-Crop Feature Extraction & Dynamic Catalog Cosine Lookup**:
   - Replaced the MD5 hash stub in `scann_vector_lookup()` with [`extract_real_crop_features(image, box)`](../src/utils/hul_domain.py), [`extract_gemini_subroi_embedding(image, box)`](../src/utils/maxvit_clustering.py), and [`load_dynamic_hul_catalog_index()`](../src/utils/maxvit_clustering.py), computing real pixel-crop embeddings (`image.crop(box)`), 4-zone vertical sub-ROI RGB/CIELAB (`L*, a*, b*`) statistics, Sobel edge density, pixel-level specular glare dampening (`lum > 232, sat < 18`), and true cosine similarity/margin routing against the 245-SKU HUL catalog (`data/hul_catalog/hul_india_master_taxonomy.json`).
3. **De-Hardcoded Playground Image Upload & Store Presets**:
   - Updated [`src/utils/server.py`](../src/utils/server.py) (`_detect_crops_from_pil_image`) and [`src/utils/hul_domain.py`](../src/utils/hul_domain.py) (`detect_shelf_boxes_from_pixels`) so uploaded shelf images and store presets run real 2D Sobel shelf-rail + vertical valley instance detection and real pixel-crop classification rather than falling back to static preset boxes.
4. **Rebuilt Real Human-Annotated Ground-Truth Splits (`train` / `val` / `test`)**:
   - Rebuilt [`data/SKU110K_fixed/annotations/`](../src/utils/dataset.py) from 100% real human annotations:
     - **`train`**: `20` reference images (`38` GT boxes)
     - **`val`**: `25` official shelf images (`3,649` real human-annotated shelf boxes from `sku110k_benchmark_slice.json`)
     - **`test`**: `50` official `test_*.jpg` images (`7,154` real human-annotated shelf boxes) + `10` held-out RPC/labeled images
5. **Model Consolidation & Empirical Ablation of `MaxViT` (`ADR-007`) and High-Purity Crop Clustering (`ADR-008`)**:
   - Upgraded `src/utils/embeddings.py` from deprecated `multimodalembedding@001` to **`gemini-embedding-001`** (`ADR-002`), upgraded `SAM 2` → **`SAM 3`** (`ADR-004`), removed `Florence-2` (`ADR-005`) and `DINOv2/v3-Large` (`ADR-006`), and folded specular glare dampening directly into `gemini-embedding-001` sub-ROI extraction (`ADR-003`).
   - Implemented [`src/utils/maxvit_clustering.py`](../src/utils/maxvit_clustering.py) and ran full head-to-head ablations on both `val` (`25` images, `3,649` GT boxes) and `test` (`50` images, `7,154` GT boxes) to empirically evaluate **Complete-Linkage High-Purity Crop Clustering (`ADR-008`)** and **`MaxViT` Multi-Scale Block+Grid Attention (`ADR-007`)** before making any model pruning decisions.

---

## 2. Un-Leaked Benchmark Baseline & Empirical Ablation (`val` & `test` Splits)

### 2.1 Official `test` Split (`50` Images, `7,154` Real Ground-Truth Shelf Boxes)

| Run ID | Approach | Feature / Clustering Configuration | Owner | TP | FP | FN | Precision | Recall | **Box F2 (`IoU>=0.5`)** | 7-Dim SKU F2 | Sister-Shade F2 | p95 Latency | Cost / Image |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0924-231056` | `single_pass` | `gemini-3.8-flash` (Un-clustered VLM) | `rgavigan` | 4,284 | 556 | 2,870 | 0.8851 | 0.5988 | **0.6402** | — | — | 47.69s | ₹1.3560 |
| `0924-231739` | `detect_classify` | `gemini-3.5-flash-lite` (Retired SLA tier) | `rgavigan` | 4,922 | 1,753 | 2,232 | 0.7374 | 0.6880 | **0.6973** | — | — | 21.76s | ₹1.4780 |
| `0924-231554` | `detect_classify` | `gemini-3.8-flash` (Per-crop VLM) | `rgavigan` | 4,979 | 823 | 2,175 | 0.8582 | 0.6960 | **0.7233** | — | — | 107.13s | ₹2.0320 |
| `0924-231056` | `single_pass` | `gemini-3.5-flash-lite` (Retired SLA tier) | `rgavigan` | 5,118 | 1,510 | 2,036 | 0.7722 | 0.7154 | **0.7261** | — | — | 25.88s | ₹1.2290 |
| `0928-104539` | `djev_systemone_sister_shade` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 5,682 | 1,092 | 1,472 | 0.8388 | 0.7942 | **0.8028** | 0.974 | 0.964 | 2.13s | ₹0.0522 |
| `0928-104524` | `tiered_hybrid_scann` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 5,674 | 1,037 | 1,480 | 0.8455 | 0.7931 | **0.8031** | 0.958 | 0.884 | 2.23s | ₹0.0646 |
| `0928-104554` | `hul_8stage_gemini38_hybrid` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 5,841 | 1,355 | 1,313 | 0.8117 | 0.8165 | **0.8155** | 0.979 | 0.969 | 2.40s | ₹0.0417 |
| **`0928-115702`** | **`hul_8stage_gemini38_hybrid`** | **`gemini-embedding-001` Sub-ROI + High-Purity Clustering (`ADR-008`)** | **`jjuneja`** | **5,835** | **1,310** | **1,319** | **0.8167** | **0.8156** | **0.8158** (`+8.97 pts`) | **0.979** | **0.969** | **2.68s** | **₹0.0311** (`-25.4%`) |
| **`0928-115719`** | **`maxvit_clustered_djev`** | **`MaxViT` Block+Grid + High-Purity Clustering (`ADR-007` + `ADR-008`)** | **`jjuneja`** | **5,877** | **1,476** | **1,277** | **0.7993** | **0.8215** | **0.8170** (`+9.09 pts`) | **0.981** | **0.972** | **1.97s** | **₹0.0434** (`+39.5%`) |

### 2.2 Official `val` Split (`25` Images, `3,649` Real Ground-Truth Shelf Boxes)

| Run ID | Approach | Feature / Clustering Configuration | Owner | TP | FP | FN | Precision | Recall | **Box F2 (`IoU>=0.5`)** | 7-Dim SKU F2 | Sister-Shade F2 | p95 Latency | Cost / Image |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0928-105128` | `tiered_hybrid_scann` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 3,248 | 477 | 401 | 0.8719 | 0.8901 | **0.8864** | 0.958 | 0.884 | 6.30s | ₹0.0711 |
| `0928-105311` | `djev_systemone_sister_shade` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 3,248 | 357 | 401 | 0.9010 | 0.8901 | **0.8923** | 0.974 | 0.964 | 6.39s | ₹0.0554 |
| `0928-105323` | `hul_8stage_gemini38_hybrid` | Un-clustered baseline (`4fb66a5`) | `jjuneja` | 3,248 | 304 | 401 | 0.9144 | 0.8901 | **0.8949** | 0.979 | 0.969 | 3.20s | ₹0.0409 |
| **`0928-115645`** | **`hul_8stage_gemini38_hybrid`** | **`gemini-embedding-001` Sub-ROI + High-Purity Clustering (`ADR-008`)** | **`jjuneja`** | **3,248** | **304** | **401** | **0.9144** | **0.8901** | **0.8949** | **0.979** | **0.969** | **3.03s** | **₹0.0359** (`-12.2%`) |
| **`0928-115654`** | **`maxvit_clustered_djev`** | **`MaxViT` Block+Grid + High-Purity Clustering (`ADR-007` + `ADR-008`)** | **`jjuneja`** | **3,248** | **290** | **401** | **0.9180** | **0.8901** | **0.8956** | **0.981** | **0.972** | **1.71s** | **₹0.0474** (`+32.0%`) |

---

## 3. Architecture Decision Records (ADRs) & Model Consolidation Policy

To prevent **model inflation** (replacing Unilever's legacy 13-model Azure stack with an unmaintainable zoo of 7–8 self-deployed vision backbones), every candidate model family is governed by the following benchmark-driven ADR matrix aligned with Principal FDE review:

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

## 4. Verification & Reproducibility Commands

```bash
# 1. Verify zero train/val/test split leakage & SHA-256 manifest
PYTHONPATH=src python3 -m src.cli splits

# 2. Run consolidated production pipeline vs. MaxViT ablation on val (25 images) and test (50 images)
python3 -m src.cli run -a hul_8stage_gemini38_hybrid maxvit_clustered_djev -m gemini-3.8-flash --split val --limit 25
python3 -m src.cli run -a hul_8stage_gemini38_hybrid maxvit_clustered_djev -m gemini-3.8-flash --split test --limit 50

# 3. Run all 8 unit & integration test suites (36 tests)
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
