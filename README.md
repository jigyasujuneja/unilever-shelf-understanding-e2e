# Unilever Retail Shelf Understanding — End-to-End Prototype & Benchmark (`SPEC-001` & `SPEC-002`)

**Owner:** Jigyasu Juneja (`jjuneja@google.com`)  
**Specifications:** [`specs/DESIGN.md`](specs/DESIGN.md) (`SPEC-001`) & [`specs/HARNESS.md`](specs/HARNESS.md) (`SPEC-002`)

---

## 1. System Intent & Production SLAs
- **Goal:** Ingest 4K retail shelf images, localize SKUs, resolve Base Pack codes, and audit promotional compliance across **4 Modern Trade use cases**:
  1. `MT MarketShare` (`resolved_skus`: `[x1, y1, x2, y2]`, `base_pack_id`, `confidence`)
  2. `MT Merchandising` (`red_line_gaps`, `width_pack_gaps`)
  3. `MT Toker Compliance` (`toker_status`, `display_status`, `coaching_message`)
  4. `MT Share of Shelf (SOS)` (`share_of_shelf_pct`)
- **Hard Cost Ceiling:** $\le$ **₹0.22 (~$0.0026 USD)** per image at 500,000 images/day scale (`1 USD = 84 INR`).
- **Hard Latency SLA:** **P95 $\le$ 20.0s (`20,000 ms`)** across **525 concurrent worker threads**.

---

## 2. Competing Architecture Tracks (`src/shelf_e2e/tracks/`)

| Track | Module | Architecture Pipeline |
| :--- | :--- | :--- |
| **Track A** | [`track_a_cascading_vit.py`](src/shelf_e2e/tracks/track_a_cascading_vit.py) | `RT-DETR` Detection $\rightarrow$ `ViT-B-16` Brand Classifier $\rightarrow$ `6x InceptionNet` Variant Classifiers |
| **Track B** | [`track_b_e2e_vlm.py`](src/shelf_e2e/tracks/track_b_e2e_vlm.py) | Single-pass End-to-End `Gemini 2.5 Flash Lite` with raw 4K shelf image + planogram rules |
| **Track C (FDE Recommended)** | [`track_c_tiered_hybrid.py`](src/shelf_e2e/tracks/track_c_tiered_hybrid.py) | `RT-DETR` (Cloud Run L4 GPU) $\rightarrow$ Vertex AI Multimodal Embeddings + `ScaNN` Vector Search $\rightarrow$ `Gemini 2.5 Flash Lite` (Promo/Toker crops only) |
| **Track D (`Jev` & `GeminiDiffusion-as-Jev`)** | [`track_d_jev_routing.py`](src/shelf_e2e/tracks/track_d_jev_routing.py) | Tier 1 `RT-DETR` $\rightarrow$ Tier 2 `ScaNN` Vector Search $\rightarrow$ `Jev` Deterministic State Machine (+ `GeminiDiffusion-as-Jev` latent de-glaring) $\rightarrow$ `Gemini 2.5 Flash Lite` promo reasoning |

---

## 3. Quickstart

### Run the `SPEC-002` Standalone Evaluation Harness (`pytest`)
```bash
pytest -v tests/benchmark_harness.py tests/test_all_tracks_and_slas.py
```

### Run the Pareto Benchmark Matrix
```bash
python3 scripts/run_pareto_benchmark.py
```
