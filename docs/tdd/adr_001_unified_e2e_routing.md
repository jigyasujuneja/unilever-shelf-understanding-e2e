# ADR-001: Unified End-to-End Routing Across Detection, Vector Matching, and VLM Fallback

* **Status:** Accepted
* **Date:** 2026-09-29
* **Owners:** Jigyasu Juneja (`jjuneja@google.com`), Riley Gavigan (`rgavigan@google.com`), Anil (`cloud-gtm`)
* **Target Branch:** `feat/unified-cloud-e2e`

## 1. Context and Problem Statement

Hindustan Unilever Limited (`HUL`) processes 500,000 retail shelf photographs daily across General Trade (`Shikhar` / `Sales EDGE`) and Modern Trade outlets. Single-pass Vision Language Model (`VLM`) calls on full-resolution shelf images exceed the `<= 10.0s` Merchandising SLA and the `<= INR 0.22/image` FinOps ceiling when shelves contain `100+` dense product facings. Conversely, pure closed-set vision detectors cannot resolve fine-grained sister shades (`Lakme 9to5 CC Almond` vs. `Honey` vs. `Beige`) or open-set competitor SKUs.

## 2. Decision

We decouple the production pipeline in [`src/core/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/) into three composable EPIC pillars:

1. **Pillar 1 — High-Recall SKU Localization ([`src/core/detection.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/detection.py))**:
   * Localizes candidate product bounding boxes and promotional `Toker` headers using tiled shelf-rail detection (`RT-DETR-v2` / `YOLOv11` / `Gemini Flash` spatial proposals) followed by Distance-IoU Non-Maximum Suppression (`DIoU-NMS`) to preserve vertically stacked sachets and tight horizontal facings.
2. **Pillar 2 — Multimodal Vector Matching ([`src/core/matching.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/matching.py))**:
   * Crops each detected bounding box, masks specular glare highlights, and embeds the crop using **Gemini Embedding 2 (`gemini-embedding-2-preview`)** via [`src/utils/embeddings.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/utils/embeddings.py).
   * Queries the HUL 7-dimension catalog via Cloud SQL (`pgvector` / `AlloyDB`) or Vertex AI Vector Search (`ScaNN`) in [`src/utils/vector_store.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/utils/vector_store.py).
   * High-confidence matches (`cosine_similarity >= 0.82` and non-colliding sister-shade margin) resolve immediately at `CONFIDENT` status without consuming generative VLM output tokens.
3. **Pillar 3 — Cost-Aware VLM Fallback & Sister-Shade Disambiguation ([`src/core/fallback.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/fallback.py))**:
   * Crops scoring in the ambiguous band (`0.65 <= similarity < 0.82` or colliding sister-shade families) escalate to the 64-token structured compound VLM gateway (`Gemini 3.8 Flash` / `Gemini 3.5 Flash Lite`) with `3x` sub-ROI zoom and CIELAB `Delta-E00` color distance features.
   * Crops scoring `< 0.65` are flagged as `UNRECOGNIZED` / open-set competitor SKUs and surfaced in the **Perfect Store Control Plane UI** ([`web/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/web/)) for human-in-the-loop audit.

## 3. Consequences

* **Latency & FinOps:** `85%–89%` of shelf facings resolve in the vector matching tier (`gemini-embedding-2-preview` + `ScaNN`/`pgvector`), reserving generative VLM calls for the `11%–15%` ambiguous or open-set tail.
* **Zero-Retraining Day-0 SKU Onboarding:** New HUL studio packshots are embedded once via `gemini-embedding-2-preview` and inserted into the vector catalog without retraining the localization detector.
* **Reviewer Ergonomics:** The Control Plane UI receives pre-classified `CONFIDENT`, `AMBIGUOUS`, and `UNRECOGNIZED` statuses alongside business KPI cards (`Share of Shelf %`, `Toker Compliance`, `Red Line Alignment`).
