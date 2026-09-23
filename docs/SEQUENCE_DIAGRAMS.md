# Unilever Shelf Understanding — Execution & Extensibility Sequence Diagrams

This document contains the complete set of sequence diagrams for the Unilever
Shelf Understanding platform:

1. **Adding a New Approach & Running It** (End-to-End Extensibility Lifecycle)
2. **Running the Benchmark System in Detail** (Deep Runtime Execution Sequence)
3. **Individual Execution Sequence Diagrams for All 8 Current Paths (`Path 1` – `Path 8`)**

---

## 1. Sequence Diagram: Adding a New Approach Plugin (Design, Implementation & Contract Binding)

This diagram details the complete workflow for an engineer authoring, scaffolding, and registering a new approach plugin (e.g., `custom_cv_vlm`). Because the framework uses an **Auto-Discovery Plugin Architecture**, new approaches are added entirely within their own module or script—**zero modifications are needed to core engine files** (`runner.py`, `config.py`, `tasks/`, or `ui/`).

*(Note: Runtime benchmark execution is intentionally excluded here and detailed separately in Section 2).*

<!-- mdformat off(reason: preserving Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    actor Eng as Engineer
    participant PluginPkg as approaches/<custom_id>/plugin.py<br/>(or standalone script)
    participant Base as approaches/base.py<br/>(SimpleShelfApproachPlugin)
    participant Ctx as CommonLayerContext<br/>(Shared Infrastructure)
    participant Pipe as pipeline.py<br/>(PipelineExecutor)
    participant Registry as approaches/registry.py<br/>(GLOBAL_APPROACH_REGISTRY)
    participant Discovery as CLI / UI Discovery<br/>(list-approaches / UI Studio)

    Note over Eng,PluginPkg: Phase 1: Architecture & Scaffolding
    Eng->>PluginPkg: 1. Scaffold plugin directory: `src/shelf_benchmark/approaches/<custom_id>/`
    Eng->>PluginPkg: 2. Create `plugin.py` subclassing `SimpleShelfApproachPlugin` (or use `@register_approach_function`)
    Eng->>PluginPkg: 3. Define metadata properties:
    Note right of PluginPkg: • approach_id: unique slug (e.g., "custom_cv_vlm")<br/>• display_name: human-readable title<br/>• category: "two_stage_vlm" | "vlm_multimodal" | "classic_cv"<br/>• task_type: "classification" (default) or "detection"<br/>• stages_description: list of stage summaries

    Note over Eng,Ctx: Phase 2: Implement Inference & Bind Shared Helpers
    Eng->>PluginPkg: 4. Implement `detect_and_classify(self, ctx, model_name, record)`
    PluginPkg->>Ctx: 5. Connect shared image loader: `ctx.load_shelf_image(record)`
    opt If chaining from a prior Stage 1 detector
        PluginPkg->>Ctx: 5a. Consume locked coordinates: `ctx.get_prior_detected_boxes(prior_detection)`
    end
    PluginPkg->>Ctx: 6. Access unified GenAI / Vertex client: `ctx.get_client()`
    PluginPkg->>Ctx: 7. Apply geometric 1D-NMS: `ctx.deduplicate_depth_stacked_facings(candidates)`
    Note right of PluginPkg: Candidate dicts populate standard keys:<br/>`bbox_2d` [ymin,xmin,ymax,xmax] (0..1000), `brand`,<br/>`product_name`, `variant`, `category`, `packaging_type`,<br/>`size`, `confidence`, plus custom attributes (>8 attrs).

    Note over PluginPkg,Pipe: Phase 3: Contract Binding & Automatic Pipeline Wrapping
    Base->>Pipe: 8. `SimpleShelfApproachPlugin.execute()` delegates to `ctx.execute_with_pipeline()`
    Pipe->>Pipe: 9. Pre-bind execution wrapper with `InvocationContext`:
    Note right of Pipe: • Config-driven RetryPolicy (offline: NO_RETRY; live: max_attempts=3)<br/>• Process CPU time measurement (`time.process_time()`)<br/>• Rule-derived size bucketing (`derive_size_bucket_from_bbox`)<br/>• HUL brand portfolio matching (`check_is_hul_brand`)<br/>• 5-Bucket GCP Cost Attribution Engine<br/>• Ground-truth accuracy evaluation (Hungarian IoU + per-attribute)<br/>• OpenTelemetry span generation (`otel_logs.jsonl`)

    Note over Registry,Discovery: Phase 4: Dynamic Auto-Discovery & Exposure
    Registry->>PluginPkg: 10. `GLOBAL_APPROACH_REGISTRY.discover_all()` dynamically scans `approaches/*/plugin.py`
    Registry->>Registry: 11. Inspect classes and factories (`get_plugins()`), validating `BaseShelfApproachPlugin` contract
    Registry->>Registry: 12. Register instance into `_plugins[approach_id]`
    Registry-->>Discovery: 13. Expose new approach to ecosystem:
    Discovery-->>Eng: • `shelf-benchmark list-approaches` outputs new ID with `source=plugin`<br/>• CLI `--approaches <custom_id>` accepts the new approach<br/>• UI `/api/dashboard` and Live Studio dropdowns auto-populate with zero code edits
```
<!-- mdformat on -->

---

## 2. Detailed Sequence Diagram: Running the Benchmark Pipeline (Runtime Deep-Dive)

This diagram isolates **runtime execution only**, detailing how `1..N` shelf
images, candidate catalog SKUs, Vertex AI structured inference, crop generation,
Hungarian bipartite Ground Truth evaluation, and OpenTelemetry billing/tracing
interact during a live run.

<!-- mdformat off(reason: preserving Mermaid detailed runtime sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    actor Caller as User / CLI /<br/>Live Studio UI
    participant Entry as cloud_run_service.py<br/>or cli.py
    participant Runner as BenchmarkRunner<br/>(runner.py)
    participant Data as ShelfAssociationProvider<br/>& CatalogProvider
    participant Worker as ThreadPoolExecutor<br/>(run_single)
    participant VLM as VertexAIClient<br/>(vertex_client.py)
    participant Cropper as PIL Image Cropper<br/>(reports/crops/)
    participant Eval as evaluate_shelf_run<br/>(metrics.py)
    participant OTel as OtelJsonlLogger &<br/>CostCalculator
    participant Store as reports/*.json &<br/>ui/server.py

    Caller->>Entry: Trigger benchmark (`approaches`, `models`, `images=1..N`, `connect_gt`)
    Entry->>Data: Load `configs/shelf_associations.json` (1..N images) & `product_catalog.json`
    Data-->>Entry: Return `list[ShelfAssociationRecord]` + candidate `CatalogProduct` list
    Entry->>Runner: `run_benchmark(task_types, models, approaches, max_workers=N)`

    par Parallel Execution across (Image_i × Model_m × Approach_a)
        Runner->>Worker: Submit `run_single(record_i, task_type, model_m, approach_a)`
        Worker->>Worker: Read `image_bytes` (local file or `gs://` blob) & start `t_start`
        Worker->>VLM: Dispatch to `_execute_inference(approach_a, image_bytes, candidate_skus)`

        alt Single-Stage Approach (Path 1, Path 5, Path 8)
            VLM->>VLM: Build multimodal prompt + JSON response schema
            VLM->>VLM: Call `client.models.generate_content(model_m, [image_part, prompt])`
            opt JSON Malformed or Truncated
                VLM->>VLM: Execute Pass-2 Self-Repair Retry with higher `max_output_tokens`
            end
            VLM-->>Worker: Return `list[DetectedFacing]` (`[ymin, xmin, ymax, xmax]` in 0..1000) + `UsageMetadata`
            Worker->>Cropper: Generate UI inspection thumbnails from full-shelf boxes (`_ensure_crop_files`)
        else Two-Stage Crop Approach (Path 2, Path 3, Path 4, Path 6, Path 7)
            VLM-->>Worker: Stage 1 Detector returns raw `list[BoundingBox]` (`[ymin, xmin, ymax, xmax]`)
            loop For each detected bounding box `idx`
                Worker->>Cropper: `_crop_image_bytes(image_bytes, bbox)` -> save `reports/crops/{run_id}_{idx}.jpg`
                Worker->>VLM: Stage 2 Classifier (`classify_single_crop` / Vector / OCR) on `crop_bytes`
                VLM-->>Worker: Return 7-Dim attributes + matched SKU (`bbox` locked to Stage 1 shelf coords)
            end
        end

        Worker->>OTel: Compute `estimate_vertex_cost(model_m, prompt_tokens, output_tokens, thinking_tokens)`
        OTel-->>Worker: Return `CostBreakdown` (USD)

        opt Ground Truth Record Exists for `record_i.ground_truth_id`
            Worker->>Eval: `evaluate_shelf_run(facings, gt_record, iou_threshold=0.50)`
            Eval->>Eval: Stage A: `match_boxes_hungarian()` ($O(n^3)$ global IoU bipartite matching)
            Eval->>Eval: Compute `AP@0.50`, `mAP@[0.50:0.95]`, `Precision`, `Recall`, `Detection F1`, `Mean IoU`
            Eval->>Eval: Stage B (on TP pairs): Compute `Brand Acc`, `Product Acc`, `7-Dim Macro Acc`, `SKU Acc`
            Eval-->>Worker: Return `AccuracyMetrics` + annotate each `DetectedFacing` (`iou_with_ground_truth`, `is_correct_detection`)
        end

        Worker->>OTel: Emit structured trace span to `reports/otel_logs.jsonl` (`trace_id`, `span_id`, `stage_latency_ms`)
        Worker-->>Runner: Return `SingleRunResult`
    end

    Runner->>Store: Aggregate across `1..N` images via `build_leaderboard_records()`
    Runner->>Store: Write `benchmark_summary.json`, `row_level_report.json`, `leaderboard.csv`
    Store-->>Caller: Serve live results via `GET /api/dashboard` & `GET /api/shelf-image`
```
<!-- mdformat on -->

---

## 3. Execution Sequence Diagrams for All 8 Current Approaches (`Path 1` – `Path 8`)

Below are the 8 dedicated execution sequence diagrams corresponding to each of
the 8 architectural paths implemented in
[`BenchmarkRunner._execute_inference`](file:///usr/local/google/home/rgavigan/unilever-shelf-understanding-with-cv/src/shelf_benchmark/runner.py#L406-L772).

### Path 1: `monolithic_llm` — Single-Shot End-to-End Multimodal LLM

One multimodal Gemini call receives the full high-resolution shelf image and
simultaneously outputs every front-facing `[ymin, xmin, ymax, xmax]` bounding
box, all 7 product taxonomy dimensions, and the matched catalog SKU.

<!-- mdformat off(reason: preserving Path 1 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant V as VertexAIClient<br/>(detect_and_classify_monolithic)
    participant G as Vertex AI Gemini<br/>(3.8-flash / 3.5-flash-lite)
    participant C as Cropper (_ensure_crop_files)

    R->>V: `detect_and_classify_monolithic(image_bytes, candidate_skus, brand_mode)`
    V->>G: `generate_content([full_shelf_image, monolithic_prompt + SKU_catalog_json])`
    G-->>V: Structured JSON array of facings (`box_2d`, `brand`, `product_name`, `attributes`[7], `matched_sku_id`)
    V->>V: Normalize `[ymin, xmin, ymax, xmax]` to `0..1000` & validate 7 taxonomy keys
    V-->>R: `list[DetectedFacing]` + `UsageMetadata` (1 API call total)
    R->>C: Slice visual thumbnails from `full_shelf_image` for UI inspection strip
    C-->>R: Populate `facing.crop_image_path` (`reports/crops/...`)
```
<!-- mdformat on -->

---

### Path 2: `two_stage_detection_first` — Two-Stage LLM Detector $\rightarrow$ Crop Classifier

Decouples spatial localization from fine-grained attribute reading: Stage 1 asks
Gemini exclusively for front-facing bounding boxes; PIL slices high-resolution
image crops; Stage 2 classifies each crop independently across all 7 dimensions.

<!-- mdformat off(reason: preserving Path 2 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant V as VertexAIClient
    participant G as Vertex AI Gemini
    participant PIL as PIL Cropper

    Note over R,G: Stage 1: Spatial Facing Localization Only (`detection_ms`)
    R->>V: `detect_only(image_bytes)`
    V->>G: `generate_content([full_shelf_image, detection_only_prompt])`
    G-->>V: JSON list of `stage1_facings` (`[ymin, xmin, ymax, xmax]`, `confidence`)
    V-->>R: Return `stage1_facings` (shelf-space coordinates)

    Note over R,G: Stage 2: Per-Crop 7-Dimension Classification (`classification_ms`)
    loop For each `s1_facing` in `stage1_facings`
        R->>PIL: `_crop_image_bytes(image_bytes, s1_facing.bbox)`
        PIL-->>R: Save `reports/crops/{run_id}_{idx}.jpg` & return `crop_bytes`
        R->>V: `classify_single_crop(crop_bytes, s1_facing.bbox, candidate_skus)`
        V->>G: `generate_content([crop_image, single_crop_7dim_prompt])`
        G-->>V: Return `brand`, `product_name`, `attributes`[7], `matched_sku_id`
        V-->>R: Return `DetectedFacing` with `bbox = s1_facing.bbox` (preserved shelf coordinates)
    end
```
<!-- mdformat on -->

---

### Path 3: `hybrid_yolo_gemini` — Local YOLOv8n Proposals $\rightarrow$ Gemini VLM Classifier

Replaces the Stage 1 LLM detector with an ultra-fast local **YOLOv8n**
convolutional detector (`yolov8n.pt`) for deterministic sub-second bounding box
proposals, followed by Gemini 7-Dimension classification on each crop.

<!-- mdformat off(reason: preserving Path 3 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant Y as YOLODetector<br/>(models/yolo_detector.py)
    participant PIL as PIL Cropper
    participant V as VertexAIClient
    participant G as Vertex AI Gemini

    Note over R,Y: Stage 1: Local Convolutional Object Detection (`detection_ms`)
    R->>Y: `detect_bounding_boxes(image_bytes)`
    Y->>Y: Run Ultralytics `YOLOv8n` forward pass + NMS + dense shelf grid synthesis
    Y-->>R: Return `boxes: list[tuple[BoundingBox, confidence]]` (0 LLM tokens)

    Note over R,G: Stage 2: Per-Crop Gemini VLM Classification (`classification_ms`)
    loop For each `(bbox, conf)` in `boxes`
        R->>PIL: `_crop_image_bytes(image_bytes, bbox)` -> save `reports/crops/{run_id}_{idx}.jpg`
        R->>V: `classify_single_crop(crop_bytes, bbox, candidate_skus)`
        V->>G: `generate_content([crop_image, single_crop_7dim_prompt])`
        G-->>V: Return 7-Dim attributes + `matched_sku_id`
        V-->>R: Return `DetectedFacing(bbox=bbox, detection_confidence=conf, ...)`
    end
```
<!-- mdformat on -->

---

### Path 4: `grounding_dino_gemini` — Zero-Shot Grounding DINO $\rightarrow$ Gemini VLM Classifier

Uses **Grounding DINO** (`IDEA-Research/grounding-dino-base`) prompted with
retail text queries (`"retail product . bottle . jar . box . deodorant .
shampoo"`) for zero-shot open-vocabulary localization, followed by Gemini crop
classification.

<!-- mdformat off(reason: preserving Path 4 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant D as GroundingDINODetector<br/>(models/grounding_dino_detector.py)
    participant PIL as PIL Cropper
    participant V as VertexAIClient
    participant G as Vertex AI Gemini

    Note over R,D: Stage 1: Open-Vocabulary Zero-Shot Detection (`detection_ms`)
    R->>D: `detect_bounding_boxes(image_bytes, text_prompt="retail product . bottle . box...")`
    D->>D: Cross-modal vision-language transformer box regression (`box_threshold=0.22`)
    D-->>R: Return `boxes: list[tuple[BoundingBox, confidence]]` (0 Vertex LLM tokens)

    Note over R,G: Stage 2: Per-Crop Gemini VLM Classification (`classification_ms`)
    loop For each `(bbox, conf)` in `boxes`
        R->>PIL: `_crop_image_bytes(image_bytes, bbox)` -> save `reports/crops/{run_id}_{idx}.jpg`
        R->>V: `classify_single_crop(crop_bytes, bbox, candidate_skus)`
        V->>G: `generate_content([crop_image, single_crop_7dim_prompt])`
        G-->>V: Return 7-Dim attributes + `matched_sku_id`
        V-->>R: Return `DetectedFacing(bbox=bbox, detection_confidence=conf, ...)`
    end
```
<!-- mdformat on -->

---

### Path 5: `geap_end_to_end` — GEAP 4-Agent Orchestration Pipeline

Executes a 4-Persona Google Enterprise Agentic Pipeline (**Agent 1: Shelf Grid
Planner** $\rightarrow$ **Agent 2: Spatial Facing Detector** $\rightarrow$
**Agent 3: Brand & 7-Dim Specialist** $\rightarrow$ **Agent 4: Catalog Critic &
SKU Reconciler**).

<!-- mdformat off(reason: preserving Path 5 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant V as VertexAIClient<br/>(run_geap_agentic_pipeline)
    participant G as Vertex AI Gemini<br/>(Multi-Agent Persona Prompting)
    participant C as Cropper (_ensure_crop_files)

    R->>V: `run_geap_agentic_pipeline(image_bytes, candidate_skus, brand_mode)`
    V->>G: Invoke 4-Agent Structured Workflow:<br/>1. Grid Planner (identifies horizontal shelf levels)<br/>2. Spatial Detector (sweeps left-to-right per shelf level)<br/>3. Taxonomy Specialist (extracts 7 Unilever dimensions)<br/>4. Catalog Critic (verifies SKU IDs & eliminates back-row depth ghosts)
    G-->>V: Return reconciled JSON array of front-row `DetectedFacing` objects + `UsageMetadata`
    V-->>R: Split stage timings (`detection_ms` = 35%, `classification_ms` = 65%) & return facings
    R->>C: Generate per-facing crop thumbnails in `reports/crops/`
```
<!-- mdformat on -->

---

### Path 6: `embedding_vector_search` — Multimodal Embedding + Vector Index Top-$K$ + VLM

Uses `multimodalembedding@001` (`1408-D` vectors) and `CatalogVectorIndex`
cosine similarity search to shortlist the top-$K$ (`K=3`) closest catalog SKUs
for each crop before invoking Gemini VLM disambiguation.

<!-- mdformat off(reason: preserving Path 6 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant V as VertexAIClient
    participant PIL as PIL Cropper
    participant IDX as CatalogVectorIndex<br/>(multimodalembedding@001)
    participant G as Vertex AI Gemini

    Note over R,IDX: Pre-Warm: Embed Catalog SKUs into 1408-D Vector Index
    R->>IDX: `warm_index(catalog_products)`
    Note over R,V: Stage 1: Detect Shelf Bounding Boxes (`detection_ms`)
    R->>V: `detect_only(image_bytes)`
    V-->>R: Return `stage1_facings` (`list[DetectedFacing]`)

    Note over R,G: Stage 2: Crop Embedding -> Cosine Top-K Retrieval -> VLM Disambiguation
    loop For each `s1_facing` in `stage1_facings`
        R->>PIL: `_crop_image_bytes(image_bytes, s1_facing.bbox)` -> `crop_bytes`
        R->>IDX: `search(crop_bytes, top_k=3)`
        IDX->>IDX: Compute 1408-D crop embedding & cosine similarity against catalog vectors
        IDX-->>R: Return `shortlisted_skus` (Top-3 `CatalogProduct` candidates + similarity scores)
        R->>V: `classify_single_crop(crop_bytes, s1_facing.bbox, shortlisted_skus)`
        V->>G: `generate_content([crop_image, prompt_with_only_top_3_skus])`
        G-->>V: Return 7-Dim attributes + disambiguated `matched_sku_id`
        V-->>R: Return `DetectedFacing(bbox=s1_facing.bbox, ...)`
    end
```
<!-- mdformat on -->

---

### Path 7: `ocr_augmented_vlm` — Crop Detection $\rightarrow$ Dedicated OCR Pre-Extraction $\rightarrow$ Grounded VLM

Adds a dedicated high-resolution crop **OCR pre-extraction step**
(`extract_ocr_from_crop`) to read tiny printed pack sizes (`750ml`, `250ml`),
multi-pack counts (`2-Pack`), and variant text before feeding those verbatim OCR
strings into the 7-Dimension VLM classifier.

<!-- mdformat off(reason: preserving Path 7 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant V as VertexAIClient
    participant PIL as PIL Cropper
    participant OCR as Vertex AI OCR Stage<br/>(extract_ocr_from_crop)
    participant G as Vertex AI 7-Dim Classifier<br/>(classify_single_crop_with_ocr)

    Note over R,V: Stage 1: Spatial Facing Detection (`detection_ms`)
    R->>V: `detect_only(image_bytes)`
    V-->>R: Return `stage1_facings` (`list[DetectedFacing]`)

    Note over R,G: Stage 2A (OCR Transcription) + Stage 2B (OCR-Grounded 7-Dim Classification)
    loop For each `s1_facing` in `stage1_facings`
        R->>PIL: `_crop_image_bytes(image_bytes, s1_facing.bbox)` -> `crop_bytes`
        R->>OCR: `extract_ocr_from_crop(crop_bytes)`
        OCR-->>R: Return verbatim `ocr_text` (e.g., `"DOVE DEEP MOISTURE 750mL 24hr"`) + `u_ocr` tokens
        R->>G: `classify_single_crop_with_ocr(crop_bytes, s1_facing.bbox, ocr_text, candidate_skus)`
        G-->>R: Return `DetectedFacing` with `size` / `variant` / `pack_type` anchored to `ocr_text`
    end
```
<!-- mdformat on -->

---

### Path 8: `critic_verify_cascade` — Self-Correction / Critic Verification Cascade

Runs a two-pass **Generator $\rightarrow$ Critic Cascade**: Pass 1 generates
initial full-shelf detections and 7-Dimension classifications; Pass 2 invokes a
Visual Critic (`verify_and_refine_shelf`) that inspects the shelf image
alongside Pass 1's JSON output to prune false positives, merge duplicate boxes,
recover missed facings, and fix misread attributes.

<!-- mdformat off(reason: preserving Path 8 Mermaid sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant Gen as Pass 1 Generator<br/>(detect_and_classify_monolithic)
    participant Crit as Pass 2 Visual Critic<br/>(verify_and_refine_shelf)
    participant C as Cropper (_ensure_crop_files)

    Note over R,Gen: Pass 1: Initial Full-Shelf Detection & Classification (`t_pass1`)
    R->>Gen: `detect_and_classify_monolithic(image_bytes, candidate_skus)`
    Gen-->>R: Return `pass1_facings` (`list[DetectedFacing]`) + `u_pass1` usage

    Note over R,Crit: Pass 2: Critic Self-Correction & Verification Audit (`t_pass2`)
    R->>Crit: `verify_and_refine_shelf(image_bytes, pass1_facings, candidate_skus)`
    Crit->>Crit: Audit `pass1_facings` against `image_bytes`:<br/>- Remove duplicate/overlapping boxes<br/>- Add missed front-row products<br/>- Correct 7-Dim attribute & SKU mismatches
    Crit-->>R: Return `refined_facings` (`list[DetectedFacing]`) + `u_pass2` usage
    R->>C: Slice per-facing crop thumbnails in `reports/crops/` for UI inspection
```
<!-- mdformat on -->
