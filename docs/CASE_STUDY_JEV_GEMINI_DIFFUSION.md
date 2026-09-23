# Engineering Case Study: Solving Retail Shelf Understanding with `Jev + Gemini` vs. `GeminiDiffusion` Acting as `Jev`

> [!NOTE]
> This standalone architectural case study is **completely separate from the 8
> baseline production approaches (`Path 1` – `Path 8`)**. It illustrates how an
> engineer uses this repository's plug-and-play benchmarking framework to
> prototype, execute, and compare two advanced next-generation vision-language
> paradigms:
>
> 1. **Pattern A (`jev_gemini_hybrid`)**: Pairing **`Jev`** (Joint
>    Embedding-Vision spatial/representation specialist) **together with
>    `Gemini`** (`gemini-3.8-flash`).
> 2. **Pattern B (`gemini_diffusion_as_jev`)**: Using **`GeminiDiffusion` acting
>    directly as `Jev`** (unified diffusion-based spatial segmentation, latent
>    glare removal / super-resolution, and 7-Dimension taxonomy readout).

---

## 1. Why an Engineer Explores `Jev + Gemini` or `GeminiDiffusion-as-Jev`

On real-world retail shelves, baseline single-pass or standard 2-stage crop
pipelines encounter three physical optical bottlenecks:

1. **Specular Overhead Glare on Curved Bottles**: Fluorescent store lighting
   reflects off curved shampoo/body-wash bottles (`Dove 750ml`, `TRESemmé`),
   obscuring pack size (`750ml` vs `500ml`) and variant text.
2. **Tightly Packed Overlapping Facings & Depth Ghosts**: Cylindrical bottles
   touch edge-to-edge on crowded shelves while recessed 2nd-row stock (`depth
   ghosts`) peeks through gaps.
3. **Fine-Grained Visual-Semantic Disambiguation**: Distinguishing two nearly
   identical SKUs (e.g., *Dove Deep Moisture Body Wash 500ml* vs. *750ml Pump*)
   requires both sub-millimeter spatial boundary awareness and structured
   catalog reasoning.

---

## 2. Side-by-Side Architecture Diagram: `Jev + Gemini` vs. `GeminiDiffusion Acting as Jev`

<!-- mdformat off(reason: preserving Mermaid architecture comparison diagram) -->
```mermaid
flowchart TB
    subgraph Input["Input Shelf Context"]
        SHELF["High-Res Shelf Image (1..N)<br/>+ Candidate Unilever Catalog SKUs"]
    end

    subgraph PatternA["Pattern A: Jev + Gemini Together (jev_gemini_hybrid)"]
        direction TB
        JEV_STAGE1["1. Jev Spatial & Representation Engine<br/>• Panoptic Front-Row Instance Masks<br/>• Amodal De-Occlusion & Depth-Ghost Filtering<br/>• Joint Visual-Semantic Token Extraction (z_jev)"]
        JEV_CROP["2. Jev-Guided High-Res Crop + Latent Feature Pack<br/>• Shelf-Space BBox [ymin, xmin, ymax, xmax] locked<br/>• Cropped RGB patch + z_jev similarity prior"]
        GEMINI_STAGE2["3. Gemini Multimodal Classifier (gemini-3.8-flash)<br/>• Consumes RGB Crop + z_jev Prior + Candidate SKUs<br/>• Structured 7-Dim Taxonomy Extraction & SKU Match"]
        JEV_STAGE1 --> JEV_CROP --> GEMINI_STAGE2
    end

    subgraph PatternB["Pattern B: GeminiDiffusion Acting as Jev (gemini_diffusion_as_jev)"]
        direction TB
        GD_STAGE1["1. GeminiDiffusion Spatial Denoising (Acting as Jev Detector)<br/>• Iterative Latent Diffusion Panoptic Mask Generation<br/>• Separates touching curved bottles & suppresses 2nd-row ghosts"]
        GD_STAGE2["2. GeminiDiffusion Generative Restoration (Acting as Jev Enhancer)<br/>• Latent Inpainting removes specular store-light glare<br/>• Diffusion Super-Resolution sharpens micro-text (750ml / 2-Pack)"]
        GD_STAGE3["3. Unified Autoregressive + Diffusion 7-Dim Readout<br/>• Directly decodes 7 Unilever Attributes + Matched SKU ID<br/>from restored latent crop representations"]
        GD_STAGE1 --> GD_STAGE2 --> GD_STAGE3
    end

    subgraph SharedEval["Unified Plug-and-Play Evaluation & UI Layer"]
        EVAL["evaluate_shelf_run (metrics.py)<br/>• Hungarian O(n³) IoU Matching (AP@0.50, mAP@[0.50:0.95], F1)<br/>• 7-Dim Attribute Macro Accuracy & SKU Matching Accuracy"]
        UI["Interactive UI Matrix & Visual Inspector<br/>Compare Pattern A vs. Pattern B vs. Paths 1–8"]
        EVAL --> UI
    end

    SHELF --> PatternA
    SHELF --> PatternB
    PatternA --> SharedEval
    PatternB --> SharedEval
```
<!-- mdformat on -->

---

## 3. Sequence Diagram 1: Engineer Uses `Jev` and `Gemini` Together (`jev_gemini_hybrid`)

In **Pattern A**, the engineer delegates all dense visual perception, panoptic
boundary separation, and joint embedding extraction to **`Jev`**, and passes
`Jev`'s locked shelf-space bounding boxes, de-glared crops, and top candidate
priors to **`Gemini`** for 7-Dimension structured reasoning.

<!-- mdformat off(reason: preserving Pattern A Jev + Gemini sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    actor Eng as Engineer
    participant Runner as BenchmarkRunner<br/>(runner.py)
    participant Jev as Jev Vision Engine<br/>(Joint Embedding-Vision)
    participant Cropper as PIL + Jev Feature Pack<br/>(reports/crops/)
    participant Gemini as VertexAIClient<br/>(Gemini 3.8 Flash)
    participant Eval as evaluate_shelf_run<br/>(Hungarian IoU + 7-Dim GT)

    Eng->>Runner: Execute `shelf-benchmark run --approaches jev_gemini_hybrid`
    Runner->>Jev: `extract_front_row_instances_and_embeddings(shelf_image_bytes)`

    Note over Jev: Stage 1: Jev Spatial & Semantic Perception (`detection_ms`)
    Jev->>Jev: 1. Compute panoptic instance masks & filter recessed 2nd-row depth ghosts
    Jev->>Jev: 2. Regress tight shelf-space boxes `[ymin, xmin, ymax, xmax]` (0..1000)
    Jev->>Jev: 3. Compute Joint Embedding-Vision vectors `z_jev` per facing & rank catalog SKUs
    Jev-->>Runner: Return `list[JevProposal(bbox, confidence, z_jev_top_skus)]`

    Note over Runner,Gemini: Stage 2: Gemini 7-Dimension Reasoning on Jev Proposals (`classification_ms`)
    loop For each `proposal` in `JevProposal` list
        Runner->>Cropper: Slice high-res crop from `proposal.bbox` -> `reports/crops/{run_id}_{idx}.jpg`
        Runner->>Gemini: `classify_crop_with_jev_prior(crop_bytes, proposal.bbox, proposal.z_jev_top_skus)`
        Gemini->>Gemini: Combine visual crop + `Jev` semantic prior to resolve:<br/>Category, Subcategory, Brand, Variant, Packaging, Pack Type, Size + SKU
        Gemini-->>Runner: Return `DetectedFacing(bbox=proposal.bbox, attributes=7_dims, matched_sku_id=sku)`
    end

    Runner->>Eval: Score `jev_gemini_hybrid` facings against Ground Truth (`AP@0.50`, `mAP`, `7-Dim Macro Acc`)
    Eval-->>Eng: Output side-by-side comparison in `reports/benchmark_summary.json` & UI
```
<!-- mdformat on -->

---

## 4. Sequence Diagram 2: `GeminiDiffusion` Acting Directly as `Jev` (`gemini_diffusion_as_jev`)

In **Pattern B**, the engineer replaces the external `Jev` module completely by
having **`GeminiDiffusion` act as `Jev`**. `GeminiDiffusion` performs iterative
denoising in latent space to segment overlapping bottles, reconstructs text
under specular shelf glare, and emits the 7-Dimension classification contract.

<!-- mdformat off(reason: preserving Pattern B GeminiDiffusion-as-Jev sequence diagram) -->
```mermaid
sequenceDiagram
    autonumber
    actor Eng as Engineer
    participant Runner as BenchmarkRunner<br/>(runner.py)
    participant GDiff as GeminiDiffusion Engine<br/>(Acting as Jev)
    participant Cropper as Latent-Restored Crop Saver<br/>(reports/crops/)
    participant Eval as evaluate_shelf_run<br/>(Hungarian IoU + 7-Dim GT)

    Eng->>Runner: Execute `shelf-benchmark run --approaches gemini_diffusion_as_jev`
    Runner->>GDiff: `run_gemini_diffusion_as_jev(shelf_image_bytes, candidate_skus)`

    Note over GDiff: Step 1 (Spatial Diffusion as Jev Detector):<br/>Denoise Panoptic Front-Facing Masks (`t = T ... 0`)
    GDiff->>GDiff: Iteratively denoise 2D spatial segmentation map on full shelf image
    GDiff->>GDiff: Isolate touching bottles & discard low-opacity back-row shelf shadows
    GDiff->>GDiff: Extract canonical shelf-space `[ymin, xmin, ymax, xmax]` per front facing

    Note over GDiff: Step 2 (Generative Restoration as Jev Enhancer):<br/>De-Glare & Super-Resolve Crop Typography
    loop For each localized front facing `i`
        GDiff->>GDiff: Inpaint specular fluorescent glare & super-resolve micro-typography (`750mL`, `2-Pack`)
        GDiff->>Cropper: Save restored crop image to `reports/crops/{run_id}_{i}.jpg`

        Note over GDiff: Step 3 (Joint Generative-Discriminative Readout):<br/>Decode 7 Unilever Taxonomy Dimensions & Catalog SKU
        GDiff->>GDiff: Decode `category`, `subcategory`, `brand`, `variant`, `packaging`, `pack_type`, `size`, `matched_sku_id`
    end

    GDiff-->>Runner: Return `list[DetectedFacing]` + diffusion step latency & token telemetry
    Runner->>Eval: Evaluate `gemini_diffusion_as_jev` vs. `jev_gemini_hybrid` vs. Paths 1–8
    Eval-->>Eng: Render comparative metrics & restored crop strips in UI (`http://127.0.0.1:8080`)
```
<!-- mdformat on -->

---

## 5. Engineer's End-to-End Experimentation Workflow Diagram

This diagram shows how the engineer takes the hypothesis (*"Does `Jev + Gemini`
or `GeminiDiffusion-as-Jev` beat our existing 8 approaches on occluded/glared
bottles?"*) and validates it in the repository:

<!-- mdformat off(reason: preserving Mermaid experimentation workflow diagram) -->
```mermaid
flowchart LR
    H["1. Formulate Hypothesis<br/>Specular glare & tight bottle overlap<br/>hurt size/variant accuracy"]
    REG["2. Register Enum Values<br/>Add JEV_GEMINI_HYBRID &<br/>GEMINI_DIFFUSION_AS_JEV<br/>in config.py"]
    IMPL["3. Implement Adapters<br/>Add Jev/GeminiDiffusion calls<br/>in vertex_client.py & runner.py"]
    RUN["4. Execute Parallel Benchmark<br/>Run across 1..N shelf images<br/> alongside Paths 1–8"]
    SCORE["5. Compare on Ground Truth<br/>• Object Detection: AP@0.50, mAP, IoU<br/>• 7-Dim Classification: Size & Variant Acc<br/>• Cost ($) & Latency (s) Trade-off"]

    H --> REG --> IMPL --> RUN --> SCORE
```
<!-- mdformat on -->

---

## 6. Trade-Off Matrix: `Jev + Gemini` vs. `GeminiDiffusion-as-Jev` vs. Baseline Paths

| Architectural Dimension | Baseline `Path 2` (`two_stage_detection_first`) | Pattern A: `Jev + Gemini` Together (`jev_gemini_hybrid`) | Pattern B: `GeminiDiffusion` Acting as `Jev` (`gemini_diffusion_as_jev`) |
| :--- | :--- | :--- | :--- |
| **Stage 1 Spatial Localization (`UC1`)** | Standard VLM bounding-box regression (`detect_only`) | **`Jev`** panoptic instance masks + amodal boundary separation | **`GeminiDiffusion`** iterative latent mask denoising |
| **Handling Specular Bottle Glare & Tiny Text** | Raw RGB crop passed directly to classifier | **`Jev`** joint visual-semantic embedding (`z_jev`) bridges missing pixels | **`GeminiDiffusion`** generative latent de-glaring & super-resolution |
| **Stage 2 7-Dimension Taxonomy & SKU (`UC2/UC3`)** | `Gemini` (`gemini-3.8-flash`) on raw RGB crop | **`Gemini`** conditioned on RGB crop + **`Jev`** top-$K$ semantic prior | **`GeminiDiffusion`** unified generative-discriminative readout |
| **Infrastructure Complexity** | 1 Model Family (`Gemini`) | 2 Models (`Jev` Vision Endpoint + `Gemini` VLM) | 1 Unified Model Family (`GeminiDiffusion` acting as both `Jev` & Classifier) |
| **Primary Strength** | Low latency, zero custom vision infra | High recall on ultra-dense shelves + fast `z_jev` vector pruning | Highest robustness on heavily glared, motion-blurred, or partially occluded labels |
