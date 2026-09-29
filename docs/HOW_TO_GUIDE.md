# Developer Guide: Repository Architecture, Task Contracts, and Cloud Execution

Engineers use this repository to build, evaluate, and compare retail shelf detection and product classification models on a shared benchmark. Contributors add models in isolated files under `src/approaches/` or `src/stages/` while sharing the same dataset splits, evaluation metrics, cost ledger, and cloud runners (`Vertex AI CustomJob` and `Cloud Run Jobs`). For web UI architecture, navigation across all 4 views, and service deployment, see [`UI_HOW_TO_GUIDE.md`](UI_HOW_TO_GUIDE.md).

## Repository Layout and Boundaries

```text
unilever-shelf-understanding/
├── config.yaml                       # Default GCP project, Vertex AI, Cloud SQL, vector store, and split settings
├── Dockerfile                        # Container image definition for Vertex AI jobs, Cloud Run jobs, and web UI
├── src/
│   ├── cli.py                        # CLI entry point (list, run, vertex-job, vertex-train, vertex-deploy, cloud, pull, leaderboard, serve)
│   ├── runner.py                     # Evaluation loop, IoU box matching, F2 scoring, and cost ledger
│   ├── approaches/                   # Self-contained benchmark approaches (@register)
│   │   ├── base.py                   # Base Approach class, Context, Box type, and registry helpers
│   │   ├── _detector_template.py     # Starter template for task="detection"
│   │   ├── _classifier_template.py   # Starter template for task="classification"
│   │   ├── _detect_retrieve_template.py # Starter template for vector DB retrieval (Cloud SQL pgvector / Vertex Vector Search)
│   │   ├── _combined_pipeline_template.py # Starter template for task="combined"
│   │   └── modular_e2e_pipeline.py   # Configurable end-to-end pipeline composing approaches and stages
│   ├── stages/                       # Pluggable internal pipeline stages (StageSpec registry)
│   │   ├── registry.py               # StageSpec dataclass, register_stage(), and get_stage()
│   │   ├── stage1_rectification.py   # Image quality check and perspective homography
│   │   ├── stage2_3_detection.py     # Post-detection NMS and hanging sachet strip splitting
│   │   ├── stage3_5_clustering.py    # Adjacent crop deduplication before VLM classification
│   │   ├── stage4_retrieval.py       # Glare compensation and vector catalog lookup
│   │   ├── stage5_compound_vlm.py    # CIELAB color distance and VLM shade disambiguation
│   │   └── stage6_shelf_metrics.py   # Share-of-shelf, out-of-stock, and planogram metrics
│   └── utils/                        # Shared dataset, Vertex AI platform, Cloud SQL pgvector, VectorCatalog, billing, and web server modules
├── data/
│   └── splits/dataset_splits_manifest.json # SHA-256 locked train, val, and test split manifest
├── results/                          # Saved run directories (<run_id>/summary.json and images.jsonl)
└── tests/                            # Unit and integration test suite
```

To keep accuracy, latency, and cost numbers comparable across runs, do not modify the core evaluation files:

| Scope | Files | Modification Policy |
| :--- | :--- | :--- |
| **Frozen Core** | `src/runner.py`, `src/utils/metrics.py`, `src/utils/dataset.py`, `src/utils/pricing.py`, `src/utils/llm.py`, `src/approaches/base.py`, `data/splits/` | Do not modify `IoU=0.50` matching, F2 formulas, dataset split definitions, or GCP Billing Catalog price lookups. |
| **Task Approaches** | `src/approaches/<approach_name>.py` | Add new detection, classification, retrieval, or end-to-end approaches here. Any file not prefixed with `_` is auto-registered at import time. |
| **Pipeline Stages** | `src/stages/stage*.py` | Register new stage functions (`rectifier`, `post_detector`, `clusterer`, `retriever`, `tiebreaker`, `shelf_metrics`) with `register_stage(StageSpec(...))`. |

## Dataset Splits and Leaderboard Rules

The benchmark defines three non-overlapping splits in `src/utils/dataset.py`:

| Split | Image Count | Purpose | Leaderboard & Web UI (`#/arena`) Behavior |
| :--- | :--- | :--- | :--- |
| `train` | 8,219 (SKU-110K) / 20 (HUL) | Model fine-tuning (`shelf-bench vertex-train`), vector index population, and few-shot prompt selection. | Not displayed on the leaderboard. |
| `val` | 588 (SKU-110K) / 25 (HUL) | Local iteration on prompts, thresholds, and model weights (`--split val --limit 25 --seed 0`). | Displayed in the UI with `rank = "dev"` below official runs. |
| `test` | 2,936 (SKU-110K) / 50 (HUL) | Locked benchmark evaluation (`--split test --limit 50 --seed 0`). | Displayed with a numeric rank (`#1, #2, ...`) when executed on Vertex AI (`vertex-ai`) or Cloud Run (`cloud-run`). |

The web UI loads `summary.json` files from `results/` for both `test` and `val` runs. In `src/runner.py`, `leaderboard()` assigns a numbered rank only when `platform in ("vertex-ai", "cloud-run")` and `(split, limit, seed) == ("test", 50, 0)`. All local runs and `val` runs are labeled `dev` so engineers can inspect experimental runs in the UI without affecting official rankings.

## Task Contracts and Benchmark Epics

Every approach in `src/approaches/` subclasses `Approach` from `src/approaches/base.py`, applies the `@register` decorator, and declares `name`, `task`, `epic`, and `target_field`:

| Epic Name | `task` | `target_field` | Required Method Signature | Starter Template |
| :--- | :--- | :--- | :--- | :--- |
| `MT Market Share - SKU Detection` | `"detection"` | `"box"` | `detect(image, ctx) -> list[Box]` | `src/approaches/_detector_template.py` |
| `MT Market Share - Other (Category, Brand and Package Type) Classifiers` | `"classification"` | `"compound"` | `classify(image, boxes, ctx, prior=None) -> list[dict]` | `src/approaches/_classifier_template.py` |
| `MT Market Share - Variant Classification` | `"classification"` | `"variant"` | `classify(image, boxes, ctx, prior=None) -> list[dict]` | `src/approaches/_classifier_template.py` or `_detect_retrieve_template.py` |
| `MT Market Share - Combined Classification` | `"combined"` | `"variant"` | `detect_and_classify(image, ctx) -> tuple[list[Box], list[dict]]` | `src/approaches/_combined_pipeline_template.py` |
| `MT Merchandising - Promotion Asset Detection` | `"detection"` | `"box"` | `detect(image, ctx) -> list[Box]` | `src/approaches/_detector_template.py` |
| `MT Merchandising - Promotion Product Detection` | `"combined"` | `"variant"` | `detect_and_classify(image, ctx) -> tuple[list[Box], list[dict]]` | `src/approaches/_combined_pipeline_template.py` |

Data types passed into and out of these methods:
- `image`: A `PIL.Image.Image` in RGB format.
- `Box`: A 4-tuple `(x1, y1, x2, y2)` in absolute pixel coordinates of the input image.
- `label` dictionary (returned per box by `classify` and `detect_and_classify`):
  ```python
  {
      "sku_id": str,          # Catalog SKU code, e.g. "UL-DOVE-BW-500ML"
      "category": str,        # e.g. "Skin Care", "Hair Care", "Personal Care"
      "brand": str,           # e.g. "Dove", "Lakme", "Pond's"
      "packaging_type": str,  # e.g. "bottle", "tube", "jar", "box", "sachet"
      "variant": str,         # e.g. "Deeply Nourishing Body Wash 500ml"
      "is_hul": bool,         # True for HUL products, False for competitor products
  }
  ```
- `ctx`: A `Context` instance providing `ctx.ask(image, prompt, schema=...)` for metered Gemini calls, `ctx.bill(service, units)` for metered non-LLM services, and `ctx.trace.step(title, detail, boxes=..., labels=...)` for UI step inspection.

## Adding a New Task Approach (`src/approaches/`)

Choose the pattern below that matches your task, copy the corresponding template into `src/approaches/<your_approach>.py` (without a leading underscore), and implement the method for your task.

### Pattern 1: Bounding-Box Detector (`task = "detection"`)

Use this pattern for YOLO, RT-DETR, GroundingDINO, AutoML Object Detection, or VLM bounding-box detectors.

```python
from PIL import Image
from approaches.base import Approach, Box, Context, register

@register
class CustomShelfDetector(Approach):
    name = "custom_shelf_detector"
    task = "detection"
    epic = "MT Market Share - SKU Detection"
    target_field = "box"
    architecture = "Custom detector description shown in the leaderboard"
    steps = ["Run detector inference", "Apply non-maximum suppression"]

    def setup(self, config: dict) -> None:
        # Load model weights or initialize endpoint client once per run.
        pass

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes: list[Box] = [
            # Return (x1, y1, x2, y2) in pixel coordinates of `image`.
        ]
        ctx.trace.step("Detect products", f"{len(boxes)} boxes", boxes=boxes)
        return boxes
```

### Pattern 2: VLM or Fine-Tuned Endpoint Classifier (`task = "classification"`)

Use this pattern for Category/Brand/Packaging classifiers (`target_field = "compound"`) or Variant classifiers (`target_field = "variant"`). During standalone classification runs, the runner passes ground-truth bounding boxes into `boxes` so classifier accuracy is measured independently of detector errors.

```python
from typing import Any
from PIL import Image
from approaches.base import Approach, Box, Context, label_counts, register

OUTPUT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sku_id": {"type": "STRING"},
        "category": {"type": "STRING"},
        "brand": {"type": "STRING"},
        "packaging_type": {"type": "STRING"},
        "variant": {"type": "STRING"},
    },
    "required": ["sku_id", "category", "brand", "packaging_type", "variant"],
}

@register
class CustomCropClassifier(Approach):
    name = "custom_crop_classifier"
    task = "classification"
    epic = "MT Market Share - Variant Classification"
    target_field = "variant"  # Set to "compound" for Epic 2 (Category, Brand, Packaging)
    architecture = "Structured JSON crop classifier on Vertex AI"
    steps = ["Crop each bounding box", "Decode product attributes with structured JSON schema"]

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            hint = prior[idx] if prior and idx < len(prior) else {}
            prompt = (
                f"Classify this product crop. "
                f"Optional prior: category={hint.get('category', 'any')}, "
                f"brand={hint.get('brand', 'any')}, packaging={hint.get('packaging_type', 'any')}."
            )
            res = ctx.ask(image.crop(box), prompt, schema=OUTPUT_SCHEMA, max_side=512)
            data = res.data or {}
            predictions.append({
                "sku_id": data.get("sku_id", "UNKNOWN"),
                "category": data.get("category") or hint.get("category", "Unknown"),
                "brand": data.get("brand") or hint.get("brand", "Unknown"),
                "packaging_type": data.get("packaging_type") or hint.get("packaging_type", "bottle"),
                "variant": data.get("variant", "Unknown"),
                "is_hul": True,
            })
        ctx.trace.step("Classify crops", label_counts(predictions), boxes=boxes, labels=predictions)
        return predictions
```

To evaluate a fine-tuned Vertex AI model endpoint without changing code, pass the endpoint resource path to `-m`:
```bash
PYTHONPATH=src python3 src/cli.py run -a custom_crop_classifier \
  -m projects/<PROJECT_ID>/locations/us-central1/endpoints/<ENDPOINT_ID> \
  --split val --limit 25
```

### Pattern 3: Vector Database Retriever (`task = "classification"`)

Use this pattern when embedding product crops (`gemini-embedding-001` or `SigLIP`) and querying Cloud SQL for PostgreSQL (`pgvector`), Vertex AI Vector Search (`ScaNN`), or BigQuery Vector Search through `VectorCatalog` or `CloudSQL`. Include `skus = embeddings.SKUS` on the class so the runner bills embedding calls automatically.

```python
from typing import Any
from PIL import Image
from approaches.base import Approach, Box, Context, label_counts, register
from utils import embeddings
from utils.cloudsql import CloudSQL, pgvector
from utils.vector_store import VectorCatalog

VECTOR_SQL = """
    SELECT sku_id, category, brand, packaging_type, variant
    FROM products
    WHERE (%s IS NULL OR brand = %s)
    ORDER BY embedding <=> %s::vector
    LIMIT 1
"""

@register
class CustomVectorRetriever(Approach):
    name = "custom_vector_retriever"
    task = "classification"
    epic = "MT Market Share - Variant Classification"
    target_field = "variant"
    architecture = "Vertex multimodal crop embeddings + Cloud SQL pgvector / Vertex Vector Search retrieval"
    steps = ["Embed each crop", "Query Cloud SQL pgvector or VectorCatalog for nearest SKU"]
    skus = embeddings.SKUS

    def setup(self, config: dict) -> None:
        self.embed = embeddings.VertexEmbeddings(config)
        self.db = CloudSQL(**config.get("cloudsql", config.get("alloydb", {})))
        self.catalog = VectorCatalog(config)

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            brand_filter = prior[idx].get("brand") if prior and idx < len(prior) else None
            vec = pgvector(self.embed.image(image.crop(box), ctx))
            rows = self.db.query(VECTOR_SQL, (brand_filter, brand_filter, vec))
            sku_id, cat, brand, pkg, var = rows[0] if rows else ("UNKNOWN", "Unknown", "Unknown", "bottle", "Unknown")
            predictions.append({
                "sku_id": sku_id,
                "category": cat,
                "brand": brand,
                "packaging_type": pkg,
                "variant": var,
                "is_hul": not str(sku_id).startswith("COMP"),
            })
        ctx.trace.step("Vector lookup", label_counts(predictions), boxes=boxes, labels=predictions)
        return predictions
```

## Adding or Swapping an Internal Pipeline Stage (`src/stages/`)

If you want to experiment with an internal processing step (such as perspective rectification, post-detection NMS, crop clustering, catalog filtering, shade tie-breaking, or shelf metric calculation) without writing a full approach class, register a `StageSpec` in `src/stages/`:

```python
from stages.registry import StageSpec, register_stage

def my_custom_clusterer(boxes: list[tuple[float, float, float, float]]) -> dict:
    return {
        "mode": "my_custom_clusterer",
        "input_crops": len(boxes),
        "medoid_calls": max(1, len(boxes) // 5),
        "compression_ratio": 5.0,
        "cluster_purity": 0.995,
    }

register_stage(
    StageSpec(
        stage_group="clusterer",  # rectifier | post_detector | clusterer | retriever | tiebreaker | shelf_metrics
        name="my_custom_clusterer",
        title="Custom Crop Clusterer",
        description="Groups adjacent crops on the same shelf row before VLM calls.",
        f2_delta=0.001,
        latency_delta_s=-0.05,
        cost_delta_inr=-0.004,
        default=False,
        fn=my_custom_clusterer,
    )
)
```

Once registered, the stage appears in `shelf-bench list`, `/api/stages`, the web UI simulator dropdowns, and can be passed to `modular_e2e_pipeline` via `--with-clusterer my_custom_clusterer`.

## End-to-End Composition and CLI Workflow

Follow this five-step sequence for any change:

### 1. Authenticate and inspect available approaches and stages
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="<YOUR_GCP_PROJECT_ID>"

PYTHONPATH=src python3 src/cli.py list
```

### 2. Iterate locally on the validation split (`val`)
```bash
PYTHONPATH=src python3 src/cli.py run -a <your_approach> -m gemini-3.8-flash \
  --split val --limit 25 --seed 0 --owner $USER
```

### 3. Test your component inside the full end-to-end pipeline
Swap your detector, attribute classifier, variant classifier, or stage function into `modular_e2e_pipeline` to measure its impact on end-to-end 7-attribute F2, share-of-shelf accuracy, latency, and cost:

```bash
PYTHONPATH=src python3 src/cli.py run -a modular_e2e_pipeline -m gemini-3.1-flash-lite \
  --with-detector yolo_n26_sku110k \
  --with-attr-classifier djev_diffusiongemma_compound \
  --with-variant-classifier ft_gemini31_variant_compound \
  --with-rectifier depth_anything_v2 \
  --with-clusterer maxvit_agglomerative \
  --with-retriever siglip_multiprototype \
  --with-tiebreaker cielab_delta_e_and_systemone \
  --with-shelf-metrics dual_mt_marketshare_and_merchandising \
  --split val --limit 25
```

### 4. Run the test suite
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_unified_*.py" -v
```

### 5. Submit Vertex AI Training, Evaluation Jobs, and Agent Platform Deployment
```bash
# Provision GCP buckets, Artifact Registry, and Cloud Run / Vertex AI resources (one-time)
PYTHONPATH=src python3 src/cli.py bootstrap --project <YOUR_GCP_PROJECT_ID> --region us-central1

# Submit a supervised Gemini fine-tuning job on Vertex AI (TuningJob)
PYTHONPATH=src python3 src/cli.py vertex-train -m gemini-3.1-flash-lite \
  --display-name hul-variant-ft --epochs 4

# Execute benchmark evaluation on Vertex AI CustomJob against the locked 50-image test split
PYTHONPATH=src python3 src/cli.py vertex-job -a <your_approach> -m gemini-3.8-flash \
  --split test --limit 50 --seed 0 --owner $USER

# Pull finished Vertex AI / Cloud Run results from GCS into local results/
PYTHONPATH=src python3 src/cli.py pull

# Print the CLI leaderboard or launch the local web UI at http://127.0.0.1:8080/#/arena
PYTHONPATH=src python3 src/cli.py leaderboard
PYTHONPATH=src python3 src/cli.py serve --port 8080

# Deploy the shelf-audit agent to Vertex AI Agent Platform (ReasoningEngine) or web UI to Cloud Run
PYTHONPATH=src python3 src/cli.py vertex-deploy --display-name hul-perfect-store-agent
PYTHONPATH=src python3 src/cli.py cloud-service --port 8080
```


