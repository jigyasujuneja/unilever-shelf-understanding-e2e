# How to Add a New Shelf Understanding Approach Plugin

The benchmark suite uses an **Auto-Discovery Plugin Architecture** (`src/shelf_benchmark/approaches/`).
You never need to modify existing approaches (`src/shelf_benchmark/tasks/`) to test a new architecture.

## 1. Create a New Directory

Create a folder under `src/shelf_benchmark/approaches/<your_approach_name>/`:

```text
src/shelf_benchmark/approaches/my_custom_approach/
├── __init__.py
└── plugin.py
```

## 2. Implement `BaseShelfApproachPlugin` in `plugin.py`

Every plugin receives `ctx: CommonLayerContext`, which provides direct API access to all shared layers:
- **`ctx.storage`**: Read/write images and artifacts to/from Google Cloud Storage (`gs://...`).
- **`ctx.deduplicate_depth_stacked_facings(items)`**: Deterministic 1D horizontal column + `y_max` shelf-base NMS so back-row depth units are never double-counted.
- **`ctx.crop_facing_images(image_bytes, facings, output_dir)`**: Physically crops each bounding box into `facing_XX.png` and generates a numbered composite montage (`montage_all_facings.png`).
- **`ctx.derive_size_bucket_from_bbox(bbox_2d, all_bboxes)`**: Rule-derived HUL size bucket calculator.
- **`ctx.check_is_hul_brand(brand_name)`**: Checks against the configured HUL portfolio brands (`configs/taxonomy.yaml`).
- **`ctx.compute_cost(tokens, model_name, product_count, extra_api_cost_usd)`**: Computes Cost per Shelf Image (`$`) and Cost per Product Facing (`$`).
- **`ctx.evaluate_accuracy(predicted_items, gt_record, depth_duplicates_filtered)`**: Evaluates IoU@50, Count Accuracy, 7-Dimension Taxonomy Accuracy, and SKU Recall against `GroundTruthShelfRecord` (or returns `PLACEHOLDER_AWAITING_GROUND_TRUTH`).
- **`ctx.telemetry`**: Logs OpenTelemetry `TraceId`, `SpanId`, latency, tokens, and costs to `reports/otel_logs.jsonl`.

### Minimal Boilerplate (`plugin.py`):

```python
from typing import List, Optional
from shelf_benchmark.approaches.base import BaseShelfApproachPlugin, CommonLayerContext
from shelf_benchmark.models import GroundTruthShelfRecord, ShelfAssociationRecord, TaskExecutionResult


class MyCustomShelfApproach(BaseShelfApproachPlugin):
    @property
    def approach_id(self) -> str:
        return "my_custom_approach"

    @property
    def display_name(self) -> str:
        return "My Custom Approach (Stage 1 -> Stage 2 -> Stage 3)"

    @property
    def category(self) -> str:
        return "classic_cv_metric_learning"

    @property
    def stages_description(self) -> List[str]:
        return [
            "Stage 1: Custom Detector + Front-Facing Depth NMS",
            "Stage 2: Custom Feature / Embedding Extraction",
            "Stage 3: Catalog Vector Search & Taxonomy Resolution",
        ]

    def execute(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
        gt_record: Optional[GroundTruthShelfRecord] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        # Use ctx.storage, ctx.deduplicate_depth_stacked_facings, ctx.crop_facing_images,
        # ctx.compute_cost, ctx.evaluate_accuracy, and ctx.telemetry!
        ...
```

Once `plugin.py` is saved, `GLOBAL_APPROACH_REGISTRY.discover_all()` automatically registers `my_custom_approach` across the CLI, BenchmarkRunner, and Interactive Web Workbench!
