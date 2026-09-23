# How to Add & Test a New Shelf Understanding Approach Plugin

The benchmark suite uses an **Auto-Discovery Plugin Architecture** (`src/shelf_benchmark/approaches/`).
You never need to modify existing approaches or core files (`src/shelf_benchmark/tasks/`, `runner.py`, or `ui/`) to test a new architecture.

---

## Option A: Fastest Path (`@register_approach_function` in a script)

Decorate a single function `(ctx, model_name, shelf_image_uri) -> List[Dict[str, Any]]`:

```python
from typing import Any, Dict, List
from shelf_benchmark import CommonLayerContext, register_approach_function


@register_approach_function(
    approach_id="my_custom_approach",
    display_name="My 2-Stage Detector + Verifier",
    category="two_stage_vlm",
    stages_description=["Stage 1: Detector + Depth NMS", "Stage 2: Attribute Verifier"],
)
def run_my_custom_approach(
    ctx: CommonLayerContext,
    model_name: str,
    shelf_image_uri: str,
) -> List[Dict[str, Any]]:
    candidates = [
        {
            "bbox_2d": [100, 100, 300, 200],
            "brand": "Pond's",
            "product_name": "Pond's Bright Beauty Face Wash",
            "category": "Skin Cleansing",
            "subcategory": "Face Wash",
            "variant": "Bright Beauty",
            "packaging_type": "tube",
            "pack_type": "Single",
            "size": "100g",
            "confidence": 0.96,
        }
    ]
    kept, filtered = ctx.deduplicate_depth_stacked_facings(candidates)
    if kept:
        kept[0]["_depth_filtered"] = filtered
    return kept
```

Run it from the CLI without moving files:
```bash
.venv/bin/shelf-benchmark run --offline \
  --plugin-module path/to/my_script.py \
  --approaches my_custom_approach
```

---

## Option B: Permanent Plugin Package (`SimpleShelfApproachPlugin` in ~15 lines)

Create `src/shelf_benchmark/approaches/my_custom_approach/plugin.py`:

```python
from typing import Any, Dict, List
from shelf_benchmark.approaches import CommonLayerContext, SimpleShelfApproachPlugin
from shelf_benchmark.models import ShelfAssociationRecord


class MyCustomShelfApproach(SimpleShelfApproachPlugin):
    @property
    def approach_id(self) -> str:
        return "my_custom_approach"

    @property
    def display_name(self) -> str:
        return "My Custom Approach (Stage 1 -> Stage 2)"

    @property
    def category(self) -> str:
        return "two_stage_vlm"

    @property
    def stages_description(self) -> List[str]:
        return [
            "Stage 1: Custom Detector + Front-Facing Depth NMS",
            "Stage 2: Custom Attribute Extraction + Rule-Derived Size Bucketing",
        ]

    def detect_and_classify(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
    ) -> List[Dict[str, Any]]:
        candidates = [
            {
                "bbox_2d": [100, 100, 300, 200],
                "brand": "Pond's",
                "product_name": "Pond's Bright Beauty Face Wash",
                "packaging_type": "tube",
                "size": "100g",
            }
        ]
        kept, filtered = ctx.deduplicate_depth_stacked_facings(candidates)
        if kept:
            kept[0]["_depth_filtered"] = filtered
        return kept
```

`SimpleShelfApproachPlugin` delegates to `ctx.finalize(...)` automatically to compute rule-derived size buckets, HUL brand attribution, 5-bucket GCP costs, ground-truth accuracy, OpenTelemetry spans (`reports/otel_logs.jsonl`), and UI execution traces.

### Shared `ctx: CommonLayerContext` Helpers
- `ctx.get_client()`: Returns the active GenAI client (honours offline test fakes, GEAP, Gemma, and Tuned Endpoints).
- `ctx.load_shelf_image(record)`: Loads the shelf image as a PIL `Image`.
- `ctx.deduplicate_depth_stacked_facings(items, x_overlap_threshold=None)`: Suppresses back-row stacked units using `config.depth_deduplication.x_overlap_threshold`.
- `ctx.crop_facing_images(shelf_image_uri, facings, model_tag, local_fallback=None)`: Crops each facing to `facing_XX.png` + `montage_all_facings.png`.
- `ctx.derive_size_bucket_from_bbox(bbox_2d, all_bboxes_on_shelf, packaging_type="tube", model_size_hint="")`: Rule-derived size bucket from `configs/taxonomy.yaml`.
- `ctx.check_is_hul_brand(brand_name)`: Portfolio lookup from `configs/taxonomy.yaml`.

---

## 3. Test Your Approach Offline in 3 Lines

```python
from shelf_benchmark.testing import run_offline_approach

res = run_offline_approach("/tmp/test-run", "my_custom_approach", with_ground_truth=True)
assert res.accuracy.detection_f1 == 1.0
print(res.execution_trace)
```
