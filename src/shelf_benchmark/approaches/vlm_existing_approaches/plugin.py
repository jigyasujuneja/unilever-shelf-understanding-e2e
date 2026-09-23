"""Plugin wrappers for the 3 existing VLM approaches (delegating 100% to untouched `ProductClassificationTask`)."""

from __future__ import annotations

from typing import List, Optional

from shelf_benchmark.approaches.base import BaseShelfApproachPlugin, CommonLayerContext
from shelf_benchmark.models import (
    ImageGroundTruth,
    ShelfAssociationRecord,
    TaskExecutionResult,
)
from shelf_benchmark.tasks.classification import ProductClassificationTask
from shelf_benchmark.tasks.detection import ProductDetectionTask


class ExistingVLMClassificationApproachPlugin(BaseShelfApproachPlugin):
    """Wraps an existing `separation_approach` in `ProductClassificationTask` without modifying it."""

    def __init__(
        self,
        sep_approach: str,
        title: str,
        stages: List[str],
    ) -> None:
        self._sep_approach = sep_approach
        self._title = title
        self._stages = stages

    @property
    def approach_id(self) -> str:
        return self._sep_approach

    @property
    def display_name(self) -> str:
        return self._title

    @property
    def category(self) -> str:
        return "vlm_multimodal"

    @property
    def stages_description(self) -> List[str]:
        return self._stages

    def execute(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
        gt_record: Optional[ImageGroundTruth] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        cls_task = ProductClassificationTask(
            ctx.config, ctx.storage, ctx.telemetry, genai_client=ctx.genai_client
        )
        detected_boxes = None
        if prior_detection is not None:
            detected_boxes = prior_detection.raw_output.get("detected_products")
        elif self._sep_approach in (
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
        ):
            det_task = ProductDetectionTask(
                ctx.config, ctx.storage, ctx.telemetry, genai_client=ctx.genai_client
            )
            det_res = det_task.execute(
                model_name=model_name,
                shelf_image_uri=record.shelf_image_uri,
                run_id=f"det-{model_name}",
                store_id=record.store_id,
                ground_truth=gt_record,
            )
            detected_boxes = det_res.raw_output.get("detected_products")
        return cls_task.execute(
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            run_id=f"cls-{self._sep_approach[:9]}-{model_name}",
            store_id=record.store_id,
            ground_truth=gt_record,
            separation_approach=self._sep_approach,
            detected_boxes=detected_boxes,
        )


class SingleStepDetectClassifyAndMatchPlugin(BaseShelfApproachPlugin):
    """Single-step unified VLM approach that performs Detection + 7-Dimension Classification + Hybrid SKU Matching in 1 API call."""

    @property
    def approach_id(self) -> str:
        return "single_step_detect_classify_and_match"

    @property
    def display_name(self) -> str:
        return "Single-Step End-to-End VLM (1 Call: Detect BBoxes + 7-Dim Classify + Hybrid SKU Match)"

    @property
    def category(self) -> str:
        return "vlm_multimodal"

    @property
    def stages_description(self) -> List[str]:
        return [
            "Stage 1 (Single Step): 1 VLM call simultaneously detects front-facing [ymin,xmin,ymax,xmax] boxes, classifies all 7 taxonomy dimensions, and emits BM25 + dense SKU matching passages",
            "Stage 2 (Post-Processing): Geometric Front-Facing Column Depth NMS + rule-derived size bucketing",
        ]

    def execute(
        self,
        ctx: CommonLayerContext,
        model_name: str,
        record: ShelfAssociationRecord,
        gt_record: Optional[ImageGroundTruth] = None,
        prior_detection: Optional[TaskExecutionResult] = None,
    ) -> TaskExecutionResult:
        from shelf_benchmark.tasks.matching import ProductMatchingTask

        mat_task = ProductMatchingTask(
            ctx.config, ctx.storage, ctx.telemetry, genai_client=ctx.genai_client
        )
        res = mat_task.execute(
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            run_id=f"single-step-{model_name}",
            store_id=record.store_id,
            ground_truth=gt_record,
            catalog_uri=record.catalog_uri,
            planogram_uri=record.planogram_uri,
            separation_approach=self.approach_id,
        )
        res.task_type = "classification"
        res.separation_approach = self.approach_id
        for r in res.row_level_items:
            r.task_type = "classification"
            r.separation_approach = self.approach_id
        if "execution_trace" in res.raw_output and isinstance(res.raw_output["execution_trace"], dict):
            res.raw_output["execution_trace"]["task_type"] = "classification"
            res.raw_output["execution_trace"]["separation_approach"] = self.approach_id
        return res


def get_plugins() -> List[BaseShelfApproachPlugin]:
    return [
        ExistingVLMClassificationApproachPlugin(
            sep_approach="single_pass_full_shelf",
            title="Single-Step Open-Vocab VLM (1 Call: BBoxes + LLM-Generated Brand & N-Dim Taxonomy)",
            stages=[
                "Stage 1: Single VLM call localizes front-facing bboxes and generates brand/attributes openly from package text",
                "Stage 2: Post-hoc Front-Facing Column Depth NMS (`deduplicate_depth_stacked_facings`)",
            ],
        ),
        ExistingVLMClassificationApproachPlugin(
            sep_approach="open_vocab_brand_plus_catalog_resolver",
            title="Open-Vocabulary Brand Generation + O(1) 2,000-Brand Master Catalog Resolver",
            stages=[
                "Stage 1: VLM generates brand and N-Dim attributes openly (zero brand list in prompt; scales to 2,000+ brands)",
                "Stage 2: Post-hoc O(1) normalized catalog resolver (`resolve_brand_against_catalog`) snaps generated brand to master catalog + Depth NMS",
            ],
        ),
        ExistingVLMClassificationApproachPlugin(
            sep_approach="configurable_multi_attribute_vlm",
            title="Configurable N-Attribute VLM (1 Call for >8 Attributes or Grouped Multi-Call by Attribute Type)",
            stages=[
                "Stage 1: Extract core + custom attributes (`taxonomy.custom_attributes`) in 1 VLM call or grouped via `taxonomy.attribute_call_groups`",
                "Stage 2: Merge attribute groups per facing (`extra_attributes`) + Front-Facing Column Depth NMS",
            ],
        ),
        SingleStepDetectClassifyAndMatchPlugin(),
        ExistingVLMClassificationApproachPlugin(
            sep_approach="two_stage_bbox_guided_nms",
            title="Two-Stage BBox-Guided NMS (Stage 1 Depth NMS -> Stage 2 Coordinate Prompt)",
            stages=[
                "Stage 1: VLM Detection + Front-Facing Column Depth NMS locks coordinates",
                "Stage 2: Full shelf image + locked bbox coordinates prompt for N-Dim Taxonomy",
            ],
        ),
        ExistingVLMClassificationApproachPlugin(
            sep_approach="two_stage_physical_crop_per_facing",
            title="Two-Stage Physical Crop per Facing (Stage 1 Depth NMS -> PIL Crops + Montage -> Stage 2 VLM)",
            stages=[
                "Stage 1: VLM Detection + Front-Facing Column Depth NMS",
                "Stage 2: Physical PIL cropping per facing (`facing_XX.png`) + numbered montage (`montage_all_facings.png`)",
                "Stage 3: VLM fine-print reading on physical crops for N-Dim Taxonomy",
            ],
        ),
    ]
