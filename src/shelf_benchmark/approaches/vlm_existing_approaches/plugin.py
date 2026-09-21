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
        cls_task = ProductClassificationTask(ctx.config, ctx.storage, ctx.telemetry)
        detected_boxes = None
        if prior_detection is not None:
            detected_boxes = prior_detection.raw_output.get("detected_products")
        elif self._sep_approach in (
            "two_stage_bbox_guided_nms",
            "two_stage_physical_crop_per_facing",
        ):
            det_task = ProductDetectionTask(ctx.config, ctx.storage, ctx.telemetry)
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


def get_plugins() -> List[BaseShelfApproachPlugin]:
    return [
        ExistingVLMClassificationApproachPlugin(
            sep_approach="single_pass_full_shelf",
            title="Approach A: Single-Pass Full Shelf VLM (1 Call: BBoxes + 7-Dim HUL Taxonomy)",
            stages=[
                "Stage 1: Single VLM call extracts front-facing bboxes + 7-Dim HUL Taxonomy simultaneously",
                "Stage 2: Post-hoc Front-Facing Column Depth NMS (`deduplicate_depth_stacked_facings`)",
            ],
        ),
        ExistingVLMClassificationApproachPlugin(
            sep_approach="two_stage_bbox_guided_nms",
            title="Approach B: Two-Stage BBox-Guided NMS (Stage 1 Depth NMS -> Stage 2 Coordinate Prompt)",
            stages=[
                "Stage 1: VLM Detection + Front-Facing Column Depth NMS locks coordinates",
                "Stage 2: Full shelf image + locked bbox coordinates prompt for 7-Dim HUL Taxonomy",
            ],
        ),
        ExistingVLMClassificationApproachPlugin(
            sep_approach="two_stage_physical_crop_per_facing",
            title="Approach C: Two-Stage Physical Crop per Facing (Stage 1 Depth NMS -> PIL Crops + Montage -> Stage 2 VLM)",
            stages=[
                "Stage 1: VLM Detection + Front-Facing Column Depth NMS",
                "Stage 2: Physical PIL cropping per facing (`facing_XX.png`) + numbered montage (`montage_all_facings.png`)",
                "Stage 3: VLM fine-print reading on physical crops for 7-Dim HUL Taxonomy",
            ],
        ),
    ]
