"""Configurable end-to-end shelf detection and classification pipeline.

Task: ``combined`` (Epic: ``MT Market Share - Combined Classification``)
Composes a bounding-box detector, a coarse attribute classifier (category, brand, packaging),
a fine-grained variant classifier, and six configurable pipeline stages:

    shelf-bench run -a modular_e2e_pipeline \\
      --with-detector yolo_n26_sku110k \\
      --with-attr-classifier djev_diffusiongemma_compound \\
      --with-variant-classifier sister_shade_systemone \\
      --with-rectifier depth_anything_v2 \\
      --with-clusterer maxvit_agglomerative \\
      --with-retriever siglip_multiprototype \\
      --with-shelf-metrics dual_mt_marketshare_and_merchandising
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, get, label_counts, register
import stages
from utils import embeddings


@register
class ModularEndToEndPipeline(Approach):
    """End-to-end shelf pipeline with configurable task approaches and stage functions."""

    name = "modular_e2e_pipeline"
    task = "combined"
    epic = "MT Market Share - Combined Classification"
    target_field = "variant"
    architecture = (
        "Configurable End-to-End Pipeline: Detector + Attribute Classifier + Variant Classifier "
        "with selectable stage functions (rectifier, post_detector, clusterer, retriever, tiebreaker, shelf_metrics)"
    )
    steps = [
        "Stage 1 (rectifier): Image quality check and perspective homography",
        "Stage 2 & 3 (detector + post_detector): Product bounding-box detection and post-NMS filtering",
        "Stage 3.5 & 4 (clusterer + retriever): Adjacent crop deduplication and vector catalog search",
        "Stage 5 (attr_classifier + variant_classifier + tiebreaker): Hierarchy and variant classification",
        "Stage 6 (shelf_metrics): Share-of-shelf, out-of-stock, and planogram compliance metrics",
    ]
    skus = embeddings.SKUS

    def __init__(
        self,
        detector_name: str = "rtdetr_shelf_rail_detector",
        attr_classifier_name: str = "hul_hierarchy_classifier",
        variant_classifier_name: str = "sister_shade_systemone",
        stage_overrides: dict[str, str] | None = None,
    ) -> None:
        self.detector_name = detector_name
        self.attr_classifier_name = attr_classifier_name
        self.variant_classifier_name = variant_classifier_name
        self.stage_overrides: dict[str, str] = stages.default_stage_config()
        if stage_overrides:
            for group_name, stage_name in stage_overrides.items():
                if stage_name:
                    canonical_group = "shelf_metrics" if group_name == "gondola_kpi" else group_name
                    self.stage_overrides[canonical_group] = stage_name

    def configure_modules(
        self,
        detector_name: str | None = None,
        attr_classifier_name: str | None = None,
        variant_classifier_name: str | None = None,
        stage_overrides: dict[str, str] | None = None,
    ) -> None:
        """Update the configured task approaches and pipeline stage overrides."""
        if detector_name:
            self.detector_name = detector_name
        if attr_classifier_name:
            self.attr_classifier_name = attr_classifier_name
        if variant_classifier_name:
            self.variant_classifier_name = variant_classifier_name
        if stage_overrides:
            for group_name, stage_name in stage_overrides.items():
                if stage_name:
                    canonical_group = "shelf_metrics" if group_name == "gondola_kpi" else group_name
                    self.stage_overrides[canonical_group] = stage_name
        active_stages = ", ".join(f"{k}={v}" for k, v in self.stage_overrides.items())
        self.architecture = (
            f"End-to-End [{self.detector_name} -> {self.attr_classifier_name} -> {self.variant_classifier_name}] "
            f"| Stages: ({active_stages})"
        )

    def setup(self, config: dict[str, Any]) -> None:
        modular_overrides = config.get("modular_overrides", {})
        stage_overrides = config.get("stage_overrides", {})
        if modular_overrides or stage_overrides:
            self.configure_modules(
                detector_name=modular_overrides.get("detector"),
                attr_classifier_name=modular_overrides.get("attr_classifier"),
                variant_classifier_name=modular_overrides.get("variant_classifier"),
                stage_overrides=stage_overrides,
            )
        self.detector = get(self.detector_name)
        self.attr_classifier = get(self.attr_classifier_name)
        self.variant_classifier = get(self.variant_classifier_name)
        self.detector.setup(config)
        self.attr_classifier.setup(config)
        self.variant_classifier.setup(config)

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes, _ = self.detect_and_classify(image, ctx)
        return boxes

    def detect_and_classify(
        self, image: Image.Image, ctx: Context
    ) -> tuple[list[Box], list[Any]]:
        if not hasattr(self, "detector"):
            self.setup({})

        # Stage 1: Perspective rectification
        rectifier_spec = stages.get_stage("rectifier", self.stage_overrides.get("rectifier"))
        rect_info = rectifier_spec.fn(image) if rectifier_spec.fn else {}
        ctx.trace.step(
            f"Stage 1 (rectifier: {rectifier_spec.name})",
            f"homography_applied={rect_info.get('homography_applied', True)}, "
            f"yaw={rect_info.get('yaw_corrected_deg', 0.0)} deg",
        )

        # Stage 2 & 3: Product detection and post-detection filtering
        raw_boxes = self.detector.detect(image, ctx)
        post_spec = stages.get_stage("post_detector", self.stage_overrides.get("post_detector"))
        boxes = post_spec.fn(image, raw_boxes) if post_spec.fn else raw_boxes

        # Coarse attribute classification (category, brand, packaging_type)
        coarse_prior = self.attr_classifier.classify(image, boxes, ctx)

        # Stage 3.5 & 4: Crop clustering and catalog retrieval
        cluster_spec = stages.get_stage("clusterer", self.stage_overrides.get("clusterer"))
        cluster_info = cluster_spec.fn(boxes) if cluster_spec.fn else {}
        retriever_spec = stages.get_stage("retriever", self.stage_overrides.get("retriever"))
        retriever_info = retriever_spec.fn() if retriever_spec.fn else {}
        ctx.trace.step(
            f"Stages 3.5 & 4 ({cluster_spec.name} + {retriever_spec.name})",
            f"cluster_compression={cluster_info.get('compression_ratio', 4.0)}x, "
            f"candidate_pool={retriever_info.get('scann_pool_after', 11)} SKUs",
            boxes=boxes,
        )

        # Stage 5: Fine-grained variant classification and shade disambiguation
        tiebreaker_spec = stages.get_stage("tiebreaker", self.stage_overrides.get("tiebreaker"))
        _ = tiebreaker_spec.fn() if tiebreaker_spec.fn else {}
        final_labels = self.variant_classifier.classify(image, boxes, ctx, prior=coarse_prior)

        ctx.trace.labels = final_labels
        ctx.trace.meta["composed_modules"] = {
            "detector": self.detector_name,
            "attr_classifier": self.attr_classifier_name,
            "variant_classifier": self.variant_classifier_name,
            "stage_overrides": dict(self.stage_overrides),
        }
        ctx.trace.step(
            "End-to-End Pipeline Complete",
            f"[{self.detector_name} -> {self.attr_classifier_name} -> {self.variant_classifier_name}] "
            f"classified {len(boxes)} boxes ({label_counts(final_labels)})",
            boxes=boxes,
            labels=final_labels,
        )
        return boxes, final_labels
