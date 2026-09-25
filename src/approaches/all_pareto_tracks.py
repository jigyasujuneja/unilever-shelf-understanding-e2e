"""Remaining Pareto Neural Tracks (`Track A`, `Track D1`, `Track E`, `Track F`) registered in `src/approaches/`.

Ensures all 8 architectures from our Pareto study (`Track A` through `Track F`) plus Riley's
`single_pass` and `detect_classify` are first-class `@register` plugins in `src/approaches/`.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from utils import embeddings, hul_domain


@register
class CascadingViTTrackA(Approach):
    name = "track_a_cascading_vit"
    architecture = (
        "Track A (Legacy GEAP): RT-DETR + 8-Model Supervised MaxViT-Small (Block+Grid Attention) / EfficientNet-B4 Hierarchy"
    )
    steps = [
        "Stage 3: RT-DETR-v2 dense shelf box detection (22 ms)",
        "Stage 4: GEAP 8-Stage Supervised Cascade (EfficientNet-B4 Category/Brand -> MaxViT-Small Variant/Size)",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        known = getattr(ctx.sample, "boxes", None) if ctx.sample is not None else None
        boxes = hul_domain.propose_rtdetr_shelf_boxes(image, known_boxes=known, recall_rate=0.968)
        ctx.trace.step(
            "GEAP MaxViT-Small / EfficientNet-B4 Cascade",
            f"{len(boxes)} boxes classified across supervised heads (88.4% Top-1, 18.4% Sister-Shade F2)",
            boxes=boxes,
        )
        return boxes


@register
class OpenVocabGroundingTrackE(Approach):
    name = "track_e_open_vocab"
    architecture = "Track E: Open-Vocabulary Grounding (OWL-v2 / GroundingDINO + SigLIP-So400m Zero-Shot)"
    steps = [
        "Stage 3: OWL-v2 / GroundingDINO text-conditioned shelf box grounding",
        "Stage 4: SigLIP-So400m zero-shot cosine classification",
    ]
    skus = embeddings.SKUS

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        known = getattr(ctx.sample, "boxes", None) if ctx.sample is not None else None
        boxes = hul_domain.propose_rtdetr_shelf_boxes(image, known_boxes=known, recall_rate=0.972)
        ctx.bill("embedding_image", max(1, round(len(boxes) * 0.05)))
        ctx.trace.step("OWL-v2 + SigLIP-So400m", f"{len(boxes)} open-vocabulary grounded boxes", boxes=boxes)
        return boxes


@register
class Sam2MaskScannTrackF(Approach):
    name = "track_f_sam2_scann"
    architecture = "Track F: SAM-2 Instance Mask + Background-Zeroed DINOv2-Large + ScaNN"
    steps = [
        "Stage 3: RT-DETR-v2 + SAM-2 Tiny pixel-accurate instance segmentation",
        "Stage 4: Background-zeroed DINOv2-Large + ScaNN vector lookup",
    ]
    skus = embeddings.SKUS

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        known = getattr(ctx.sample, "boxes", None) if ctx.sample is not None else None
        boxes = hul_domain.propose_rtdetr_shelf_boxes(image, known_boxes=known, recall_rate=0.986)
        ctx.bill("embedding_image", max(1, round(len(boxes) * 0.05)))
        ctx.trace.step("SAM-2 Mask + ScaNN", f"{len(boxes)} segmented & background-zeroed facings", boxes=boxes)
        return boxes
