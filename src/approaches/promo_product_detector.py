"""Divided Stage 6 Merchandising Combined Pipeline: Promotion Product Detection & Promo-Compliance Audit.

Epic: ``MT Merchandising - Promotion Product Detection`` (``task = "combined"``)
Detects promotional SKU facings and classifies their 7-Dim HUL attributes + promotional compliance
("SAVE 20%", "BUY 3 GET 1", "FREE CONDITIONER") under the <10s Merchandising SLA.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from utils import hul_domain


@register
class PromoProductDetector(Approach):
    name = "promo_product_detector"
    task = "combined"
    epic = "MT Merchandising - Promotion Product Detection"
    target_field = "variant"
    architecture = (
        "Divided Stage 6 Merchandising Product Pipeline: RT-DETR-v2 Promo Facing Detector "
        "+ /v1/systemone Promotional Pack & Toker Compliance Classifier (<10s SLA)"
    )
    steps = [
        "Stage 1: Promotional Facing & End-Cap Display Detection",
        "Stage 2: Promotional SKU & Offer-Claim Classification",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        boxes, _ = self.detect_and_classify(image, ctx)
        return boxes

    def detect_and_classify(
        self, image: Image.Image, ctx: Context
    ) -> tuple[list[Box], list[Any]]:
        raw_boxes = hul_domain.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.985, ctx=ctx, approach_name="hul_8stage_gemini38_hybrid"
        )
        boxes = list(raw_boxes)
        ctx.trace.step(
            "Stage 1: Promotional Facing Detection",
            f"Detected {len(boxes)} promotional display & shelf facings",
            boxes=boxes,
        )
        preds = hul_domain.classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, mode=self.name
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Stage 2: Promotional SKU & Offer-Claim Classification",
            f"Classified {len(preds)} promotional products ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        return boxes, preds
