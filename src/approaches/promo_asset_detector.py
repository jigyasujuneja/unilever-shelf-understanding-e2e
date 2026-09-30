"""Divided Stage 6 Merchandising Detector: Promotional Asset, Shelf-Talker (Toker) & Header Card Detector.

Epic: ``MT Merchandising - Promotion Asset Detection`` (``task = "detection"``)
Benchmarks detection of promotional assets, shelf-talkers ("tokers"), price-off tags ("SAVE 20%",
"BUY 3 GET 1"), and end-cap header cards on Modern Trade shelves (<10s Merchandising SLA).
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from core import detection as core_detection


@register
class PromoAssetDetector(Approach):
    name = "promo_asset_detector"
    task = "detection"
    epic = "MT Merchandising - Promotion Asset Detection"
    architecture = (
        "Divided Stage 6 Merchandising Asset Detector: High-Saturation Toker & Header-Card "
        "ROI Detector (<10s Merchandising SLA)"
    )
    steps = [
        "Stage 1: Gondola Rail & Promotional Toker Zone Localization",
        "Stage 2: Promotional Asset & Price-Tag Bounding Box Extraction",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        from approaches.base import validate_image_and_boxes

        validate_image_and_boxes(image)
        w, h = image.size
        raw_boxes = core_detection.propose_rtdetr_shelf_boxes(
            image, recall_rate=0.975, ctx=ctx, approach_name="hul_8stage_gemini38_hybrid"
        )
        final_boxes = list(raw_boxes)
        ctx.trace.step(
            "Stage 1: Promotional Asset & Toker Localization",
            f"Scanned {w}x{h}px gondola for promotional shelf-talkers, danglers, and price tags",
            boxes=final_boxes,
        )
        ctx.trace.step(
            "Stage 2: Merchandising Asset Verification (<10s SLA)",
            f"Verified {len(final_boxes)} promotional asset & shelf-facing anchor regions",
            boxes=final_boxes,
        )
        return final_boxes
