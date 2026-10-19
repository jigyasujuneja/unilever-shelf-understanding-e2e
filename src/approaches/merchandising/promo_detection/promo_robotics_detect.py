"""Track 2 Merchandising: Embodied Spatial Bay & SKU Detector (`promo_robotics_spatial_detect`, EXP-38).

Tests Gemini Robotics-ER (`gemini-robotics-er-2.0-preview`) embodied 2D/3D spatial pointing priors
on dense Kirana shelf localization, crowded sachet separation, and tilted (`>8°`) shelf-plank
Out-of-Stock void reasoning. Includes automatic runtime fallback to `gemini-2.5-pro` when the
preview model is not yet allowlisted in the active GCP project.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register
from approaches.market_share.detection.single_pass_dedup import NMS_IOU, dedup_traced
from approaches.merchandising.promo_detection.promo_asset_sku_detect import (
    DETECTION_MAX_OUTPUT_TOKENS,
    SCHEMA,
    parse_promo_and_sku_boxes,
)
from utils import metrics
from utils.photometry import dual_buffer_shelf_photometry

ROBOTICS_PRIMARY_MODEL = "gemini-robotics-er-2.0-preview"
ROBOTICS_FALLBACK_MODEL = "gemini-2.5-pro"

SPATIAL_PROMPT = (
    "Use 3D physical shelf-plane and object-boundary reasoning to detect every promotional display "
    "fixture and every individual retail product package in this store image.\n"
    "1. 'promo_asset': rigid branded promotional display bays, countertop display units, or wall "
    "display frames (Class 0 container assets). Trace the outer 3D frame boundary including top "
    "header sign and side wings, even when the camera is tilted (>8 degrees).\n"
    "2. 'product_sku': each individual physical 3D retail product package, bottle, carton, jar, "
    "tube, pouch, or hanging sachet (Class 1 SKUs). Separate touching or stacked identical sachets "
    "into individual tight boxes using 3D pack seams and physical depth cues.\n"
    "3. 'not_a_product': flat 2D printed graphics on cardboard display wings, price tags, or "
    "shelf-rail strips.\n"
    "List all 'promo_asset' boxes first, followed by every 'product_sku' box. "
    'Return ONLY a JSON array like [{"box_2d": [ymin, xmin, ymax, xmax], "label": "product_sku"}] '
    "with coordinates normalized to 0-1000."
)


def _is_preview_access_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(
        tok in msg
        for tok in ("404", "not found", "not_found", "403", "permission", "allowlist", "access")
    )


@register
class PromoRoboticsSpatialDetect(Approach):
    """EXP-38: Embodied 2D/3D Spatial Detector (gemini-robotics-er-2.0-preview with gemini-2.5-pro fallback)."""

    name = "promo_robotics_spatial_detect"
    use_case = "merchandising"
    task = "detection"
    dataset = "gtmerch_golden"
    architecture = (
        f"EXP-38 Embodied Spatial Detector ({ROBOTICS_PRIMARY_MODEL} with {ROBOTICS_FALLBACK_MODEL} "
        "fallback, 8192 output tokens) -> Stage 0 CLAHE + 3D seam sachet separation"
    )
    steps = [
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear color_image)",
        f"Stage 1: Prompt {ROBOTICS_PRIMARY_MODEL} (or fallback {ROBOTICS_FALLBACK_MODEL}) with 3D shelf-plane and pack-seam pointing rubric",
        f"Deduplicate Class 0 promo_asset bays (NMS IoU {NMS_IOU}) and clean Class 1 product_sku boxes",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        det_image, color_image, photo_stats = dual_buffer_shelf_photometry(image)
        ctx.detector_image = det_image
        ctx.color_image = color_image
        ctx.trace.step(
            "Stage 0 Dual-Buffer Photometry",
            f"Shades-of-Gray gains={photo_stats['gains']}, CIELAB L* CLAHE std {photo_stats['l_std_before']:.1f} -> {photo_stats['l_std_after']:.1f}",
            info=photo_stats,
        )

        active_model = ctx.model
        fallback_used = False
        try:
            res = ctx.ask(
                det_image,
                SPATIAL_PROMPT,
                schema=SCHEMA,
                max_side=2048,
                max_output_tokens=DETECTION_MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            if ROBOTICS_PRIMARY_MODEL in str(ctx.model) and _is_preview_access_error(exc):
                from utils.llm import Gemini

                fallback_used = True
                active_model = ROBOTICS_FALLBACK_MODEL
                prev_llm = ctx.llm
                prev_model = ctx.model
                try:
                    ctx.model = ROBOTICS_FALLBACK_MODEL
                    ctx.llm = Gemini(ROBOTICS_FALLBACK_MODEL)
                    res = ctx.ask(
                        det_image,
                        SPATIAL_PROMPT,
                        schema=SCHEMA,
                        max_side=2048,
                        max_output_tokens=DETECTION_MAX_OUTPUT_TOKENS,
                    )
                finally:
                    ctx.model = prev_model
                    ctx.llm = prev_llm
            else:
                raise

        raw_assets, raw_skus = parse_promo_and_sku_boxes(res.data, w, h)
        if raw_assets:
            kept_asset_idxs = metrics.nms(raw_assets, thr=NMS_IOU)
            ctx.detected_asset_boxes = [raw_assets[i] for i in kept_asset_idxs]
        else:
            ctx.detected_asset_boxes = []

        ctx.trace.step(
            "Embodied spatial promo asset + SKU detection",
            f"model={active_model} (fallback={fallback_used}), {res.seconds:.1f}s -> "
            f"{len(ctx.detected_asset_boxes)} Class 0 bays, {len(raw_skus)} Class 1 SKU boxes",
            boxes=raw_skus,
            info={
                "active_model": active_model,
                "fallback_used": fallback_used,
                "promo_asset_count": len(ctx.detected_asset_boxes),
                "raw_sku_count": len(raw_skus),
                "detected_asset_boxes": [
                    [round(v, 1) for v in b] for b in ctx.detected_asset_boxes
                ],
            },
        )
        return dedup_traced(raw_skus, ctx)
