"""Track 2 Merchandising: 3-Class Shelf-Talker & Price-Tag Detector (`talker_price_tag_detect`).

Localizes printed promotional shelf-talker cards (`promo_talker`), printed store rail price tags
(`printed_price_tag`), and handwritten marker price tags (`handwritten_price_tag`) on Kirana and
Modern Trade shelves (`hul_talkers_golden`, EXP-47).
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Box, Context, register, to_pixels
from utils import metrics
from utils.photometry import dual_buffer_shelf_photometry

TALKER_CLASSES = ("promo_talker", "printed_price_tag", "handwritten_price_tag", "not_a_tag")

TALKER_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {"type": "array", "items": {"type": "integer"}},
            "label": {"type": "string", "enum": list(TALKER_CLASSES)},
        },
        "required": ["box_2d", "label"],
    },
}

TALKER_PROMPT = (
    "Detect every promotional shelf-talker card and every price tag visible on the shelves in this "
    "retail store photo. Classify each box into one of three classes:\n"
    "1. 'promo_talker': printed promotional shelf-talker cards, danglers, or hanging offer wobblers.\n"
    "2. 'printed_price_tag': printed paper or plastic store price tags clipped to the shelf rail.\n"
    "3. 'handwritten_price_tag': handwritten marker or pen price tags / discount signs.\n"
    "4. 'not_a_tag': regular product packaging with no shelf-talker or price tag.\n"
    "Give each physical card or tag its own tight bounding box. "
    'Return ONLY a JSON array like [{"box_2d": [ymin, xmin, ymax, xmax], "label": "printed_price_tag"}] '
    "with coordinates normalized to 0-1000."
)

TALKER_MAX_OUTPUT_TOKENS = 8192
TALKER_NMS_IOU = 0.50


def parse_talker_boxes(
    raw: list | dict | None,
    w: float,
    h: float,
) -> tuple[list[Box], list[str]]:
    """Convert Gemini's 3-class talker/price-tag JSON array into pixel boxes and class labels."""
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    boxes: list[Box] = []
    labels: list[str] = []
    for item in raw or []:
        if not isinstance(item, (dict, list, tuple)):
            continue
        px = to_pixels([item], 0, 0, w, h)
        if not px:
            continue
        lab = str(item.get("label") or "printed_price_tag") if isinstance(item, dict) else "printed_price_tag"
        if lab == "not_a_tag":
            continue
        if lab not in TALKER_CLASSES:
            lab = "printed_price_tag"
        for b in px:
            boxes.append(b)
            labels.append(lab)
    return boxes, labels


@register
class TalkerPriceTagDetect(Approach):
    """EXP-47: 3-Class Promotional Shelf-Talker & Printed/Handwritten Price-Tag Detector."""

    name = "talker_price_tag_detect"
    use_case = "merchandising"
    task = "detection"
    dataset = "hul_talkers_golden"
    architecture = (
        "EXP-47: Stage 0 Dual-Buffer CLAHE -> Gemini 3-class detector (promo_talker, "
        "printed_price_tag, handwritten_price_tag, 8192 output tokens) -> class-wise NMS"
    )
    steps = [
        "Stage 0: Build dual photometric buffers (CIELAB L* CLAHE detector_image + linear color_image)",
        "Stage 1: Prompt Gemini on detector_image for promo_talker, printed_price_tag, and handwritten_price_tag boxes",
        f"Stage 1 post-filter: Class-wise greedy NMS at IoU {TALKER_NMS_IOU}",
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

        res = ctx.ask(
            det_image,
            TALKER_PROMPT,
            schema=TALKER_SCHEMA,
            max_side=2048,
            max_output_tokens=TALKER_MAX_OUTPUT_TOKENS,
        )
        raw_boxes, raw_labels = parse_talker_boxes(res.data, w, h)

        kept_boxes: list[Box] = []
        kept_labels: list[str] = []
        for cls_name in ("promo_talker", "printed_price_tag", "handwritten_price_tag"):
            cls_idxs = [i for i, lab in enumerate(raw_labels) if lab == cls_name]
            if not cls_idxs:
                continue
            cls_boxes = [raw_boxes[i] for i in cls_idxs]
            nms_idxs = metrics.nms(cls_boxes, thr=TALKER_NMS_IOU)
            for ni in nms_idxs:
                kept_boxes.append(cls_boxes[ni])
                kept_labels.append(cls_name)

        counts = {
            c: sum(1 for lab in kept_labels if lab == c)
            for c in ("promo_talker", "printed_price_tag", "handwritten_price_tag")
        }
        ctx.trace.step(
            "Gemini 3-class talker + price-tag detection",
            f"1 call, {res.seconds:.1f}s -> {len(kept_boxes)} tags "
            f"(talkers={counts['promo_talker']}, printed={counts['printed_price_tag']}, handwritten={counts['handwritten_price_tag']})",
            boxes=kept_boxes,
            info={"class_counts": counts, "raw_count": len(raw_boxes)},
        )
        return kept_boxes
