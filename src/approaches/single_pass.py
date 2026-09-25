"""Baseline: one Gemini call on the whole shelf image that detects AND classifies every product."""

from __future__ import annotations

from PIL import Image

from approaches.base import (
    CATEGORIES,
    NOT_PRODUCT,
    Approach,
    Box,
    Context,
    label_counts,
    register,
    to_pixels,
)

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {"type": "array", "items": {"type": "integer"}},
            "label": {"type": "string", "enum": CATEGORIES},
        },
        "required": ["box_2d", "label"],
    },
}

PROMPT = (
    "Detect every individual retail product visible on the shelves in this image - every "
    "box, bottle, can, bag and package, including small, partially visible and edge-of-frame "
    "items. Each physical item gets its own tight box. Do not merge adjacent identical products. "
    f"Classify each one as one of: {', '.join(CATEGORIES)} (not_a_product = price tag, "
    "shelf edge or other non-product). "
    'Return ONLY a JSON array like [{"box_2d": [ymin, xmin, ymax, xmax], "label": "food"}] '
    "with boxes normalized to 0-1000."
)


def labelled_boxes(raw, w: float, h: float) -> tuple[list[Box], list[str]]:
    """Gemini's [{box_2d, label}] -> pixel boxes + labels (missing labels -> other_product)."""
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    boxes, labels = [], []
    for item in raw or []:
        b = to_pixels([item], 0, 0, w, h)
        if b:
            boxes += b
            lab = item.get("label") if isinstance(item, dict) else None
            labels.append(lab if lab in CATEGORIES else "other_product")
    return boxes, labels


@register
class SinglePass(Approach):
    name = "single_pass"
    architecture = "Gemini, one call detects and classifies every product"
    steps = [
        "Downscale the image (longest side 2048 px)",
        "One Gemini call returns every product box with its class",
        "Map boxes back to original pixels, drop any labelled not_a_product",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        res = ctx.ask(image, PROMPT, schema=SCHEMA, max_side=2048)
        boxes, labels = labelled_boxes(res.data, w, h)
        kept = [b for b, n in zip(boxes, labels, strict=True) if n != NOT_PRODUCT]
        ctx.trace.step(
            "Gemini detection + classification",
            f"1 call, {res.usage.input_tokens} in / {res.usage.output_tokens} out tokens, "
            f"{res.seconds:.1f}s -> {len(kept)} products ({label_counts(labels) or 'none'})",
            boxes=kept,
        )
        return kept
