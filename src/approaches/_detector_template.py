"""TEMPLATE (not registered: files starting with ``_`` are skipped).
Copy to ``src/approaches/my_detector.py`` to benchmark a new SKU or Promo Asset detector.

Run locally on validation (25 images) or on Cloud Run (50 test images):
    shelf-bench run -a my_detector -m gemini-3.5-flash-lite --split val --limit 25 --owner riley
    shelf-bench cloud-run -a my_detector -m gemini-3.5-flash-lite --owner riley
"""

from __future__ import annotations

from PIL import Image

from approaches.base import (
    BOX_LIST_SCHEMA,
    DETECT_PROMPT,
    Approach,
    Box,
    Context,
    register,
    to_pixels,
)
from utils import metrics


@register
class MyDetector(Approach):
    name = "my_detector"
    task = "detection"
    epic = "MT Market Share - SKU Detection"  # or "MT Merchandising - Promotion Asset Detection"
    architecture = "Describe your detector (e.g. YOLO-N26 / Gemini 2 Robotics / custom tiling)"
    steps = [
        "Propose candidate bounding boxes on the shelf image",
        "Apply Non-Maximum Suppression (NMS) and return [x1, y1, x2, y2] boxes",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        res = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=2048)
        boxes = to_pixels(res.data, 0, 0, w, h)
        kept = [boxes[i] for i in metrics.nms(boxes, thr=0.5)]
        ctx.trace.step("Detect products", f"{len(kept)} boxes after NMS", boxes=kept)
        return kept
