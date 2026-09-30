"""One Gemini call detects and classifies every product, then geometric clean-up (Detection tab).

One call on the whole (downscaled) shelf returns every box with a coarse class; boxes labelled
``not_a_product`` are dropped. Then the post-filters ported from Jigyasu's detection stages
(``hul_domain`` container suppression, ``shelf_e2e.geometry`` depth-ghost suppression) remove
duplicate boxes without another call:

1. Container boxes: a box that has a much smaller box (< 65% of its area) inside it is usually a
   whole group or shelf section, so it is dropped, unless it is tall and narrow (w/h <= 0.72,
   e.g. a bottle with its label boxed separately).
2. Greedy NMS at IoU 0.58.
3. Depth ghosts: a smaller box at about the same height and mostly the same columns as a
   bigger box nearer the front (the product behind it, or a double detection) is dropped.

The plain call without these filters (the former ``single_pass`` approach) scored lower F2 at the
same cost, so only this version is kept.
"""

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
from utils import metrics

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

CONTAINED_AREA = 0.65  # a box inside another is "much smaller" below this area ratio
CONTAINED_SLACK = 8.0  # px a contained box may stick out of its container
TALL_ASPECT = 0.72     # containers with w/h at most this are kept (bottles, tall packs)
NMS_IOU = 0.58
GHOST_X_OVERLAP = 0.55  # share of the narrower box's width that overlaps
GHOST_Y_CENTRE = 0.65   # centre-height difference, as a share of the bigger box's height
GHOST_AREA = 0.92       # the ghost is at most this share of the bigger box's area


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def drop_containers(boxes: list[Box]) -> list[Box]:
    def contains(b: Box, c: Box) -> bool:
        s = CONTAINED_SLACK
        return (_area(c) < CONTAINED_AREA * _area(b) and c[0] >= b[0] - s and c[1] >= b[1] - s
                and c[2] <= b[2] + s and c[3] <= b[3] + s)

    def tall(b: Box) -> bool:
        return (b[2] - b[0]) <= TALL_ASPECT * (b[3] - b[1])

    return [b for i, b in enumerate(boxes)
            if tall(b) or not any(contains(b, c) for j, c in enumerate(boxes) if j != i)]


def drop_depth_ghosts(boxes: list[Box]) -> list[Box]:
    """Keep front boxes first (lowest bottom edge, then biggest); drop smaller ones behind them."""
    kept: list[Box] = []
    for c in sorted(boxes, key=lambda b: (b[3], _area(b)), reverse=True):
        def ghost(f: Box, c: Box = c) -> bool:
            x_overlap = max(0.0, min(c[2], f[2]) - max(c[0], f[0]))
            narrower = min(c[2] - c[0], f[2] - f[0]) or 1.0
            dy = abs((c[1] + c[3]) / 2 - (f[1] + f[3]) / 2)
            return (x_overlap / narrower >= GHOST_X_OVERLAP and dy <= GHOST_Y_CENTRE * (f[3] - f[1])
                    and _area(c) <= GHOST_AREA * _area(f))

        if not any(ghost(f) for f in kept):
            kept.append(c)
    return kept


def dedup(boxes: list[Box]) -> tuple[list[Box], list[Box], list[Box]]:
    """(after container rule, after NMS, after depth-ghost rule)."""
    a = drop_containers(boxes)
    b = [a[i] for i in metrics.nms(a, thr=NMS_IOU)]
    return a, b, drop_depth_ghosts(b)


def dedup_traced(boxes: list[Box], ctx: Context) -> list[Box]:
    """``dedup`` + one trace step per filter that removed something (no-op filters stay hidden)."""
    a, b, kept = dedup(boxes)
    for name, before, after in (("Container boxes removed", boxes, a),
                                (f"Duplicates removed (NMS at IoU {NMS_IOU})", a, b),
                                ("Depth ghosts removed", b, kept)):
        if len(after) < len(before):
            left = {tuple(x) for x in after}
            gone = [x for x in before if tuple(x) not in left]
            ctx.trace.step(name, f"{len(before) - len(after)} removed -> {len(after)}", boxes=after,
                           info={"removed": [[round(v, 1) for v in x] for x in gone]})
    return kept


@register
class SinglePassDedup(Approach):
    name = "single_pass_dedup"
    architecture = "Gemini, one call detects every product -> geometric duplicate removal"
    steps = [
        "Downscale the image (longest side 2048 px); one Gemini call returns every product box",
        f"Drop container boxes (hold a box < {CONTAINED_AREA:.0%} of their area), unless tall",
        f"Greedy NMS at IoU {NMS_IOU}",
        "Drop depth ghosts: smaller boxes at the same height, behind a bigger front box",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        res = ctx.ask(image, PROMPT, schema=SCHEMA, max_side=2048)
        boxes, labels = labelled_boxes(res.data, w, h)
        raw = [b for b, n in zip(boxes, labels, strict=True) if n != NOT_PRODUCT]
        ctx.trace.step("Gemini detection + classification",
                       f"1 call, {res.seconds:.1f}s -> {len(raw)} products "
                       f"({label_counts(labels) or 'none'})", boxes=raw)
        return dedup_traced(raw, ctx)
