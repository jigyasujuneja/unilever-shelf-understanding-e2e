"""Two passes: Gemini detects product boxes, then a second Gemini pass classifies each box.

Pass 2 crops every box, lays the crops out as numbered contact sheets (48 per sheet) and asks
Gemini what each crop is. Boxes classified as ``not_a_product`` (shelf edges, price tags,
gaps, background) are dropped, so pass 2 can only improve precision. SKU-110K has a single
"object" class, so the category labels are recorded for inspection but not scored.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from approaches.base import (
    BOX_LIST_SCHEMA,
    CATEGORIES,
    DETECT_PROMPT,
    Approach,
    Box,
    Context,
    label_counts,
    register,
    to_pixels,
)

CELL, COLS, ROWS = 160, 8, 6  # contact sheet: 8x6 crops of 160 px -> 1280x960 image
PER_SHEET = COLS * ROWS

CLASSIFY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "label": {"type": "string", "enum": CATEGORIES}},
        "required": ["id", "label"],
    },
}

CLASSIFY_PROMPT = (
    "This is a contact sheet of numbered crops from a retail shelf photo. Each crop should "
    "contain exactly one product. For every numbered crop, classify what it shows: "
    f"one of {', '.join(CATEGORIES)}. Use not_a_product when the crop is mainly a shelf edge, "
    "price tag, empty space, background, or only a sliver of a product. Return ONLY a JSON "
    'array like [{"id": 0, "label": "food"}], one entry per crop.'
)


def contact_sheet(image: Image.Image, boxes: list[Box], start: int) -> Image.Image:
    """Crops of ``boxes`` (padded 10%) in a numbered grid; numbering starts at ``start``."""
    sheet = Image.new("RGB", (COLS * CELL, ROWS * CELL), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        px, py = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        crop = image.crop((max(0, x1 - px), max(0, y1 - py),
                           min(image.width, x2 + px), min(image.height, y2 + py)))
        crop.thumbnail((CELL - 8, CELL - 8))
        cx, cy = (k % COLS) * CELL, (k // COLS) * CELL
        sheet.paste(crop, (cx + (CELL - crop.width) // 2, cy + (CELL - crop.height) // 2))
        draw.rectangle((cx, cy, cx + CELL - 1, cy + CELL - 1), outline="#999")
        draw.rectangle((cx, cy, cx + 34, cy + 22), fill="black")
        draw.text((cx + 3, cy + 1), str(start + k), fill="yellow", font=font)
    return sheet


@register
class DetectClassify(Approach):
    name = "detect_classify"
    architecture = "Gemini, pass 1 detects boxes, pass 2 classifies each box"
    steps = [
        "Pass 1: one Gemini call on the full image returns every product box",
        "Pass 2: crop each box onto numbered contact sheets, Gemini labels every crop",
        "Drop boxes labelled not_a_product",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        res = ctx.ask(image, DETECT_PROMPT, schema=BOX_LIST_SCHEMA, max_side=2048)
        boxes = to_pixels(res.data, 0, 0, w, h)
        ctx.trace.step("Pass 1: detection",
                       f"1 call, {res.seconds:.1f}s -> {len(boxes)} boxes", boxes=boxes)
        if not boxes:
            return []

        starts = list(range(0, len(boxes), PER_SHEET))

        def classify(start: int) -> dict[int, str]:
            sheet = contact_sheet(image, boxes[start:start + PER_SHEET], start)
            out = ctx.ask(sheet, CLASSIFY_PROMPT, schema=CLASSIFY_SCHEMA).data
            return {int(d["id"]): d["label"] for d in out or []
                    if isinstance(d, dict) and "id" in d and d.get("label") in CATEGORIES}

        labels: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(starts)) as pool:
            for part in pool.map(classify, starts):
                labels.update(part)

        # A crop the model skipped keeps its box (treated as a product).
        names = [labels.get(i, "other_product") for i in range(len(boxes))]
        kept = [b for b, n in zip(boxes, names, strict=True) if n != "not_a_product"]
        ctx.trace.step("Pass 2: classification",
                       f"{len(starts)} contact sheet call(s) -> {label_counts(names)}; "
                       f"dropped {len(boxes) - len(kept)} -> {len(kept)} boxes",
                       boxes=kept)
        return kept
