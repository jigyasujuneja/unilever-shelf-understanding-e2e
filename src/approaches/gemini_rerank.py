"""Embedding shortlist, then Gemini picks the product (the Retrieval tab).

Embeddings are cheap and good at "roughly this kind of pack"; Gemini is better at reading the
label (brand, flavour, size). So: take the 5 products whose reference photos are closest to the
crop (``embedding_retrieval``), put the crop and one reference photo of each candidate on a
numbered sheet, and ask Gemini which candidate it is, or none. One Gemini call per crop.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from approaches.base import Context, register
from approaches.embedding_retrieval import MODEL, EmbeddingRetrieval

K = 5       # candidates shown to Gemini
CELL = 320  # px per image on the sheet

SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]}
PROMPT = (
    f"The image is a row of {K + 1} panels. Panel Q is one product cut out of a checkout-counter "
    f"photo. Panels 1-{K} are reference photos of {K} different products from the catalog. "
    "Which numbered panel shows the same product as Q: same brand, product, flavour or variant, "
    "and pack size? The reference may be photographed from another angle. If none of them is "
    'the same product, answer 0. Return ONLY JSON like {"choice": 2}.'
)


def sheet(query: Image.Image, refs: list[Image.Image]) -> Image.Image:
    """Q followed by the numbered candidates, each fitted into a CELL x CELL panel."""
    out = Image.new("RGB", (CELL * (len(refs) + 1), CELL + 30), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for k, im in enumerate([query, *refs]):
        im = im.copy()
        im.thumbnail((CELL - 10, CELL - 10))
        x = k * CELL
        out.paste(im, (x + (CELL - im.width) // 2, 30 + (CELL - im.height) // 2))
        draw.rectangle((x, 0, x + CELL - 1, CELL + 29), outline="#999")
        draw.text((x + 6, 2), "Q" if k == 0 else str(k), fill="red" if k == 0 else "black", font=font)
    return out


@register
class GeminiRerank(EmbeddingRetrieval):
    name = "gemini_rerank"
    models: list[str] | None = None  # any Gemini for the pick
    also_calls = [MODEL]  # the shortlist
    architecture = f"Embedding shortlist (top {K}) -> Gemini picks the product"
    steps = [
        "Setup (once per run, reported apart from cost/img): embed every reference photo",
        f"Per crop: embed it and shortlist the {K} products with the most similar reference photo",
        f"One Gemini call per crop: the crop next to the {K} candidates; Gemini picks one (or none)",
    ]

    def identify(self, image: Image.Image, ctx: Context) -> int | None:
        top = self.ranked(image, ctx)[:K]
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        data = ctx.ask(sheet(image, refs), PROMPT, schema=SCHEMA).data
        choice = data.get("choice") if isinstance(data, dict) else None
        return top[choice - 1][1] if isinstance(choice, int) and 1 <= choice <= K else None
