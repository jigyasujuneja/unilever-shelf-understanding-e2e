"""Gemini picks the product in a photo from the closed catalog (the Classification tab).

One call per photo: the image plus the numbered catalog (184 products: brand, name with size,
category), and Gemini answers with the catalog id. Scored on the labelled products set in
``utils/dataset.py``: exact product, and brand / category as coarser checks.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Context, register
from utils import dataset

SCHEMA = {"type": "object", "properties": {"sku_id": {"type": "integer"}}, "required": ["sku_id"]}

PROMPT = (
    "The photo shows one retail product. Identify it in this catalog, where each line is "
    "'id: brand | product name and size | category'. Match the exact product, including its "
    "size or variant. If it isn't in the catalog, answer -1. Return ONLY JSON like "
    '{{"sku_id": 72}}.\n\nCatalog:\n{catalog}'
)


@register
class GeminiClassify(Approach):
    name = "gemini_classify"
    task = "classification"
    dataset = "products"
    architecture = "Gemini picks the product from the catalog"
    steps = [
        "One Gemini call: the photo plus the numbered catalog (brand | product and size | category)",
        "Gemini returns a catalog id (or -1 if the product isn't listed)",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.catalog = dataset.catalog()
        self.prompt = PROMPT.format(catalog="\n".join(
            f"{i}: {p['brand']} | {p['product']} | {p['category']}" for i, p in self.catalog.items()))

    def identify(self, image: Image.Image, ctx: Context) -> int | None:
        data = ctx.ask(image, self.prompt, schema=SCHEMA, max_side=1024).data
        sku = data.get("sku_id") if isinstance(data, dict) else None
        p = self.catalog.get(sku)
        ctx.trace.step("Gemini picks from the catalog",
                       f"id {sku}: {p['brand']} | {p['product']}" if p else f"id {sku}: not in catalog")
        return sku if p else None
