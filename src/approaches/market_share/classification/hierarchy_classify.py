"""Brand first, then size, then product (Classification tab).

Ported from the idea in Jigyasu's ``hul_hierarchy_classifier``: narrow the catalog step by step
instead of showing Gemini all 184 products at once. His version guessed the labels from box
positions; this one reads the pack:

1. Gemini call 1: which catalog brand is this (from the list of brands, or "not listed"), and
   what pack size is printed on it.
2. Keep the catalog products of that brand; if the size read matches some of them, keep only
   those.
3. One product left: that's the answer, no second call. Otherwise Gemini call 2 picks from the
   short list (same question as ``gemini_classify``). If the brand isn't in the catalog, call 2
   gets the whole catalog, and the trace says so.
"""

from __future__ import annotations

import re

from PIL import Image

from approaches.base import Approach, Context, register
from approaches.market_share.classification.gemini_classify import PROMPT as PICK_PROMPT
from approaches.market_share.classification.gemini_classify import SCHEMA as PICK_SCHEMA
from utils import dataset

NOT_LISTED = "not listed"
BRAND_PROMPT = (
    "The photo shows one retail product. Which of these brands is it? {brands}. If it's none of "
    f"them, answer \"{NOT_LISTED}\". Also copy the net size printed on the pack (e.g. 385ml, "
    '1kg), or "" if you can\'t read one. Return ONLY JSON like {{"brand": "Knorr", "size": "8g"}}.'
)


def norm_size(s: str) -> str:
    """'385 mL' -> '385ml'; '1.0 L' -> '1l'. Empty if there's no number + unit."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|g|l|kg)\b", s.lower())
    return f"{float(m.group(1)):g}{m.group(2)}" if m else ""


@register
class HierarchyClassify(Approach):
    name = "hierarchy_classify"
    task = "classification"
    dataset = "products"
    architecture = "Gemini reads brand + size -> catalog filter -> Gemini picks from the short list"
    steps = [
        "Gemini call 1: the photo and the list of catalog brands; returns the brand and pack size",
        "Filter the catalog to that brand, then to that size when it matches",
        "One product left: done. Otherwise Gemini call 2 picks from the short list",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.catalog = dataset.catalog(name=self.dataset)
        self.brands = sorted({p.get("brand") or p.get("category", "") for p in self.catalog.values() if p.get("brand") or p.get("category")})
        self.brand_schema = {
            "type": "object",
            "properties": {"brand": {"type": "string", "enum": [*self.brands, NOT_LISTED]},
                           "size": {"type": "string"}},
            "required": ["brand", "size"],
        }
        self.brand_prompt = BRAND_PROMPT.format(brands=", ".join(self.brands))

    def candidates(self, brand: str, size: str,
                   allowed_ids: set[int] | None = None) -> tuple[dict[int, dict], str]:
        """Catalog products left after the brand and size filters, and how they were chosen."""
        pool = ({i: p for i, p in self.catalog.items() if i in allowed_ids}
                if allowed_ids else self.catalog) or self.catalog
        by_brand = {i: p for i, p in pool.items()
                    if (p.get("brand") or p.get("category")) == brand}
        if not by_brand:
            return pool, f"brand {brand!r} not in the catalog: whole catalog"
        sized = {i: p for i, p in by_brand.items()
                 if size and norm_size(p.get("size") or p["product"]) == size}
        if sized:
            return sized, f"{len(by_brand)} {brand} products, {len(sized)} of size {size}"
        return by_brand, f"{len(by_brand)} {brand} products (size {size or '?'} matches none)"

    def narrow(self, image: Image.Image, catalog: dict[int, dict], ctx: Context) -> set[int] | None:
        data = ctx.ask(image, self.brand_prompt, schema=self.brand_schema, max_side=1024).data
        data = data if isinstance(data, dict) else {}
        brand, size = str(data.get("brand", NOT_LISTED)), norm_size(str(data.get("size", "")))
        cands, how = self.candidates(brand, size)
        ctx.trace.step("Gemini reads brand and size", f"brand {brand}, size {size or '?'} -> {how}")
        return set(cands)

    def identify(self, image: Image.Image, ctx: Context,
                 allowed_ids: set[int] | None = None) -> int | None:
        data = ctx.ask(image, self.brand_prompt, schema=self.brand_schema, max_side=1024).data
        data = data if isinstance(data, dict) else {}
        brand, size = str(data.get("brand", NOT_LISTED)), norm_size(str(data.get("size", "")))
        cands, how = self.candidates(brand, size, allowed_ids=allowed_ids)
        ctx.trace.step("Gemini reads brand and size", f"brand {brand}, size {size or '?'} -> {how}")
        if len(cands) == 1:
            sku = next(iter(cands))
            ctx.trace.step("One candidate left", f"id {sku}: {cands[sku]['product']}")
            return sku
        prompt = PICK_PROMPT.format(catalog="\n".join(
            f"{i}: {p.get('brand', '')} | {p['product']} | {p.get('category', '')}"
            for i, p in cands.items()))
        data = ctx.ask(image, prompt, schema=PICK_SCHEMA, max_side=1024).data
        sku = data.get("sku_id") if isinstance(data, dict) else None
        p = cands.get(sku)
        ctx.trace.step(f"Gemini picks from {len(cands)} candidates",
                       f"id {sku}: {p['product']}" if p else f"id {sku}: none of them")
        return sku if p else None
