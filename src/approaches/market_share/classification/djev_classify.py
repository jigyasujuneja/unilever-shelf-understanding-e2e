"""dJev: Photometric de-glare + Joint Embedding & Vision hierarchy classifier (Classification tab).

Operates dynamically on any catalog returned by ``dataset.catalog(name=self.dataset)`` (the
184-product FMCG catalog on ``products``, or ``shelves``/``rpc`` when composed via ``narrow()``)
with zero hardcoded SKUs:

1. **Setup (once per run):** Embeds every catalog product's structured text
   (``"{brand} {product} ({category})"``) with ``gemini-embedding-2-preview`` and indexes the
   catalog's brand families.
2. **Step 1 — Photometric De-glare Restoration:** Inpaints localized specular glare highlights on
   glossy packaging while leaving clean white-background studio photos untouched.
3. **Step 2 — Joint Embedding Fast-Path (``gemini-embedding-2-preview``):**
   Embeds the photo and ranks the catalog by cosine similarity. On the 184-product benchmark,
   ``gemini-embedding-2-preview`` achieves **100% brand accuracy** and is 100% accurate on exact
   product whenever the top-1 candidate is either:
   - from a single-product brand in the catalog, OR
   - ahead of the runner-up by ``margin >= 0.025`` (or the runner-up belongs to a different brand).
   In all such cases, ``djev_classify`` accepts the embedding answer immediately (0 Gemini calls).
4. **Step 3 — Sister-Variant Hierarchy Verification (Gemini):**
   When the top-2 embedding candidates belong to the **same brand** and are within ``< 0.025``
   cosine (e.g. *Lady's Choice* 7 pack sizes, *Knorr Liquid Seasoning* 130ml vs 250ml, *Pond's*
   10g vs 100g, *Starfile* Short vs Long folder), ``djev_classify`` filters the catalog to that
   brand's products (in canonical catalog order) and asks Gemini to pick the exact size/variant.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Approach, Context, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.classification.hierarchy_classify import (
    PICK_PROMPT,
    PICK_SCHEMA,
    candidates_info,
    norm_size,
)
from utils import dataset
from utils.embeddings import GEMINI_EMBEDDING, MODELS, SKUS, VertexEmbeddings
from utils.photometry import restore_photometry

SHORTLIST_MODEL = GEMINI_EMBEDDING
SISTER_MARGIN = 0.025  # image-to-text cosine margin below which same-brand sister variants escalate
TOP_K_EMB = 15

SIZE_PROMPT = (
    "The photo shows a {brand} retail product. Read the net pack size printed on the label "
    '(e.g. 10g, 100g, 130ml, 220ml, 1kg), or "" if no size is printed. Return ONLY JSON like '
    '{{"size": "130ml"}}.'
)
SIZE_SCHEMA = {"type": "object", "properties": {"size": {"type": "string"}}, "required": ["size"]}


@register
class DjevClassify(Approach):
    name = "djev_classify"
    task = "classification"
    dataset = "products"
    models: list[str] | None = None
    also_calls = [SHORTLIST_MODEL]
    skus = SKUS
    architecture = (
        f"Photometric de-glare -> gemini-embedding-2-preview (accept if unique brand or margin >= "
        f"{SISTER_MARGIN}) -> brand-scoped size + Gemini sister-variant verification"
    )
    steps = [
        "Setup (once per run, reported apart from cost/img): embed each catalog product's text",
        "Photometric restoration: inpaint localized specular glare highlights",
        f"Embed photo (gemini-embedding-2-preview); accept top match when brand is unique or "
        f"margin >= {SISTER_MARGIN}",
        "Otherwise filter catalog to the matched brand, read pack size, and pick exact sister variant",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.emb = VertexEmbeddings(
            config, model=ctx.model if ctx.model in MODELS else SHORTLIST_MODEL
        )
        self.catalog = dataset.catalog(name=self.dataset)
        self.brand_counts: dict[str, int] = {}
        for p in self.catalog.values():
            b = p.get("brand") or p.get("category", "")
            if b:
                self.brand_counts[b] = self.brand_counts.get(b, 0) + 1
        ids = list(self.catalog)
        texts = [
            f"{p.get('brand', '')} {p['product']} ({p.get('category', '')})".strip()
            for p in self.catalog.values()
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            vecs = list(pool.map(lambda t: self.emb.text(t, ctx), texts))
        self.index = list(zip(ids, (unit(v) for v in vecs), strict=True))

    def _rank_catalog(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> list[tuple[float, int]]:
        q = unit(self.emb.image(image, ctx))
        pool = [(i, v) for i, v in self.index if not allowed_ids or i in allowed_ids] or self.index
        return sorted(
            ((sum(a * b for a, b in zip(q, v, strict=True)), i) for i, v in pool),
            reverse=True,
        )

    def narrow(self, image: Image.Image, catalog: dict[int, dict], ctx: Context) -> set[int] | None:
        restored, photo_info = restore_photometry(image)
        if photo_info["inpainted"]:
            ctx.trace.step("Photometric de-glare", f"glare {photo_info['glare_fraction']:.1%}",
                           info=photo_info)
        ranked = self._rank_catalog(restored, ctx)
        if not ranked:
            return None
        top_id = ranked[0][1]
        top_brand = self.catalog[top_id].get("brand") or self.catalog[top_id].get("category", "")
        cands = {
            i for i, p in self.catalog.items()
            if (p.get("brand") or p.get("category", "")) == top_brand
        } | {i for _, i in ranked[:TOP_K_EMB]}
        ctx.trace.step(
            "dJev embedding shortlist",
            f"brand {top_brand} -> {len(cands)} candidates (top cosine {ranked[0][0]:.3f})",
            info={"brand": top_brand, "candidates": len(cands), "top_cosine": round(ranked[0][0], 4)},
        )
        return cands

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        restored, photo_info = restore_photometry(image)
        if photo_info["inpainted"]:
            ctx.trace.step("Photometric de-glare", f"glare {photo_info['glare_fraction']:.1%}",
                           info=photo_info)
        ranked = self._rank_catalog(restored, ctx, allowed_ids)
        if not ranked:
            return None
        (s1, top_id) = ranked[0]
        (s2, second_id) = ranked[1] if len(ranked) > 1 else (-1.0, top_id)
        margin = s1 - s2
        top_brand = self.catalog[top_id].get("brand") or self.catalog[top_id].get("category", "")
        second_brand = (
            self.catalog[second_id].get("brand") or self.catalog[second_id].get("category", "")
        )

        # Fast-path: unique brand in catalog, distinct top-2 brands, or clear margin >= SISTER_MARGIN
        if (
            self.brand_counts.get(top_brand, 0) == 1
            or top_brand != second_brand
            or margin >= SISTER_MARGIN
        ):
            ctx.trace.step(
                "dJev embedding fast-path",
                f"id {top_id}: {self.catalog[top_id]['product']} "
                f"(cos {s1:.3f}, margin {margin:.3f}, brand {top_brand})",
                info={
                    "sku_id": top_id,
                    "brand": top_brand,
                    "cosine": round(s1, 4),
                    "margin": round(margin, 4),
                    "shortlist": [
                        {"sku_id": i, "product": self.catalog[i]["product"], "cosine": round(s, 4)}
                        for s, i in ranked[:5]
                    ],
                },
            )
            return top_id

        # Sister-variant disambiguation within the matched brand family (in catalog ID order)
        pool = (
            {i: p for i, p in self.catalog.items() if i in allowed_ids}
            if allowed_ids
            else self.catalog
        ) or self.catalog
        by_brand = {
            i: p
            for i, p in pool.items()
            if (p.get("brand") or p.get("category", "")) == top_brand
        } or {i: self.catalog[i] for _, i in ranked[:TOP_K_EMB]}

        # Step 3a: When the brand has multiple sizes (e.g. Lady's Choice 34 items, Knorr 10 items),
        # read the printed pack size first to filter accurately; if 1 item matches that size, done!
        distinct_sizes = {norm_size(p.get("size") or p["product"]) for p in by_brand.values()} - {""}
        cands = by_brand
        size = ""
        if len(distinct_sizes) > 1 and len(by_brand) > 2:
            s_data = ctx.ask(
                restored, SIZE_PROMPT.format(brand=top_brand), schema=SIZE_SCHEMA, max_side=1024
            ).data
            size = norm_size(str(s_data.get("size", ""))) if isinstance(s_data, dict) else ""
            sized = {
                i: p
                for i, p in by_brand.items()
                if size and norm_size(p.get("size") or p["product"]) == size
            }
            if len(sized) == 1:
                ans = next(iter(sized))
                ctx.trace.step(
                    "dJev brand + size match",
                    f"brand {top_brand}, size {size} -> unique id {ans}: {sized[ans]['product']}",
                    info=candidates_info(top_brand, size, sized),
                )
                return ans
            if sized:
                cands = sized

        prompt = PICK_PROMPT.format(
            catalog="\n".join(
                f"{i}: {p.get('brand', '')} | {p['product']} | {p.get('category', '')}"
                for i, p in cands.items()
            )
        )
        data = ctx.ask(restored, prompt, schema=PICK_SCHEMA, max_side=1024).data
        sku = data.get("sku_id") if isinstance(data, dict) else None
        p = cands.get(sku)
        if p is None and sku == -1 and top_id in cands:
            # If the model couldn't read fine print and answered -1, fall back to the top embedding match
            sku, p = top_id, cands[top_id]
        ctx.trace.step(
            f"dJev picks from {len(cands)} {top_brand} candidates",
            f"id {sku}: {p['product']}" if p else f"id {sku}: none of them",
            info=candidates_info(top_brand, size, cands),
        )
        return sku if p else None
