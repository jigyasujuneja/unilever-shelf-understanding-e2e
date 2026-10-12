"""Standalone CLEF (Constrained Logprob-Evaluated Flash) Classifier (``clef_standalone_classify``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Pure System-2 CLEF 3-Panel Zoom-Canvas Classifier (Zero JEV Coupling)     │
  │  • Inherits directly from Approach (no Stage-A Coarse JEV fast-path,       │
  │    no JEV embedding fallback override)                                     │
  │  • Enriched Catalog Shortlist -> 100% evaluated on 3-Panel Zoom Canvas     │
  │    (Full Pack + Center Variant Crop + Bottom-35% Net-Weight Crop)          │
  │  • Grounded {"pack_size", "sku_id"} schema + direct token logprob readout  │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Approach, Context, register
from approaches.market_share.classification.embedding_text_match import (
    scoped_catalog_for_ctx,
    unit,
)
from approaches.market_share.classification.shelf_l8_ultimate import (
    build_pack_and_zoom_canvas,
    enrich_catalog_entry,
)
from approaches.market_share.retrieval.clef_standalone import clef_logprob_select
from utils.embeddings import GEMINI_EMBEDDING, MODELS, SKUS, VertexEmbeddings

CLEF_CLASSIFY_SHORTLIST = 10


@register
class ClefStandaloneClassify(Approach):
    """Standalone System-2 CLEF 3-Panel Zoom-Canvas Classifier (100% CLEF evaluation, zero JEV coupling)."""

    name = "clef_standalone_classify"
    task = "classification"
    dataset = "products"
    models: list[str] | None = None
    also_calls = [GEMINI_EMBEDDING]
    skus = SKUS
    architecture = (
        f"Standalone CLEF: Enriched Catalog ({GEMINI_EMBEDDING} top-{CLEF_CLASSIFY_SHORTLIST}) -> "
        "100% CLEF 3-Panel Pack+Variant+Size Zoom-Canvas Decision Head (zero JEV bypass)"
    )
    steps = [
        "Setup: auto-enrich catalog (brand, variant, pack form, size) and embed structured text",
        f"Retrieve top-{CLEF_CLASSIFY_SHORTLIST} catalog candidates (+ same-brand sisters) by embedding cosine",
        "Build 3-Panel Zoom Canvas (Full Pack + Variant Zoom + Bottom-35% Size Zoom) for 100% of queries",
        "Execute CLEF constrained {'pack_size', 'sku_id'} schema + direct token logprob readout",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.emb = VertexEmbeddings(
            config, model=ctx.model if ctx.model in MODELS else GEMINI_EMBEDDING
        )
        raw_cat = scoped_catalog_for_ctx(self.dataset, ctx)
        self.catalog = {i: enrich_catalog_entry(p) for i, p in raw_cat.items()}
        ids = list(self.catalog)
        texts = [self.catalog[i]["enriched_text"] for i in ids]
        with ThreadPoolExecutor(max_workers=8) as pool:
            vecs = list(pool.map(lambda t: self.emb.text(t, ctx), texts))
        self.index = list(zip(ids, (unit(v) for v in vecs), strict=True))

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        q = unit(self.emb.image(image, ctx))
        pool_idx = [(i, v) for i, v in self.index if not allowed_ids or i in allowed_ids] or self.index
        ranked = sorted(
            ((sum(a * b for a, b in zip(q, v, strict=True)), i) for i, v in pool_idx),
            reverse=True,
        )
        if not ranked:
            return None

        top_id = ranked[0][1]
        top_brand = str(self.catalog[top_id].get("brand") or "")
        cand_ids: list[int] = []
        for _, i in ranked:
            if len(cand_ids) < CLEF_CLASSIFY_SHORTLIST or (
                top_brand and self.catalog[i].get("brand") == top_brand and len(cand_ids) < 14
            ):
                cand_ids.append(i)
        cands = {i: self.catalog[i] for i in cand_ids}

        zoom_canvas = build_pack_and_zoom_canvas(image)
        lines = "\n".join(
            f"  {i}: {p['product']} (brand: {p.get('brand', '')}, variant: {p.get('variant', '')}, "
            f"form: {p.get('pack_form', '')}, size: {p.get('size', '')})"
            for i, p in cands.items()
        )
        prompt = (
            "Left panel: full product photo. Top-right panel: variant zoom. Bottom-right panel: "
            "pack-size/net-weight zoom. First read the visible net weight/pack size printed on the "
            "label (or '' if none), then pick the exact matching catalog SKU ID from the candidates below:\n"
            f"{lines}\n"
            'Return ONLY JSON {{"pack_size": "...", "sku_id": <integer>}}.'
        )
        schema = {
            "type": "object",
            "properties": {
                "pack_size": {"type": "string"},
                "sku_id": {"type": "integer"},
            },
            "required": ["pack_size", "sku_id"],
        }
        res = ctx.ask(zoom_canvas, prompt, schema=schema, max_side=768, logprobs=8)
        data = res.data if isinstance(res.data, dict) else {}
        raw_sku = data.get("sku_id")
        slots = clef_logprob_select(res.meta.get("logprobs"), set(cands.keys()), n_slots=1)
        picked = (
            slots[0]["choice"]
            if slots and slots[0]["choice"] in cands
            else (int(raw_sku) if isinstance(raw_sku, int) and raw_sku in cands else top_id)
        )
        prob = slots[0]["top1_prob"] if slots else 1.0
        ctx.trace.step(
            "Standalone CLEF 3-Panel Zoom-Canvas",
            f"id {picked}: {cands[picked]['product']} (P_token={prob:.2f}, size_read={data.get('pack_size', '')!r})",
            info={
                "sku_id": picked,
                "clef_prob": prob,
                "size_read": data.get("pack_size", ""),
                "candidates": len(cands),
            },
        )
        return picked
