"""Standalone CLM (Contrastive-LM) Cross-Modal Classifier (``clm_standalone_classify``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Pure System-1 Cross-Modal Contrastive-LM Classifier (0 VLM Calls)         │
  │  • Inherits directly from EmbeddingTextMatch (zero DjevClassify coupling)  │
  │  • Dual-Prototype Text Index: raw catalog string + enriched taxonomy text  │
  │    (brand, variant, pack_form, size, HUL ownership) + Brand Centroid       │
  │  • Cross-Modal InfoNCE Energy:                                             │
  │      E(x, y_i) = 0.55*cos_raw + 0.30*cos_enriched + 0.15*cos_brand_cent    │
  │  • Temperature-scaled Softmax Posterior P_CLM(y_1|x) at 1/tau = 32.0       │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Context, register
from approaches.market_share.classification.embedding_text_match import (
    EmbeddingTextMatch,
    scoped_catalog_for_ctx,
    unit,
    visual_gallery_centroids_for_catalog,
)
from approaches.market_share.classification.shelf_l8_ultimate import (
    enrich_catalog_entry,
)
from utils.embeddings import MODELS, SKUS, VertexEmbeddings
from utils.photometry import restore_photometry

CLM_TEXT_INV_TEMP = 32.0


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@register
class ClmStandaloneClassify(EmbeddingTextMatch):
    """Standalone System-1 Cross-Modal CLM InfoNCE Classifier (0 VLM calls, zero JEV coupling)."""

    name = "clm_standalone_classify"
    task = "classification"
    dataset = "products"
    models = MODELS
    skus = SKUS
    architecture = (
        "Standalone CLM (Contrastive-LM) Cross-Modal Dual-Prototype + Brand-Centroid InfoNCE "
        f"Classifier (1/tau={CLM_TEXT_INV_TEMP}, 0 VLM calls)"
    )
    steps = [
        "Setup: enrich catalog (brand, variant, pack form, size) and embed raw + enriched text prototypes & brand centroids",
        "Photometric de-glare + embed query photo once",
        "Score all catalog SKUs via CLM cross-modal energy E(x, y_i) = 0.55*cos_raw + 0.30*cos_enriched + 0.15*cos_brand",
        f"Return top-1 catalog SKU with calibrated InfoNCE posterior P_CLM(y_1|x) at 1/tau={CLM_TEXT_INV_TEMP} in 0 VLM calls",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.emb = VertexEmbeddings(config, model=ctx.model if ctx.model in MODELS else MODELS[0])
        raw_cat = scoped_catalog_for_ctx(self.dataset, ctx)
        self.catalog = {i: enrich_catalog_entry(p) for i, p in raw_cat.items()}
        ids = list(self.catalog)
        raw_texts = [
            f"{p.get('brand', '')} {p['product']} ({p.get('category', '')})".strip()
            for p in self.catalog.values()
        ]
        enr_texts = [self.catalog[i]["enriched_text"] for i in ids]
        with ThreadPoolExecutor(max_workers=8) as pool:
            raw_vecs = [unit(v) for v in pool.map(lambda t: self.emb.text(t, ctx), raw_texts)]
            enr_vecs = [unit(v) for v in pool.map(lambda t: self.emb.text(t, ctx), enr_texts)]

        vis_cents = visual_gallery_centroids_for_catalog(self.dataset, ids, self.emb, ctx)
        if vis_cents:
            raw_vecs = [
                unit([0.85 * vc + 0.15 * rv for vc, rv in zip(vis_cents[i], rv_vec, strict=True)])
                if i in vis_cents
                else rv_vec
                for i, rv_vec in zip(ids, raw_vecs, strict=True)
            ]
            enr_vecs = [
                unit([0.85 * vc + 0.15 * ev for vc, ev in zip(vis_cents[i], ev_vec, strict=True)])
                if i in vis_cents
                else ev_vec
                for i, ev_vec in zip(ids, enr_vecs, strict=True)
            ]

        self.dual_index = list(zip(ids, raw_vecs, enr_vecs, strict=True))
        by_brand: dict[str, list[list[float]]] = {}
        for i, rv, ev in self.dual_index:
            b = str(self.catalog[i].get("brand") or self.catalog[i].get("category") or "")
            if b:
                by_brand.setdefault(b, []).append(unit([0.6 * r + 0.4 * e for r, e in zip(rv, ev, strict=True)]))
        self.brand_centroids: dict[str, list[float]] = {
            b: unit([sum(col) / len(vs) for col in zip(*vs, strict=True)])
            for b, vs in by_brand.items()
        }

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        restored, _ = restore_photometry(image)
        q = unit(self.emb.image(restored, ctx))
        pool = (
            [(i, rv, ev) for i, rv, ev in self.dual_index if i in allowed_ids]
            if allowed_ids
            else []
        ) or self.dual_index

        scored: list[tuple[float, int]] = []
        for i, rv, ev in pool:
            b = str(self.catalog[i].get("brand") or self.catalog[i].get("category") or "")
            bc = self.brand_centroids.get(b)
            s_raw = _dot(q, rv)
            s_enr = _dot(q, ev)
            s_brand = _dot(q, bc) if bc is not None else s_raw
            energy = 0.55 * s_raw + 0.30 * s_enr + 0.15 * s_brand
            scored.append((energy, i))

        scored.sort(reverse=True)
        if not scored:
            return None

        top_k = scored[:6]
        e1, top_id = top_k[0]
        e2 = top_k[1][0] if len(top_k) > 1 else -1.0
        exps = [math.exp(max(-30.0, CLM_TEXT_INV_TEMP * (e - e1))) for e, _ in top_k]
        z = sum(exps) or 1.0
        p1 = round(exps[0] / z, 4)

        ctx.trace.step(
            "Standalone CLM Cross-Modal InfoNCE",
            f"id {top_id}: {self.catalog[top_id]['product']} (E={e1:.3f}, margin={e1 - e2:.3f}, P_CLM={p1:.2f})",
            info={
                "sku_id": top_id,
                "energy": round(e1, 4),
                "margin": round(e1 - e2, 4),
                "clm_prob": p1,
                "shortlist": [
                    {"sku_id": i, "product": self.catalog[i]["product"], "energy": round(s, 4)}
                    for s, i in scored[:8]
                ],
            },
        )
        return top_id
