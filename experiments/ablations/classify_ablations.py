"""Classification Ablation Studies (``clef_flash_classify``, ``clm_classify``).

Kept in ``experiments/ablations/`` for reproducibility and Vertex AI Experiments lineage.
See ``experiments/ablations/__init__.py`` for why ``djev_classify``'s 2-field ``{"pack_size", "choice"}``
grounding outperformed ``clef_flash_classify`` on ``gemini-3.5-flash-lite`` (88.0% vs 84.0%).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Context, register
from approaches.market_share.classification.coarse_jev_gemini import CoarseJevGemini
from approaches.market_share.classification.djev_classify import (
    SHORTLIST_MODEL,
    TOP_K_EMB,
    candidates_info,
)
from approaches.market_share.classification.embedding_text_match import (
    scoped_catalog_for_ctx,
    unit,
)
from approaches.market_share.classification.shelf_l8_ultimate import (
    build_pack_and_zoom_canvas,
    enrich_catalog_entry,
)
from utils.embeddings import MODELS, VertexEmbeddings
from utils.openjev import extract_slot_distributions, jev_entropy_confidence
from utils.photometry import restore_photometry


@register
class ClefFlashClassify(CoarseJevGemini):
    """Ablation #6 & #8 (0.8400 flash-lite / 0.9200 flash): Enriched Catalog + Clef-Flash Zoom-Canvas."""

    name = "clef_flash_classify"
    task = "classification"
    dataset = "products"
    architecture = (
        f"Exp #8 Enriched Catalog ({SHORTLIST_MODEL} over brand+variant+form+size) -> "
        "Stage A Coarse JEV triage -> Stage B Clef-Flash Zoom-Canvas Decision Head"
    )
    steps = [
        "Setup (Exp #8): auto-enrich 184-SKU catalog (recover missing sizes, variants, pack forms & HUL flags) + embed",
        f"Stage A: Photometric de-glare + Coarse JEV triage ({SHORTLIST_MODEL}) over enriched catalog",
        "Stage B: Build Pack + Bottom-35% Size Zoom Canvas and resolve sister SKUs via Clef-Flash typed decision head",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.emb = VertexEmbeddings(
            config, model=ctx.model if ctx.model in MODELS else SHORTLIST_MODEL
        )
        raw_cat = scoped_catalog_for_ctx(self.dataset, ctx)
        self.catalog = {i: enrich_catalog_entry(p) for i, p in raw_cat.items()}
        self.brand_counts: dict[str, int] = {}
        for p in self.catalog.values():
            b = p.get("brand") or p.get("category", "")
            if b:
                self.brand_counts[b] = self.brand_counts.get(b, 0) + 1
        ids = list(self.catalog)
        texts = [self.catalog[i]["enriched_text"] for i in ids]
        with ThreadPoolExecutor(max_workers=8) as pool:
            vecs = list(pool.map(lambda t: self.emb.text(t, ctx), texts))
        self.index = list(zip(ids, (unit(v) for v in vecs), strict=True))

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        restored, photo_info = restore_photometry(image)
        if photo_info["inpainted"]:
            ctx.trace.step("Photometric de-glare", f"glare {photo_info['glare_fraction']:.1%}", info=photo_info)

        ranked = self._rank_catalog(restored, ctx, allowed_ids)
        if not ranked:
            return None
        pool = (
            {i: p for i, p in self.catalog.items() if i in allowed_ids}
            if allowed_ids
            else self.catalog
        ) or self.catalog

        top_id = ranked[0][1]
        s1 = ranked[0][0]
        s2 = ranked[1][0] if len(ranked) > 1 else -1.0
        second_id = ranked[1][1] if len(ranked) > 1 else top_id
        margin = s1 - s2
        top_brand = str(pool[top_id].get("brand") or "")
        second_brand = str(pool[second_id].get("brand") or "")

        if self.brand_counts.get(top_brand, 0) == 1 or (top_brand != second_brand and margin >= 0.018) or margin >= 0.028:
            ctx.trace.step(
                "Stage A Enriched Coarse JEV fast-path",
                f"id {top_id}: {pool[top_id]['product']} (size={pool[top_id]['size']}, HUL={pool[top_id]['is_unilever']}, cos={s1:.3f}, margin={margin:.3f})",
                info={"sku_id": top_id, "cosine": round(s1, 4), "margin": round(margin, 4)},
            )
            return top_id

        by_brand = {
            i: p for i, p in pool.items() if p.get("brand") == top_brand
        } or {i: pool[i] for _, i in ranked[:TOP_K_EMB]}
        cand_ids = [i for _, i in ranked if i in by_brand][:8]
        cands = {i: pool[i] for i in cand_ids}

        zoom_canvas = build_pack_and_zoom_canvas(image)
        lines = "\n".join(
            f"  {i}: {p['product']} (variant: {p.get('variant', '')}, form: {p.get('pack_form', '')}, size: {p.get('size', '')})"
            for i, p in cands.items()
        )
        prompt = (
            "Left panel: full product photo. Top-right panel: variant zoom. Bottom-right panel: "
            "pack-size/net-weight zoom. Pick the exact matching catalog SKU ID from the candidates below "
            "(match brand, variant, packaging form, and exact gram/ml size):\n"
            f"{lines}\nReturn ONLY JSON {{\"sku_id\": <integer>}}."
        )
        schema = {"type": "object", "properties": {"sku_id": {"type": "integer"}}, "required": ["sku_id"]}
        res = ctx.ask(zoom_canvas, prompt, schema=schema, max_side=768, logprobs=min(len(cands) + 1, 8))
        raw_pick = res.data.get("sku_id") if isinstance(res.data, dict) else None
        slots = extract_slot_distributions(res.meta.get("logprobs"), valid_options=set(cands), n_options=len(cands))
        jev_conf = slots[0]["confidence"] if slots else 1.0
        picked = int(raw_pick) if isinstance(raw_pick, int) and raw_pick in cands else top_id
        ctx.trace.step(
            "Stage B Clef-Flash Zoom-Canvas decision",
            f"id {picked}: {pool[picked]['product']} (jev_conf {jev_conf:.2f}, {len(cands)} sister candidates)",
            info={"sku_id": picked, "jev_confidence": jev_conf, "candidates": candidates_info(top_brand, "", cands)},
        )
        return picked


@register
class ClmClassify(ClefFlashClassify):
    """Ablation #7 (0.8800 / 0.8800 Acc, 16.86s p95 on Flash): CLM InfoNCE Cross-Modal Classifier."""

    name = "clm_classify"
    clm_inv_temp = 32.0
    clm_min_prob = 0.52
    architecture = (
        f"CLM (Contrastive-LM) InfoNCE Cross-Modal Scorer ({SHORTLIST_MODEL}, 1/tau={clm_inv_temp}, "
        f"P_CLM >= {clm_min_prob}) -> Clef-Flash Zoom-Canvas fallback"
    )
    steps = [
        "Setup: auto-enrich 184-SKU catalog with recovered sizes, variants, pack forms & HUL ownership",
        f"Compute CLM InfoNCE softmax posterior P_CLM(y_k | x) at 1/tau={clm_inv_temp} over top-K catalog embeddings",
        f"Accept in 0 VLM calls when P_CLM(y_1 | x) >= {clm_min_prob} (or brand is unique)",
        "Escalate high-entropy sister SKUs to Clef-Flash Pack+Size Zoom Canvas",
    ]

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        restored, _ = restore_photometry(image)
        ranked = self._rank_catalog(restored, ctx, allowed_ids)
        if not ranked:
            return None
        pool = (
            {i: p for i, p in self.catalog.items() if i in allowed_ids}
            if allowed_ids
            else self.catalog
        ) or self.catalog

        top_k = ranked[:6]
        s1, top_id = top_k[0]
        logits = [self.clm_inv_temp * (s - s1) for s, _ in top_k]
        exps = [math.exp(max(-30.0, x)) for x in logits]
        z = sum(exps) or 1.0
        probs = [e / z for e in exps]
        p1 = probs[0]
        ent_conf = jev_entropy_confidence(probs, n_options=len(top_k))

        top_brand = str(pool[top_id].get("brand") or "")
        second_brand = str(pool[top_k[1][1]].get("brand") or "") if len(top_k) > 1 else top_brand

        if self.brand_counts.get(top_brand, 0) == 1 or top_brand != second_brand or p1 >= self.clm_min_prob:
            ctx.trace.step(
                "CLM InfoNCE fast-path",
                f"id {top_id}: {pool[top_id]['product']} (P_CLM={p1:.2f}, ent_conf={ent_conf:.2f}, cos={s1:.3f})",
                info={"sku_id": top_id, "clm_prob": round(p1, 4), "entropy_conf": ent_conf, "cosine": round(s1, 4)},
            )
            return top_id

        return super().identify(image, ctx, allowed_ids=allowed_ids)
