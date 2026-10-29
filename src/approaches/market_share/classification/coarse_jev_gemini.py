"""Two-Stage Hybrid Classification: Stage A Coarse JEV Triage + Stage B Gemini Sister-SKU Verification.

Combines System-1 Coarse JEV Multi-Attribute Triage (`coarse_jev_triage` / `OpenJevClient`) for
fast coarse attribute classification (`category`, `brand`, and intra-brand family) with System-2
Gemini fine-print verification on sister SKUs:

1. **Stage A — Coarse JEV Multi-Attribute Triage:**
   - Applies photometric glare restoration (`restore_photometry`) for the embedding vector lookup
     while preserving the crisp original image for any downstream fine-print OCR.
   - Computes Boltzmann probability distributions and `1 - H(p)/ln(N)` entropy confidence across
     hierarchical slots (`category`, `brand`, and `sku_id`) via `coarse_jev_triage`.
   - Fast-path accepts `top_id` (0 VLM calls) whenever the coarse brand has 1 SKU, the top-2
     matches belong to different brands, or the intra-brand sister margin is `>= 0.025`.
2. **Stage B — Gemini Fine-Grained Sister-Variant Disambiguation:**
   - Triggered only when sister SKUs of the same coarse brand compete within `< 0.025` cosine.
   - Uses the crisp un-inpainted original `image` so specular inpainting never blurs white
     printed grammage/ml text (such as `"130ml"` on dark bottles), reads the pack size, and
     resolves remaining sister variants with `djev` token logprob entropy calibration.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Context, register
from approaches.market_share.classification.djev_classify import (
    MIN_JEV_CONFIDENCE,
    PICK_PROMPT,
    PICK_SCHEMA,
    SHORTLIST_MODEL,
    SISTER_MARGIN,
    SIZE_PROMPT,
    SIZE_SCHEMA,
    TOP_K_EMB,
    DjevClassify,
    candidates_info,
    norm_size,
)
from utils.openjev import OpenJevClient, coarse_jev_triage, extract_slot_distributions
from utils.photometry import restore_photometry


@register
class CoarseJevGemini(DjevClassify):
    name = "coarse_jev_gemini"
    task = "classification"
    dataset = "products"
    min_jev_confidence = MIN_JEV_CONFIDENCE
    architecture = (
        f"Stage A Coarse JEV multi-attribute triage ({SHORTLIST_MODEL} + 1-H(p)/ln(N) gate) -> "
        "Stage B Gemini crisp-image size & sister-variant verification"
    )
    steps = [
        "Setup (once per run, reported apart from cost/img): embed each catalog product's text",
        f"Stage A: Photometric de-glare + Coarse JEV triage ({SHORTLIST_MODEL}) over (category, brand, sku_id)",
        f"Accept Stage A immediately when brand is unique, top-2 brands differ, or intra-brand margin >= {SISTER_MARGIN}",
        "Stage B: Crisp-image Gemini pack-size read + logprob-calibrated sister-SKU verification",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        self.openjev = OpenJevClient(config)

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        restored, photo_info = restore_photometry(image)
        if photo_info["inpainted"]:
            ctx.trace.step(
                "Photometric de-glare",
                f"glare {photo_info['glare_fraction']:.1%}",
                info=photo_info,
            )
        ranked = self._rank_catalog(restored, ctx, allowed_ids)
        if not ranked:
            return None

        pool = (
            {i: p for i, p in self.catalog.items() if i in allowed_ids}
            if allowed_ids
            else self.catalog
        ) or self.catalog

        triage = coarse_jev_triage(ranked, pool)
        top_id = triage["top_id"]
        s1 = triage["top_score"]
        margin = triage["margin"]
        intra_brand_margin = triage["intra_brand_margin"]
        top_brand = triage["top_brand"]
        top_cat = triage["top_category"]
        cat_conf = triage["category_confidence"]
        brand_conf = triage["brand_confidence"]

        second_id = ranked[1][1] if len(ranked) > 1 else top_id
        second_brand = str(
            pool[second_id].get("brand") or pool[second_id].get("category") or ""
        )

        # Stage A Coarse JEV Fast-Path Gate
        if (
            self.brand_counts.get(top_brand, 0) == 1
            or top_brand != second_brand
            or margin >= SISTER_MARGIN
            or intra_brand_margin >= SISTER_MARGIN
        ):
            ctx.trace.step(
                "Stage A Coarse JEV fast-path",
                f"id {top_id}: {pool[top_id]['product']} "
                f"(cat {top_cat} conf {cat_conf:.2f}, brand {top_brand} conf {brand_conf:.2f}, "
                f"cos {s1:.3f}, margin {margin:.3f})",
                info={
                    "sku_id": top_id,
                    "category": top_cat,
                    "category_confidence": cat_conf,
                    "brand": top_brand,
                    "brand_confidence": brand_conf,
                    "cosine": round(s1, 4),
                    "margin": round(margin, 4),
                    "intra_brand_margin": round(intra_brand_margin, 4),
                    "shortlist": [
                        {"sku_id": i, "product": pool[i]["product"], "cosine": round(s, 4)}
                        for s, i in ranked[:5]
                    ],
                },
            )
            return top_id

        # Stage B: Crisp-image Gemini fine-print disambiguation within the coarse brand cluster
        by_brand = {
            i: p
            for i, p in pool.items()
            if (p.get("brand") or p.get("category", "")) == top_brand
        } or {i: pool[i] for _, i in ranked[:TOP_K_EMB]}

        distinct_sizes = {norm_size(p.get("size") or p["product"]) for p in by_brand.values()} - {""}
        cands = by_brand
        size = ""
        if len(distinct_sizes) > 1 and len(by_brand) > 2:
            # Use original crisp `image` so specular inpainting never blurs white fine-print text
            s_data = ctx.ask(
                image, SIZE_PROMPT.format(brand=top_brand), schema=SIZE_SCHEMA, max_side=1024
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
                    "Stage B Gemini size match",
                    f"brand {top_brand}, size {size} -> unique id {ans}: {sized[ans]['product']}",
                    info={
                        **candidates_info(top_brand, size, sized),
                        "category": top_cat,
                        "category_confidence": cat_conf,
                        "brand_confidence": brand_conf,
                    },
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
        res = ctx.ask(
            image,
            prompt,
            schema=PICK_SCHEMA,
            max_side=1024,
            logprobs=min(len(cands) + 1, 10),
        )
        data = res.data
        sku = data.get("sku_id") if isinstance(data, dict) else None
        slots = extract_slot_distributions(
            res.meta.get("logprobs"),
            valid_options=set(cands.keys()) | {-1},
            n_options=len(cands) + 1,
        )
        slot = slots[0] if slots else None
        p = cands.get(sku)
        valid_pick = p is not None
        cos_by_id = {cid: cos for cos, cid in ranked}
        chosen_cos = cos_by_id.get(sku if isinstance(sku, int) else top_id, s1)
        cos_clarity = max(0.08, min(0.985, (chosen_cos - 0.42) / 0.54))
        if slot is not None:
            conf = round(min(0.985, float(slot["confidence"])), 4)
        elif valid_pick:
            conf = round(max(0.45, min(0.96, 0.50 * brand_conf + 0.45 * cos_clarity)), 4)
        else:
            conf = round(max(0.05, min(0.32, 0.35 * cos_clarity)), 4)
        if (p is None or conf < self.min_jev_confidence) and top_id in cands:
            sku, p = top_id, cands[top_id]
            if not valid_pick:
                conf = round(max(0.05, min(0.32, 0.35 * cos_clarity)), 4)
        step_info = {
            **candidates_info(top_brand, size, cands),
            "category": top_cat,
            "category_confidence": cat_conf,
            "brand_confidence": brand_conf,
            "jev_confidence": conf,
            "abstain_recommended": bool(not valid_pick or conf < self.min_jev_confidence),
        }
        if slot:
            step_info["jev_probabilities"] = {
                str(k): round(v, 4) for k, v in (slot.get("probabilities") or {}).items()
            }
        ctx.trace.step(
            f"Stage B Gemini picks from {len(cands)} {top_brand} sister SKUs",
            f"id {sku}: {p['product']} (jev_conf {conf:.2f})" if p else f"id {sku}: none of them",
            info=step_info,
        )
        return sku if p else None
