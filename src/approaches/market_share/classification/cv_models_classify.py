"""``cv_classify``: the legacy InceptionV3 Keras brand, category-routed variant, and pack classifiers.

The ``.h5`` models run on Vertex AI Endpoints (``cv_models`` in config.yaml, see
``utils/cv_models.py``). The models are coupled:
1. The brand model runs first, predicting ``brand``.
2. When variant models are configured (``wash_variant``, ``skin_variant``, ``p5_variant``,
   ``oral_variant``, ``gen_variant``, ``nonhul_p6_variant``, ``nonhul_gen_variant``), the predicted
   ``brand`` selects the matching variant endpoint via ``gt-mapping.yml`` and feeds the ``brand``
   one-hot vector + image to predict ``base_variant``, ``product``, and the variant's category.
3. The pack model takes the category output from the variant model (falling back to the
   brand-implied category when variant models are not run) as its 10-d one-hot input to predict
   ``packaging_type`` and ``category``.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Context, register
from utils.cv_models import (
    VARIANT_ROLES,
    CVPipeline,
    configured_variant_roles,
    resolve_evaluated_fields,
)


@register
class CVClassify(Approach):
    name = "cv_classify"
    task = "classification"
    dataset = "hul_golden"
    models = (
        "keras-h5",
        "marketshare_brand.h5+marketshare_packtype.h5",
        "wash_variant.h5+skin_variant.h5+p5_variant.h5+oral_variant.h5+gen_variant.h5+nonhul_p6_variant.h5+nonhul_gen_variant.h5",
    )
    architecture = (
        "Phase-1 Coupled 3-Stage InceptionV3 Cascade on Vertex AI Endpoints: "
        "Stage 1 Brand Classifier (67 classes, RGB [-1,1]) → "
        "Stage 2 Silo-Routed Variant Classifier (7 HUL & Non-HUL variant endpoints selected & conditioned by brand one-hot + BGR [0,1]) → "
        "Stage 3 Variant-Category-Conditioned Pack Classifier (46 classes, 10-d category one-hot from variant output + BGR [0,1])"
    )
    steps = [
        "Stage 1 · Brand Classifier (marketshare_brand.h5): 299×299 PIL RGB NEAREST [-1, 1] InceptionV3 predicts brand (67 classes)",
        "Stage 2 · Silo Router (gt-mapping.yml): Uses predicted brand to select the matching specialist endpoint among 5 HUL models (wash, skin, p5, oral, gen) and 2 Non-HUL models (nonhul_p6, nonhul_gen)",
        "Stage 3 · Brand-Conditioned Variant Classifier (*_variant.h5): 299×299 OpenCV BGR BILINEAR [0, 1] + brand one-hot vector predicts base_variant, product (Brand__Variant), and gt-mapping.yml category",
        "Stage 4 · Category-Conditioned Pack Classifier (marketshare_packtype.h5): 299×299 OpenCV BGR BILINEAR [0, 1] + 10-d category one-hot from variant model output predicts packaging_type and category (46 classes)",
        "Stage 5 · 5-Attribute Taxonomy Scoring: Evaluates product, brand, base_variant, packaging_type, and category against ground truth",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        var_roles = configured_variant_roles(config)
        self.evaluated_fields = tuple(
            self.evaluated_fields or resolve_evaluated_fields(self.dataset, with_variants=bool(var_roles))
        )
        fset = set(self.evaluated_fields)
        self.with_pack = bool({"packaging_type", "category"} & fset)
        self.with_variant = bool({"product", "base_variant", "variant"} & fset) and (
            bool(var_roles) or self.dataset == "hul_variants"
        )
        roles = ["brand"]
        if self.with_pack:
            roles.append("pack")
        if self.with_variant:
            roles.extend(var_roles or list(VARIANT_ROLES))
        self.pipeline = CVPipeline.from_config(config, roles)
        runners = list(self.pipeline.runners.values())
        self.resolved_model = "+".join(r.name for r in runners)
        self.model_uris = [r.source_uri for r in runners]
        chain = "brand -> routed variant -> pack" if self.with_variant else "brand -> pack"
        self.architecture = (
            "Phase-1 Coupled 3-Stage InceptionV3 Cascade on Vertex AI Endpoints: "
            "Stage 1 Brand Classifier (67 classes, RGB [-1,1]) → "
            "Stage 2 Silo-Routed Variant Classifier (7 HUL & Non-HUL variant endpoints selected & conditioned by brand one-hot + BGR [0,1]) → "
            "Stage 3 Variant-Category-Conditioned Pack Classifier (46 classes, 10-d category one-hot from variant output + BGR [0,1])"
            if self.with_variant
            else "Phase-1 Coupled 2-Stage InceptionV3 Cascade on Vertex AI Endpoints: Stage 1 Brand Classifier → Stage 2 Category-Conditioned Pack Classifier"
        )
        ctx.trace.step(
            f"Legacy CV models: {self.resolved_model}",
            f"{chain} for {self.dataset} ({', '.join(self.evaluated_fields)})",
            info={"models": self.model_uris, "fields": list(self.evaluated_fields)},
        )

    def identify(self, image: Image.Image, ctx: Context) -> dict | None:
        out = self.pipeline.predict_coupled(
            image,
            with_pack=self.with_pack,
            with_variant=self.with_variant,
        )
        ctx.trace.step(
            f"brand {out.get('brand_class') or '?'} ({out.get('brand_confidence', 0.0):.2f})",
            f"brand={out.get('brand') or '?'} -> variant role {out.get('variant_role') or 'none'}",
            info={k: out[k] for k in ("brand_class", "brand_confidence", "brand") if k in out},
        )
        if self.with_variant and out.get("variant_role"):
            ctx.trace.step(
                f"{out['variant_role']} {out.get('variant_class') or '?'} ({out.get('variant_confidence', 0.0):.2f})",
                f"routed_category={out.get('routed_category') or '?'}, brand_hint={out.get('brand_hint') or '?'} "
                f"-> base_variant={out.get('base_variant') or '?'}, product={out.get('product') or '?'}",
                info={
                    k: out[k]
                    for k in (
                        "variant_role",
                        "routed_category",
                        "brand_hint",
                        "variant_class",
                        "variant_confidence",
                        "base_variant",
                        "product",
                    )
                    if k in out
                },
            )
        if self.with_pack:
            ctx.trace.step(
                f"pack {out.get('pack_class') or '?'} ({out.get('pack_confidence', 0.0):.2f})",
                f"category_hint={out.get('category_hint') or '?'} -> packaging_type={out.get('packaging_type') or '?'}, category={out.get('category') or '?'}",
                info={
                    k: out[k]
                    for k in ("category_hint", "pack_class", "pack_confidence", "packaging_type", "category")
                    if k in out
                },
            )
        pred = {f: out.get(f, "") for f in self.evaluated_fields}
        return pred if any(pred.values()) else None


@register
class CVPromoClassify(CVClassify):
    """Legacy 9-Model InceptionV3 Cascade evaluated on W9 Promotional & Multipack SKUs (`hul_promos`)."""

    name = "cv_promo_classify"
    use_case = "merchandising"
    task = "promo_detection"
    dataset = "hul_promos"

    def setup(self, config: dict, ctx: Context) -> None:
        if not self.evaluated_fields:
            self.evaluated_fields = (
                "product",
                "brand",
                "base_variant",
                "category",
                "packaging_type",
                "is_hul",
                "promo_type",
            )
        super().setup(config, ctx)

    def identify(self, image: Image.Image, ctx: Context) -> dict | None:
        from approaches.merchandising.promo_detection.jev_promo_verifier import _normalize_promo_token

        out = self.pipeline.predict_coupled(
            image,
            with_pack=self.with_pack,
            with_variant=self.with_variant,
        )
        promo_tok = _normalize_promo_token(
            f"{out.get('variant') or ''} {out.get('gt_variant') or ''} {out.get('variant_class') or ''}"
        )
        out["promo_type"] = promo_tok
        if "is_hul" not in out:
            from utils.taxonomy.brands import HUL_BRANDS

            out["is_hul"] = str(out.get("brand") or "").lower() in HUL_BRANDS
        ctx.trace.step(
            f"Legacy 9-Model CV Promo Cascade: {out.get('brand') or '?'} / {out.get('base_variant') or '?'}",
            f"brand={out.get('brand') or '?'}, variant={out.get('base_variant') or '?'}, pack={out.get('packaging_type') or '?'}, promo_type={promo_tok}",
            info={
                k: out[k]
                for k in (
                    "brand",
                    "base_variant",
                    "product",
                    "packaging_type",
                    "category",
                    "is_hul",
                    "promo_type",
                    "brand_confidence",
                    "variant_confidence",
                    "pack_confidence",
                )
                if k in out
            },
        )
        pred = {f: out.get(f, "") for f in self.evaluated_fields}
        return pred if any(v not in ("", None) for v in pred.values()) else None

