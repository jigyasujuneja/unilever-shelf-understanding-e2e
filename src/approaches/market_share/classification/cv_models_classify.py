"""``cv_classify``: the legacy InceptionV3 Keras brand, pack, and category-routed variant classifiers.

The ``.h5`` models run on Vertex AI Endpoints (``cv_models`` in config.yaml, see
``utils/cv_models.py``). The models are coupled:
1. The brand model runs first, predicting ``brand`` and implying a category + HUL ownership hint.
2. The pack model takes that category as its 10-d one-hot input to predict ``packaging_type`` and ``category``.
3. When variant models are configured (``wash_variant``, ``skin_variant``, ``p5_variant``,
   ``oral_variant``, ``gen_variant``, ``nonhul_p6_variant``, ``nonhul_gen_variant``), each image is
   routed by its category and brand to the matching variant endpoint (with the predicted brand as
   its one-hot brand input) to predict ``base_variant`` and ``product``.
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
        "Stage 2 Category-Conditioned Pack Classifier (46 classes, 10-d category one-hot + BGR [0,1]) → "
        "Stage 3 Silo-Routed Variant Classifier (7 HUL & Non-HUL variant endpoints with brand one-hot + BGR [0,1])"
    )
    steps = [
        "Stage 1 · Brand Classifier (marketshare_brand.h5): 299×299 PIL RGB NEAREST [-1, 1] InceptionV3 predicts brand (67 classes) and infers canonical category + HUL ownership hint",
        "Stage 2 · Category-Conditioned Pack Classifier (marketshare_packtype.h5): 299×299 OpenCV BGR BILINEAR [0, 1] + 10-d category one-hot predicts packaging_type and category (46 classes)",
        "Stage 3 · Silo Router (gt-mapping.yml): Routes by HUL ownership, category, and brand to the matching specialist endpoint among 5 HUL models (wash, skin, p5, oral, gen) and 2 Non-HUL models (nonhul_p6, nonhul_gen)",
        "Stage 4 · Brand-Conditioned Variant Classifier (*_variant.h5): 299×299 OpenCV BGR BILINEAR [0, 1] + brand one-hot vector predicts base_variant and product (Brand__Variant); re-conditions pack model if variant category differs",
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
        chain = "brand -> pack -> routed variant" if self.with_variant else "brand -> pack"
        self.architecture = (
            "Phase-1 Coupled 3-Stage InceptionV3 Cascade on Vertex AI Endpoints: "
            "Stage 1 Brand Classifier (67 classes, RGB [-1,1]) → "
            "Stage 2 Category-Conditioned Pack Classifier (46 classes, 10-d category one-hot + BGR [0,1]) → "
            "Stage 3 Silo-Routed Variant Classifier (7 HUL & Non-HUL variant endpoints with brand one-hot + BGR [0,1])"
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
            f"brand={out.get('brand') or '?'} -> category hint {out.get('category_hint') or out.get('category') or 'none'}",
            info={k: out[k] for k in ("brand_class", "brand_confidence", "brand") if k in out},
        )
        if self.with_pack:
            ctx.trace.step(
                f"pack {out.get('pack_class') or '?'} ({out.get('pack_confidence', 0.0):.2f})",
                f"packaging_type={out.get('packaging_type') or '?'}, category={out.get('category') or '?'}",
                info={
                    k: out[k]
                    for k in ("category_hint", "pack_class", "pack_confidence", "packaging_type", "category")
                    if k in out
                },
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
        pred = {f: out.get(f, "") for f in self.evaluated_fields}
        return pred if any(pred.values()) else None
