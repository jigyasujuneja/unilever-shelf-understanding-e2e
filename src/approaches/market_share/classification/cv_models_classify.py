"""``cv_classify``: the legacy InceptionV3 Keras brand + pack classifiers on the HUL datasets.

The ``.h5`` models run on Vertex AI Endpoints (``cv_models`` in config.yaml, see
``utils/cv_models.py``). The two models are coupled: the brand model runs first, and the category
implied by its brand is the pack model's category input. ``brand`` is scored from the brand model,
``packaging_type`` and ``category`` from the pack model.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Approach, Context, register
from utils.cv_models import CVPipeline, resolve_evaluated_fields


@register
class CVClassify(Approach):
    name = "cv_classify"
    task = "classification"
    dataset = "hul_golden"
    models = ("keras-h5", "marketshare_brand.h5+marketshare_packtype.h5")
    architecture = "Legacy InceptionV3 Keras models on Vertex AI Endpoints: brand -> pack (category hint)"
    steps = [
        "Brand Vertex AI Endpoint (legacy .h5): 299x299 InceptionV3 argmax -> brand_data class -> brand",
        "Map the predicted brand to its category (main's legacy brand -> category taxonomy)",
        "Pack Vertex AI Endpoint (legacy .h5), with that category as its one-hot category input",
        "Score brand from the brand model; packaging_type and category from the pack's Category__PackType",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.evaluated_fields = tuple(self.evaluated_fields or resolve_evaluated_fields(self.dataset))
        self.with_pack = bool({"packaging_type", "category"} & set(self.evaluated_fields))
        # The brand model always runs: it is scored, or its category feeds the pack model.
        self.pipeline = CVPipeline.from_config(config, ["brand", "pack"] if self.with_pack else ["brand"])
        runners = list(self.pipeline.runners.values())
        self.resolved_model = "+".join(r.name for r in runners)
        self.model_uris = [r.source_uri for r in runners]
        self.architecture = f"Legacy InceptionV3 Keras models, brand -> pack ({' -> '.join(self.model_uris)})"
        ctx.trace.step(
            f"Legacy CV models: {self.resolved_model}",
            f"brand -> pack (category hint) for {self.dataset} ({', '.join(self.evaluated_fields)})",
            info={"models": self.model_uris, "fields": list(self.evaluated_fields)},
        )

    def identify(self, image: Image.Image, ctx: Context) -> dict | None:
        out = self.pipeline.predict_coupled(image, with_pack=self.with_pack)
        ctx.trace.step(
            f"brand {out['brand_class'] or '?'} ({out['brand_confidence']:.2f})",
            f"brand={out['brand'] or '?'} -> category hint {out.get('category_hint') or 'none'}",
            info={k: out[k] for k in ("brand_class", "brand_confidence", "brand") if k in out},
        )
        if self.with_pack:
            ctx.trace.step(
                f"pack {out['pack_class'] or '?'} ({out['pack_confidence']:.2f})",
                f"packaging_type={out['packaging_type'] or '?'}, category={out['category'] or '?'}",
                info={k: out[k] for k in ("category_hint", "pack_class", "pack_confidence",
                                          "packaging_type", "category")},
            )
        pred = {f: out.get(f, "") for f in self.evaluated_fields}
        return pred if any(pred.values()) else None
