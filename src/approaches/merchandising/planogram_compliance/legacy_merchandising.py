"""Legacy GT & MT Merchandising Pipelines (`legacy_gtmerch_planogram`, `legacy_gtmerch_e2e`, `legacy_mt_planogram`, `legacy_mt_e2e`).

Benchmarks the legacy customer CV models across General Trade (GT) and Modern Trade (MT)
Merchandising use cases:
- GT Merchandising:
  - Detector: ``gtmerch_asset_sku_intermediary_17092026.pt`` (2-class YOLOv8n: Class 0 = promo/display
    asset bay, Class 1 = SKU facing, exported to ONNX).
  - Classifier: ``17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5`` (209-class InceptionV3 Keras model
    trained on ``gtmerch-marketshare/product__classification.zip``).
- MT Merchandising / Market Share:
  - Detector: ``marketshare_yolo.pt`` (1-class YOLOv8n exported to ONNX).
  - Classifier: 9-model coupled InceptionV3 Keras cascade (``marketshare_brand.h5`` + 7 ``*_variant.h5``
    silo models + ``marketshare_packtype.h5``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from approaches.base import Approach, Box, Context, crop, register
from approaches.market_share.detection.onnx_yolo_detector import (
    LegacyGtmerchYolo,
    LegacyMtYolo,
)
from approaches.merchandising.planogram_compliance.bay_planogram_verifier import (
    compute_planogram_metrics,
)
from utils.awp_template_index import resolve_awp_planogram_spec
from utils.cv_models import (
    VARIANT_ROLES,
    CVPipeline,
    configured_variant_roles,
)
from utils.taxonomy.brands import HUL_BRANDS


def _attach_planogram_step(
    boxes: list[Box],
    ids: list[dict | int | None],
    image: Image.Image,
    ctx: Context,
    catalog: dict | None = None,
) -> None:
    asset_boxes = getattr(ctx, "detected_asset_boxes", None)
    spec_hint = getattr(ctx, "awp_code", None) or getattr(ctx, "image_id", None)
    planogram_spec = getattr(ctx, "planogram_spec", None) or resolve_awp_planogram_spec(spec_hint)
    plano = compute_planogram_metrics(
        boxes,
        ids,  # type: ignore[arg-type]
        catalog or {},
        asset_boxes=asset_boxes,
        image_size=(image.width, image.height),
        planogram_spec=planogram_spec,
    )
    spec_check = plano.get("spec_compliance") or {}
    spec_str = (
        f" | AWP={spec_check.get('spec_id')} ({'PASS' if spec_check.get('is_compliant') else 'FAIL'})"
        if spec_check.get("has_spec")
        else ""
    )
    ctx.trace.step(
        "Verify Bay Planogram & Share of Shelf",
        f"Scope={plano['scope_mode']}{spec_str} | HUL SOS={plano['hul_facing_sos']*100:.1f}% "
        f"({plano['hul_facings']}/{plano['total_facings']} facings), "
        f"brand-block purity={plano['brand_block_purity']*100:.1f}%, "
        f"vertical alignment={plano['vertical_block_alignment']*100:.1f}%, "
        f"OOS voids={plano['oos_void_count']} "
        f"(empty share={plano['empty_shelf_share']*100:.1f}%), {plano['distinct_brands']} brands",
        boxes=boxes,
        info=plano,
    )


@register
class LegacyGtmerchPlanogram(Approach):
    """Legacy GT Merchandising 209-Class `.h5` Classifier (`17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5`) + Stage 6 Planogram Compliance."""

    name = "legacy_gtmerch_planogram"
    use_case = "merchandising"
    task = "planogram_compliance"
    dataset = "hul_sessions"
    evaluated_fields = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul")
    models = (
        "keras-h5",
        "17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5",
    )
    architecture = (
        "Legacy GT Merchandising InceptionV3 Classifier (`17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5`, "
        "209 classes, 299x299 RGB [0,255]) -> Stage 6 Per-Plank Brand-Block, OOS Void & Class 0 Bay Planogram Engine"
    )
    steps = [
        "Setup: load legacy 209-class GT Merchandising Keras model (`17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5`) and 209 class folder mappings",
        "Stage 1: Classify every shelf facing crop into 209 GT Merchandising classes (brand, base_variant, product, category, packaging_type, is_hul)",
        "Stage 6: Route Class 0 bay vs full-image hanger scope, cluster shelf planks, detect per-row OOS void boxes, and score brand-block & bay purity",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.pipeline = CVPipeline.from_config(config, ["gtmerch"])
        runner = self.pipeline.runners["gtmerch"]
        self.resolved_model = runner.name
        self.model_uris = [runner.source_uri]
        ctx.trace.step(
            f"Legacy GT Merchandising .h5: {self.resolved_model}",
            f"209-class classifier on {self.dataset}",
            info={"models": self.model_uris, "fields": list(self.evaluated_fields or ())},
        )

    def identify(self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None) -> dict | None:  # type: ignore[override]
        out = self.pipeline.predict_gtmerch(image)
        return out if out.get("brand") or out.get("product") else None

    def identify_boxes(self, image: Image.Image, boxes: list[Box], ctx: Context) -> list[int | None]:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda b: self.identify(crop(image, b), ctx), boxes))
        ctx.trace.step(
            "Legacy GT Merchandising 209-Class Identification",
            f"Classified {sum(i is not None for i in ids)}/{len(boxes)} shelf crops with 17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5",
            boxes=boxes,
        )
        _attach_planogram_step(boxes, ids, image, ctx)  # type: ignore[arg-type]
        return ids  # type: ignore[return-value]


@register
class LegacyGtmerchE2E(Approach):
    """Legacy GT Merchandising End-to-End Pipeline: `gtmerch_asset_sku_intermediary_17092026.pt` (2-Class YOLOv8n) + `17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5` (209-Class `.h5`)."""

    name = "legacy_gtmerch_e2e"
    use_case = "merchandising"
    task = "end_to_end"
    dataset = "gtmerch_golden"
    evaluated_fields = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul")
    models = (
        "keras-h5",
        "gtmerch-yolo-legacy-pt",
        "17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5",
    )
    also_calls = [
        "gtmerch_asset_sku_intermediary_17092026.pt",
        "17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5",
    ]
    architecture = (
        "Legacy GT Merchandising E2E Cascade: Stage 1 2-Class YOLOv8n (`gtmerch_asset_sku_intermediary_17092026.pt`: "
        "Class 0 Promo Bay + Class 1 SKU) -> Stage 2 209-Class InceptionV3 (`17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5`) "
        "-> Stage 6 Bay Planogram Engine"
    )
    steps = [
        "Stage 1 · Legacy GT Merchandising YOLOv8n (`gtmerch_asset_sku_intermediary_17092026.pt`): 2-class detection separating Class 0 display/promo bays from Class 1 SKU facings",
        "Stage 2 · Legacy GT Merchandising InceptionV3 (`17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5`): 299x299 RGB [0,255] 209-class softmax over each detected SKU crop",
        "Stage 6 · Bay Planogram & OOS Compliance: Scopes SKUs to detected Class 0 display bays (or full-image hanger fallback) and scores brand-block purity & OOS voids",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.detector = LegacyGtmerchYolo()
        self.detector.setup(config, ctx)
        self.classifier = LegacyGtmerchPlanogram()
        self.classifier.dataset = self.dataset
        self.classifier.setup(config, ctx)
        self.resolved_model = "gtmerch_asset_sku_intermediary_17092026.pt+17Sep26-moc10-model-gtmerch-pr-0050-0.9480.h5"

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return self.detector.detect(image, ctx)

    def identify_boxes(self, image: Image.Image, boxes: list[Box], ctx: Context) -> list[int | None]:
        return self.classifier.identify_boxes(image, boxes, ctx)


@register
class LegacyMtPlanogram(Approach):
    """Legacy Modern Trade 9-Model `.h5` Cascade (`marketshare_brand.h5` + 7 `*_variant.h5` + `marketshare_packtype.h5`) + Stage 6 Planogram Compliance."""

    name = "legacy_mt_planogram"
    use_case = "merchandising"
    task = "planogram_compliance"
    dataset = "mt_json_golden"
    evaluated_fields = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul")
    models = (
        "keras-h5",
        "marketshare_brand.h5+marketshare_packtype.h5",
    )
    architecture = (
        "Legacy MT 9-Model Coupled InceptionV3 Cascade (`marketshare_brand.h5` -> 7 Silo-Routed `*_variant.h5` "
        "-> `marketshare_packtype.h5`) -> Stage 6 Per-Plank Brand-Block & OOS Void Planogram Engine"
    )
    steps = [
        "Stage 1 · Brand Classifier (`marketshare_brand.h5`): 299x299 RGB [-1,1] InceptionV3 predicts brand",
        "Stage 2 · Silo-Routed Variant Classifier (7 `*_variant.h5` models): Brand one-hot + 299x299 BGR [0,1] predicts base_variant, product, and category",
        "Stage 3 · Category-Conditioned Pack Classifier (`marketshare_packtype.h5`): 10-d category one-hot + 299x299 BGR [0,1] predicts packaging_type",
        "Stage 6 · Planogram & OOS Compliance: Clusters shelf planks, detects per-row OOS void boxes, and scores brand-block purity & HUL Share of Shelf",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        var_roles = configured_variant_roles(config) or list(VARIANT_ROLES)
        roles = ["brand", "pack", *var_roles]
        self.pipeline = CVPipeline.from_config(config, roles)
        runners = list(self.pipeline.runners.values())
        self.resolved_model = "+".join(r.name for r in runners)
        self.model_uris = [r.source_uri for r in runners]

    def identify(self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None) -> dict | None:  # type: ignore[override]
        out = self.pipeline.predict_coupled(image, with_pack=True, with_variant=True)
        if "is_hul" not in out:
            out["is_hul"] = str(out.get("brand") or "").lower() in HUL_BRANDS
        pred = {f: out.get(f, "") for f in (self.evaluated_fields or ())}
        return pred if any(v not in ("", None) for v in pred.values()) else None

    def identify_boxes(self, image: Image.Image, boxes: list[Box], ctx: Context) -> list[int | None]:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda b: self.identify(crop(image, b), ctx), boxes))
        ctx.trace.step(
            "Legacy MT 9-Model Cascade Identification",
            f"Classified {sum(i is not None for i in ids)}/{len(boxes)} shelf crops with {self.resolved_model}",
            boxes=boxes,
        )
        _attach_planogram_step(boxes, ids, image, ctx)  # type: ignore[arg-type]
        return ids  # type: ignore[return-value]


@register
class LegacyMtE2E(Approach):
    """Legacy Modern Trade End-to-End Pipeline: `marketshare_yolo.pt` (1-Class YOLOv8n) + 9-Model `.h5` Cascade (`marketshare_brand.h5` + 7 `*_variant.h5` + `marketshare_packtype.h5`)."""

    name = "legacy_mt_e2e"
    use_case = "merchandising"
    task = "end_to_end"
    dataset = "mt_json_golden"
    evaluated_fields = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul")
    models = (
        "keras-h5",
        "marketshare-yolo-legacy-pt",
        "marketshare_brand.h5+marketshare_packtype.h5",
    )
    also_calls = [
        "marketshare_yolo.pt",
        "marketshare_brand.h5",
        "marketshare_packtype.h5",
    ]
    architecture = (
        "Legacy MT E2E Cascade: Stage 1 1-Class YOLOv8n (`marketshare_yolo.pt`) -> "
        "Stage 2 9-Model Coupled InceptionV3 Cascade (`marketshare_brand.h5` + 7 `*_variant.h5` + `marketshare_packtype.h5`) "
        "-> Stage 6 Planogram Engine"
    )
    steps = [
        "Stage 1 · Legacy MT YOLOv8n (`marketshare_yolo.pt`): 1-class 640x640 shelf object detection",
        "Stage 2 · Legacy MT 9-Model InceptionV3 Cascade: Brand -> Silo-Routed Variant -> Category-Conditioned Pack classification on each detected box",
        "Stage 6 · Planogram & OOS Compliance: Clusters shelf planks, detects OOS void boxes, and scores brand-block purity & HUL Share of Shelf",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        self.detector = LegacyMtYolo()
        self.detector.setup(config, ctx)
        self.classifier = LegacyMtPlanogram()
        self.classifier.dataset = self.dataset
        self.classifier.setup(config, ctx)
        self.resolved_model = f"marketshare_yolo.pt+{self.classifier.resolved_model}"

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        return self.detector.detect(image, ctx)

    def identify_boxes(self, image: Image.Image, boxes: list[Box], ctx: Context) -> list[int | None]:
        return self.classifier.identify_boxes(image, boxes, ctx)
