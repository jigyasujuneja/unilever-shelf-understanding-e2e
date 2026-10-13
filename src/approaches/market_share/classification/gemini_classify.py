"""Single-Pass Zero-Shot & SFT Gemini Classification (Classification tab).

Evaluates the base or LoRA-tuned Gemini model (``gemini-3.5-flash-lite``, ``gemini-3.8-flash``,
``marketshare-hul-golden-5attr-sft-euw4-gemini-3-5-flash-lite``, or
``marketshare-hul-golden-clean-sft-euw4-gemini-3-5-flash-lite``) in a single structured VLM call per
photo using the task's structured JSON schema:
  - ``hul_golden`` / ``hul_variants``:
      * 5-attribute mode (default): predicts all 5 fields
        ``{"product": "<brand>__<base_variant>", "brand": "<brand>", "base_variant": "<base_variant>", "packaging_type": "<pt>", "category": "<cat>"}``
        with catalog-grounded variant canonicalization and taxonomy consistency projection.
      * 3-attribute mode (``--fields 3`` / ``--fields brand,packaging_type,category``): predicts
        ``{"brand": "<brand>", "packaging_type": "<pt>", "category": "<cat>"}``
  - ``hul_packs``:
      * 3-attribute mode: predicts ``{"brand": "<brand>", "packaging_type": "<pt>", "category": "<cat>"}``
  - ``hul_brands``:
      * 1-attribute mode: predicts ``{"brand": "<brand>"}``
  - ``products``: picks ``sku_id`` from the catalog in one call.
"""

from __future__ import annotations

from collections import defaultdict

from PIL import Image

from approaches.base import Approach, Context, register
from approaches.market_share.classification.hierarchy_classify import PICK_PROMPT, PICK_SCHEMA
from utils import dataset
from utils.cv_models import resolve_evaluated_fields
from utils.gemini_fine_tuning.hul_golden_sft import (
    ALLOWED_CATEGORIES as HUL_GOLDEN_CATEGORIES,
    ALLOWED_PACKAGING_TYPES as HUL_GOLDEN_PACKAGING_TYPES,
    HUL_GOLDEN_5ATTR_INFERENCE_PROMPT,
    HUL_GOLDEN_INFERENCE_PROMPT as HUL_GOLDEN_SINGLE_PROMPT,
    catalog_variant_lookup,
    normalize_hul_golden_brand,
    normalize_hul_golden_variant,
)
from utils.gemini_fine_tuning.tuning import (
    VARIANT_BRANDS,
    VARIANT_CATEGORIES,
    clean_variant_name,
)
from utils.taxonomy.packaging import project_to_valid_category_pack

HUL_GOLDEN_5ATTR_SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "brand": {"type": "string"},
        "base_variant": {"type": "string"},
        "packaging_type": {"type": "string", "enum": list(HUL_GOLDEN_PACKAGING_TYPES)},
        "category": {"type": "string", "enum": list(HUL_GOLDEN_CATEGORIES)},
    },
    "required": ["product", "brand", "base_variant", "packaging_type", "category"],
}

HUL_GOLDEN_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
        "packaging_type": {"type": "string", "enum": list(HUL_GOLDEN_PACKAGING_TYPES)},
        "category": {"type": "string", "enum": list(HUL_GOLDEN_CATEGORIES)},
    },
    "required": ["brand", "packaging_type", "category"],
}

INPUT_SINGLE_PROMPT = (
    "Classify the retail product shown in the image by its category, brand, and variant all together.\n"
    f"Valid categories: {', '.join(VARIANT_CATEGORIES)} "
    "(gen=hair care shampoos/conditioners/oils; oral=toothpaste/toothbrush; "
    "p5=foods/tea/coffee/health drinks/dishwash/surface cleaners; "
    "skin=skin care creams/lotions/face wash/deodorants/cosmetics; "
    "wash=laundry detergents/fabric care/bathing soap bars).\n"
    f"Valid brands: {', '.join(VARIANT_BRANDS)}.\n"
    "For variant, read the sub-brand/variant name and pack format printed on the pack and format it as "
    "Title_Case_With_Underscores (e.g. Active_Care_Shampoo, Black_Shine_Sachet, Easy_Wash_Pouch).\n"
    'Return ONLY JSON like {"category": "gen", "brand": "Clear", "variant": "Active_Care_Shampoo"}.'
)
INPUT_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(VARIANT_CATEGORIES)},
        "brand": {"type": "string", "enum": list(VARIANT_BRANDS)},
        "variant": {"type": "string"},
    },
    "required": ["category", "brand", "variant"],
}

INPUT_CAT_BRAND_PROMPT = (
    "Classify the retail product shown in the image by its category and brand together.\n"
    f"Valid categories: {', '.join(VARIANT_CATEGORIES)} "
    "(gen=hair care shampoos/conditioners/oils; oral=toothpaste/toothbrush; "
    "p5=foods/tea/coffee/health drinks/dishwash/surface cleaners; "
    "skin=skin care creams/lotions/face wash/deodorants/cosmetics; "
    "wash=laundry detergents/fabric care/bathing soap bars).\n"
    f"Valid brands: {', '.join(VARIANT_BRANDS)}.\n"
    'Return ONLY JSON like {"category": "gen", "brand": "Clear"}.'
)
INPUT_CAT_BRAND_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(VARIANT_CATEGORIES)},
        "brand": {"type": "string", "enum": list(VARIANT_BRANDS)},
    },
    "required": ["category", "brand"],
}

INPUT_BRAND_ONLY_PROMPT = (
    "Classify the retail product shown in the image by its brand.\n"
    f"Valid brands: {', '.join(VARIANT_BRANDS)}.\n"
    'Return ONLY JSON like {"brand": "Clear"}.'
)
INPUT_BRAND_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string", "enum": list(VARIANT_BRANDS)},
    },
    "required": ["brand"],
}

BRAND_SINGLE_PROMPT = (
    "Classify the retail product shown in the image by its brand.\n"
    'Return ONLY JSON like {"brand": "Dove"}.'
)
BRAND_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
    },
    "required": ["brand"],
}


@register
class GeminiClassify(Approach):
    name = "gemini_classify"
    task = "classification"
    dataset = "hul_golden"
    architecture = (
        "Gemini single-pass 5-attribute classification (product, brand, base_variant, packaging_type, category) "
        "with catalog-grounded variant canonicalization and taxonomy consistency"
    )
    steps = [
        "Single Gemini call with the image and structured 5-attribute JSON schema",
        "Normalize brand and snap base_variant to canonical catalog/gt-mapping.yml SKU variant",
        "Enforce (brand, base_variant, category, packaging_type) taxonomy consistency",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        if self.dataset != "products" and not self.evaluated_fields:
            self.evaluated_fields = tuple(resolve_evaluated_fields(self.dataset, with_variants=True))
        self.catalog = dataset.catalog(name=self.dataset) if self.dataset in ("products", *dataset.RPC_FORMAT_SETS) else {}
        self.variants_by_brand, self.sku_meta_by_bv = catalog_variant_lookup(self.catalog)
        self.brand_cats: dict[str, set[str]] = defaultdict(set)
        self.valid_pairs: set[tuple[str, str]] = set()
        for p in self.catalog.values():
            b, c = normalize_hul_golden_brand(p.get("brand")), str(p.get("category") or "")
            if b and c:
                self.brand_cats[b].add(c)
            for pt in [p.get("packaging_type"), *(p.get("allowed_packaging_types") or [])]:
                if c and pt:
                    self.valid_pairs.add((c, str(pt)))

    def _project_5attr(self, brand: str, raw_bv: str, pt: str, cat: str) -> dict[str, str]:
        bv = normalize_hul_golden_variant(
            raw_bv,
            brand=brand,
            catalog_variants_by_brand=self.variants_by_brand,
        )
        sku_meta = self.sku_meta_by_bv.get((brand.lower(), bv.lower())) if (brand and bv) else None
        if sku_meta and sku_meta.get("category"):
            cat = sku_meta["category"]
        else:
            cats = self.brand_cats.get(brand)
            if cats and len(cats) == 1 and cat not in cats:
                cat = next(iter(cats))
        if cat and pt and self.valid_pairs and (cat, pt) not in self.valid_pairs:
            proj = project_to_valid_category_pack(cat, pt)
            if proj != pt and (cat, proj) in self.valid_pairs:
                pt = proj
        prod = f"{brand}__{bv}" if (brand and bv and bv != brand) else (brand or bv)
        return {
            "product": prod,
            "brand": brand,
            "base_variant": bv,
            "variant": bv,
            "packaging_type": pt,
            "category": cat,
        }

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | dict | None:
        fields = tuple(self.evaluated_fields or ())
        fset = set(fields)

        if self.dataset == "hul_brands" or fields == ("brand",):
            if self.dataset == "hul_variants":
                res = ctx.ask(image, INPUT_BRAND_ONLY_PROMPT, schema=INPUT_BRAND_ONLY_SCHEMA, max_side=1024)
            else:
                res = ctx.ask(image, BRAND_SINGLE_PROMPT, schema=BRAND_SINGLE_SCHEMA, max_side=1024)
            data = res.data if isinstance(res.data, dict) else {}
            raw_b = str(data.get("brand") or "")
            brand = normalize_hul_golden_brand(raw_b) if self.dataset != "hul_variants" else raw_b
            pred = {"brand": brand or raw_b}
            ctx.trace.step(
                "Gemini single-pass (brand)",
                f"brand={pred['brand'] or '?'}",
                info=pred,
            )
            return pred if pred["brand"] else None

        if self.dataset == "hul_variants" and fields == ("category", "brand"):
            res = ctx.ask(image, INPUT_CAT_BRAND_PROMPT, schema=INPUT_CAT_BRAND_SCHEMA, max_side=1024)
            data = res.data if isinstance(res.data, dict) else {}
            cat = str(data.get("category") or "")
            brand = str(data.get("brand") or "")
            pred = {"category": cat, "brand": brand}
            ctx.trace.step(
                "Gemini single-pass (category + brand)",
                f"category={cat or '?'}, brand={brand or '?'}",
                info=pred,
            )
            return pred if (cat or brand) else None

        if self.dataset == "hul_variants" and fields == ("category", "brand", "variant"):
            res = ctx.ask(image, INPUT_SINGLE_PROMPT, schema=INPUT_SINGLE_SCHEMA, max_side=1024)
            data = res.data if isinstance(res.data, dict) else {}
            cat = str(data.get("category") or "")
            brand = str(data.get("brand") or "")
            raw_var = str(data.get("variant") or "")
            var = clean_variant_name(raw_var) or raw_var.strip()
            pred = {"category": cat, "brand": brand, "variant": var}
            ctx.trace.step(
                "Gemini single-pass (category + brand + variant)",
                f"category={cat or '?'}, brand={brand or '?'}, variant={var or '?'}",
                info=pred,
            )
            return pred if (cat or brand or var) else None

        # 5-attribute mode (default for hul_golden and hul_variants when product/base_variant are evaluated)
        if self.dataset in ("hul_golden", "hul_variants") and (not fset or bool({"product", "base_variant"} & fset)):
            res = ctx.ask(image, HUL_GOLDEN_5ATTR_INFERENCE_PROMPT, schema=HUL_GOLDEN_5ATTR_SCHEMA, max_side=1024)
            data = res.data if isinstance(res.data, dict) else {}
            brand = normalize_hul_golden_brand(str(data.get("brand") or ""))
            raw_bv = str(data.get("base_variant") or data.get("variant") or data.get("product") or "")
            pt = str(data.get("packaging_type") or "")
            cat = str(data.get("category") or "")
            pred = self._project_5attr(brand, raw_bv, pt, cat)
            ctx.trace.step(
                "Gemini single-pass (5-attr: product + brand + base_variant + packaging_type + category)",
                f"product={pred['product'] or '?'}, packaging_type={pred['packaging_type'] or '?'}, category={pred['category'] or '?'}",
                info=pred,
            )
            return pred if any(pred.get(k) for k in ("product", "brand", "base_variant", "packaging_type", "category")) else None

        # 3-attribute mode (--fields 3 / hul_packs)
        if self.dataset in ("hul_golden", "hul_packs") or ("packaging_type" in fset and not ({"product", "base_variant"} & fset)):
            res = ctx.ask(image, HUL_GOLDEN_SINGLE_PROMPT, schema=HUL_GOLDEN_SINGLE_SCHEMA, max_side=1024)
            data = res.data if isinstance(res.data, dict) else {}
            brand = normalize_hul_golden_brand(str(data.get("brand") or ""))
            pt = str(data.get("packaging_type") or "")
            cat = str(data.get("category") or "")
            pred = {"brand": brand, "packaging_type": pt, "category": cat}
            ctx.trace.step(
                "Gemini single-pass (brand + packaging_type + category)",
                f"brand={brand or '?'}, packaging_type={pt or '?'}, category={cat or '?'}",
                info=pred,
            )
            return pred if (brand or pt or cat) else None

        pool = (
            {i: p for i, p in self.catalog.items() if i in allowed_ids}
            if allowed_ids
            else self.catalog
        ) or self.catalog
        prompt = PICK_PROMPT.format(
            catalog="\n".join(
                f"{i}: {p.get('brand', '')} | {p['product']} | {p.get('category', '')}"
                for i, p in pool.items()
            )
        )
        data = ctx.ask(image, prompt, schema=PICK_SCHEMA, max_side=1024).data
        sku = data.get("sku_id") if isinstance(data, dict) else None
        p = pool.get(sku)
        ctx.trace.step(
            f"Gemini single-pass ({len(pool)} catalog items)",
            f"id {sku}: {p['product']}" if p else f"id {sku}: none of them",
        )
        return sku if p else None
