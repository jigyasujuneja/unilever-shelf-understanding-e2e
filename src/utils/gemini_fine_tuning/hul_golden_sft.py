"""HUL Golden 5-attribute (product, brand, base_variant, packaging_type, category) & 3-attribute Gemini SFT helpers.

``build_clean_splits_in_gcp`` (``shelf-bench tune --clean-hul-golden``) works entirely in GCS:
1. Reads the canonical ``hul_golden/rpc.json`` splits (already cleaned by ``golden_builder``).
2. Enforces strict URI disjointness AND store-visit ``session_id`` disjointness across ``val``,
   ``test``, and ``train`` (preventing any capture-session overlap with held-out splits).
3. Validates all 5 fields (``product``, ``brand``, ``base_variant``, ``packaging_type``, ``category``)
   against canonical taxonomy rules (excluding synthetic catch-all placeholders like ``Others``,
   ``Unknown``, ``Brand_Level``, ``Non_HUL_Competitor``).
4. Stratifies round-robin across ``(category, packaging_type, brand, base_variant)`` and writes
   Vertex AI SFT JSONL to ``derived_manifests/hul_golden/sft_5attr/`` (or ``sft_3attr/`` when
   ``fields_mode=3``).
5. Does NOT launch a Vertex AI tuning job unless ``launch_sft=True`` is explicitly passed.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from utils.taxonomy.categories import CANONICAL_CATEGORIES
from utils.taxonomy.packaging import CANONICAL_PACKAGING_TYPES

HUL_GOLDEN_5ATTR_FIELDS: tuple[str, ...] = (
    "product",
    "brand",
    "base_variant",
    "packaging_type",
    "category",
)
HUL_GOLDEN_3ATTR_FIELDS: tuple[str, ...] = (
    "brand",
    "packaging_type",
    "category",
)

ALLOWED_CATEGORIES: tuple[str, ...] = tuple(sorted(CANONICAL_CATEGORIES))
ALLOWED_PACKAGING_TYPES: tuple[str, ...] = tuple(
    sorted(set(CANONICAL_PACKAGING_TYPES) - {"Pack", "Spray", "Stick"})
)

FORBIDDEN_LABELS: frozenset[str] = frozenset({
    "",
    "Unknown",
    "Others",
    "Other",
    "unknown",
    "others",
    "other",
    "None",
    "null",
    "NA",
    "na",
    "Brand_Level",
    "Non_HUL_Competitor",
})

DEFAULT_PROMPT_TEXT: str = (
    "Classify the retail product in this image by its brand, packaging_type, and category. "
    f"Allowed categories: {', '.join(ALLOWED_CATEGORIES)}. "
    f"Allowed packaging_types: {', '.join(ALLOWED_PACKAGING_TYPES)}. "
    'Return ONLY valid JSON: {"brand": "<brand>", "packaging_type": "<packaging_type>", "category": "<category>"}'
)

DEFAULT_5ATTR_PROMPT_TEXT: str = (
    "Classify the retail product in this image by all 5 attributes: product (Brand__Variant), "
    "brand, base_variant, packaging_type, and category. "
    f"Allowed categories: {', '.join(ALLOWED_CATEGORIES)}. "
    f"Allowed packaging_types: {', '.join(ALLOWED_PACKAGING_TYPES)}. "
    "Return ONLY valid JSON: "
    '{"product": "<brand>__<base_variant>", "brand": "<brand>", "base_variant": "<base_variant>", '
    '"packaging_type": "<packaging_type>", "category": "<category>"}'
)

_PACKAGING_CUES_SUFFIX: str = (
    "Always output the canonical English brand and base_variant in Title_Case_With_Underscores "
    "(never regional script, e.g. Red_Label, Taj_Mahal, 3_Roses, Taaza, Biolage; "
    "base_variant e.g. Daily_Shine_Conditioner_Tube, Easy_Wash_Pouch, Classic_Salted). "
    "Packaging cues: Bottle=narrow neck or cap narrower than body; Jar=wide-mouth screw lid nearly as wide as body; "
    "Tube=squeezable tapered body; Pouch=flexible stand-up/spout/refill bag; "
    "Wrapper=flat flow-wrapped bar or noodle pillow pack; Sachet=small single-serve flat tear packet."
)

HUL_GOLDEN_INFERENCE_PROMPT: str = (
    f"{DEFAULT_PROMPT_TEXT} "
    "Always output the canonical English brand name in Title_Case_With_Underscores "
    "(never regional script, e.g. Red_Label, Taj_Mahal, 3_Roses, Taaza, Biolage). "
    "Packaging cues: Bottle=narrow neck or cap narrower than body; Jar=wide-mouth screw lid nearly as wide as body; "
    "Tube=squeezable tapered body; Pouch=flexible stand-up/spout/refill bag; "
    "Wrapper=flat flow-wrapped bar or noodle pillow pack; Sachet=small single-serve flat tear packet."
)

HUL_GOLDEN_5ATTR_INFERENCE_PROMPT: str = f"{DEFAULT_5ATTR_PROMPT_TEXT} {_PACKAGING_CUES_SUFFIX}"

_REGIONAL_AND_OCR_BRAND_ALIASES: dict[str, str] = {
    "লাল_label": "Red_Label",
    "লাল": "Red_Label",
    "redlabel": "Red_Label",
    "brooke_bond_red_label": "Red_Label",
    "তাজ": "Taj_Mahal",
    "তাজ_মহল": "Taj_Mahal",
    "taj": "Taj_Mahal",
    "tajmahal": "Taj_Mahal",
    "brooke_bond_taj_mahal": "Taj_Mahal",
    "তাজা": "Taaza",
    "brooke_bond_taaza": "Taaza",
    "biolane": "Biolage",
    "love_beauty_and_planet": "Lbp",
    "love_beauty_&_planet": "Lbp",
    "love_beauty_planet": "Lbp",
    "glow_&_lovely": "Glow_And_Lovely",
    "fair_&_lovely": "Fair_And_Lovely",
    "head_&_shoulders": "Head_And_Shoulders",
    "head_and_shoulder": "Head_And_Shoulders",
}


def normalize_hul_golden_brand(raw_brand: str | None) -> str:
    """Normalize a predicted brand string to its canonical English HUL Golden brand name."""
    from utils.taxonomy.brands import BRAND_ALIAS_MAP, HUL_BRANDS, canonicalize_brand_name
    from utils.taxonomy.categories import NON_HUL_BRAND_CATEGORY_MAP, _damerau_levenshtein_le1

    cleaned = re.sub(r"[\s\-]+", "_", str(raw_brand or "").strip()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        return ""
    low = cleaned.lower()
    if low in _REGIONAL_AND_OCR_BRAND_ALIASES:
        return _REGIONAL_AND_OCR_BRAND_ALIASES[low]
    canon = canonicalize_brand_name(cleaned)
    canon_low = canon.lower()
    if canon_low in _REGIONAL_AND_OCR_BRAND_ALIASES:
        return _REGIONAL_AND_OCR_BRAND_ALIASES[canon_low]
    known_brands = HUL_BRANDS | set(NON_HUL_BRAND_CATEGORY_MAP)
    if canon_low in known_brands:
        return canon
    if len(canon_low) >= 6:
        matches = [b for b in known_brands if _damerau_levenshtein_le1(canon_low, b)]
        if len(matches) == 1:
            hit = matches[0]
            return BRAND_ALIAS_MAP.get(hit, hit.replace("_", " ").title().replace(" ", "_"))
    return canon


def catalog_variant_lookup(
    catalog: dict[int, dict] | None = None,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], dict[str, str]]]:
    """Build ``(variants_by_brand, sku_meta_by_bv)`` from the dataset catalog and ``gt-mapping.yml``."""
    from utils.cv_models import VARIANT_ROLES, gt_variant_classes, parse_gt_variant_label

    by_brand: dict[str, set[str]] = defaultdict(set)
    sku_meta: dict[tuple[str, str], dict[str, str]] = {}

    for p in (catalog or {}).values():
        b = normalize_hul_golden_brand(p.get("brand"))
        bv = str(p.get("base_variant") or p.get("variant") or "").strip()
        if b and bv and bv not in FORBIDDEN_LABELS:
            by_brand[b.lower()].add(bv)
            sku_meta[(b.lower(), bv.lower())] = {
                "brand": b,
                "base_variant": bv,
                "product": str(p.get("product") or f"{b}__{bv}"),
                "category": str(p.get("category") or ""),
                "packaging_type": str(p.get("packaging_type") or ""),
            }

    for role in VARIANT_ROLES:
        sz = 367 if role == "nonhul_p6_variant" else None
        for cls_name in gt_variant_classes(role, size=sz):
            parsed = parse_gt_variant_label(role, cls_name)
            b = normalize_hul_golden_brand(parsed.get("brand"))
            bv = str(parsed.get("base_variant") or "").strip()
            if b and bv and bv not in FORBIDDEN_LABELS:
                by_brand[b.lower()].add(bv)
                sku_meta.setdefault(
                    (b.lower(), bv.lower()),
                    {
                        "brand": b,
                        "base_variant": bv,
                        "product": f"{b}__{bv}",
                        "category": str(parsed.get("category") or ""),
                        "packaging_type": str(parsed.get("packaging_type") or ""),
                    },
                )

    return {b: sorted(vs) for b, vs in by_brand.items()}, sku_meta


def normalize_hul_golden_variant(
    raw_variant: str | None,
    brand: str | None = None,
    catalog_variants_by_brand: dict[str, list[str]] | None = None,
) -> str:
    """Normalize a raw predicted variant string and snap to the brand's canonical catalog ``base_variant`` when matched."""
    from utils.gemini_fine_tuning.tuning import clean_variant_name
    from utils.metrics import _variant_tokens_match
    from utils.taxonomy.categories import normalize_variant_tokens

    raw = str(raw_variant or "").strip()
    if not raw:
        return ""
    if "__" in raw:
        prefix, rest = raw.split("__", 1)
        if not brand or normalize_hul_golden_brand(prefix).lower() == normalize_hul_golden_brand(brand).lower():
            raw = rest.strip()

    cleaned = clean_variant_name(raw) or re.sub(r"[\s\-]+", "_", raw).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        return ""

    norm_b = normalize_hul_golden_brand(brand) if brand else ""
    if norm_b and catalog_variants_by_brand:
        cands = catalog_variants_by_brand.get(norm_b.lower()) or catalog_variants_by_brand.get(norm_b) or []
        if cands:
            for c in cands:
                if c.lower() == cleaned.lower():
                    return c
            b_toks = {t for t in normalize_variant_tokens(norm_b).split("_") if t}
            pred_toks = {t for t in normalize_variant_tokens(cleaned).split("_") if t} - b_toks
            best_cand: str | None = None
            best_score: tuple[int, float, int] = (0, 0.0, 0)
            for c in cands:
                is_tok_match = int(_variant_tokens_match(cleaned, c, norm_b))
                c_toks = {t for t in normalize_variant_tokens(c).split("_") if t} - b_toks
                if pred_toks and c_toks:
                    inter = len(pred_toks & c_toks)
                    f1 = (2.0 * inter) / (len(pred_toks) + len(c_toks))
                else:
                    f1 = 0.0
                if is_tok_match or f1 >= 0.65:
                    score = (is_tok_match, round(f1, 4), -abs(len(c) - len(cleaned)))
                    if score > best_score:
                        best_score = score
                        best_cand = c
            if best_cand is not None:
                return best_cand

    return cleaned


@dataclass(frozen=True)
class HulGoldenSftConfig:
    """Where the HUL Golden 5-attribute (or 3-attribute) SFT splits come from and go to, and how to tune on them."""

    bucket_name: str = "marketshare-datasets"
    manifest_prefix: str = "derived_manifests/hul_golden"  # rpc.json (input)
    output_prefix: str = "derived_manifests/hul_golden/sft_5attr"  # SFT JSONL (output)
    fields_mode: int = 5  # 5 = (product, brand, base_variant, packaging_type, category); 3 = (brand, packaging_type, category)
    region: str = "europe-west4"
    base_model: str = "gemini-3.5-flash-lite"
    tuned_model_display_name: str = "marketshare-hul-golden-5attr-sft-euw4-gemini-3-5-flash-lite"
    epoch_count: int = 3
    learning_rate_multiplier: float = 1.0
    adapter_size: int = 4
    max_train: int | None = 3200
    max_val: int | None = 500
    prompt_text: str = DEFAULT_5ATTR_PROMPT_TEXT


def is_valid_label(value: str | None) -> bool:
    """Returns True if a taxonomy label is non-empty and not a forbidden fallback."""
    if value is None:
        return False
    cleaned = str(value).strip()
    return bool(cleaned) and cleaned not in FORBIDDEN_LABELS


def validate_triplet(
    uri: str | None,
    brand: str | None,
    packaging_type: str | None,
    category: str | None,
    *,
    strict_taxonomy: bool = True,
) -> bool:
    """Validates that a crop URI and its (brand, packaging_type, category) triplet pass quality gates."""
    if not uri or not str(uri).startswith("gs://"):
        return False
    if not (is_valid_label(brand) and is_valid_label(packaging_type) and is_valid_label(category)):
        return False
    if strict_taxonomy:
        if category not in ALLOWED_CATEGORIES:
            return False
        if packaging_type not in ALLOWED_PACKAGING_TYPES:
            return False
    return True


def validate_quintet(
    uri: str | None,
    brand: str | None,
    base_variant: str | None,
    packaging_type: str | None,
    category: str | None,
    *,
    strict_taxonomy: bool = True,
) -> bool:
    """Validates that a crop URI and all 5 classification attributes pass quality gates."""
    if not validate_triplet(uri, brand, packaging_type, category, strict_taxonomy=strict_taxonomy):
        return False
    if not is_valid_label(base_variant):
        return False
    bv_low = str(base_variant or "").strip().lower()
    if "brand_level" in bv_low or "non_hul_competitor" in bv_low:
        return False
    return True


def sft_record(
    uri: str,
    brand: str,
    packaging_type: str,
    category: str,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
) -> dict[str, Any]:
    """One Vertex AI Gemini 3-attribute SFT example (``tuning._sft_example``) for a validated triplet."""
    from utils.gemini_fine_tuning.tuning import _sft_example

    if not validate_triplet(uri, brand, packaging_type, category):
        raise ValueError(
            f"Invalid SFT record inputs: uri={uri!r}, brand={brand!r}, "
            f"packaging_type={packaging_type!r}, category={category!r}"
        )
    return _sft_example(uri, prompt_text, {"brand": brand, "packaging_type": packaging_type, "category": category})


def sft_5attr_record(
    uri: str,
    brand: str,
    base_variant: str,
    packaging_type: str,
    category: str,
    prompt_text: str = DEFAULT_5ATTR_PROMPT_TEXT,
    product: str | None = None,
) -> dict[str, Any]:
    """One Vertex AI Gemini 5-attribute SFT example for a validated (product, brand, base_variant, packaging_type, category) tuple."""
    from utils.gemini_fine_tuning.tuning import _sft_example

    if not validate_quintet(uri, brand, base_variant, packaging_type, category):
        raise ValueError(
            f"Invalid 5-attr SFT record inputs: uri={uri!r}, brand={brand!r}, "
            f"base_variant={base_variant!r}, packaging_type={packaging_type!r}, category={category!r}"
        )
    prod = product or (f"{brand}__{base_variant}" if base_variant != brand else brand)
    return _sft_example(
        uri,
        prompt_text,
        {
            "product": prod,
            "brand": brand,
            "base_variant": base_variant,
            "packaging_type": packaging_type,
            "category": category,
        },
    )


def verify_split_disjointness(split_uris: dict[str, set[str]]) -> None:
    """Raises ValueError if any URI appears in more than one dataset split."""
    split_names = sorted(split_uris.keys())
    for i, name_a in enumerate(split_names):
        for name_b in split_names[i + 1 :]:
            overlap = split_uris[name_a] & split_uris[name_b]
            if overlap:
                sample = sorted(overlap)[:3]
                raise ValueError(
                    f"Cross-split leakage detected between '{name_a}' and '{name_b}': "
                    f"{len(overlap)} overlapping URIs (sample: {sample})"
                )


def _stratified_interleave(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically interleaves items round-robin across `(category, packaging_type, brand, base_variant)` strata."""
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for it in sorted(items, key=lambda x: x["uri"]):
        if "base_variant" in it:
            key: tuple[str, ...] = (it["category"], it["packaging_type"], it["brand"], it["base_variant"])
        else:
            key = (it["category"], it["packaging_type"], it["brand"])
        by_stratum[key].append(it)
    strata_keys = sorted(by_stratum.keys())
    out: list[dict[str, Any]] = []
    idx = 0
    while True:
        added = False
        for k in strata_keys:
            bucket = by_stratum[k]
            if idx < len(bucket):
                out.append(bucket[idx])
                added = True
        if not added:
            break
        idx += 1
    return out


def build_clean_splits_in_gcp(
    config: HulGoldenSftConfig | None = None,
    *,
    launch_sft: bool = False,
    app_config: dict | None = None,
) -> dict[str, Any]:
    """Write ``{train,val,test}_clean.jsonl`` in Vertex 5-attribute (or 3-attribute) SFT format from
    ``hul_golden/rpc.json`` to ``config.output_prefix`` in GCS and optionally launch Vertex AI SFT."""
    from utils import dataset
    from utils.taxonomy.categories import normalize_category_name
    from utils.taxonomy.filenames import parse_crop_filename
    from utils.taxonomy.packaging import normalize_packaging_type

    cfg = config or HulGoldenSftConfig()
    is_5attr = int(cfg.fields_mode) >= 5
    client = dataset._gcs()
    bucket = client.bucket(cfg.bucket_name)

    rpc_blob = bucket.blob(f"{cfg.manifest_prefix}/rpc.json")
    rpc_data = json.loads(rpc_blob.download_as_bytes().decode("utf-8"))
    classes = rpc_data.get("classes", {})
    splits_data = rpc_data.get("splits", {})

    def _session_of(item: dict[str, Any]) -> str:
        sid = str(item.get("session_id") or "").strip()
        if sid:
            return sid
        uri = str(item.get("uri") or "")
        if not uri:
            return ""
        try:
            return str(parse_crop_filename(uri.rsplit("/", 1)[-1]).get("session_id") or "")
        except Exception:  # noqa: BLE001
            return ""

    held_out_sessions: set[str] = {
        _session_of(it)
        for sp in ("val", "test")
        for it in splits_data.get(sp, [])
    } - {""}

    used_uris: set[str] = set()
    cleaned_splits: dict[str, list[dict[str, Any]]] = {"val": [], "test": [], "train": []}
    stats: dict[str, Any] = {"fields_mode": 5 if is_5attr else 3, "splits": {}, "session_filtered_train": 0}

    # Process val and test first so they never lose items to train deduplication
    for split_name in ("val", "test", "train"):
        records: list[dict[str, Any]] = []
        for item in splits_data.get(split_name, []):
            uri = str(item.get("uri") or "")
            if not uri or uri in used_uris:
                continue
            if split_name == "train":
                sid = _session_of(item)
                if sid and sid in held_out_sessions:
                    stats["session_filtered_train"] += 1
                    continue
            labels = item.get("labels") or []
            prods = item.get("products") or []
            lbl = labels[0] if labels else (classes.get(str(prods[0])) if prods else None)
            if not lbl:
                continue
            brand = normalize_hul_golden_brand(lbl.get("brand"))
            raw_bv = str(lbl.get("base_variant") or lbl.get("variant") or "").strip()
            bv = normalize_hul_golden_variant(raw_bv, brand=brand)
            cat = normalize_category_name(str(lbl.get("category") or ""))
            pack = normalize_packaging_type(
                lbl.get("packaging_type"),
                category=cat,
                brand=brand,
                variant=bv or raw_bv,
            )
            if is_5attr:
                if not validate_quintet(uri, brand, bv, pack, cat):
                    continue
                prod = f"{brand}__{bv}" if bv != brand else brand
                used_uris.add(uri)
                records.append(
                    {
                        "uri": uri,
                        "product": prod,
                        "brand": brand,
                        "base_variant": bv,
                        "packaging_type": pack,
                        "category": cat,
                        "sft": sft_5attr_record(uri, brand, bv, pack, cat, cfg.prompt_text, product=prod),
                    }
                )
            else:
                if not validate_triplet(uri, brand, pack, cat):
                    continue
                used_uris.add(uri)
                records.append(
                    {
                        "uri": uri,
                        "brand": brand,
                        "packaging_type": pack,
                        "category": cat,
                        "sft": sft_record(uri, brand, pack, cat, cfg.prompt_text),
                    }
                )
        cleaned_splits[split_name] = _stratified_interleave(records)
        split_recs = cleaned_splits[split_name]
        split_stat: dict[str, Any] = {
            "total_clean": len(split_recs),
            "unique_brands": len({r["brand"] for r in split_recs}),
            "unique_packaging_types": len({r["packaging_type"] for r in split_recs}),
            "unique_categories": len({r["category"] for r in split_recs}),
        }
        if is_5attr:
            split_stat["unique_base_variants"] = len({r["base_variant"] for r in split_recs})
            split_stat["unique_products"] = len({r["product"] for r in split_recs})
        stats["splits"][split_name] = split_stat

    verify_split_disjointness({k: {r["uri"] for r in v} for k, v in cleaned_splits.items()})

    uploaded_uris: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        clean_fname = f"{cfg.output_prefix}/{split_name}_clean.jsonl"
        clean_body = "".join(json.dumps(r["sft"], ensure_ascii=False) + "\n" for r in cleaned_splits[split_name])
        bucket.blob(clean_fname).upload_from_string(clean_body, content_type="application/jsonl")
        uploaded_uris[f"{split_name}_clean"] = f"gs://{cfg.bucket_name}/{clean_fname}"

    train_job_uri = uploaded_uris["train_clean"]
    val_job_uri = uploaded_uris["val_clean"]
    if cfg.max_train and cfg.max_train > 0:
        sub_train_recs = cleaned_splits["train"][: cfg.max_train]
        sub_train_fname = f"{cfg.output_prefix}/temp_train_clean_{cfg.max_train}.jsonl"
        sub_train_body = "".join(json.dumps(r["sft"], ensure_ascii=False) + "\n" for r in sub_train_recs)
        bucket.blob(sub_train_fname).upload_from_string(sub_train_body, content_type="application/jsonl")
        train_job_uri = f"gs://{cfg.bucket_name}/{sub_train_fname}"
        uploaded_uris["train_subset"] = train_job_uri
    if cfg.max_val and cfg.max_val > 0:
        sub_val_recs = cleaned_splits["val"][: cfg.max_val]
        sub_val_fname = f"{cfg.output_prefix}/temp_val_clean_{cfg.max_val}.jsonl"
        sub_val_body = "".join(json.dumps(r["sft"], ensure_ascii=False) + "\n" for r in sub_val_recs)
        bucket.blob(sub_val_fname).upload_from_string(sub_val_body, content_type="application/jsonl")
        val_job_uri = f"gs://{cfg.bucket_name}/{sub_val_fname}"
        uploaded_uris["val_subset"] = val_job_uri

    stats["uploaded_uris"] = uploaded_uris

    if launch_sft:
        from utils.gemini_fine_tuning.tuning import launch_sft_job

        job = launch_sft_job(
            train_job_uri,
            val_job_uri,
            base_model=cfg.base_model,
            display_name=cfg.tuned_model_display_name,
            epochs=cfg.epoch_count,
            adapter_size=cfg.adapter_size,
            learning_rate_multiplier=cfg.learning_rate_multiplier,
            config=app_config,
            region=cfg.region,
        )
        stats["sft_job"] = job
        bucket.blob(f"{cfg.output_prefix}/clean_sft_job.json").upload_from_string(
            json.dumps(job, indent=2),
            content_type="application/json",
        )

    bucket.blob(f"{cfg.output_prefix}/clean_splits_summary.json").upload_from_string(
        json.dumps(stats, indent=2),
        content_type="application/json",
    )
    return stats
