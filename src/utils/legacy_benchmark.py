"""Standalone Legacy Mapping Table, Prediction Joiner & Dual-Accept Benchmark Evaluator.

Keeps all legacy ``ABrand`` (``AClinic``, ``ADove``, ``A_3_Roses``, ``ABrook_Bond``) and
``Category__PackType`` (``Hair__Bottle``, ``PW__Bar``, ``Oral__Tube``) mappings in a clean,
self-contained table separate from the core canonical taxonomy and data pipelines.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from utils import metrics
from utils.taxonomy.brands import HUL_BRANDS, canonicalize_brand_name
from utils.taxonomy.categories import (
    CANONICAL_CATEGORIES,
    _infer_brand_category,
    normalize_category_name,
)
from utils.taxonomy.filenames import normalize_brand_folder, parse_variant_folder
from utils.taxonomy.packaging import (
    CANONICAL_PACKAGING_TYPES,
    normalize_packaging_type,
    parse_pack_folder,
)

# Canonical Brand (lowercase) -> Exact Legacy brand_data Folder Name (where irregular)
CANONICAL_TO_LEGACY_BRAND_MAP: dict[str, str] = {
    "3_roses": "A_3_Roses",
    "brooke_bond": "ABrook_Bond",
    "bru": "ABru",
    "clinic_plus": "AClinic",
    "close_up": "AClose_Up",
    "comfort": "AComfort",
    "dove": "ADove",
    "dr_wash": "ADr_Wash",
    "glow_and_handsome": "AFair_and_Lovely",
    "glow_and_lovely": "AFair_and_Lovely",
    "horlicks": "AHorlicks",
    "indulekha": "AIndulekha",
    "jam": "AKissan",
    "ketchup": "AKissan",
    "kissan": "AKissan",
    "knorr": "AKnorr",
    "kwality_walls": "AKwality_Walls",
    "lakme": "ALakme",
    "lifebuoy": "ALifebuoy",
    "lipton": "ALipton",
    "liril": "ALiril",
    "love_beauty_and_planet": "ALove_Beauty_and_Planet",
    "lux": "ALux",
    "moti": "AMoti",
    "nature_protect": "ANature_Protect",
    "oziva": "AOziva",
    "pears": "APears",
    "pepsodent": "APepsodent",
    "plush": "APlush",
    "ponds": "APonds",
    "pureit": "APureit",
    "red_label": "ARed_Label",
    "rexona": "ARexona",
    "rin": "ARin",
    "simple": "ASimple",
    "sunsilk": "ASunsilk",
    "surf_excel": "ASurf_Excel",
    "taaza": "ATaaza",
    "taj_mahal": "ATaj_Mahal",
    "tresemme": "ATresemme",
    "vaseline": "AVaseline",
    "vim": "AVim",
    "wheel": "AWheel",
}

# Canonical Category -> Legacy pack_data Category Prefix ("Category__PackType")
CANONICAL_TO_LEGACY_PACK_CATEGORY: dict[str, str] = {
    "Skin": "Skin",
    "Skin_Care": "Skin",
    "PW": "PW",
    "Skin_Cleansing": "PW",
    "Hair_Care": "Hair",
    "Oral_Care": "Oral_Care",
    "DMT": "DMT",
    "Deodorants": "Deo",
    "Laundry": "Laundry",
    "Fabric_Care": "FAB_SOL",
    "HHC": "HHC",
    "Home_Care": "FSOL",
    "Foods": "Foods",
    "Beverages": "Beverages",
    "HFD": "HFD",
    "Health_Wellbeing": "HFD",
}

LEGACY_PACK_CATEGORY_TO_CANONICAL: dict[str, str] = {
    "skin": "Skin",
    "skin_care": "Skin",
    "pw": "PW",
    "skin_cleansing": "PW",
    "hair": "Hair_Care",
    "hair_care": "Hair_Care",
    "oral": "Oral_Care",
    "oral_care": "Oral_Care",
    "dmt": "DMT",
    "deo": "DMT",
    "deodorants": "DMT",
    "laundry": "Laundry",
    "fab_sol": "Laundry",
    "fabric_care": "Laundry",
    "hhc": "HHC",
    "fsol": "HHC",
    "home_care": "HHC",
    "foods": "Foods",
    "savoury": "Foods",
    "beverages": "Beverages",
    "tea": "Beverages",
    "coffee": "Beverages",
    "hfd": "HFD",
    "health_wellbeing": "HFD",
}


def parse_legacy_pack_folder(folder: str) -> dict[str, str]:
    """Parse a legacy ``Category__PackType`` folder (including ``TEA__Carton``, ``Hair__Bottle``, ``PW__Bar``)."""
    raw = str(folder or "").strip()
    if "__" in raw:
        raw_cat, raw_pkg = raw.split("__", 1)
        cat = LEGACY_PACK_CATEGORY_TO_CANONICAL.get(raw_cat.lower(), normalize_category_name(raw_cat))
        pkg = normalize_packaging_type(raw_pkg, category=cat)
        return {"raw_folder": raw, "category": cat, "packaging_type": pkg}
    p_info = parse_pack_folder(raw)
    return {
        "raw_folder": raw,
        "category": p_info["category"],
        "packaging_type": p_info["packaging_type"],
    }


def resolve_legacy_brand_name(
    brand: str,
    is_hul: bool | None = None,
    raw_folder: str | None = None,
) -> str:
    """Resolve the legacy ``brand_data`` folder name (e.g. ``Clinic_Plus -> AClinic``, ``Pantene -> Pantene``)."""
    canon = canonicalize_brand_name(brand)
    if not canon or canon == "Unknown":
        return "Unknown"
    low = canon.lower()
    if is_hul is None:
        is_hul = low in HUL_BRANDS
    if raw_folder:
        raw_clean = str(raw_folder).strip()
        if raw_clean and "__" not in raw_clean:
            if (is_hul and (raw_clean.startswith("A_") or (raw_clean.startswith("A") and len(raw_clean) > 1 and raw_clean[1].isupper()))) or not is_hul:
                return raw_clean
    if not is_hul:
        return canon
    if low in CANONICAL_TO_LEGACY_BRAND_MAP:
        return CANONICAL_TO_LEGACY_BRAND_MAP[low]
    if canon[0].isdigit():
        return f"A_{canon}"
    return f"A{canon}"


def resolve_legacy_pack_folder(
    category: str,
    packaging_type: str,
    raw_folder: str | None = None,
) -> str:
    """Resolve the legacy ``pack_data`` ``Category__PackType`` folder name (e.g. ``Hair__Bottle``, ``PW__Wrapper``)."""
    if raw_folder and "__" in str(raw_folder):
        parts = str(raw_folder).strip().split("__", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[0]}__{parts[1]}"
    from utils.taxonomy.packaging import project_to_valid_category_pack

    cat_norm = normalize_category_name(category)
    pkg_norm = normalize_packaging_type(packaging_type, category=cat_norm)
    pkg_valid = project_to_valid_category_pack(cat_norm, pkg_norm)
    prefix = CANONICAL_TO_LEGACY_PACK_CATEGORY.get(category, CANONICAL_TO_LEGACY_PACK_CATEGORY.get(cat_norm, cat_norm))
    return f"{prefix}__{pkg_valid}"


def build_legacy_mapping_table(
    extra_brands: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build the standalone Canonical <-> Legacy mapping table for brands and the 46 Category__PackType folders."""
    from utils.taxonomy.packaging import VALID_CATEGORY_PACKAGING_TYPES

    seen_brands: dict[str, dict[str, Any]] = {}
    for b_low in sorted(HUL_BRANDS):
        canon = canonicalize_brand_name(b_low)
        seen_brands[canon.lower()] = {
            "mapping_type": "brand",
            "canonical_brand": canon,
            "legacy_brand_name": resolve_legacy_brand_name(canon, is_hul=True),
            "is_hul": True,
            "category": normalize_category_name(_infer_brand_category(canon.lower())),
        }
    for row in extra_brands or ():
        b_raw = str(row.get("brand") or "").strip()
        if not b_raw:
            continue
        canon = canonicalize_brand_name(b_raw)
        is_hul = bool(row.get("is_hul")) if "is_hul" in row else (canon.lower() in HUL_BRANDS)
        seen_brands[canon.lower()] = {
            "mapping_type": "brand",
            "canonical_brand": canon,
            "legacy_brand_name": resolve_legacy_brand_name(canon, is_hul=is_hul),
            "is_hul": is_hul,
            "category": normalize_category_name(str(row.get("category") or _infer_brand_category(canon.lower()))),
        }

    pack_rows: list[dict[str, Any]] = []
    for cat in sorted(CANONICAL_CATEGORIES):
        for pkg in VALID_CATEGORY_PACKAGING_TYPES.get(cat, sorted(CANONICAL_PACKAGING_TYPES)):
            pack_rows.append({
                "mapping_type": "category_pack",
                "canonical_category": cat,
                "canonical_packaging_type": pkg,
                "legacy_pack_folder": resolve_legacy_pack_folder(cat, pkg),
            })

    return {
        "brands": [seen_brands[k] for k in sorted(seen_brands)],
        "category_packs": pack_rows,
    }


def export_legacy_mapping_jsonl(
    output_path: str | Path,
    extra_brands: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Export the standalone Canonical <-> Legacy mapping table as a ``.jsonl`` file (1 JSON object per line)."""
    table = build_legacy_mapping_table(extra_brands=extra_brands)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in table["brands"]] + [json.dumps(r) for r in table["category_packs"]]
    out.write_text("\n".join(lines) + "\n")
    return out


def format_legacy_mapping_table_jsonl(
    classes_or_brands: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Format the standalone Canonical <-> Legacy mapping table as a JSONL string."""
    extra = list(classes_or_brands.values()) if isinstance(classes_or_brands, Mapping) else list(classes_or_brands or ())
    table = build_legacy_mapping_table(extra_brands=extra)
    lines = [json.dumps(r) for r in table["brands"]] + [json.dumps(r) for r in table["category_packs"]]
    return "\n".join(lines) + "\n"


def write_legacy_mapping_table_jsonl(
    classes_or_brands: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    output_path: str | Path,
) -> int:
    """Write ``legacy_mapping_table.jsonl`` to disk and return the number of JSONL rows written."""
    txt = format_legacy_mapping_table_jsonl(classes_or_brands)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    return len([ln for ln in txt.splitlines() if ln.strip()])


def legacy_field_matches(pred: Mapping[str, Any] | None, gt: Mapping[str, Any], field: str) -> bool:
    """Dual-accept comparator consulted by ``metrics._field_matches`` without modifying dataset dicts."""
    if not pred:
        return False
    p_val = pred.get(field)
    g_val = gt.get(field)

    if field == "brand":
        p_raw = str(p_val or pred.get("legacy_brand_name") or "").strip()
        g_raw = str(g_val or "").strip()
        if not p_raw or not g_raw:
            return False
        g_is_hul = gt.get("is_hul") if isinstance(gt.get("is_hul"), bool) else None
        g_legacy = str(gt.get("legacy_brand_name") or resolve_legacy_brand_name(g_raw, is_hul=g_is_hul)).strip()
        p_legacy = str(pred.get("legacy_brand_name") or "").strip()
        if p_raw.lower() in (g_raw.lower(), g_legacy.lower()):
            return True
        if p_legacy and p_legacy.lower() in (g_raw.lower(), g_legacy.lower()):
            return True
        p_canon = canonicalize_brand_name(p_raw)
        g_canon = canonicalize_brand_name(g_raw)
        if p_canon != "Unknown" and g_canon != "Unknown" and p_canon.lower() == g_canon.lower():
            return True
        return False

    if field == "category":
        p_raw = str(p_val or pred.get("legacy_pack_folder") or "").strip()
        g_raw = str(g_val or "").strip()
        if not p_raw or not g_raw:
            return False
        p_cat = parse_legacy_pack_folder(p_raw)["category"] if "__" in p_raw else normalize_category_name(p_raw)
        g_cat = parse_legacy_pack_folder(g_raw)["category"] if "__" in g_raw else normalize_category_name(g_raw)
        return p_cat != "Unknown" and p_cat == g_cat

    if field == "packaging_type":
        p_raw = str(p_val or pred.get("legacy_pack_folder") or "").strip()
        g_raw = str(g_val or "").strip()
        if not p_raw or not g_raw:
            return False
        p_pkg = parse_legacy_pack_folder(p_raw)["packaging_type"] if "__" in p_raw else normalize_packaging_type(p_raw)
        g_pkg = parse_legacy_pack_folder(g_raw)["packaging_type"] if "__" in g_raw else normalize_packaging_type(g_raw)
        return p_pkg != "Unknown" and p_pkg == g_pkg

    return False


def normalize_legacy_prediction(
    brand_pred: str | None = None,
    pack_pred: str | None = None,
    variant_pred: str | None = None,
) -> dict[str, Any]:
    """Combine predictions from legacy Brand, Pack (``Category__PackType``), and Variant models into one dict."""
    brand = "Unknown"
    legacy_brand = "Unknown"
    is_hul = False
    category = "Unknown"
    pkg_type = "Unknown"
    legacy_pack = "Unknown"
    variant = "Brand_Level"
    base_variant = "Brand_Level"

    if variant_pred and str(variant_pred).strip():
        v_info = parse_variant_folder("variant_skin", str(variant_pred).strip())
        brand = v_info["brand"]
        is_hul = v_info["is_hul"]
        legacy_brand = resolve_legacy_brand_name(brand, is_hul=is_hul)
        category = v_info["category"]
        pkg_type = v_info["packaging_type"]
        legacy_pack = resolve_legacy_pack_folder(category, pkg_type)
        variant = v_info["variant"]
        base_variant = v_info["base_variant"]

    if brand_pred and str(brand_pred).strip():
        raw_b = str(brand_pred).strip()
        b_info = normalize_brand_folder(raw_b)
        if b_info is not None:
            brand = b_info["brand"]
            is_hul = b_info["is_hul"]
            legacy_brand = resolve_legacy_brand_name(brand, is_hul=is_hul, raw_folder=raw_b)
            if category == "Unknown":
                category = b_info["category"]
        else:
            brand = canonicalize_brand_name(raw_b)
            is_hul = brand.lower() in HUL_BRANDS
            legacy_brand = resolve_legacy_brand_name(brand, is_hul=is_hul, raw_folder=raw_b)
            if category == "Unknown":
                category = normalize_category_name(_infer_brand_category(brand.lower()))

    if pack_pred and str(pack_pred).strip():
        raw_p = str(pack_pred).strip()
        p_info = parse_legacy_pack_folder(raw_p)
        category = p_info["category"]
        pkg_type = p_info["packaging_type"]
        legacy_pack = resolve_legacy_pack_folder(category, pkg_type, raw_folder=raw_p)

    if legacy_pack == "Unknown" and category != "Unknown" and pkg_type != "Unknown":
        legacy_pack = resolve_legacy_pack_folder(category, pkg_type)

    product = (
        f"{brand}__{base_variant}"
        if brand != "Unknown" and base_variant not in ("Brand_Level", brand)
        else (brand if brand != "Unknown" else legacy_pack)
    )
    return {
        "product": product,
        "brand": brand,
        "legacy_brand_name": legacy_brand,
        "variant": variant,
        "base_variant": base_variant,
        "category": category,
        "packaging_type": pkg_type,
        "legacy_pack_folder": legacy_pack,
        "is_hul": is_hul,
    }


def _lookup_box_pred(
    pred_map: Mapping[str, Any] | None,
    image_id: str,
    box_idx: int,
) -> str | None:
    """Retrieve a prediction string for ``(image_id, box_idx)`` from a flat or list-valued dict."""
    if not pred_map:
        return None
    if f"{image_id}:{box_idx}" in pred_map:
        val = pred_map[f"{image_id}:{box_idx}"]
        return str(val) if val is not None else None
    if image_id in pred_map:
        val = pred_map[image_id]
        if isinstance(val, list):
            return str(val[box_idx]) if box_idx < len(val) and val[box_idx] is not None else None
        return str(val) if val is not None else None
    return None


def evaluate_joined_legacy_models(
    samples: Sequence[Any],
    brand_preds: Mapping[str, Any] | None = None,
    pack_preds: Mapping[str, Any] | None = None,
    variant_preds: Mapping[str, Any] | None = None,
    fields: Sequence[str] = ("product", "brand", "base_variant", "category", "packaging_type", "is_hul"),
) -> dict[str, Any]:
    """Join legacy Brand + Pack (+ optional Variant) model outputs per box and score individual & compound accuracy."""
    all_labels: list[dict[str, Any]] = []
    image_bcp_scores: list[float] = []
    image_brand_scores: list[float] = []

    for sample in samples:
        img_id = str(getattr(sample, "image_id", ""))
        gt_labels = list(getattr(sample, "labels", []) or [])
        img_bcp: list[int] = []
        img_brand: list[int] = []

        for idx, gt in enumerate(gt_labels):
            b_str = _lookup_box_pred(brand_preds, img_id, idx)
            p_str = _lookup_box_pred(pack_preds, img_id, idx)
            v_str = _lookup_box_pred(variant_preds, img_id, idx)
            pred = (
                normalize_legacy_prediction(brand_pred=b_str, pack_pred=p_str, variant_pred=v_str)
                if (b_str or p_str or v_str)
                else None
            )
            mk = metrics.match_product(pred, gt, fields)
            corr = mk["correct"]
            all_labels.append({"pred": pred, "gt": gt, "correct": corr})
            b_ok = int(bool(corr.get("brand")))
            c_ok = int(bool(corr.get("category")))
            p_ok = int(bool(corr.get("packaging_type")))
            img_brand.append(b_ok)
            img_bcp.append(int(b_ok and c_ok and p_ok))

        if img_bcp:
            image_bcp_scores.append(sum(img_bcp) / len(img_bcp))
            image_brand_scores.append(sum(img_brand) / len(img_brand))

    n_boxes = max(1, len(all_labels))
    field_accuracy = {
        f: round(sum(bool(lab["correct"].get(f)) for lab in all_labels) / n_boxes, 4)
        for f in fields
    }
    hier = metrics.hierarchical_field_scores(all_labels)
    return {
        "evaluated_images": len(samples),
        "evaluated_boxes": len(all_labels),
        "field_accuracy": field_accuracy,
        "hierarchical_accuracy": hier,
        "compound_brand_category_pack": hier["compound_brand_category_pack"],
        "ci95": {
            "compound_brand_category_pack": metrics.bootstrap_ci95(image_bcp_scores),
            "brand": metrics.bootstrap_ci95(image_brand_scores),
        },
        "labels": all_labels,
    }
