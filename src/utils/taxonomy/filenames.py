"""Filename, Folder & Client LabelMe Shape Parsers (Zero Default Fallback Guarantee)."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from utils.taxonomy.brands import (
    A_PREFIX_TWINS,
    BRAND_ALIAS_MAP,
    HUL_BRAND_ONLY_NO_VARIANT_TREE,
    HUL_BRANDS,
    NOISE_BRAND_FOLDERS,
    canonicalize_brand_name,
)
from utils.taxonomy.categories import (
    _FUZZY_CATEGORY_TARGETS,
    BRAND_CATEGORY_MAP,
    HFD_SUPPLEMENT_BRANDS,
    MULTI_CATEGORY_BRAND_PROFILES,
    NON_HUL_BRAND_CATEGORY_MAP,
    TREE_DEFAULT_CATEGORY,
    VARIANT_CATEGORY_KEYWORD_RULES,
    _damerau_levenshtein_le1,
    _infer_brand_category,
    normalize_category_name,
    normalize_variant_tokens,
)
from utils.taxonomy.packaging import normalize_packaging_type, resolve_packaging_type

_NON_PROMO_FREE_PREFIX = (
    r"(?<!Oil_)(?<!Sugar_)(?<!Sulfate_)(?<!Sulphate_)(?<!Soap_)(?<!Paraben_)"
    r"(?<!Alcohol_)(?<!Acne_)(?<!Dandruff_)(?<!Chemical_)(?<!Cruelty_)(?<!Gluten_)"
    r"(?<!Caffeine_)(?<!Dust_)(?<!Stain_)(?<!Tears_)(?<!Frizz_)(?<!Shine_)(?<!Blemish_)"
)

# Expanded promotional & multipack token regex covering both HUL and Non-HUL rival conventions
PROMO_TOKEN_RE = re.compile(
    rf"(?:^|_)(B[1-9]G[1-9]|PO[2-9]|Pack_Of_\d+|{_NON_PROMO_FREE_PREFIX}Free_[A-Za-z0-9_]+|Combo|Super_Saver|Value_Pack|Extra_\d+|Offer)(?:$|_)",
    re.I,
)
PRICE_TOKEN_RE = re.compile(r"(?:^|_)Rs_?(\d+)(?:$|_)", re.I)
ARTWORK_TOKEN_RE = re.compile(r"_(New|Old|Square)(?:_\d+)?$", re.I)
STRIP_PROMO_ART_SUFFIX_RE = re.compile(
    rf"(?:_(?:with_)?{_NON_PROMO_FREE_PREFIX}Free_[A-Za-z0-9_]+|_(?:New|Old|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Combo|Super_Saver|Value_Pack|Extra_\d+|Offer|Rs_?\d+|copy\d+)(?:_\d+)?)+$",
    re.I,
)

AUGMENTED_FILE_RE = re.compile(r"^DA_(?:TBC_)?", re.I)
COPY_FILE_RE = re.compile(
    r"(?:^copy\d*\.(?:jpe?g|png)$|\s-\s*Copy(?:\s*\(\d+\))?\.(?:jpe?g|png)$|\(copy\)\.(?:jpe?g|png)$|^DA_(?:TBC_)?|_\d+\s+\(\d+\)\.(?:jpe?g|png)$)",
    re.I,
)
BBOX_FILE_RE = re.compile(
    r"^(?:processed_|\d+\.\d+____)?(?P<sess>[A-Za-z0-9_-]+?)(?:__|_)0*(?P<idx>\d+)__(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)$"
)
STD_FILE_RE = re.compile(
    r"^(?:processed_|DA_TBC_|DA_|GT\d+_PCLW_|GT\d+_|PCLW_|Personal_Wash|sc_\d+_|sc_|\d+\.\d+_)?(?P<sess>[A-Za-z0-9_-]+?)_(?P<idx>\d+)$",
    re.I,
)
SPACE_BRACKET_SESS_RE = re.compile(
    r"(?:^|[\s_\-\[\]\(\)])(?P<sess>\d{3,})(?:\s*[\[\(]\d+[\]\)])?_(?P<idx>\d+)(?:\s*[\[\(]\d+[\]\)])?$"
)
PACK_BV_FILE_RE = re.compile(
    r"^(?P<brand>[A-Za-z0-9][A-Za-z0-9_]*?)__(?P<variant>[A-Za-z0-9_]+?)(?:_(?:image|img|copy)_?(?P<img_idx>\d+)|_(?:GT\d+_)?(?:PCLW_)?(?P<sess>\d{2,})_(?P<idx>\d+)|_(?P<idx_only>\d+))?$",
    re.I,
)

GTMERCH_OUTLET_RE = re.compile(
    r"^(?P<outlet>(?:U-\d+|IN\d+))_(?P<asset>A-\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<ts>\d{10,16})$",
    re.I,
)
GTMERCH_NUMERIC_SESS_RE = re.compile(
    r"^(?:processed_)?(?P<sess>\d{3,22})(?:_(?P<dup>\d{3,22}))?$",
    re.I,
)
GTMERCH_ARTIFACT_FILENAMES = {
    ".amlignore", ".amlignore.amltmp", "labels.cache", "train.cache", "test.cache", "val.cache",
}

# Canonical GTMerch campaign bay brand rules (ppvwash -> V_Wash separated from Lifebuoy; strictly canonical categories & packs; ZERO Dove fallback!)
GTMERCH_BAY_BRAND_RULES: tuple[tuple[tuple[str, ...], str, str, str, bool], ...] = (
    (("ppvwash", "vwash", "v_wash"), "V_Wash", "PW", "Bottle", True),
    (("lifebuoy",), "Lifebuoy", "PW", "Bar", True),
    (("indulekha",), "Indulekha", "Hair_Care", "Bottle", True),
    (("lux_asset", "lux_", "lux-", "lux"), "Lux", "PW", "Bar", True),
    (("dove",), "Dove", "Skin", "Bottle", True),
    (("ponds", "pppon"), "Ponds", "Skin", "Tube", True),
    (("lakme", "gtchlp"), "Lakme", "Skin", "Tube", True),
    (("hellmann", "hellmans"), "Hellmans", "Foods", "Jar", True),
    (("knorr",), "Knorr", "Foods", "Pouch", True),
    (("boost", "gtchboost"), "Boost", "HFD", "Jar", True),
    (("mother", "horlick", "choch"), "Horlicks", "HFD", "Jar", True),
    (("sunsilk", "gtchss", "ssm63", "ssmph"), "Sunsilk", "Hair_Care", "Bottle", True),
    (("lbp", "love_beauty"), "LBP", "Hair_Care", "Bottle", True),
    (("brnbg", "bru_", "bru-"), "Bru", "Beverages", "Pouch", True),
    (("vaseline",), "Vaseline", "Skin", "Bottle", True),
    (("clinic",), "Clinic_Plus", "Hair_Care", "Bottle", True),
    (("tresemme",), "Tresemme", "Hair_Care", "Bottle", True),
    (("closeup",), "Closeup", "Oral_Care", "Tube", True),
    (("pepsodent",), "Pepsodent", "Oral_Care", "Tube", True),
    (("surf",), "Surf_Excel", "Laundry", "Pouch", True),
    (("vim",), "Vim", "HHC", "Bar", True),
    (("rin_", "rin-"), "Rin", "Laundry", "Bar", True),
    (("pears",), "Pears", "PW", "Bar", True),
    (("rexona",), "Rexona", "DMT", "Roll_On", True),
    (("axe_", "axe-"), "Axe", "DMT", "Bottle", True),
    (("kissan",), "Kissan", "Foods", "Bottle", True),
    (("shilajit", "dabur"), "Dabur", "HFD", "Bottle", False),
)


def normalize_brand_folder(folder: str) -> dict | None:
    """Normalize a ``brand_data/`` folder into canonical ``(brand, is_hul, category, legacy_tier)``."""
    raw = folder.strip()
    if not raw or raw in NOISE_BRAND_FOLDERS:
        return None
    raw_low = raw.lower()
    if raw_low in BRAND_ALIAS_MAP:
        brand = BRAND_ALIAS_MAP[raw_low]
        tier = "alias_mapped"
    elif raw_low in A_PREFIX_TWINS:
        brand = A_PREFIX_TWINS[raw_low]
        tier = "a_prefix_twin"
    elif raw.startswith("Z_"):
        brand = raw[2:]
        tier = "numeric_z_prefix"
    elif len(raw) >= 3 and raw[0] == "A" and raw[1].isupper():
        brand = raw[1:]
        tier = "priority_a_prefix"
    else:
        brand = re.sub(r"[\s_]+", "_", raw).strip("_")
        tier = "standard"
    key = brand.lower()
    is_hul = key in HUL_BRANDS
    if is_hul and key in HUL_BRAND_ONLY_NO_VARIANT_TREE and tier == "standard":
        tier = "hul_umbrella_brand_only"
    return {
        "raw_folder": raw,
        "brand": brand,
        "is_hul": is_hul,
        "category": _infer_brand_category(key),
        "legacy_tier": tier,
    }


def parse_variant_folder(tree: str, folder: str, crc_votes: dict[str, int] | None = None) -> dict:
    """Parse an ``input_data/variant_*/train/<Brand>__<Variant>`` folder into canonical W1-W9 attributes."""
    raw = folder.strip()
    if "__" in raw:
        raw_brand, raw_variant = raw.split("__", 1)
    else:
        raw_brand, raw_variant = raw, "Standard"
    brand = canonicalize_brand_name(raw_brand)
    variant = re.sub(r"[\s_]+", "_", raw_variant.strip()).strip("_") or "Standard"
    b_key = brand.lower()
    raw_silo_category = tree[len("variant_") :] if tree.startswith("variant_") else tree
    norm_var = normalize_variant_tokens(variant)
    if b_key in MULTI_CATEGORY_BRAND_PROFILES:
        allowed_cats: tuple[str, ...] | None = MULTI_CATEGORY_BRAND_PROFILES[b_key]
    elif b_key in BRAND_CATEGORY_MAP:
        allowed_cats = (BRAND_CATEGORY_MAP[b_key],)
    elif b_key in NON_HUL_BRAND_CATEGORY_MAP:
        allowed_cats = (NON_HUL_BRAND_CATEGORY_MAP[b_key],)
    else:
        allowed_cats = None

    category = None
    if b_key in HFD_SUPPLEMENT_BRANDS:
        category = "HFD"
    elif b_key == "ayush" and "rock_salt" in norm_var:
        category = "Oral_Care"
    elif b_key == "tata" and ("salt" in norm_var.split("_") or "sampann" in norm_var or "soulful" in norm_var or "quisine" in norm_var):
        category = "Foods"
    else:
        for kw_re, cat_override in VARIANT_CATEGORY_KEYWORD_RULES:
            if kw_re.search(variant) or kw_re.search(norm_var):
                if allowed_cats is None or cat_override in allowed_cats:
                    category = cat_override
                    break
        if category is None:
            for tok in norm_var.split("_"):
                if len(tok) < 5:
                    continue
                for target_word, target_cat in _FUZZY_CATEGORY_TARGETS:
                    if _damerau_levenshtein_le1(tok, target_word):
                        if allowed_cats is None or target_cat in allowed_cats:
                            category = target_cat
                            break
                if category is not None:
                    break
    if category is None:
        brand_cat = _infer_brand_category(b_key)
        category = brand_cat if brand_cat != "Unknown" else TREE_DEFAULT_CATEGORY.get(tree, "Foods")
    category = normalize_category_name(category)

    pkg_type = resolve_packaging_type(brand=brand, variant=variant, category=category, crc_votes=crc_votes)

    promo_m = PROMO_TOKEN_RE.search(variant)
    price_m = PRICE_TOKEN_RE.search(variant)
    art_m = ARTWORK_TOKEN_RE.search(variant)
    price_inr = int(price_m.group(1)) if price_m else None
    promo_type = promo_m.group(1).upper() if promo_m else (f"RS{price_inr}" if price_inr else None)
    artwork_gen = art_m.group(1).lower() if art_m else "standard"

    base_variant = STRIP_PROMO_ART_SUFFIX_RE.sub("", variant) or variant
    bv_canon_brand = canonicalize_brand_name(base_variant).lower()
    is_catchall = (
        base_variant.lower() in ("others", "other", "na", "unknown", "non_hul_competitor", raw_brand.strip().lower(), b_key, "standard")
        or bv_canon_brand == b_key
    )
    is_hul = (
        any(tok in variant.lower().replace("_", " ") for tok in ("taj mahal", "bru", "red label", "taaza", "3 roses", "lipton", "kissan", "knorr", "annapurna"))
        if b_key == "tata"
        else (b_key in HUL_BRANDS)
    )
    return {
        "tree": tree,
        "raw_silo_category": raw_silo_category,
        "raw_folder": raw,
        "category": category,
        "brand": brand,
        "variant": variant,
        "base_variant": base_variant,
        "packaging_type": pkg_type,
        "is_hul": is_hul,
        "has_promo": promo_type is not None,
        "promo_type": promo_type,
        "has_price_tag": price_inr is not None,
        "artwork_gen": artwork_gen,
        "is_catchall_variant": is_catchall,
    }


def parse_ms_json_shape_label(label_str: str) -> dict:
    """Parse a 4-part ``<Brand>____<Variant>____<Packaging_Type>____<Confidence>`` or ``S-T-P-S-T_<conf>_<Brand>____<Variant>____<Pkg>`` label from ``gs://ms-json-dataset``."""
    parts = label_str.strip().split("____")
    if len(parts) == 4:
        raw_b, raw_v, raw_p, raw_c = parts
        try:
            conf = float(raw_c)
        except ValueError:
            conf = 0.0
    elif len(parts) == 3 and parts[0].startswith("S-T-P-S-T_"):
        prefix_m = re.match(r"^S-T-P-S-T_([0-9.]+)_(.+)$", parts[0])
        if not prefix_m:
            return {
                "raw_label": label_str,
                "is_valid_format": False,
                "is_others_catchall": True,
                "confidence": 0.0,
            }
        conf = float(prefix_m.group(1))
        raw_b = prefix_m.group(2)
        raw_v, raw_p = parts[1], parts[2]
    else:
        return {
            "raw_label": label_str,
            "is_valid_format": False,
            "is_others_catchall": True,
            "confidence": 0.0,
        }

    is_others = (
        raw_b.lower() in ("others", "other", "na", "unknown")
        or raw_v.lower() in ("others", "other", "na", "unknown")
        or "others" in raw_v.lower().split("_")
    )
    info = parse_variant_folder("ms_json", f"{raw_b}__{raw_v}")
    pkg = normalize_packaging_type(None if raw_p.upper() == "NA" else raw_p, category=info["category"], brand=info["brand"], variant=info["variant"])
    return {
        "raw_label": label_str,
        "is_valid_format": True,
        "is_others_catchall": is_others,
        "confidence": conf,
        "brand": info["brand"],
        "variant": info["variant"],
        "base_variant": info["base_variant"],
        "product": f"{info['brand']}__{info['base_variant']}",
        "basepack_id": f"{info['brand']}__{info['base_variant']}__{pkg}",
        "category": info["category"],
        "packaging_type": pkg,
        "is_hul": info["is_hul"],
        "has_promo": info["has_promo"],
        "promo_type": info["promo_type"],
    }


def parse_gtmerch_bay_brand(fname: str) -> dict | None:
    """Return campaign bay brand metadata ONLY when the filename contains an explicit brand token (ZERO Dove fallback!)."""
    low = fname.lower()
    has_promo = "promo" in low or "win" in low or "free" in low
    promo_type = "Free_Gift" if has_promo else None
    for tokens, brand, category, pkg, is_hul in GTMERCH_BAY_BRAND_RULES:
        if any(t in low for t in tokens):
            return {
                "brand": brand,
                "category": category,
                "packaging_type": pkg,
                "is_hul": is_hul,
                "has_promo": has_promo,
                "promo_type": promo_type,
            }
    return None


def parse_pack_filename_brand_variant(filename: str) -> dict | None:
    """Extract embedded ``Brand__Variant`` metadata from ``pack_data`` filenames when present."""
    stem = PurePosixPath(filename).stem
    stem_clean = re.sub(r"(?:\s-\s*Copy(?:\s*\(\d+\))?|\s*\(copy\)|\s+\(\d+\))$", "", stem, flags=re.I)
    if BBOX_FILE_RE.match(stem_clean):
        return None
    m_bv = PACK_BV_FILE_RE.match(stem_clean)
    if not m_bv:
        return None
    raw_b = m_bv.group("brand")
    if (raw_b.isdigit() and raw_b != "3_Roses") or raw_b.lower() == "processed":
        return None
    var = m_bv.group("variant")
    return parse_variant_folder("pack_data", f"{raw_b}__{var}")


def parse_crop_filename(filename: str) -> dict:
    """Extract ``(session_id, crop_idx, bbox, is_synthetic_copy, is_augmented_copy)``."""
    name = PurePosixPath(filename).name
    is_aug = bool(AUGMENTED_FILE_RE.search(name))
    is_copy = bool(COPY_FILE_RE.search(name))
    stem = PurePosixPath(name).stem
    stem_clean = re.sub(r"(?:\s-\s*Copy(?:\s*\(\d+\))?|\s*\(copy\)|\s+\(\d+\))$", "", stem, flags=re.I)

    m_bbox = BBOX_FILE_RE.match(stem_clean)
    if m_bbox:
        sess = m_bbox.group("sess")
        parts = sess.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            sess = parts[0]
        return {
            "session_id": sess,
            "crop_idx": int(m_bbox.group("idx")),
            "bbox": (int(m_bbox.group("x1")), int(m_bbox.group("y1")), int(m_bbox.group("x2")), int(m_bbox.group("y2"))),
            "is_synthetic_copy": is_copy,
            "is_augmented_copy": is_aug,
        }

    m_bv = PACK_BV_FILE_RE.match(stem_clean)
    if m_bv and (not m_bv.group("brand").isdigit() or m_bv.group("brand") == "3_Roses") and m_bv.group("brand").lower() != "processed":
        raw_b = m_bv.group("brand")
        brand = BRAND_ALIAS_MAP.get(raw_b.lower(), raw_b)
        sess_num = m_bv.group("sess")
        idx = int(m_bv.group("idx") or m_bv.group("img_idx") or m_bv.group("idx_only") or 0)
        return {
            "session_id": sess_num if sess_num else f"pack_{brand.lower()}_{idx // 10}",
            "crop_idx": idx,
            "bbox": None,
            "is_synthetic_copy": is_copy,
            "is_augmented_copy": is_aug,
        }

    m_std = STD_FILE_RE.match(stem_clean)
    if m_std and not stem_clean.lower().startswith("copy"):
        sess = m_std.group("sess")
        parts = sess.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            sess = parts[0]
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and len(parts[0]) <= 2 and len(parts[1]) >= 3:
            sess = parts[1]
        return {
            "session_id": sess,
            "crop_idx": int(m_std.group("idx")),
            "bbox": None,
            "is_synthetic_copy": is_copy,
            "is_augmented_copy": is_aug,
        }

    m_sb = SPACE_BRACKET_SESS_RE.search(stem_clean)
    if m_sb and not stem_clean.lower().startswith("copy"):
        return {
            "session_id": m_sb.group("sess"),
            "crop_idx": int(m_sb.group("idx")),
            "bbox": None,
            "is_synthetic_copy": is_copy,
            "is_augmented_copy": is_aug,
        }

    fallback_sess = hashlib.sha1(stem_clean.encode("utf-8")).hexdigest()[:12]
    return {
        "session_id": f"single_{fallback_sess}",
        "crop_idx": 0,
        "bbox": None,
        "is_synthetic_copy": is_copy,
        "is_augmented_copy": is_aug,
    }


def parse_gtmerch_filename(filename: str) -> dict:
    """Parse a ``gtmerch`` YOLO shelf image or label filename into outlet/asset/session metadata."""
    base = filename.rsplit("/", 1)[-1].strip()
    base_low = base.lower()
    if not base or base.startswith(".") or base_low in GTMERCH_ARTIFACT_FILENAMES or base_low.endswith((".cache", ".amltmp", ".amlignore")):
        return {
            "raw_filename": base, "stem": base, "outlet_id": None, "asset_id": None,
            "capture_date": None, "timestamp_ms": None, "session_id": None,
            "is_copy_variant": False, "is_artifact_file": True,
        }

    stem = base.rsplit(".", 1)[0] if "." in base else base
    stem_clean = re.sub(r"(?:\s-\s*Copy(?:\s*\(\d+\))?|\s*\(copy\)|\s+\(\d+\))$", "", stem, flags=re.I).strip()
    is_copy_var = stem_clean != stem.strip() or bool(COPY_FILE_RE.search(base))

    m_out = GTMERCH_OUTLET_RE.match(stem_clean)
    if m_out:
        outlet = m_out.group("outlet").upper()
        asset = m_out.group("asset").upper()
        return {
            "raw_filename": base, "stem": stem, "outlet_id": outlet, "asset_id": asset,
            "capture_date": m_out.group("date"), "timestamp_ms": int(m_out.group("ts")),
            "session_id": outlet, "is_copy_variant": is_copy_var, "is_artifact_file": False,
        }

    m_num = GTMERCH_NUMERIC_SESS_RE.match(stem_clean)
    if m_num:
        sess = m_num.group("sess")
        return {
            "raw_filename": base, "stem": stem, "outlet_id": f"STORE_{sess}", "asset_id": None,
            "capture_date": None, "timestamp_ms": None, "session_id": sess,
            "is_copy_variant": is_copy_var, "is_artifact_file": False,
        }

    crop_info = parse_crop_filename(base)
    sid = str(crop_info["session_id"])
    return {
        "raw_filename": base, "stem": stem, "outlet_id": sid, "asset_id": None,
        "capture_date": None, "timestamp_ms": None, "session_id": sid,
        "is_copy_variant": is_copy_var or bool(crop_info["is_synthetic_copy"]), "is_artifact_file": False,
    }
