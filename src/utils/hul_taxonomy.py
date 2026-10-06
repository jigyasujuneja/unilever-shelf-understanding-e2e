"""Client Data Taxonomy Normalization & Store-Session Filename Parser (``gs://marketshare-datasets``).

Normalizes folder and filename conventions across the three siloed trees:
  1. ``brand_data/brand/brand_data/<Brand>/<file>.jpg``
  2. ``input_data/variant_<tree>/train/<Brand>__<Variant>/<file>.jpg``
  3. ``pack_data/pack/train/<Category>__<PkgType>/<file>.jpg``
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

NOISE_BRAND_FOLDERS = {"W2", "Z", "An", "Fr", "He"}

BRAND_ALIAS_MAP = {
    "a_3_roses": "3_Roses",
    "3_roses": "3_Roses",
    "abrown&polson": "Brown_And_Polson",
    "brown_and_polson": "Brown_And_Polson",
    "acif": "CIF",
    "cif": "CIF",
    "aclinic": "Clinic_Plus",
    "clinic_plus": "Clinic_Plus",
    "ahellmanns": "Hellmans",
    "hellmans": "Hellmans",
    "abrook_bond": "Brooke_Bond",
    "brooke_bond": "Brooke_Bond",
}

A_PREFIX_TWINS = {
    "adr_wash": "Dr_Wash",
    "agenteel": "Genteel",
    "amr_muscle": "Mr_Muscle",
    "amysore_sandal": "Mysore_Sandal",
    "arevive": "Revive",
    "asafe_wash": "Safe_Wash",
    "avanish": "Vanish",
}

HUL_BRANDS = {
    "3_roses", "annapurna", "axe", "ayush", "boost", "breeze", "brooke_bond",
    "brown_and_polson", "bru", "brut", "cif", "clear", "clinic_plus", "closeup",
    "comfort", "cornetto", "domex", "dove", "elle18", "fair_and_lovely",
    "freshwrapp", "glow_and_lovely", "hamam", "hellmans", "horlicks", "indulekha",
    "kissan", "knorr", "kwality", "lakme", "lbp", "lifebuoy", "lipton", "liril",
    "lux", "maltova", "moti", "nature_protect", "novology", "oralb", "pears",
    "pepsodent", "ponds", "pukka", "red_label", "rex", "rexona", "rin", "ruby",
    "simple", "st_ives", "sun", "sunlight", "sunsilk", "surf_excel", "taaza",
    "taj_mahal", "tresemme", "v_wash", "vaseline", "vim", "viva",
    "wellbeing_nutrition", "wheel",
}

HUL_BRAND_ONLY_NO_VARIANT_TREE = {"axe", "brut", "brooke_bond"}

BRAND_CATEGORY_MAP = {
    "ayush": "Skin", "dove": "Skin", "elle18": "Skin", "glow_and_lovely": "Skin",
    "lakme": "Skin", "novology": "Skin", "ponds": "Skin", "simple": "Skin", "vaseline": "Skin",
    "comfort": "Laundry", "rin": "Laundry", "sunlight": "Laundry", "surf_excel": "Laundry", "wheel": "Laundry",
    "hamam": "PW", "lifebuoy": "PW", "liril": "PW", "lux": "PW", "moti": "PW",
    "nature_protect": "PW", "pears": "PW", "st_ives": "PW", "v_wash": "PW",
    "clear": "Hair_Care", "clinic_plus": "Hair_Care", "indulekha": "Hair_Care", "lbp": "Hair_Care",
    "sunsilk": "Hair_Care", "tresemme": "Hair_Care", "rexona": "DMT", "axe": "DMT", "brut": "DMT",
    "closeup": "Oral_Care", "oralb": "Oral_Care", "pepsodent": "Oral_Care",
    "3_roses": "Beverages", "brooke_bond": "Beverages", "bru": "Beverages", "lipton": "Beverages",
    "pukka": "Beverages", "red_label": "Beverages", "taaza": "Beverages", "taj_mahal": "Beverages",
    "boost": "HFD", "horlicks": "HFD", "maltova": "HFD", "viva": "HFD", "wellbeing_nutrition": "HFD",
    "annapurna": "Foods", "brown_and_polson": "Foods", "hellmans": "Foods",
    "kissan": "Foods", "knorr": "Foods", "rex": "Foods",
    "cif": "HHC", "domex": "HHC", "freshwrapp": "HHC", "ruby": "HHC", "sun": "HHC", "vim": "HHC",
}

TREE_DEFAULT_CATEGORY = {
    "variant_skin": "Skin",
    "variant_wash": "PW",
    "variant_p4": "Oral_Care",
    "variant_p5": "Foods",
    "variant_gen": "Hair_Care",
    "variant_oral": "Oral_Care",
}

PACK_CATEGORY_ALIAS = {
    "Hair": "Hair_Care",
}

VARIANT_CATEGORY_KEYWORD_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:^|_)(toothpaste|tooth_paste|toothbrush|tooth_brush|mouthwash|anti_cavity)(?:$|_)", re.I), "Oral_Care"),
    (re.compile(r"(?:^|_)(shampoo|conditioner|hair_mask|hair_oil|scalp|dandruff|bringha|hair_fall)(?:$|_)", re.I), "Hair_Care"),
    (re.compile(r"(?:^|_)(roll_on|deodorant|deo_spray)(?:$|_)", re.I), "DMT"),
]

PKG_SUFFIX_RE = re.compile(
    r"(?:^|_)(Bottle|Sachet|Pouch|Jar|Tub|Tube|Bar|Box|Can|Carton|Stick|Roll_On|Pump|Tin|Wrapper|Tetrapack)(?:_(?:New|Old|\d*Old_Pack|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+|Rs_?\d+|\d+))*$",
    re.I,
)
PRODUCT_FORM_PACK_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:^|_)(soap|bathing_bar)(?:$|_)", re.I), "Bar"),
    (re.compile(r"(?:^|_)(toothpaste|face_wash|facewash|cream|ointment)(?:$|_)", re.I), "Tube"),
    (re.compile(r"(?:^|_)(roll_on|hair_oil|oil|serum|mouthwash|shampoo|conditioner)(?:$|_)", re.I), "Bottle"),
    (re.compile(r"(?:^|_)(atta|salt|poly|detergent_powder)(?:$|_)", re.I), "Pouch"),
]
PROMO_TOKEN_RE = re.compile(r"(?:^|_)(B[1-9]G[1-9]|PO[2-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+)(?:$|_)", re.I)
PRICE_TOKEN_RE = re.compile(r"(?:^|_)Rs_?(\d+)(?:$|_)", re.I)
ARTWORK_TOKEN_RE = re.compile(r"_(New|Old|\d*Old_Pack|Square)(?:_\d+)?$", re.I)
STRIP_PROMO_ART_SUFFIX_RE = re.compile(
    r"(?:_(?:New|Old|\d*Old_Pack|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+|Rs_?\d+)(?:_\d+)?)+$",
    re.I,
)

COPY_FILE_RE = re.compile(
    r"(?:^copy\d*\.(?:jpe?g|png)$|\s-\s*Copy(?:\s*\(\d+\))?|\(copy\)\.(?:jpe?g|png)$|^DA_(?:TBC_)?|(?:^|_)\d+\s+\(\d+\)\.(?:jpe?g|png)$)",
    re.I,
)
BBOX_FILE_RE = re.compile(
    r"^(?:processed_|\d+\.\d+____)?(?P<sess>[A-Za-z0-9_-]+?)(?:__|_)0*(?P<idx>\d+)__(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)$"
)
STD_FILE_RE = re.compile(
    r"^(?:processed_|DA_TBC_|DA_|GT\d+_PCLW_|GT\d+_|PCLW_|Personal_Wash|sc_\d+_|sc_|\d+\.\d+_)?(?P<sess>[A-Za-z0-9_-]+?)_(?P<idx>\d+)$",
    re.I,
)
PACK_BV_FILE_RE = re.compile(
    r"^(?P<brand>[A-Za-z0-9][A-Za-z0-9_]*?)__(?P<variant>[A-Za-z0-9_]+?)(?:_(?:image|img)_?(?P<img_idx>\d+)|_(?:GT\d+_)?(?:PCLW_)?(?P<sess>\d{2,})_(?P<idx>\d+)|_(?P<idx_only>\d+))?$",
    re.I,
)


def is_copy_or_synthetic(uri_or_filename: str) -> bool:
    """True when the image filename is a duplicate copy (`copy*.jpg`, `- Copy`, `(copy)`, `0 (2).jpg`) or synthetic `DA_TBC_*` crop."""
    name = PurePosixPath(uri_or_filename).name
    return bool(COPY_FILE_RE.search(name))


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
        brand = raw
        tier = "standard"
    key = brand.lower()
    is_hul = key in HUL_BRANDS
    if is_hul and key in HUL_BRAND_ONLY_NO_VARIANT_TREE and tier == "standard":
        tier = "hul_umbrella_brand_only"
    return {
        "raw_folder": raw,
        "brand": brand,
        "is_hul": is_hul,
        "category": BRAND_CATEGORY_MAP.get(key, "Unknown"),
        "legacy_tier": tier,
    }


def parse_variant_folder(tree: str, folder: str) -> dict:
    """Parse an ``input_data/variant_*/train/<Brand>__<Variant>`` folder into orthogonal W1-W9 attributes."""
    raw = folder.strip()
    if "__" in raw:
        raw_brand, raw_variant = raw.split("__", 1)
    else:
        raw_brand, raw_variant = raw, "Standard"
    variant = re.sub(r"_+", "_", raw_variant.replace(" ", "_")).strip("_")
    brand = BRAND_ALIAS_MAP.get(raw_brand.lower(), raw_brand)
    b_key = brand.lower()

    # 1. Category resolution: product-form keywords override brand defaults (fixes Ayush oral & Dove hair in variant_skin)
    category = None
    for kw_re, cat_override in VARIANT_CATEGORY_KEYWORD_RULES:
        if kw_re.search(variant):
            category = cat_override
            break
    if category is None:
        category = BRAND_CATEGORY_MAP.get(b_key) or TREE_DEFAULT_CATEGORY.get(tree, "Unknown")

    # 2. Pack-type resolution: strict suffix -> product-form noun -> Unknown
    pkg_m = PKG_SUFFIX_RE.search(variant)
    if pkg_m:
        pkg_type = pkg_m.group(1).title()
    else:
        pkg_type = "Unknown"
        for pf_re, pf_pack in PRODUCT_FORM_PACK_RULES:
            if pf_re.search(variant):
                pkg_type = pf_pack
                break

    # 3. Promo / multipack / price-point / artwork extraction (Workflow 9)
    promo_m = PROMO_TOKEN_RE.search(variant)
    price_m = PRICE_TOKEN_RE.search(variant)
    art_m = ARTWORK_TOKEN_RE.search(variant)
    price_inr = int(price_m.group(1)) if price_m else None
    promo_type = promo_m.group(1).upper() if promo_m else (f"RS{price_inr}" if price_inr else None)
    artwork_gen = art_m.group(1).lower() if art_m else "standard"

    # 4. Canonical base_variant strips promo, price, and artwork suffixes while preserving core SKU variant
    base_variant = STRIP_PROMO_ART_SUFFIX_RE.sub("", variant) or variant
    return {
        "tree": tree,
        "raw_folder": raw,
        "category": category,
        "brand": brand,
        "variant": variant,
        "base_variant": base_variant,
        "packaging_type": pkg_type,
        "is_hul": b_key in HUL_BRANDS,
        "has_promo": promo_type is not None,
        "promo_type": promo_type,
        "has_price_tag": price_inr is not None,
        "artwork_gen": artwork_gen,
    }


def parse_pack_folder(folder: str) -> dict:
    """Parse a ``pack_data/pack/train/<Category>__<PkgType>`` folder into ``(category, packaging_type)``."""
    raw = folder.strip()
    if "__" in raw:
        raw_cat, pkg_type = raw.split("__", 1)
        category = PACK_CATEGORY_ALIAS.get(raw_cat, raw_cat)
    else:
        category, pkg_type = "Unknown", raw
    return {"raw_folder": raw, "category": category, "packaging_type": pkg_type}


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
    """Extract ``(session_id, crop_idx, bbox, is_synthetic_copy)``."""
    name = PurePosixPath(filename).name
    is_copy = is_copy_or_synthetic(name)
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
            "bbox": (
                int(m_bbox.group("x1")),
                int(m_bbox.group("y1")),
                int(m_bbox.group("x2")),
                int(m_bbox.group("y2")),
            ),
            "is_synthetic_copy": is_copy,
        }

    m_bv = PACK_BV_FILE_RE.match(stem_clean)
    if (
        m_bv
        and (not m_bv.group("brand").isdigit() or m_bv.group("brand") == "3_Roses")
        and m_bv.group("brand").lower() != "processed"
    ):
        raw_b = m_bv.group("brand")
        brand = BRAND_ALIAS_MAP.get(raw_b.lower(), raw_b)
        sess_num = m_bv.group("sess")
        idx = int(m_bv.group("idx") or m_bv.group("img_idx") or m_bv.group("idx_only") or 0)
        return {
            "session_id": sess_num if sess_num else f"pack_{brand.lower()}_{idx // 10}",
            "crop_idx": idx,
            "bbox": None,
            "is_synthetic_copy": is_copy,
        }

    m_std = STD_FILE_RE.match(stem_clean)
    if m_std and not stem_clean.lower().startswith("copy"):
        sess = m_std.group("sess")
        parts = sess.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            sess = parts[0]
        elif (
            len(parts) == 2
            and parts[0].isdigit()
            and parts[1].isdigit()
            and len(parts[0]) <= 2
            and len(parts[1]) >= 3
        ):
            sess = parts[1]
        return {
            "session_id": sess,
            "crop_idx": int(m_std.group("idx")),
            "bbox": None,
            "is_synthetic_copy": is_copy,
        }

    fallback_sess = hashlib.sha1(stem_clean.encode("utf-8")).hexdigest()[:12]
    return {
        "session_id": f"single_{fallback_sess}",
        "crop_idx": 0,
        "bbox": None,
        "is_synthetic_copy": is_copy,
    }
