"""Client Data Taxonomy Normalization & Store-Session Filename Parser (``gs://marketshare-datasets``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Input: Raw GCS paths across 3 siloed trees (1,002,209 crops, 2,091 dirs)  │
  │    1. brand_data/brand/brand_data/<Brand>/<file>.jpg        (941 folders)  │
  │    2. input_data/variant_<tree>/train/<Brand>__<Variant>/   (1,104 folders)│
  │    3. pack_data/pack/train/<Category>__<PkgType>/           (46 folders)   │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Output: Canonical W1-W9 record + store-visit session_id & bbox metadata   │
  │    • Strips vendor 'A'/'Z_' sort prefixes & maps spelling aliases          │
  │    • Unifies 7 A-prefix twin brand pairs (ADr_Wash -> Dr_Wash)             │
  │    • Classifies 884 unmatched brand folders (862 Non-HUL rivals, Axe/Brut) │
  │    • Overrides misplaced oral/hair folders in variant_skin via keywords    │
  │    • Decomposes base_variant, packaging_type, promo_type, price, artwork   │
  │    • Detects synthetic copies (copy116.jpg, - Copy.jpg, (copy), DA_TBC_*)  │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

NOISE_BRAND_FOLDERS: set[str] = set()

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

# Fallback-only mapping used strictly on line 268 when BOTH product-form keyword
# rules (VARIANT_CATEGORY_KEYWORD_RULES) AND brand-level defaults (BRAND_CATEGORY_MAP)
# fail to match an unseen folder in a future data drop.
# Note: The 5 client `input_data/variant_*` directories are legacy vendor training-batch
# silos rather than 1:1 retail categories (37,796 MD5 matches to `pack_data` prove
# `variant_p5` mixes Beverages/HFD/Foods/HHC, `variant_wash` mixes Laundry/PW, and
# `variant_gen` mixes Hair_Care/DMT). Each legacy tree defaults here to its modal category.
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

# Product-form keyword overrides for multi-category umbrella brands (e.g. `Dove`
# spanning Skin/Hair_Care/PW/DMT, or `Ayush` spanning Skin/Oral_Care/Hair_Care/PW)
# that were placed together inside `variant_skin` by the upstream vendor.
# Includes both US and UK/Commonwealth spellings (e.g. moisturizer/moisturiser,
# color/colour, sanitizer/sanitiser) plus known client folder typos (`conditionar`
# in `Dove__Intense_Repair__Conditionar`). In `hul_builder.py`, these rules are
# additionally cross-checked against byte-identical MD5 consensus from `pack_data`.
VARIANT_CATEGORY_KEYWORD_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:^|_)(toothpaste|tooth_paste|tooth_gel|toothbrush|tooth_brush|mouthwash|mouth_wash|dental|anti_cavity|freshness_gel_cardamom|rock_salt|purifying_turmeric_b4g1)(?:$|_)",
            re.I,
        ),
        "Oral_Care",
    ),
    (
        re.compile(
            r"(?:^|_)(facewash|face_wash|face_cream|day_cream|night_cream|sunscreen|toner|cleanser|bodylotion|body_lotion|body_lotions|bodyscrub|body_scrub|sheet_mask|clay_mask|lip_mask|sleeping_mask|moisturizer|moisturiser|moisturising|moisturizing)(?:$|_)",
            re.I,
        ),
        "Skin",
    ),
    (
        # Note: `conditionar` (with 'a') is intentional to match the literal client
        # folder typo `gs://marketshare-datasets/input_data/variant_skin/train/Dove__Intense_Repair__Conditionar`.
        re.compile(
            r"(?:^|_)(shampoo|shampoos|conditioner|conditioners|conditionar|hair_mask|deep_repair_mask|moisture_mask|strength_mask|hair_oil|hair_colour|hair_color|scalp|dandruff|anti_dandruff|antidandruff|bringha|bhringaraj|hair_fall|hairfall|beautiful_curls|healthy_rituals|daily_shine|nourishing_oil_care|oxygen_moisture|environment_defence|peptide_bond_strength|mencare_[a-z0-9_]*2_in_1|intense_damage_repair__?mask|intense_repair__?mask)(?:$|_)",
            re.I,
        ),
        "Hair_Care",
    ),
    (
        re.compile(
            r"(?:^|_)(roll_on|deodorant|deodrant|deo_spray|body_spray|antiperspirant|anti_perspirant)(?:$|_)",
            re.I,
        ),
        "DMT",
    ),
    (
        re.compile(
            r"(?:^|_)(surface_cleaner|floor_cleaner|glass_cleaner|kitchen_cleaner|fruit_and_vegetable_wash|dishwash|dish_wash|toilet_cleaner|sanitiser|sanitizer)(?:$|_)",
            re.I,
        ),
        "HHC",
    ),
    (
        re.compile(
            r"(?:^|_)(germ_removal_detergent|detergent_powder|detergent_liquid|liquid_detergent|washing_powder|fabric_conditioner|fabric_wash|stain_remover)(?:$|_)",
            re.I,
        ),
        "Laundry",
    ),
    (
        re.compile(
            r"(?:^|_)(soap|soaps|bath_soap|bathing_bar|bodywash|body_wash|shower_gel|handwash|hand_wash|hand_sanitizer|hand_sanitiser)(?:$|_)",
            re.I,
        ),
        "PW",
    ),
]

PKG_SUFFIX_RE = re.compile(
    r"(?:^|_)(Bottle|Sachet|Sachets|Pouch|Jar|Tub|Tube|Bar|Box|Can|Carton|Stick|Roll_On|Pump|Tin|Wrapper|Tetrapack|Tetra)(?:_(?:New|Old|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+|Rs_?\d+|\d+))*$",
    re.I,
)
PKG_SUFFIX_NORM = {
    "sachets": "Sachet",
    "tetra": "Tetrapack",
    "roll_on": "Roll_On",
}
PRODUCT_FORM_PACK_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:^|_)(sachet|sachets|sheet_mask)(?:$|_)", re.I), "Sachet"),
    (re.compile(r"(?:^|_)(tetra|tetrapack|milkshake)(?:$|_)", re.I), "Tetrapack"),
    (re.compile(r"(?:^|_)(tea_bags)(?:$|_)", re.I), "Carton"),
    (re.compile(r"(?:^|_)(biscuits)(?:$|_)", re.I), "Wrapper"),
    (re.compile(r"(?:^|_)(soap|soaps|bathing_bar)(?:$|_)", re.I), "Bar"),
    # Note: `conditionar` matches the literal client folder `Dove__Intense_Repair__Conditionar` (a tube SKU).
    (re.compile(r"(?:^|_)(toothpaste|face_wash|facewash|cream|ointment|conditionar)(?:$|_)", re.I), "Tube"),
    (re.compile(r"(?:^|_)(hair_mask|moisture_mask|strength_mask|clay_mask|sleeping_mask|repair_mask|_?mask)(?:$|_)", re.I), "Jar"),
    (
        re.compile(
            r"(?:^|_)(roll_on|hair_oil|oil|serum|mouthwash|shampoo|conditioner|bodywash|body_wash|handwash|hand_wash|bodylotion|body_lotion|lotions)(?:$|_)",
            re.I,
        ),
        "Bottle",
    ),
    (re.compile(r"(?:^|_)(atta|salt|poly|doy|powder|detergent|soup)(?:$|_)", re.I), "Pouch"),
]
PROMO_TOKEN_RE = re.compile(r"(?:^|_)(B[1-9]G[1-9]|PO[2-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+)(?:$|_)", re.I)
PRICE_TOKEN_RE = re.compile(r"(?:^|_)Rs_?(\d+)(?:$|_)", re.I)
ARTWORK_TOKEN_RE = re.compile(r"_(New|Old|Square)(?:_\d+)?$", re.I)
STRIP_PROMO_ART_SUFFIX_RE = re.compile(
    r"_(?:New|Old|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+|Rs_?\d+)(?:_\d+)?$",
    re.I,
)

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
PACK_BV_FILE_RE = re.compile(
    r"^(?P<brand>[A-Za-z0-9][A-Za-z0-9_]*?)__(?P<variant>[A-Za-z0-9_]+?)(?:_(?:image|img)_?(?P<img_idx>\d+)|_(?:GT\d+_)?(?:PCLW_)?(?P<sess>\d{2,})_(?P<idx>\d+)|_(?P<idx_only>\d+))?$",
    re.I,
)


def normalize_brand_folder(folder: str) -> dict | None:
    """Normalize a ``brand_data/`` folder into canonical ``(brand, is_hul, category, legacy_tier)``.

    While ``input_data/variant_*`` contains only HUL brands (63 brands, ``is_hul=True``),
    ``brand_data/brand/brand_data/`` spans 941 raw folders (934 canonical brands):
      - 60 HUL brands (303,727 crops) -> ``is_hul=True``
      - 874 Non-HUL competitor brands (280,145 crops, e.g. Dettol, Colgate, Pantene,
        Nivea, Ariel, Tide, Maggi, Nescafe) -> ``is_hul=False``
    Tracking ``is_hul`` is required to compute Share of Shelf (SOS = HUL facings / Total
    category facings), where Non-HUL competitor facings belong in the denominator only.
    """
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
        # Strip vendor 'A' sort-priority prefix (e.g. 'ADove' -> 'Dove') only when
        # len >= 3 and followed by an uppercase letter so 2-letter brands like 'AX' are safe.
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
    """Parse an ``input_data/variant_*/train/<Brand>__<Variant>`` folder into orthogonal W1-W9 attributes.

    Category resolution follows a 3-tier hierarchy because the 5 ``variant_*`` directories
    are legacy training-batch silos rather than pure retail categories:
      1. Product-form keywords in ``VARIANT_CATEGORY_KEYWORD_RULES`` (routes multi-category
         umbrella brands like Dove shampoo -> ``Hair_Care``, Dove soap/bodywash -> ``PW``,
         Ayush toothpaste -> ``Oral_Care``).
      2. Brand default category in ``BRAND_CATEGORY_MAP``.
      3. ``TREE_DEFAULT_CATEGORY`` as a last-resort fallback for unseen brands.
    """
    raw = folder.strip()
    if "__" in raw:
        raw_brand, variant = raw.split("__", 1)
    else:
        raw_brand, variant = raw, "Standard"
    brand = BRAND_ALIAS_MAP.get(raw_brand.lower(), raw_brand)
    b_key = brand.lower()
    raw_silo_category = tree[len("variant_") :] if tree.startswith("variant_") else tree

    # 1. Category resolution: product-form keywords override brand defaults
    # (except nutrition supplements like Wellbeing_Nutrition which stay in HFD).
    category = None
    if b_key == "wellbeing_nutrition":
        category = "HFD"
    else:
        for kw_re, cat_override in VARIANT_CATEGORY_KEYWORD_RULES:
            if kw_re.search(variant):
                category = cat_override
                break
    if category is None:
        category = BRAND_CATEGORY_MAP.get(b_key) or TREE_DEFAULT_CATEGORY.get(tree, "Unknown")

    # 2. Pack-type resolution: strict suffix -> product-form noun -> Unknown (later backfilled by MD5/CLM)
    pkg_m = PKG_SUFFIX_RE.search(variant)
    if pkg_m:
        raw_pkg = pkg_m.group(1)
        pkg_type = PKG_SUFFIX_NORM.get(raw_pkg.lower(), raw_pkg.title())
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
        "raw_silo_category": raw_silo_category,
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
            "bbox": (
                int(m_bbox.group("x1")),
                int(m_bbox.group("y1")),
                int(m_bbox.group("x2")),
                int(m_bbox.group("y2")),
            ),
            "is_synthetic_copy": is_copy,
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
        }

    m_std = STD_FILE_RE.match(stem_clean)
    if m_std and not stem_clean.lower().startswith("copy"):
        sess = m_std.group("sess")
        parts = sess.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            sess = parts[0]
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and len(parts[0]) <= 2 and len(parts[1]) >= 3:
            # Strip detector class-id prefix e.g. 50_1317_53.jpg -> session_id="1317"
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

