"""Canonical Retail Packaging Type Resolver (Zero 'Unknown' Guarantee & Toothbrush/Talc Fixes)."""

from __future__ import annotations

import re

from utils.taxonomy.categories import (
    PACK_CATEGORY_ALIAS,
    normalize_category_name,
    normalize_variant_tokens,
)

CANONICAL_PACKAGING_TYPES: tuple[str, ...] = (
    "Bottle",
    "Tube",
    "Pouch",
    "Sachet",
    "Jar",
    "Tub",
    "Bar",
    "Carton",
    "Can",
    "Tin",
    "Pump",
    "Spray",
    "Stick",
    "Box",
    "Pack",
    "Wrapper",
    "Tetrapack",
    "Roll_On",
)

PKG_SYNONYM_MAP: dict[str, str] = {
    "sachets": "Sachet",
    "sachet": "Sachet",
    "tetra": "Tetrapack",
    "tetrapack": "Tetrapack",
    "roll_on": "Roll_On",
    "polly": "Pouch",
    "poly": "Pouch",
    "pouch": "Pouch",
    "bars": "Bar",
    "soapbars": "Bar",
    "bar_block": "Bar",
    "bar": "Bar",
    "tablets": "Bottle",
    "bottle": "Bottle",
    "jar_tub": "Jar",
    "jar": "Jar",
    "tub": "Tub",
    "tube": "Tube",
    "carton_box": "Carton",
    "carton": "Carton",
    "box": "Box",
    "can": "Can",
    "tin": "Tin",
    "pump": "Pump",
    "spray": "Spray",
    "stick": "Stick",
    "pack": "Pack",
    "wrapper": "Wrapper",
}

PKG_SUFFIX_RE = re.compile(
    r"(?:^|_)(Bottle|Sachet|Sachets|Pouch|Polly|Poly|Jar|Tub|Tube|Bar|Bars|Soapbars|Box|Can|Carton|Stick|Roll_On|Pump|Spray|Tin|Wrapper|Tetrapack|Tetra)(?:_(?:New|Old|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+|Rs_?\d+|copy\d+|\d+))*$",
    re.I,
)

PKG_SUFFIX_NORM: dict[str, str] = {
    "sachets": "Sachet",
    "tetra": "Tetrapack",
    "roll_on": "Roll_On",
    "polly": "Pouch",
    "poly": "Pouch",
    "bars": "Bar",
    "soapbars": "Bar",
    "tablets": "Bottle",
}

EXPLICIT_CONTAINER_TOKEN_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:^|_)(refill|refil|doy|polly|poly|pouch|coffee_pouch|surface_wipe|wipe|wipes|pichkoo)(?:$|_)", re.I), "Pouch"),
    (re.compile(r"(?:^|_)(roll_on)(?:$|_)", re.I), "Roll_On"),
    (re.compile(r"(?:^|_)(tetra|tetrapack)(?:$|_)", re.I), "Tetrapack"),
    (re.compile(r"(?:^|_)(sachet|sachets|sheet_mask)(?:$|_)", re.I), "Sachet"),
    (re.compile(r"(?:^|_)(peel_off_mask)(?:$|_)", re.I), "Tube"),
    (re.compile(r"(?:^|_)(instant_jar|bru_instant_jar)(?:$|_)", re.I), "Jar"),
    (re.compile(r"(?:^|_)(cup_noodles|cuppa_noodles|cuppa)(?:$|_)", re.I), "Tub"),
    (re.compile(r"(?:^|_)(smart_shots|tea_bags|descaler_tablets|cube_vegetarian|bouillon_cube)(?:$|_)", re.I), "Carton"),
    (re.compile(r"(?:^|_)(disinfectant_spray|disinfictant_spray|sunscreen_spray|smart_spray|shudhham_spray)(?:$|_)", re.I), "Spray"),
]

# Ordered from most specific physical form (`toothbrush` -> Carton, `talc` -> Bottle) to general forms.
# Brand names are intentionally excluded here and handled in Step 5 (`BRAND_DEFAULT_PACKAGING`) so
# `pack_data` MD5 consensus (`crc_votes`) and variant form tokens are never shadowed by brand defaults.
PRODUCT_FORM_PACK_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:^|_)(toothbrush|tooth_brush|toothbrushes|tooth_brushes|manual_toothbrush|electric_toothbrush|blade|razor|shaver|mosquito_coil|good_knight|smart_shots|descaler_tablets|cube_vegetarian)(?:$|_)",
            re.I,
        ),
        "Carton",
    ),
    (
        re.compile(
            r"(?:^|_)(dreamflower|aloe_cooling|star_light|radiance_sandal|talcum|talc|prickly_heat|dermi_cool|nycil|toothpowder|tooth_powder|dant_manjan|laundry_sanitiser|laundry_sanitizer|laundry_sanetizer)(?:$|_)",
            re.I,
        ),
        "Bottle",
    ),
    (re.compile(r"(?:^|_)(sachet|sachets|sheet_mask|shaving_foam)(?:$|_)", re.I), "Sachet"),
    (re.compile(r"(?:^|_)(tetra|tetrapack|milkshake)(?:$|_)", re.I), "Tetrapack"),
    (re.compile(r"(?:^|_)(roll_on)(?:$|_)", re.I), "Roll_On"),
    (
        re.compile(
            r"(?:^|_)(deodorant|deodrant|deo_spray|body_spray|men_deo)(?:$|_)",
            re.I,
        ),
        "Can",
    ),
    (re.compile(r"(?:^|_)(cold_coffee)(?:$|_)", re.I), "Tin"),
    (
        re.compile(
            r"(?:^|_)(refill|refil|doy|polly|poly|pouch|wipe|wipes|surface_wipe|soup|cream_of|coffee_pouch|pichkoo)(?:$|_)",
            re.I,
        ),
        "Pouch",
    ),
    (re.compile(r"(?:^|_)(peel_off_mask)(?:$|_)", re.I), "Tube"),
    (
        re.compile(
            r"(?:^|_)(hair_cream|anti_dandruff_cream|radiance_gel_cream|gel_cream|day_cream|night_cream)(?:$|_)",
            re.I,
        ),
        "Jar",
    ),
    (
        re.compile(
            r"(?:^|_)(tea_bags|clear_and_light|honey_lemon|darjeeling|earl_grey|chamomile|ashwagandha|custard_powder|corn_flour|ragi_bites|corn_flakes|muesli|oats|chocos)(?:$|_)",
            re.I,
        ),
        "Carton",
    ),
    (re.compile(r"(?:^|_)(biscuits|noodles|2_mins|choco_pie|monaco|oreo|kitkat|snickers|twix)(?:$|_)", re.I), "Wrapper"),
    (re.compile(r"(?:^|_)(soap|soaps|soap_bars?|bathing_bar|detergen_bar|detergent_bar|wash_bar|vim_bar|exo_bar|rin_bar)(?:$|_)", re.I), "Bar"),
    (
        re.compile(
            r"(?:^|_)(toothpaste|tooth_paste|tooth_gel|sensitivity_relief|purifying_turmeric|anti_cavity|dant_kanti|meswak|babool|face_wash|facewash|facwash|facescrub|face_scrub|shaving_cream|shavingcream|face_pack|facepack|scrub|cream|creame|creme|ointment|gel|conditionar|bb_natural|fairness|bright_beauty)(?:$|_)",
            re.I,
        ),
        "Tube",
    ),
    (
        re.compile(
            r"(?:^|_)(hair_mask|moisture_mask|strength_mask|clay_mask|sleeping_mask|repair_mask|_?mask|petroleum_jelly|petroleumjelly|peanut_butter|mayonnaise|spread|jam|pickle|chutney|ghee|honey|nutella|nescafe_classic|bru_instant_jar|instant_jar)(?:$|_)",
            re.I,
        ),
        "Jar",
    ),
    (
        re.compile(
            r"(?:^|_)(hair_oil|almond_drops|cool_oil|oil|bringha|bhringaraj|shikakai|serum|toner|astringent|mouthwash|listerine|shampoo|conditioner|bodywash|body_wash|handwash|hand_wash|hand_sanitizer|hand_sanitiser|hand_sanetizer|bodylotion|body_lotion|lotions|lotion|disinfectant|surface_cleaner|toilet_cleaner|toilet_cleanser|bathroom_cleanser|dishwash_liquid|matic_liquid|matic_front_load_liquid|matic_top_load_liquid|liquid|tablets|capsules|gummies|multivitamin|biotin|ketchup|sauce|syrup|vinegar)(?:$|_)",
            re.I,
        ),
        "Bottle",
    ),
    (
        re.compile(
            r"(?:^|_)(atta|salt|rock_salt|salt_crystal|sampann|powder|detergent|matic|masala|spices|chai|leaf|dust|natural_care|granules)(?:$|_)",
            re.I,
        ),
        "Pouch",
    ),
]

BRAND_DEFAULT_PACKAGING: dict[str, str] = {
    "ariel": "Pouch",
    "surf_excel": "Pouch",
    "tide": "Pouch",
    "wheel": "Pouch",
    "rin": "Pouch",
    "sunlight": "Pouch",
    "ghadi": "Pouch",
    "nirma": "Pouch",
    "henko": "Pouch",
    "fena": "Pouch",
    "comfort": "Bottle",
    "ezee": "Bottle",
    "genteel": "Bottle",
    "vanish": "Bottle",
    "ujala": "Bottle",
    "robin": "Bottle",
    "revive": "Pouch",
    "red_label": "Pouch",
    "taj_mahal": "Carton",
    "3_roses": "Pouch",
    "taaza": "Pouch",
    "lipton": "Carton",
    "pukka": "Carton",
    "tetley": "Carton",
    "twinings": "Carton",
    "wagh_bakri": "Pouch",
    "avt": "Pouch",
    "society": "Pouch",
    "girnar": "Carton",
    "organic_india": "Carton",
    "bru": "Pouch",
    "nescafe": "Jar",
    "sunrise": "Pouch",
    "continental": "Pouch",
    "levista": "Pouch",
    "cothas": "Pouch",
    "narasus": "Pouch",
    "horlicks": "Jar",
    "boost": "Jar",
    "bournvita": "Jar",
    "complan": "Jar",
    "pediasure": "Carton",
    "ensure": "Carton",
    "protinex": "Tin",
    "maltova": "Jar",
    "viva": "Jar",
    "hk_vitals": "Bottle",
    "oziva": "Bottle",
    "wellbeing_nutrition": "Bottle",
    "centrum": "Bottle",
    "plix": "Bottle",
    "fast_and_up": "Bottle",
    "bio_health": "Bottle",
    "fixuphealth": "Bottle",
    "eno": "Bottle",
    "colgate": "Tube",
    "closeup": "Tube",
    "pepsodent": "Tube",
    "sensodyne": "Tube",
    "dabur": "Tube",
    "meswak": "Tube",
    "babool": "Tube",
    "vicco": "Tube",
    "oralb": "Carton",
    "listerine": "Bottle",
    "clinic_plus": "Bottle",
    "sunsilk": "Bottle",
    "tresemme": "Bottle",
    "clear": "Bottle",
    "indulekha": "Bottle",
    "lbp": "Bottle",
    "love_beauty_and_planet": "Bottle",
    "head_and_shoulders": "Bottle",
    "pantene": "Bottle",
    "loreal": "Bottle",
    "biolage": "Bottle",
    "matrix": "Bottle",
    "garnier": "Tube",
    "parachute": "Bottle",
    "bajaj": "Bottle",
    "navratna": "Bottle",
    "kesh_king": "Bottle",
    "sesa": "Bottle",
    "vatika": "Bottle",
    "meera": "Bottle",
    "nyle": "Bottle",
    "chik": "Sachet",
    "livon": "Bottle",
    "streax": "Bottle",
    "ashwini": "Bottle",
    "lux": "Bar",
    "lifebuoy": "Bar",
    "pears": "Bar",
    "hamam": "Bar",
    "liril": "Bar",
    "breeze": "Bar",
    "moti": "Bar",
    "santoor": "Bar",
    "mysore_sandal": "Bar",
    "medimix": "Bar",
    "margo": "Bar",
    "cinthol": "Bar",
    "godrej_no_1": "Bar",
    "dettol": "Bar",
    "savlon": "Bar",
    "vivel": "Bar",
    "fiama": "Bar",
    "dyna": "Bar",
    "v_wash": "Bottle",
    "clean_and_dry": "Bottle",
    "ponds": "Tube",
    "lakme": "Tube",
    "glow_and_lovely": "Tube",
    "fair_and_lovely": "Tube",
    "vaseline": "Bottle",
    "simple": "Tube",
    "novology": "Tube",
    "elle18": "Bottle",
    "nivea": "Bottle",
    "himalaya": "Tube",
    "biotique": "Bottle",
    "lotus": "Tube",
    "joy": "Tube",
    "everyuth": "Tube",
    "clean_and_clear": "Bottle",
    "neutrogena": "Tube",
    "cetaphil": "Bottle",
    "mamaearth": "Tube",
    "plum": "Bottle",
    "dermaco": "Bottle",
    "dot_and_key": "Tub",
    "foxtale": "Bottle",
    "pilgrim": "Bottle",
    "olay": "Jar",
    "boroplus": "Tube",
    "boroline": "Tube",
    "lacto_calamine": "Bottle",
    "veet": "Tube",
    "fem": "Carton",
    "axe": "Bottle",
    "brut": "Bottle",
    "rexona": "Roll_On",
    "fogg": "Bottle",
    "denver": "Bottle",
    "engage": "Bottle",
    "wildstone": "Bottle",
    "vim": "Bar",
    "domex": "Bottle",
    "cif": "Bottle",
    "harpic": "Bottle",
    "lizol": "Bottle",
    "colin": "Bottle",
    "pril": "Bottle",
    "exo": "Bar",
    "giffy": "Bottle",
    "mr_muscle": "Bottle",
    "nimyle": "Bottle",
    "kissan": "Bottle",
    "knorr": "Pouch",
    "hellmans": "Jar",
    "brown_and_polson": "Carton",
    "annapurna": "Pouch",
    "cornetto": "Wrapper",
    "kwality": "Tub",
    "cup_noodles": "Tub",
    "maggi": "Wrapper",
    "yippee": "Wrapper",
    "wai_wai": "Wrapper",
    "top_ramen": "Wrapper",
    "chings": "Pouch",
    "veeba": "Bottle",
    "funfoods": "Jar",
    "del_monte": "Bottle",
    "hersheys": "Bottle",
    "nutella": "Jar",
    "pintola": "Jar",
    "myfitness": "Jar",
    "sundrop": "Jar",
    "keya": "Bottle",
    "weikfield": "Carton",
    "bakers": "Carton",
}

CATEGORY_DEFAULT_PACKAGING: dict[str, str] = {
    "Skin": "Tube",
    "PW": "Bar",
    "Hair_Care": "Bottle",
    "Oral_Care": "Tube",
    "Foods": "Pouch",
    "Beverages": "Pouch",
    "HFD": "Jar",
    "Laundry": "Pouch",
    "HHC": "Bottle",
    "DMT": "Bottle",
}


def normalize_packaging_type(raw_pkg: str | None, category: str = "Foods", brand: str = "", variant: str = "") -> str:
    """Normalize a raw packaging string into a canonical packaging type with 0 'Unknown' guarantee."""
    full_str = f"{brand}__{variant}".strip("_")
    norm_full = normalize_variant_tokens(full_str)

    # High-priority overrides for known mislabeled forms (e.g., toothbrushes or Pond's talc cans)
    if re.search(r"(?:^|_)(toothbrush|tooth_brush|toothbrushes|manual_toothbrush)(?:$|_)", norm_full, re.I):
        return "Carton"
    if brand.lower() == "ponds" and re.search(r"(?:^|_)(dreamflower|aloe_cooling|star_light|radiance_sandal|sandal|magic|talc)(?:$|_)", norm_full, re.I):
        return "Bottle"

    if raw_pkg:
        low = raw_pkg.strip().lower().replace(" ", "_")
        if low in PKG_SYNONYM_MAP:
            return PKG_SYNONYM_MAP[low]
        if raw_pkg in CANONICAL_PACKAGING_TYPES:
            return raw_pkg

    return resolve_packaging_type(brand=brand, variant=variant, category=category)


def resolve_packaging_type(
    brand: str,
    variant: str,
    category: str = "Foods",
    crc_votes: dict[str, int] | None = None,
) -> str:
    """Resolve canonical packaging_type for ``(brand, variant, category)`` with 0 'Unknown' output."""
    norm_var = normalize_variant_tokens(variant)
    full_bv = f"{brand}__{variant}"
    norm_full = normalize_variant_tokens(full_bv)
    cat_canon = normalize_category_name(category)

    # 1. Hard physical-form overrides (toothbrush -> Carton, Pond's talc -> Bottle)
    if re.search(r"(?:^|_)(toothbrush|tooth_brush|toothbrushes|manual_toothbrush)(?:$|_)", norm_full, re.I):
        return "Carton"
    if brand.lower() == "ponds" and re.search(r"(?:^|_)(dreamflower|aloe_cooling|star_light|radiance_sandal|sandal|magic|talc)(?:$|_)", norm_full, re.I):
        return "Bottle"

    # 2. Explicit suffix match on variant
    pkg_m = PKG_SUFFIX_RE.search(variant) or PKG_SUFFIX_RE.search(norm_var)
    if pkg_m:
        raw_pkg = pkg_m.group(1)
        return PKG_SUFFIX_NORM.get(raw_pkg.lower(), raw_pkg.title())

    # 2b. Explicit container token in variant (e.g. Refill/Doy/Polly/Roll_On/Tetrapack/Peel_Off_Mask/Cup_Noodles)
    for ct_re, ct_pack in EXPLICIT_CONTAINER_TOKEN_RULES:
        if ct_re.search(variant) or ct_re.search(norm_var):
            return ct_pack

    # 3. High-consensus MD5/CRC32 vote from pack_data (when >= 65% agreement on actual crop bytes)
    if crc_votes:
        total_v = sum(crc_votes.values())
        if total_v > 0:
            top_p, top_c = max(crc_votes.items(), key=lambda kv: kv[1])
            if top_c / total_v >= 0.65 and top_p != "Unknown":
                return PKG_SYNONYM_MAP.get(top_p.lower(), top_p)

    # 4. Category-scoped flavor/form rules (prevent cross-category token collisions)
    if cat_canon == "Beverages" and re.search(r"(?:^|_)(green_tea)(?:$|_)", norm_var, re.I):
        return "Carton"
    if cat_canon == "HHC":
        if re.search(r"(?:^|_)(dish_?wash_gel|gel_dish_?wash|liquid_gel|dish_?wash_liquid|liquid)(?:$|_)", norm_var, re.I):
            return "Bottle"
        if re.search(r"(?:^|_)(scrub_easy|scrub_magic|scrub_pad|scrubber|scrub)(?:$|_)", norm_var, re.I):
            return "Wrapper"
    if cat_canon == "Oral_Care" and re.search(r"(?:^|_)(rock_salt|clove_and_salt|salt)(?:$|_)", norm_var, re.I):
        if not re.search(r"(?:^|_)(mouthwash|toothpowder|tooth_powder)(?:$|_)", norm_var, re.I):
            return "Tube"
    if cat_canon == "Skin":
        if re.search(r"(?:^|_)(powder_finish_creme|powder_finish_cream)(?:$|_)", norm_var, re.I):
            return "Tube"
        if re.search(r"(?:^|_)(baby_powder|face_powder|faceit_powder|compact_powder|talcum_powder|dusting_powder|powder)(?:$|_)", norm_var, re.I):
            return "Bottle"
    if cat_canon == "Foods":
        if re.search(r"(?:^|_)(spread)(?:$|_)", norm_var, re.I):
            return "Jar"
        if re.search(r"(?:^|_)(syrup)(?:$|_)", norm_var, re.I):
            return "Bottle"
        if re.search(r"(?:^|_)(chocolate)(?:$|_)", norm_var, re.I):
            return "Wrapper"

    # 5. Product-form rules matched strictly on variant tokens (never brand name!)
    for pf_re, pf_pack in PRODUCT_FORM_PACK_RULES:
        if pf_re.search(variant) or pf_re.search(norm_var):
            return pf_pack

    # 6. Brand-level default packaging (with Dove category disambiguation)
    b_low = brand.lower()
    if b_low == "dove":
        if cat_canon == "PW":
            return "Bar"
        if cat_canon == "Hair_Care":
            return "Bottle"
        if cat_canon == "DMT":
            return "Roll_On"
        return "Bottle"
    if b_low in BRAND_DEFAULT_PACKAGING:
        return BRAND_DEFAULT_PACKAGING[b_low]

    # 7. Category-level deterministic default (guarantees 0 'Unknown')
    return CATEGORY_DEFAULT_PACKAGING.get(cat_canon, "Pouch")


def parse_pack_folder(folder: str) -> dict:
    """Parse a ``pack_data/pack/train/<Category>__<PkgType>`` folder into ``(category, packaging_type)``."""
    raw = folder.strip()
    if "__" in raw:
        raw_cat, raw_pkg = raw.split("__", 1)
        category = normalize_category_name(PACK_CATEGORY_ALIAS.get(raw_cat, raw_cat))
        pkg_type = normalize_packaging_type(raw_pkg, category=category)
    else:
        category = "Foods"
        pkg_type = normalize_packaging_type(raw, category=category)
    return {"raw_folder": raw, "category": category, "packaging_type": pkg_type}


# Exact 46 valid (Category, PackType) combinations in gs://marketshare-datasets/pack_data/pack/train/
VALID_CATEGORY_PACKAGING_TYPES: dict[str, tuple[str, ...]] = {
    "Beverages": ("Carton", "Jar", "Pouch", "Tetrapack", "Tin"),
    "DMT": ("Bottle", "Can", "Pump"),
    "Foods": ("Bottle", "Carton", "Jar", "Pouch", "Tub"),
    "HFD": ("Bottle", "Carton", "Jar", "Pouch", "Sachet", "Tetrapack", "Wrapper"),
    "HHC": ("Bar", "Bottle", "Box", "Pouch"),
    "Hair_Care": ("Bottle", "Jar", "Sachet", "Tube"),
    "Laundry": ("Bar", "Bottle", "Can", "Carton", "Pouch", "Sachet"),
    "Oral_Care": ("Bottle", "Can", "Sachet", "Tube"),
    "PW": ("Bottle", "Can", "Pouch", "Wrapper"),
    "Skin": ("Bottle", "Jar", "Sachet", "Tube"),
}

CATEGORY_PACK_PROJECTION_MAP: dict[tuple[str, str], str] = {
    ("PW", "Bar"): "Wrapper",
    ("PW", "Tube"): "Bottle",
    ("PW", "Jar"): "Bottle",
    ("PW", "Box"): "Wrapper",
    ("PW", "Carton"): "Wrapper",
    ("PW", "Sachet"): "Pouch",
    ("PW", "Spray"): "Bottle",
    ("DMT", "Roll_On"): "Bottle",
    ("DMT", "Sachet"): "Bottle",
    ("DMT", "Carton"): "Bottle",
    ("DMT", "Spray"): "Can",
    ("Oral_Care", "Carton"): "Sachet",
    ("Oral_Care", "Box"): "Sachet",
    ("Oral_Care", "Wrapper"): "Sachet",
    ("Oral_Care", "Pouch"): "Sachet",
    ("Oral_Care", "Jar"): "Bottle",
    ("Oral_Care", "Spray"): "Bottle",
    ("Beverages", "Sachet"): "Pouch",
    ("Beverages", "Bottle"): "Jar",
    ("Beverages", "Wrapper"): "Pouch",
    ("Beverages", "Tub"): "Jar",
    ("Foods", "Wrapper"): "Pouch",
    ("Foods", "Tube"): "Bottle",
    ("Foods", "Sachet"): "Pouch",
    ("Foods", "Can"): "Bottle",
    ("Foods", "Tin"): "Jar",
    ("Skin", "Pouch"): "Sachet",
    ("Skin", "Pump"): "Bottle",
    ("Skin", "Spray"): "Bottle",
    ("Skin", "Tub"): "Jar",
    ("Skin", "Tin"): "Jar",
    ("Skin", "Carton"): "Bottle",
    ("Skin", "Box"): "Bottle",
    ("Skin", "Bar"): "Jar",
    ("Skin", "Wrapper"): "Sachet",
    ("Skin", "Roll_On"): "Bottle",
    ("HHC", "Tube"): "Bottle",
    ("HHC", "Tub"): "Box",
    ("HHC", "Can"): "Bottle",
    ("HHC", "Spray"): "Bottle",
    ("HHC", "Wrapper"): "Pouch",
    ("HHC", "Carton"): "Box",
    ("HHC", "Jar"): "Bottle",
    ("HHC", "Sachet"): "Pouch",
    ("HFD", "Tub"): "Carton",
    ("HFD", "Tin"): "Jar",
    ("HFD", "Tube"): "Bottle",
    ("Hair_Care", "Pouch"): "Sachet",
    ("Hair_Care", "Carton"): "Bottle",
    ("Hair_Care", "Box"): "Bottle",
    ("Hair_Care", "Can"): "Bottle",
    ("Hair_Care", "Pump"): "Bottle",
    ("Hair_Care", "Spray"): "Bottle",
    ("Hair_Care", "Tub"): "Jar",
    ("Laundry", "Tub"): "Carton",
    ("Laundry", "Box"): "Carton",
    ("Laundry", "Wrapper"): "Bar",
    ("Laundry", "Jar"): "Bottle",
    ("Laundry", "Spray"): "Bottle",
}


def project_to_valid_category_pack(
    category: str,
    packaging_type: str,
    crc_votes: dict[str, int] | None = None,
) -> str:
    """Project ``(category, packaging_type)`` onto the exact 46 ``Category__PackType`` domain from ``pack_data``."""
    cat_canon = normalize_category_name(category)
    valid_set = set(VALID_CATEGORY_PACKAGING_TYPES.get(cat_canon, ()))
    if crc_votes:
        valid_votes = {p: c for p, c in crc_votes.items() if p in valid_set and c > 0}
        if valid_votes:
            return sorted(valid_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if not valid_set or packaging_type in valid_set:
        return packaging_type
    return CATEGORY_PACK_PROJECTION_MAP.get((cat_canon, packaging_type), VALID_CATEGORY_PACKAGING_TYPES[cat_canon][0])

