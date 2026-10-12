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
    r"(?:^|_)(Bottle|Sachet|Sachets|Pouch|Polly|Poly|Jar|Tub|Tube|Bar|Bars|Soapbars|Box|Can|Carton|Stick|Roll_On|Pump|Tin|Wrapper|Tetrapack|Tetra|Tablets)(?:_(?:New|Old|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+|Rs_?\d+|\d+))*$",
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

# Ordered from most specific physical form (`toothbrush` -> Carton, `talc` -> Bottle) to general forms
PRODUCT_FORM_PACK_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:^|_)(toothbrush|tooth_brush|toothbrushes|tooth_brushes|manual_toothbrush|electric_toothbrush|blade|razor|shaver|mosquito_coil|good_knight)(?:$|_)",
            re.I,
        ),
        "Carton",
    ),
    (
        re.compile(
            r"(?:^|_)(dreamflower|aloe_cooling|star_light|radiance_sandal|talcum|talc|prickly_heat|dermi_cool|nycil)(?:$|_)",
            re.I,
        ),
        "Bottle",
    ),
    (re.compile(r"(?:^|_)(sachet|sachets|sheet_mask|shaving_foam)(?:$|_)", re.I), "Sachet"),
    (re.compile(r"(?:^|_)(tetra|tetrapack|milkshake)(?:$|_)", re.I), "Tetrapack"),
    (
        re.compile(
            r"(?:^|_)(tea_bags|green_tea|clear_and_light|honey_lemon|darjeeling|earl_grey|chamomile|ashwagandha|pukka|twinnings|twinings|tetley|custard_powder|corn_flour|ragi_bites|corn_flakes|muesli|oats|chocos)(?:$|_)",
            re.I,
        ),
        "Carton",
    ),
    (re.compile(r"(?:^|_)(biscuits|noodles|2_mins|yippee|wai_wai|maggi|choco_pie|monaco|oreo|chocolate|kitkat|snickers|twix)(?:$|_)", re.I), "Wrapper"),
    (re.compile(r"(?:^|_)(soap|soaps|soap_bars?|bathing_bar|detergen_bar|wash_bar|vim_bar|exo_bar|rin_bar)(?:$|_)", re.I), "Bar"),
    (
        re.compile(
            r"(?:^|_)(toothpaste|tooth_paste|tooth_gel|sensitivity_relief|purifying_turmeric|anti_cavity|dant_kanti|meswak|babool|sensodyne|closeup|pepsodent|face_wash|facewash|facwash|shaving_cream|shavingcream|face_pack|facepack|scrub|cream|creame|ointment|gel|conditionar|bb_natural|fairness|bright_beauty|glow_and_lovely|fair_and_lovely)(?:$|_)",
            re.I,
        ),
        "Tube",
    ),
    (
        re.compile(
            r"(?:^|_)(hair_mask|moisture_mask|strength_mask|clay_mask|sleeping_mask|repair_mask|_?mask|petroleum_jelly|petroleumjelly|peanut_butter|mayonnaise|jam|pickle|chutney|ghee|honey|bournvita|horlicks|complan|pediasure|ensure|protinex|boost|maltova|viva|nutella|nescafe_classic|bru_instant_jar|instant_jar)(?:$|_)",
            re.I,
        ),
        "Jar",
    ),
    (
        re.compile(
            r"(?:^|_)(roll_on|deodorant|deodrant|deo_spray|body_spray|axe|brut|rexona|fogg|denver|engage|wildstone|hair_oil|almond_drops|cool_oil|oil|bringha|bhringaraj|shikakai|serum|toner|astringent|mouthwash|listerine|shampoo|conditioner|bodywash|body_wash|handwash|hand_wash|hand_sanitizer|hand_sanitiser|hand_sanetizer|bodylotion|body_lotion|lotions|lotion|disinfectant|surface_cleaner|toilet_cleaner|bathroom_cleanser|dishwash_liquid|matic_liquid|matic_front_load_liquid|matic_top_load_liquid|liquid|tablets|capsules|gummies|multivitamin|biotin|hk_vitals|centrum|oziva|wellbeing_nutrition|plix|fast_and_up|bio_health|ketchup|sauce|vinegar|vwash|v_wash)(?:$|_)",
            re.I,
        ),
        "Bottle",
    ),
    (
        re.compile(
            r"(?:^|_)(atta|salt|rock_salt|salt_crystal|sampann|polly|poly|doy|powder|detergent|matic|surf_excel|ariel|tide|rin|wheel|ghadi|nirma|henko|soup|masala|spices|chai|leaf|dust|natural_care|red_label|taj_mahal|3_roses|taaza|wagh_bakri|society|avt|cothas|levista|narasus|mili|girnar|coffee_pouch|granules|refill)(?:$|_)",
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
    "3_roses": "Carton",
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

    # 3. Product-form rules matched against BOTH variant AND full Brand__Variant
    for pf_re, pf_pack in PRODUCT_FORM_PACK_RULES:
        if pf_re.search(variant) or pf_re.search(norm_var) or pf_re.search(full_bv) or pf_re.search(norm_full):
            return pf_pack

    # 4. High-consensus CRC32 vote from pack.zip (when >= 80% agreement)
    if crc_votes:
        total_v = sum(crc_votes.values())
        top_p, top_c = max(crc_votes.items(), key=lambda kv: kv[1])
        if total_c := total_v:
            if top_c / total_c >= 0.80 and top_p != "Unknown":
                return PKG_SYNONYM_MAP.get(top_p.lower(), top_p)

    # 5. Brand-level default packaging (with Dove category disambiguation)
    b_low = brand.lower()
    cat_canon = normalize_category_name(category)
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

    # 6. Category-level deterministic default (guarantees 0 'Unknown')
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
