"""Canonical Unilever (HUL) & Competitor Brand Registry, Alias Normalization & M&A Ownership Ledger."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

NOISE_BRAND_FOLDERS: set[str] = set()

BRAND_ALIAS_MAP: dict[str, str] = {
    "a_3_roses": "3_Roses",
    "3_roses": "3_Roses",
    "three_roses": "3_Roses",
    "abrown&polson": "Brown_And_Polson",
    "brown_and_polson": "Brown_And_Polson",
    "acif": "CIF",
    "cif": "CIF",
    "aclinic": "Clinic_Plus",
    "clinic_plus": "Clinic_Plus",
    "ahellmanns": "Hellmans",
    "hellmans": "Hellmans",
    "hellmanns": "Hellmans",
    "abrook_bond": "Brooke_Bond",
    "brooke_bond": "Brooke_Bond",
    "atetley": "Tetley",
    "tetely": "Tetley",
    "tetley": "Tetley",
    "twinnings": "Twinings",
    "twinings": "Twinings",
    "ayippee": "Yippee",
    "yipee": "Yippee",
    "yippee": "Yippee",
    "oral_b": "Oralb",
    "oralb": "Oralb",
    "vwash": "V_Wash",
    "v_wash": "V_Wash",
    "ppvwash": "V_Wash",
    "lactoclamine": "Lacto_Calamine",
    "lacto_calamine": "Lacto_Calamine",
    "mamearth": "Mamaearth",
    "mamaearth": "Mamaearth",
    "enchateur": "Enchanteur",
    "enchanteur": "Enchanteur",
    "listerene": "Listerine",
    "listerine": "Listerine",
    "softouch": "Soft_Touch",
    "soft_touch": "Soft_Touch",
}

A_PREFIX_TWINS: dict[str, str] = {
    "adr_wash": "Dr_Wash",
    "agenteel": "Genteel",
    "amr_muscle": "Mr_Muscle",
    "amysore_sandal": "Mysore_Sandal",
    "arevive": "Revive",
    "asafe_wash": "Safe_Wash",
    "avanish": "Vanish",
}

HUL_BRANDS: set[str] = {
    "3_roses", "annapurna", "axe", "ayush", "boost", "breeze", "brooke_bond",
    "brown_and_polson", "bru", "brut", "cif", "clear", "clinic_plus", "closeup",
    "comfort", "cornetto", "domex", "dove", "elle18", "fair_and_lovely",
    "glow_and_lovely", "hamam", "hellmans", "horlicks", "indulekha",
    "kissan", "knorr", "kwality", "lakme", "lbp", "lifebuoy", "lipton", "liril",
    "lux", "maltova", "moti", "nature_protect", "novology", "oziva", "pears",
    "pepsodent", "ponds", "pukka", "red_label", "rex", "rexona", "rin", "ruby",
    "simple", "st_ives", "sun", "sunlight", "sunsilk", "surf_excel", "taaza",
    "taj_mahal", "tresemme", "v_wash", "vaseline", "vim", "viva",
    "wellbeing_nutrition", "wheel", "love_beauty_and_planet",
}

HUL_BRAND_ONLY_NO_VARIANT_TREE: set[str] = {"axe", "brut", "brooke_bond"}

TATA_HUL_VARIANTS: tuple[str, ...] = (
    "taj mahal", "bru", "red label", "taaza", "3 roses", "lipton", "kissan", "knorr", "annapurna",
)


def canonicalize_brand_name(raw_brand: str) -> str:
    """Normalize a raw brand token or folder name to its canonical spelling."""
    raw = raw_brand.strip()
    if not raw:
        return "Unknown"
    raw_low = raw.lower()
    if raw_low in BRAND_ALIAS_MAP:
        return BRAND_ALIAS_MAP[raw_low]
    if raw_low in A_PREFIX_TWINS:
        return A_PREFIX_TWINS[raw_low]
    if raw.startswith("Z_"):
        return raw[2:]
    if len(raw) >= 3 and raw[0] == "A" and raw[1].isupper():
        return raw[1:]
    return raw


@lru_cache(maxsize=4)
def load_live_brand_registry(path: str = "data/hul_golden/brand_sku_live_registry.json") -> dict:
    """Load the versioned Brand & SKU Live Registry JSON snapshot (or build in-memory if absent)."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    from utils.brand_registry_crawler import build_live_brand_sku_registry

    return build_live_brand_sku_registry(crawl_online=False)


def verify_brand_ownership(
    brand: str,
    variant: str = "",
    as_of_date: str | None = None,
    registry_path: str = "data/hul_golden/brand_sku_live_registry.json",
    infer_category_fn=None,
) -> dict:
    """Verify HUL vs Competitor brand ownership, parent conglomerate, and temporal M&A validity."""
    from utils.brand_registry_crawler import BRAND_TEMPORAL_OWNERSHIP, resolve_parent_company

    reg = load_live_brand_registry(registry_path)
    brands_map = reg.get("brands") or {}

    raw_b = re.sub(r"[_\-]+", " ", brand).strip()
    canon_brand = canonicalize_brand_name(raw_b.replace(" ", "_"))
    b_key = canon_brand.lower()
    entry = brands_map.get(b_key) or brands_map.get(raw_b.lower()) or {}

    is_hul = bool(entry.get("is_hul", b_key in HUL_BRANDS))
    default_cat = infer_category_fn(b_key) if infer_category_fn is not None else "Foods"
    category = str(entry.get("category") or default_cat or "Foods")
    valid_from = str(entry.get("valid_from") or "1956-01-01")
    valid_to = entry.get("valid_to")
    source = str(entry.get("verification_source") or "hul_taxonomy")
    dual_umbrella = bool(entry.get("dual_umbrella") or b_key == "tata")

    v_low = re.sub(r"[_\-]+", " ", variant).strip().lower()
    if b_key == "tata" or raw_b.lower() == "tata":
        canon_brand = "Tata"
        dual_umbrella = True
        is_hul = any(tok in v_low for tok in TATA_HUL_VARIANTS)
        source = "dual_umbrella_variant_resolver"

    parent_company = resolve_parent_company(canon_brand, is_hul=is_hul, variant=variant)

    t_info = BRAND_TEMPORAL_OWNERSHIP.get(b_key) or BRAND_TEMPORAL_OWNERSHIP.get(raw_b.lower())
    if t_info:
        valid_from = str(t_info.get("hul_valid_from") or valid_from)
        valid_to = t_info.get("hul_valid_to")
        source = "temporal_ma_ownership_ledger"
        if as_of_date:
            in_window = as_of_date >= valid_from and (valid_to is None or as_of_date <= str(valid_to))
            is_hul = in_window
            if is_hul:
                parent_company = "Hindustan Unilever Limited (HUL)"
            elif valid_to and as_of_date > str(valid_to):
                parent_company = str(t_info.get("divested_to") or "Divested Competitor FMCG")
            else:
                parent_company = str(t_info.get("acquired_from") or "Pre-Acquisition Competitor FMCG")
        elif valid_to is not None:
            is_hul = False
            parent_company = str(t_info.get("divested_to") or "Divested Competitor FMCG")

    return {
        "brand": canon_brand,
        "is_hul": is_hul,
        "parent_company": parent_company,
        "category": category,
        "dual_umbrella": dual_umbrella,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "verification_source": source,
    }
