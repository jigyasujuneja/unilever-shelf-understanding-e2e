"""Unilever 7-Dimension Product Taxonomy, Brand Canonicalization, and Rule-Derived Size Buckets.

Ported and unified from Riley's `shelf_benchmark/_resources/taxonomy.yaml`, `size_rules.py`,
and `text_normalization.py` for SPEC-005.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "configs" / "unilever_taxonomy.json"

HUL_BRANDS_CANONICAL = {
    "dove",
    "tresemme",
    "sunsilk",
    "pond's",
    "vaseline",
    "lux",
    "lifebuoy",
    "lakme",
    "clinic plus",
    "simple",
    "love beauty and planet",
    "indulekha",
    "pears",
    "closeup",
    "pepsodent",
    "radox",
    "surf excel",
    "surf",
    "breeze",
    "vim",
    "comfort",
    "domex",
    "knorr",
    "rexona",
    "lady's choice",
    "clear",
    "axe",
    "kissan",
    "bru",
    "horlicks",
    "lipton",
    "glow & lovely",
    "fair & lovely",
    "wheel",
    "rin",
    "brooke bond",
    "red label",
    "taj mahal",
    "taaza",
    "boost",
    "hamam",
    "elle 18",
}

BRAND_ALIASES = {
    "tresemmé": "Tresemme",
    "tresemme": "Tresemme",
    "ponds": "Pond's",
    "pond": "Pond's",
    "pond's": "Pond's",
    "glow and lovely": "Glow & Lovely",
    "glow & lovely": "Glow & Lovely",
    "fair & lovely": "Glow & Lovely",
    "fair and lovely": "Glow & Lovely",
    "lipton": "Lipton",
    "lipton green tea": "Lipton",
    "loreal": "L'Oreal",
    "l'oréal": "L'Oreal",
    "l'oreal": "L'Oreal",
    "l'oreal paris": "L'Oreal",
    "h&s": "Head & Shoulders",
    "head and shoulders": "Head & Shoulders",
    "head & shoulders": "Head & Shoulders",
    "lbp": "Love Beauty and Planet",
}


@dataclass(frozen=True)
class SevenDimensionProductAttributes:
    """Official Unilever 7-Dimension shelf recognition hierarchy."""

    category: str
    subcategory: str
    brand: str
    is_hul_brand: bool
    variant: str
    packaging_type: str
    pack_type: str
    rule_derived_size_bucket: str


def normalize_brand_and_hul_flag(raw_brand: Optional[str]) -> Tuple[str, bool]:
    """Canonicalize brand names and deterministically resolve HUL ownership (`is_hul_brand`)."""
    if not raw_brand or not raw_brand.strip():
        return ("Unknown", False)
    cleaned = re.sub(r"\s+", " ", raw_brand.strip())
    lower = cleaned.lower().replace("’", "'")
    canonical = BRAND_ALIASES.get(lower, cleaned)
    is_hul = canonical.lower() in HUL_BRANDS_CANONICAL
    return (canonical, is_hul)


def resolve_rule_derived_size_bucket(
    product_name_or_size_text: str = "",
    bbox_2d: Optional[List[int]] = None,
) -> str:
    """Deterministic size bucket resolver (Small <=55g/ml, Medium 56-110g/ml, Large >110g/ml).

    Uses explicit ml/g regex extraction first, falling back to normalized bounding-box height priors.
    """
    text = (product_name_or_size_text or "").lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ml|g|gm|grams|l|kg)\b", text)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit in ("l", "kg"):
            val *= 1000.0
        if val <= 55.0:
            return "Small / Trial / Sachet (<=55g/ml)"
        if val <= 110.0:
            return "Medium / Regular (56-110g/ml)"
        return "Large / Family (>110g/ml)"

    if bbox_2d and len(bbox_2d) == 4:
        height_norm = max(1, bbox_2d[2] - bbox_2d[0])
        if height_norm <= 48:
            return "Small / Trial / Sachet (<=55g/ml)"
        if height_norm <= 85:
            return "Medium / Regular (56-110g/ml)"
    return "Large / Family (>110g/ml)"


def enrich_with_7dim_taxonomy(
    catalog_entry: Dict[str, Any],
    bbox_2d: Optional[List[int]] = None,
) -> SevenDimensionProductAttributes:
    """Convert a catalog item + bounding box into a complete 7-Dimension Unilever attribute record."""
    brand_raw = str(catalog_entry.get("brand", "Unknown"))
    canonical_brand, is_hul = normalize_brand_and_hul_flag(brand_raw)
    prod_name = str(catalog_entry.get("product_name", ""))
    size_bucket = str(
        catalog_entry.get("size_bucket")
        or resolve_rule_derived_size_bucket(prod_name, bbox_2d)
    )
    return SevenDimensionProductAttributes(
        category=str(catalog_entry.get("category", "Personal Care")),
        subcategory=str(catalog_entry.get("subcategory", "General")),
        brand=canonical_brand,
        is_hul_brand=bool(catalog_entry.get("is_hul_brand", is_hul)),
        variant=str(catalog_entry.get("variant", "Standard")),
        packaging_type=str(catalog_entry.get("packaging_type", "bottle")),
        pack_type=str(catalog_entry.get("pack_type", "Single")),
        rule_derived_size_bucket=size_bucket,
    )
