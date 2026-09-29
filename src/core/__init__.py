"""Core EPIC pipeline pillars (Detection, Vector Matching, and Cost-Aware VLM Fallback)."""

from core.detection import detect_shelf_skus
from core.fallback import resolve_ambiguous_skus
from core.matching import match_sku_vectors

__all__ = [
    "detect_shelf_skus",
    "match_sku_vectors",
    "resolve_ambiguous_skus",
]
