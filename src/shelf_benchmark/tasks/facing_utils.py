"""Utilities for Front-Facing-Only Depth De-duplication, Rule-Derived Size Bucketing, and Physical Bounding-Box Cropping."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

from shelf_benchmark.config import TaxonomyConfig
from shelf_benchmark.cropping import crop_detected_facings, load_pil_image
from shelf_benchmark.geometry import (
    BBox,
    deduplicate_depth_stacked_facings,
    horizontal_overlap_ratio,
)
from shelf_benchmark.size_rules import derive_size_bucket_from_bbox
from shelf_benchmark.text_normalization import normalize_text

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _default_taxonomy() -> TaxonomyConfig:
    """Packaged taxonomy used when a caller does not pass one explicitly.

    Resolved lazily rather than at import time: doing it at import time meant merely importing
    `shelf_benchmark.tasks` read a YAML off disk relative to the current working directory, and
    froze the result for the process -- so `SDK(taxonomy_path=...)` could not override it.
    """
    return TaxonomyConfig()



def _norm_brand_key(value: str) -> str:
    """Normalize a brand for lookup.

    Delegates to the single shared normalizer. This used to be a second implementation that did
    not fold accents and stripped a trailing possessive as a word, so `Pond's` normalized to
    `pond` here but to `ponds` in the scorer -- meaning the catalog resolver and the grader
    disagreed about what the same string was.
    """
    return normalize_text(value)


def _tokens(value: str) -> List[str]:
    return [t for t in normalize_text(value).split() if t]


def _is_whole_token_subsequence(needle: str, haystack: str) -> bool:
    """True when every token of `needle` appears, in order and whole, inside `haystack`.

    Whole-token rather than raw substring: plain `in` matching made `"Lux"` resolve to
    `"Deluxe"` and `"Axe"` to `"Waxed"`, silently rewriting a prediction into a different brand
    before it was scored.
    """
    n, h = _tokens(needle), _tokens(haystack)
    if not n or len(n) > len(h):
        return False
    return any(h[i : i + len(n)] == n for i in range(len(h) - len(n) + 1))


def resolve_brand_against_catalog(
    generated_brand: str,
    taxonomy: Optional[TaxonomyConfig] = None,
) -> str:
    """Resolve an LLM-generated open-vocabulary brand string against an optional master brand catalog (scales to 2,000+ brands in O(1)).

    Resolution is deliberately conservative, because it rewrites a prediction *before* it is
    scored. An exact normalized match always wins. Otherwise the generated string must contain a
    catalog brand as a whole-token run (so `"Dove Men+Care"` resolves to `"Dove"`), and only if
    exactly one catalog brand qualifies -- an ambiguous match is left alone rather than resolved
    to whichever entry happened to come first in the list.
    """
    raw = (generated_brand or "").strip()
    if not raw or taxonomy is None:
        return raw
    all_catalog_brands = list(taxonomy.hul_brands or []) + list(taxonomy.non_hul_brands or [])
    if not all_catalog_brands:
        return raw
    norm_raw = _norm_brand_key(raw)
    if not norm_raw:
        return raw
    exact_map = {_norm_brand_key(b): b for b in all_catalog_brands if _norm_brand_key(b)}
    if norm_raw in exact_map:
        return exact_map[norm_raw]

    candidates = {
        canonical
        for norm_cat, canonical in exact_map.items()
        if _is_whole_token_subsequence(norm_cat, norm_raw)
    }
    if len(candidates) == 1:
        return candidates.pop()
    if len(candidates) > 1:
        logger.debug(
            "Brand '%s' matches %d catalog entries (%s); leaving it unresolved rather than "
            "picking one arbitrarily.",
            raw,
            len(candidates),
            sorted(candidates),
        )
    return raw


def check_is_hul_brand(
    brand_name: str,
    taxonomy: Optional[TaxonomyConfig] = None,
    model_predicted: Optional[bool] = None,
) -> bool:
    """Determine whether a predicted brand belongs to HUL without requiring predefined brand lists in YAML.

    - If the user has explicitly configured `hul_brands` (or `non_hul_brands`) in `TaxonomyConfig`, those lists take precedence.
    - Otherwise (when `hul_brands` is empty `[]` by default), relies on `model_predicted` (`is_hul_brand` returned by the VLM or Catalog join).

    Matching is whole-token, for the same reason as `resolve_brand_against_catalog`: raw
    substring matching made any brand whose name merely contained a catalog brand's letters
    inherit that brand's HUL attribution.
    """
    norm = _norm_brand_key(brand_name)
    if not norm:
        return False

    if taxonomy is not None and taxonomy.non_hul_brands:
        non_hul = [_norm_brand_key(b) for b in taxonomy.non_hul_brands if _norm_brand_key(b)]
        if any(
            nh == norm or _is_whole_token_subsequence(nh, norm) or _is_whole_token_subsequence(norm, nh)
            for nh in non_hul
        ):
            return False

    if taxonomy is not None and taxonomy.hul_brands:
        hul = [_norm_brand_key(b) for b in taxonomy.hul_brands if _norm_brand_key(b)]
        return any(
            h == norm or _is_whole_token_subsequence(h, norm) or _is_whole_token_subsequence(norm, h)
            for h in hul
        )

    if model_predicted is not None:
        return bool(model_predicted)

    return False



__all__ = [
    "BBox",
    "check_is_hul_brand",
    "crop_detected_facings",
    "deduplicate_depth_stacked_facings",
    "derive_size_bucket_from_bbox",
    "horizontal_overlap_ratio",
    "load_pil_image",
    "resolve_brand_against_catalog",
]
