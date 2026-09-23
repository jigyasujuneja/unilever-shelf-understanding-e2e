"""Single source of truth for brand/product string normalization.

There used to be two different normalizers. `evaluation/metrics.normalize_text` folded accents
(NFKD) and deleted apostrophes, so `Pond's` became `ponds`. `tasks/facing_utils._norm_brand_key`
did not fold accents and stripped a trailing possessive as a word, so the same input became
`pond`. The first is what the scorer compares with; the second is what the catalog resolver used
to rewrite a prediction *before* the scorer ever saw it. A brand could therefore be resolved
against the catalog under one set of rules and then graded under another, and `Pond's` vs `Ponds`
would score as a miss for reasons that had nothing to do with the model.

Anything that compares two brand or product strings must use `normalize_text` from this module.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional


def normalize_text(text: Optional[str]) -> str:
    """Normalize a brand/product string for comparison.

    Strips accents (NFKD), apostrophes, punctuation and case, and collapses whitespace, so
    punctuation and diacritic variants match across any brand portfolio without hardcoded alias
    lists.

    >>> normalize_text("Pond's")
    'ponds'
    >>> normalize_text("L'Oréal")
    'loreal'
    >>> normalize_text("  Dove   Men+Care ")
    'dove men care'
    >>> normalize_text(None)
    ''
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_apostrophes = re.sub(r"['\u2019`]", "", ascii_str.lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", no_apostrophes)
    return " ".join(cleaned.split())


def canonical_brand(brand: Optional[str], aliases: Optional[Dict[str, str]] = None) -> str:
    """Normalize a brand and fold it onto its canonical form via the optional configured alias table.

    >>> canonical_brand("Pond's", {"ponds": "Ponds"})
    'ponds'
    >>> canonical_brand("")
    ''
    """
    norm = normalize_text(brand)
    if not norm:
        return ""
    table = aliases or {}
    if norm in table:
        return normalize_text(table[norm])
    return norm
