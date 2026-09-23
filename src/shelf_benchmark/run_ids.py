"""Construction of `run_id` values.

A `run_id` is the join key between predictions, the OpenTelemetry spans, the
GCP billing labels, and the generated reports. Two different runs that share a
`run_id` silently merge everywhere downstream, so uniqueness is a correctness
property, not a cosmetic one.

This module exists because the codebase previously built run ids inline with
`approach[:9]`, which is not injective over the shipped approach ids::

    "two_stage_bbox_guided_nms"[:9]          == "two_stage"
    "two_stage_physical_crop_per_facing"[:9] == "two_stage"

Those are two different algorithms. Benchmarking them in the same batch produced
byte-identical run ids and merged their rows.

The rule here: never truncate a semantic identifier without appending a digest of
the full value. `_shorten` does exactly that, so shortening stays injective.
"""

from __future__ import annotations

import hashlib
import re

# GCP resource labels accept lowercase letters, digits, `-` and `_`, up to 63
# characters. run_id is emitted as a label by evaluation/gcp_billing.py, so the
# whole id has to fit inside that budget.
MAX_RUN_ID_LENGTH = 63

_DISALLOWED = re.compile(r"[^a-z0-9_-]+")
_DIGEST_LENGTH = 6


def slugify(value: str) -> str:
    """Lowercases `value` and replaces every label-unsafe run of characters with `-`."""
    return _DISALLOWED.sub("-", value.strip().lower()).strip("-")


def _shorten(slug: str, limit: int) -> str:
    """Shortens `slug` to at most `limit` characters, injectively.

    Truncation alone is not injective, so anything that loses characters gets a
    short digest of the *full* slug appended. Distinct inputs therefore keep
    distinct outputs even when both are shortened.
    """
    if limit <= 0:
        return ""
    if len(slug) <= limit:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    keep = max(0, limit - _DIGEST_LENGTH - 1)
    if keep == 0:
        return digest[:limit]
    return f"{slug[:keep]}-{digest}"


def build_run_id(
    *parts: str,
    max_length: int = MAX_RUN_ID_LENGTH,
) -> str:
    """Builds a label-safe run id from `parts`, preserving distinctness.

    Every part is slugified and joined with `-`. If the result exceeds
    `max_length`, the longest parts are shortened first -- each via `_shorten`, so
    two different inputs cannot collapse onto the same id.

    >>> build_run_id("batch1", "cls", "two_stage_bbox_guided_nms", "flash")
    'batch1-cls-two_stage_bbox_guided_nms-flash'
    >>> a = build_run_id("cls", "two_stage_bbox_guided_nms", "gemini-3-flash-preview-x" * 3)
    >>> b = build_run_id("cls", "two_stage_physical_crop_per_facing", "gemini-3-flash-preview-x" * 3)
    >>> a != b and len(a) <= 63 and len(b) <= 63
    True
    """
    slugs = [slugify(p) for p in parts if p and slugify(p)]
    if not slugs:
        return ""

    candidate = "-".join(slugs)
    if len(candidate) <= max_length:
        return candidate

    # Budget: distribute the available characters, shortening the longest parts
    # first so short discriminating parts (like the approach suffix) survive.
    separators = len(slugs) - 1
    budget = max_length - separators
    order = sorted(range(len(slugs)), key=lambda i: len(slugs[i]), reverse=True)
    shortened = list(slugs)
    for rank, idx in enumerate(order):
        remaining_parts = len(order) - rank
        per_part = max(1, budget // remaining_parts)
        shortened[idx] = _shorten(slugs[idx], per_part)
        budget -= len(shortened[idx])
    return "-".join(s for s in shortened if s)[:max_length].strip("-")
