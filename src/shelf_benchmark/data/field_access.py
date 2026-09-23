"""Shared field extraction for rows arriving in a schema this suite does not own.

Both the association layer and the ground-truth layer ingest rows from external systems
(BigQuery, CSV, JSON/JSONL, AutoML manifests), and both need the same lookup rule: try the
mapped field name, then a list of well-known aliases, supporting dot-paths into nested
objects. That rule previously existed as two byte-identical private copies
(`associations._extract_nested` and `ground_truth.BaseGroundTruthProvider._extract_field`),
so a fix applied to one did not apply to the other. It lives here once.
"""

from __future__ import annotations

from typing import Any, Dict


def extract_field(row: Dict[str, Any], field_spec: str, *fallbacks: str) -> Any:
    """Return the first non-empty value found for `field_spec` or any fallback alias.

    A candidate may be a plain key (``store_id``) or a dot-path into nested dictionaries
    (``metadata.store_id``). ``None`` and ``""`` are both treated as "absent" so that a
    blank CSV cell falls through to the next alias rather than winning.

    Returns ``None`` when no candidate yields a value. Callers decide whether an absent
    field is an error - this helper deliberately does not invent a default.
    """
    for candidate in (field_spec, *fallbacks):
        if not candidate:
            continue
        if candidate in row and row[candidate] is not None and row[candidate] != "":
            return row[candidate]
        if "." in candidate:
            cur: Any = row
            for part in candidate.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            if cur is not None and cur != "":
                return cur
    return None
