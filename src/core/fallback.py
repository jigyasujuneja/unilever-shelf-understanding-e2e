"""Pillar 3 of EPIC (src/core/fallback.py): Cost-Aware Gemini 3.8 Flash & Sister-Shade Fallback Gateway.

Escalates only ambiguous (`AMBIGUOUS`) or unrecognized (`UNRECOGNIZED`) shelf crops from the
vector retrieval tier (`src/core/matching.py`) to Gemini 3.8 Flash / `/v1/systemone` structured
classification, preserving the <= INR 0.22/image FinOps ceiling.
"""

from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from approaches.base import Box, Context
from core.catalog import _build_real_catalog_prototype_bank

FALLBACK_PROMPT = (
    "You are the HUL Perfect Store Stage 5 Sister-Shade & Open-Set Fallback Verifier. "
    "Given the retail shelf image and candidate SKU list, return a JSON object with key 'items' "
    "containing [{'category', 'brand', 'packaging_type', 'variant', 'is_hul'}] for the ambiguous crops."
)


def resolve_ambiguous_skus(
    image: Image.Image,
    boxes: list[Box],
    initial_labels: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    ctx: Context,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Escalate ambiguous or low-confidence crops to Gemini 3.8 Flash and merge results."""
    ambiguous_indices = [
        i for i, d in enumerate(diagnostics) if d.get("status") in ("AMBIGUOUS", "UNRECOGNIZED")
    ]
    if not ambiguous_indices:
        return initial_labels, {
            "escalated_count": 0,
            "total_boxes": len(boxes),
            "escalation_rate": 0.0,
            "latency_ms": 0,
        }

    catalog = _build_real_catalog_prototype_bank()
    catalog_summary = ", ".join(
        f"{s['brand']} {s['variant']} ({s['category']}, {s['packaging_type']})"
        for s in catalog
    )
    prompt = f"{FALLBACK_PROMPT}\nCatalog options: {catalog_summary}"
    res = ctx.ask(image, prompt, max_side=2048)

    updated = [dict(lbl) for lbl in initial_labels]
    parsed_items: list[dict[str, Any]] = []
    match = re.search(r"\{.*\}", res.text or "", re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            parsed_items = payload.get("items", []) if isinstance(payload, dict) else []
        except Exception:
            parsed_items = []

    for offset, box_idx in enumerate(ambiguous_indices):
        if offset < len(parsed_items) and isinstance(parsed_items[offset], dict):
            item = parsed_items[offset]
            base = updated[box_idx]
            updated[box_idx] = {
                "category": str(item.get("category") or base.get("category", "Personal Care")),
                "brand": str(item.get("brand") or base.get("brand", "HUL")),
                "packaging_type": str(item.get("packaging_type") or base.get("packaging_type", "Bottle")),
                "variant": str(item.get("variant") or base.get("variant", "Unknown")),
                "is_hul": bool(item.get("is_hul", base.get("is_hul", True))),
            }

    return updated, {
        "escalated_count": len(ambiguous_indices),
        "total_boxes": len(boxes),
        "escalation_rate": round(len(ambiguous_indices) / max(1, len(boxes)), 4),
        "latency_ms": round(res.seconds * 1000),
    }


__all__ = [
    "FALLBACK_PROMPT",
    "resolve_ambiguous_skus",
]
