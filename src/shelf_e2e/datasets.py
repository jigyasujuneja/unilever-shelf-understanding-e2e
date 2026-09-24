"""Public Dataset Adapters for SKU-110k (Dense Shelf Detection) and RPC (Retail Product Checkout Catalog)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class CatalogSKUEntry:
    """Single SKU record in the RPC / Unilever Master Product Catalog."""

    base_pack_id: str
    brand: str
    category: str
    subcategory: str
    variant: str
    packaging_type: str
    pack_type: str
    size: str
    expected_min_width_px: float
    is_hul: bool


@dataclass(frozen=True)
class GroundTruthShelfAnnotation:
    """Ground-truth annotation combining SKU-110k spatial boxes and RPC catalog IDs."""

    image_path: str
    boxes_xyxy: List[List[float]]
    base_pack_ids: List[str]
    categories: List[str]
    is_hul_flags: List[bool]
    expected_share_of_shelf_pct: float
    expected_toker_compliant: bool
    expected_display_compliant: bool
    expected_red_line_gaps: List[str] = field(default_factory=list)


class RPCCatalogAdapter:
    """Loads and validates Base Pack IDs against the RPC Master Catalog."""

    def __init__(self, entries: List[CatalogSKUEntry]):
        self.entries: Dict[str, CatalogSKUEntry] = {e.base_pack_id: e for e in entries}

    @classmethod
    def from_json(cls, json_path: str | Path) -> "RPCCatalogAdapter":
        resolved = Path(json_path).resolve()
        data = json.loads(resolved.read_text(encoding="utf-8"))
        entries: List[CatalogSKUEntry] = []
        for item in data.get("skus", []):
            entries.append(CatalogSKUEntry(**item))
        existing_ids = {e.base_pack_id for e in entries}
        for item in data.get("items", []):
            bp_id = item.get("base_pack_id") or item.get("base_pack_code")
            if bp_id and bp_id not in existing_ids:
                existing_ids.add(bp_id)
                entries.append(
                    CatalogSKUEntry(
                        base_pack_id=str(bp_id),
                        brand=str(item.get("brand", "Dove")),
                        category=str(item.get("category", "Personal Care")),
                        subcategory=str(item.get("subcategory", "General")),
                        variant=str(item.get("variant", "Standard")),
                        packaging_type=str(item.get("packaging_type", "bottle")),
                        pack_type=str(item.get("pack_type", "Single")),
                        size=str(item.get("size") or item.get("size_bucket") or "500ml"),
                        expected_min_width_px=float(item.get("expected_min_width_px", 55.0)),
                        is_hul=bool(item.get("is_hul", item.get("is_hul_brand", False))),
                    )
                )
        return cls(entries)

    def valid_base_pack_ids(self) -> Set[str]:
        return set(self.entries.keys())

    def is_hul_sku(self, base_pack_id: str) -> bool:
        entry = self.entries.get(base_pack_id)
        return bool(entry.is_hul) if entry else False

    def get_category(self, base_pack_id: str) -> str:
        entry = self.entries.get(base_pack_id)
        return entry.category if entry else "Unknown"


def calculate_linear_share_of_shelf(
    boxes_xyxy: List[List[float]],
    base_pack_ids: List[str],
    catalog: RPCCatalogAdapter,
) -> float:
    """Compute linear Share-of-Shelf (SOS %) as HUL horizontal width divided by total detected width."""
    if not boxes_xyxy or not base_pack_ids:
        return 0.0
    total_width = 0.0
    hul_width = 0.0
    for box, sku_id in zip(boxes_xyxy, base_pack_ids):
        width = max(0.0, float(box[2]) - float(box[0]))
        total_width += width
        if catalog.is_hul_sku(sku_id):
            hul_width += width
    if total_width <= 0.0:
        return 0.0
    return round((hul_width / total_width) * 100.0, 2)
