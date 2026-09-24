"""Base abstract class for all competing retail shelf understanding tracks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Tuple

from shelf_e2e.datasets import RPCCatalogAdapter, calculate_linear_share_of_shelf
from shelf_e2e.schemas import (
    CompliancePayload,
    ComplianceStatus,
    InputContract,
    MarketSharePayload,
    OutputContract,
    ResolvedSKU,
)


class BaseTrackPipeline(ABC):
    """Enforces the SPEC-001 Universal I/O Contract across all architecture tracks."""

    def __init__(self, catalog: RPCCatalogAdapter):
        self.catalog = catalog

    @property
    @abstractmethod
    def track_id(self) -> str:
        """Unique identifier for the track."""

    @property
    @abstractmethod
    def track_name(self) -> str:
        """Human-readable architecture description."""

    @abstractmethod
    def run(self, payload: InputContract) -> OutputContract:
        """Execute end-to-end shelf understanding and return a validated OutputContract."""

    def evaluate_planogram_and_compliance(
        self,
        payload: InputContract,
        resolved_skus: List[ResolvedSKU],
        toker_detected_text: str,
    ) -> Tuple[float, MarketSharePayload, CompliancePayload]:
        """Shared business logic for SOS %, Red-Line Gaps, Width-Pack Gaps, and Toker Compliance."""
        boxes = [s.box_xyxy for s in resolved_skus]
        pack_ids = [s.base_pack_id for s in resolved_skus]
        sos_pct = calculate_linear_share_of_shelf(boxes, pack_ids, self.catalog)

        sku_counts = Counter(pack_ids)
        min_count = payload.planogram_contract.promo_rules.min_display_count

        red_line_gaps: List[str] = []
        width_pack_gaps: List[str] = []
        for target_sku in payload.planogram_contract.target_skus:
            count = sku_counts.get(target_sku, 0)
            if count == 0:
                red_line_gaps.append(target_sku)
            elif count < min_count:
                width_pack_gaps.append(target_sku)

        expected_toker = payload.planogram_contract.promo_rules.toker_text.strip().lower()
        toker_ok = bool(expected_toker and expected_toker in toker_detected_text.strip().lower())
        display_ok = len(red_line_gaps) == 0 and len(width_pack_gaps) == 0

        toker_status = ComplianceStatus.COMPLIANT if toker_ok else ComplianceStatus.NON_COMPLIANT
        display_status = ComplianceStatus.COMPLIANT if display_ok else ComplianceStatus.NON_COMPLIANT

        if toker_ok and display_ok:
            coaching_msg = (
                f"Store {payload.store_metadata.store_id} is 100% compliant on planogram "
                f"{payload.store_metadata.planogram_id} (SOS: {sos_pct:.1f}%)."
            )
        else:
            issues: List[str] = []
            if red_line_gaps:
                issues.append(f"Restock missing Red-Line SKUs: {', '.join(red_line_gaps)}")
            if width_pack_gaps:
                issues.append(
                    f"Increase facing count to >= {min_count} for: {', '.join(width_pack_gaps)}"
                )
            if not toker_ok:
                issues.append(
                    f"Place promotional Toker '{payload.planogram_contract.promo_rules.toker_text}' on shelf strip"
                )
            coaching_msg = " | ".join(issues)

        return (
            sos_pct,
            MarketSharePayload(
                resolved_skus=resolved_skus,
                red_line_gaps=red_line_gaps,
                width_pack_gaps=width_pack_gaps,
            ),
            CompliancePayload(
                toker_status=toker_status,
                display_status=display_status,
                coaching_message=coaching_msg,
            ),
        )
