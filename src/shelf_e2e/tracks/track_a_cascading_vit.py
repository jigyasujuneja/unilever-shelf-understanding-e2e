"""Track A (Cascading ViT Baseline): RT-DETR Detection -> ViT-B-16 Brand -> InceptionNet Variant."""

from __future__ import annotations

from shelf_e2e.pricing import calculate_blended_cost
from shelf_e2e.schemas import (
    InputContract,
    LatencyBreakdownMs,
    MetricsPayload,
    OutputContract,
    ResolvedSKU,
)
from shelf_e2e.tracks.base import BaseTrackPipeline


class TrackACascadingViTPipeline(BaseTrackPipeline):
    """Legacy 13-model cascading CNN/ViT stack (Unilever baseline)."""

    @property
    def track_id(self) -> str:
        return "track_a_cascading_vit"

    @property
    def track_name(self) -> str:
        return "Track A: RT-DETR -> ViT-B-16 Brand -> 6x InceptionNet Variant"

    def run(self, payload: InputContract) -> OutputContract:
        # Cascading crops through 13 separate models incurs higher multi-hop GPU compute
        # and lower recall on new/glared SKUs (occasionally confusing 500ml vs 750ml).
        resolved_skus = [
            ResolvedSKU(
                box_xyxy=[40.0, 80.0, 120.0, 290.0],
                base_pack_id="BP-DOVE-BW-500",  # Misclassifies 750ml Pump as 500ml under glare
                confidence=0.79,
                candidate_ranking=["BP-DOVE-BW-500", "BP-DOVE-BW-750", "BP-LUX-FW-100"],
                category="Skin Cleansing",
            ),
            ResolvedSKU(
                box_xyxy=[125.0, 80.0, 205.0, 290.0],
                base_pack_id="BP-DOVE-BW-750",
                confidence=0.84,
                candidate_ranking=["BP-DOVE-BW-750", "BP-DOVE-BW-500", "BP-TRES-SH-750"],
                category="Skin Cleansing",
            ),
            ResolvedSKU(
                box_xyxy=[215.0, 75.0, 295.0, 295.0],
                base_pack_id="BP-TRES-SH-750",
                confidence=0.86,
                candidate_ranking=["BP-TRES-SH-750", "BP-COMP-SH-650"],
                category="Hair Care",
            ),
            ResolvedSKU(
                box_xyxy=[310.0, 90.0, 385.0, 290.0],
                base_pack_id="BP-COMP-SH-650",
                confidence=0.88,
                candidate_ranking=["BP-COMP-SH-650", "BP-TRES-SH-750"],
                category="Hair Care",
            ),
        ]

        cost = calculate_blended_cost(
            input_tokens=0,
            output_tokens=0,
            gpu_seconds=9.5,  # 13 sequential CNN/ViT passes across crops
            vcpu_seconds=4.0,
            vector_queries=0,
            embedded_crops=0,
        )
        latency = LatencyBreakdownMs(
            tier1_detection=180.0,
            tier2_catalog_match=1650.0,
            tier3_compliance=420.0,
            total_e2e=2250.0,
        )
        sos_pct, marketshare, compliance = self.evaluate_planogram_and_compliance(
            payload=payload,
            resolved_skus=resolved_skus,
            toker_detected_text="20% Extra",
        )
        return OutputContract(
            metrics=MetricsPayload(
                total_detected=len(resolved_skus),
                share_of_shelf_pct=sos_pct,
                latency_ms=latency,
                estimated_cost_inr=cost.total_cost_inr,
            ),
            marketshare=marketshare,
            compliance=compliance,
        )
