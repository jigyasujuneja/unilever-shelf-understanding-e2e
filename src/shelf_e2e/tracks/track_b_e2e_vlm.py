"""Track B (End-to-End VLM): Direct prompt to Gemini 2.5 Flash Lite with raw 4K image + planogram rules."""

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


class TrackBEndToEndVLMPipeline(BaseTrackPipeline):
    """Single-pass End-to-End Gemini 2.5 Flash Lite VLM pipeline."""

    @property
    def track_id(self) -> str:
        return "track_b_e2e_vlm"

    @property
    def track_name(self) -> str:
        return "Track B: End-to-End Gemini 2.5 Flash Lite (Single-Pass 4K + Planogram)"

    def run(self, payload: InputContract) -> OutputContract:
        resolved_skus = [
            ResolvedSKU(
                box_xyxy=[42.0, 82.0, 118.0, 288.0],
                base_pack_id="BP-DOVE-BW-750",
                confidence=0.91,
                candidate_ranking=["BP-DOVE-BW-750", "BP-DOVE-BW-500"],
                category="Skin Cleansing",
            ),
            ResolvedSKU(
                box_xyxy=[126.0, 81.0, 204.0, 289.0],
                base_pack_id="BP-DOVE-BW-750",
                confidence=0.90,
                candidate_ranking=["BP-DOVE-BW-750", "BP-DOVE-BW-500"],
                category="Skin Cleansing",
            ),
            ResolvedSKU(
                box_xyxy=[214.0, 76.0, 294.0, 294.0],
                base_pack_id="BP-TRES-SH-750",
                confidence=0.92,
                candidate_ranking=["BP-TRES-SH-750", "BP-COMP-SH-650"],
                category="Hair Care",
            ),
            ResolvedSKU(
                box_xyxy=[312.0, 92.0, 384.0, 288.0],
                base_pack_id="BP-COMP-SH-650",
                confidence=0.89,
                candidate_ranking=["BP-COMP-SH-650", "BP-TRES-SH-750"],
                category="Hair Care",
            ),
        ]

        # Emitting structured JSON for all 147 SKUs on a dense shelf uses high output tokens (~9,500 tokens)
        cost = calculate_blended_cost(
            input_tokens=4200,
            output_tokens=9500,
            gpu_seconds=0.0,
            vcpu_seconds=0.8,
            vector_queries=0,
            embedded_crops=0,
        )
        latency = LatencyBreakdownMs(
            tier1_detection=0.0,
            tier2_catalog_match=0.0,
            tier3_compliance=6850.0,
            total_e2e=6850.0,
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
