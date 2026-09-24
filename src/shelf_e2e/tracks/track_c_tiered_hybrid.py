"""Track C (Tiered Hybrid Engine - FDE Recommended):
Tier 1 RT-DETR / SKU-110k Detector -> Tier 2 Multimodal Embeddings + ScaNN Vector Search -> Tier 3 Promo Crops Only.
Wired with Plug-and-Play Backends (`DetectorBackend`, `VectorIndexBackend`, `PromoComplianceBackend`) so it runs 100%
offline on open-source SKU-110k + RPC data and swaps to live Vertex AI / Cloud Run endpoints via config.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from shelf_e2e.backends import (
    DetectorBackend,
    LocalCosineScaNNBackend,
    LocalRuleTokerVerifierBackend,
    LocalSKU110kDetectorBackend,
    PromoComplianceBackend,
    VectorIndexBackend,
)
from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.pricing import calculate_blended_cost
from shelf_e2e.schemas import (
    InputContract,
    LatencyBreakdownMs,
    MetricsPayload,
    OutputContract,
    ResolvedSKU,
)
from shelf_e2e.tracks.base import BaseTrackPipeline


class TrackCTieredHybridPipeline(BaseTrackPipeline):
    """FDE Recommended Tiered Hybrid Engine executing on SKU-110k + RPC via Plug-and-Play Backends."""

    def __init__(
        self,
        catalog: RPCCatalogAdapter,
        detector_backend: Optional[DetectorBackend] = None,
        vector_backend: Optional[VectorIndexBackend] = None,
        promo_backend: Optional[PromoComplianceBackend] = None,
        sku110k_slice_path: Optional[Path] = None,
    ):
        super().__init__(catalog)
        default_slice = sku110k_slice_path or (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "sku110k"
            / "sku110k_benchmark_slice.json"
        )
        self.detector: DetectorBackend = detector_backend or LocalSKU110kDetectorBackend(
            default_slice if default_slice.exists() else None
        )
        self.vector_index: VectorIndexBackend = vector_backend or LocalCosineScaNNBackend(catalog)
        self.promo_verifier: PromoComplianceBackend = (
            promo_backend or LocalRuleTokerVerifierBackend()
        )

    @property
    def track_id(self) -> str:
        return "track_c_tiered_hybrid"

    @property
    def track_name(self) -> str:
        return "Track C (FDE Recommended): Tier 1 Detector -> Tier 2 ScaNN Vector Search -> Tier 3 Promo Crops"

    def run(self, payload: InputContract) -> OutputContract:
        # Tier 1: Spatial Facing Detection on SKU-110k shelf image
        proposals, t1_ms = self.detector.detect_shelf_facings(payload.image_path)

        # Tier 2: ScaNN Vector Search over RPC Master Catalog for each crop
        resolved_skus: List[ResolvedSKU] = []
        t2_ms_total = 0.0
        for prop in proposals:
            candidates, step_ms = self.vector_index.search_top_k(prop, k=5)
            t2_ms_total += step_ms
            top_hit = candidates[0]
            resolved_skus.append(
                ResolvedSKU(
                    box_xyxy=prop.box_xyxy,
                    base_pack_id=top_hit.base_pack_id,
                    confidence=top_hit.cosine_similarity,
                    candidate_ranking=[c.base_pack_id for c in candidates],
                    category=top_hit.category,
                )
            )

        # Tier 3: Promotional Toker verification crop
        toker_text, in_toks, out_toks, t3_ms = self.promo_verifier.verify_toker_crop(
            payload.image_path, payload.planogram_contract.promo_rules.toker_text
        )

        cost = calculate_blended_cost(
            input_tokens=in_toks,
            output_tokens=out_toks,
            gpu_seconds=0.14,
            vcpu_seconds=0.25,
            vector_queries=len(resolved_skus),
            embedded_crops=len(resolved_skus),
        )
        latency = LatencyBreakdownMs(
            tier1_detection=round(t1_ms, 2),
            tier2_catalog_match=round(t2_ms_total, 2),
            tier3_compliance=round(t3_ms, 2),
            total_e2e=round(t1_ms + t2_ms_total + t3_ms, 2),
        )
        sos_pct, marketshare, compliance = self.evaluate_planogram_and_compliance(
            payload=payload,
            resolved_skus=resolved_skus,
            toker_detected_text=toker_text,
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
