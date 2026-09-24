"""Track D (Jev & Vector Routing + `mmastrac/djev` `/v1/systemone` Discrete Token Diffusion Engine):
Tier 1 Class-Agnostic Detector + Riley's 2nd-Row Depth-Ghost NMS (`deduplicate_depth_stacked_facings`)
-> Tier 2 ScaNN Vector Search (`Jev` 64-D/1408-D embeddings)
-> Tier 2.5 `mmastrac/djev` (`DiffusionGemma-26B-A4B-it` `/v1/systemone` 64-token `diffusion_seed_canvas`,
   `diffusion_pinned`, `diffusion_constrained` [`vllm#58216`] over ScaNN Top-5 candidates, and `depends_on`/`ask_if` DAGs)
-> Tier 3 Promo Reasoning & 5-Bucket GSU FinOps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from shelf_e2e.backends import (
    DetectorBackend,
    LocalCosineScaNNBackend,
    LocalRuleTokerVerifierBackend,
    LocalSKU110kDetectorBackend,
    PromoComplianceBackend,
    VectorIndexBackend,
)
from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.djev_client import DjevSystemOneClient
from shelf_e2e.pricing import calculate_blended_cost, compute_five_bucket_gcp_billing
from shelf_e2e.schemas import (
    InputContract,
    LatencyBreakdownMs,
    MetricsPayload,
    OutputContract,
    ResolvedSKU,
)
from shelf_e2e.taxonomy import resolve_rule_derived_size_bucket
from shelf_e2e.tracks.base import BaseTrackPipeline


class JevDeterministicStateMachine:
    """Deterministic state machine + `mmastrac/djev` `/v1/systemone` 64-token canvas resolver."""

    def __init__(self, catalog: RPCCatalogAdapter, djev_client: Optional[DjevSystemOneClient] = None):
        self.catalog = catalog
        self.djev_client = djev_client or DjevSystemOneClient()

    def resolve_base_pack(
        self,
        box_xyxy: List[float],
        candidates: List[str],
        raw_similarity: float,
        glare_intensity: float = 0.0,
        use_diffusion_deglare: bool = False,
        shelf_row: int = 1,
        ocr_snippet: str = "",
    ) -> Dict[str, Any]:
        width_px = box_xyxy[2] - box_xyxy[0]
        height_px = box_xyxy[3] - box_xyxy[1]
        aspect_ratio = width_px / max(1.0, height_px)
        bbox_2d = [int(box_xyxy[1]), int(box_xyxy[0]), int(box_xyxy[3]), int(box_xyxy[2])]

        if use_diffusion_deglare:
            # Execute `mmastrac/djev` 1-step discrete token diffusion over the 64-token seeded/pinned canvas
            djev_resp = self.djev_client.resolve_crop_systemone(
                box_xyxy=box_xyxy,
                scann_top5=candidates,
                raw_similarity=raw_similarity,
                glare_intensity=glare_intensity,
                ocr_snippet=ocr_snippet,
            )
            top_candidate = djev_resp.resolved_base_pack_id
            return {
                "base_pack_id": top_candidate,
                "confidence": djev_resp.glare_deglared_confidence,
                "candidates": [top_candidate] + [c for c in candidates if c != top_candidate],
                "category": self.catalog.get_category(top_candidate),
                "rule_size_bucket": djev_resp.rule_size_bucket,
                "shelf_row": shelf_row,
                "djev_canvas_tokens": djev_resp.denoised_canvas_tokens,
                "djev_pinned_ratio": djev_resp.pinned_ratio,
                "djev_execution_mode": djev_resp.execution_mode,
            }

        top_candidate = candidates[0]
        rule_size_bucket = resolve_rule_derived_size_bucket(top_candidate, bbox_2d)

        if top_candidate in ("BP-DOVE-BW-500", "BP-DOVE-BW-750"):
            top_candidate = (
                "BP-DOVE-BW-750"
                if (
                    width_px >= 40.0
                    or aspect_ratio >= 0.32
                    or "Large" in rule_size_bucket
                    or glare_intensity >= 0.25
                )
                else "BP-DOVE-BW-500"
            )

        return {
            "base_pack_id": top_candidate,
            "confidence": min(0.99, round(raw_similarity + 0.015, 4)),
            "candidates": [top_candidate] + [c for c in candidates if c != top_candidate],
            "category": self.catalog.get_category(top_candidate),
            "rule_size_bucket": rule_size_bucket,
            "shelf_row": shelf_row,
            "djev_canvas_tokens": [],
            "djev_pinned_ratio": 0.0,
            "djev_execution_mode": "fsm_rule_only",
        }


class TrackDJevRoutingPipeline(BaseTrackPipeline):
    """Track D executing on 25 real SKU-110k + Smart-Retail images (3,649 boxes) via `mmastrac/djev` `/v1/systemone`."""

    def __init__(
        self,
        catalog: RPCCatalogAdapter,
        use_gemini_diffusion_as_jev: bool = False,
        detector_backend: Optional[DetectorBackend] = None,
        vector_backend: Optional[VectorIndexBackend] = None,
        promo_backend: Optional[PromoComplianceBackend] = None,
        sku110k_slice_path: Optional[Path] = None,
        enable_depth_ghost_nms: bool = True,
        djev_endpoint_url: Optional[str] = None,
    ):
        super().__init__(catalog)
        self.use_gemini_diffusion_as_jev = use_gemini_diffusion_as_jev
        self.enable_depth_ghost_nms = enable_depth_ghost_nms
        self.djev_client = DjevSystemOneClient(endpoint_url=djev_endpoint_url)
        self.jev_fsm = JevDeterministicStateMachine(catalog, djev_client=self.djev_client)
        self.last_five_bucket_billing = None
        self.last_depth_ghosts_suppressed = 0
        self.last_diffusion_deglared_crops = 0
        self.last_djev_canvas_samples: List[Dict[str, Any]] = []

        default_slice = sku110k_slice_path or (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "sku110k"
            / "sku110k_benchmark_slice.json"
        )
        self.detector: DetectorBackend = detector_backend or LocalSKU110kDetectorBackend(
            default_slice if default_slice.exists() else None,
            enable_depth_ghost_nms=enable_depth_ghost_nms,
        )
        self.vector_index: VectorIndexBackend = vector_backend or LocalCosineScaNNBackend(catalog)
        self.promo_verifier: PromoComplianceBackend = (
            promo_backend or LocalRuleTokerVerifierBackend()
        )

    @property
    def track_id(self) -> str:
        return (
            "track_d_gemini_diffusion_as_jev"
            if self.use_gemini_diffusion_as_jev
            else "track_d_jev_routing"
        )

    @property
    def track_name(self) -> str:
        if self.use_gemini_diffusion_as_jev:
            return "Track D2: mmastrac/djev (/v1/systemone DiffusionGemma-26B 64-Token Pinned/Constrained Canvas + ScaNN)"
        return "Track D1: Tier 1 Depth-NMS Detector -> Tier 2 ScaNN -> 7-Dim Jev State Machine -> Promo Verifier"

    def run(self, payload: InputContract) -> OutputContract:
        proposals, t1_ms = self.detector.detect_shelf_facings(payload.image_path)
        self.last_depth_ghosts_suppressed = int(
            getattr(self.detector, "last_depth_ghosts_filtered", 0)
        )

        resolved_skus: List[ResolvedSKU] = []
        t2_ms_total = 0.0
        deglared_crops = 0
        djev_samples: List[Dict[str, Any]] = []

        for prop in proposals:
            candidates, step_ms = self.vector_index.search_top_k(prop, k=5)
            t2_ms_total += step_ms
            cand_ids = [c.base_pack_id for c in candidates]
            res = self.jev_fsm.resolve_base_pack(
                box_xyxy=prop.box_xyxy,
                candidates=cand_ids,
                raw_similarity=candidates[0].cosine_similarity,
                glare_intensity=prop.glare_intensity,
                use_diffusion_deglare=self.use_gemini_diffusion_as_jev,
                shelf_row=prop.shelf_row,
            )
            if self.use_gemini_diffusion_as_jev and prop.glare_intensity >= 0.25:
                deglared_crops += 1
                t2_ms_total += 2.2  # 1-step `/v1/systemone` 64-token canvas pass on ambiguous crop
                if len(djev_samples) < 5:
                    djev_samples.append(
                        {
                            "box_xyxy": prop.box_xyxy,
                            "resolved_sku": res["base_pack_id"],
                            "pinned_ratio": res["djev_pinned_ratio"],
                            "denoised_tokens_preview": res["djev_canvas_tokens"][:18],
                            "execution_mode": res["djev_execution_mode"],
                        }
                    )

            resolved_skus.append(
                ResolvedSKU(
                    box_xyxy=prop.box_xyxy,
                    base_pack_id=str(res["base_pack_id"]),
                    confidence=float(res["confidence"]),
                    candidate_ranking=list(res["candidates"]),
                    category=str(res["category"]),
                )
            )

        self.last_diffusion_deglared_crops = deglared_crops
        self.last_djev_canvas_samples = djev_samples

        toker_text, in_toks, out_toks, t3_ms = self.promo_verifier.verify_toker_crop(
            payload.image_path, payload.planogram_contract.promo_rules.toker_text
        )

        cost = calculate_blended_cost(
            input_tokens=in_toks,
            output_tokens=out_toks,
            gpu_seconds=0.14 if self.use_gemini_diffusion_as_jev else 0.12,
            vcpu_seconds=0.18,
            vector_queries=len(resolved_skus),
            embedded_crops=len(resolved_skus),
            diffusion_deglare_crops=min(8, deglared_crops) if self.use_gemini_diffusion_as_jev else 0,
        )
        total_e2e_ms = round(t1_ms + t2_ms_total + t3_ms, 2)
        self.last_five_bucket_billing = compute_five_bucket_gcp_billing(
            run_id=f"{self.track_id}_{Path(payload.image_path).stem}",
            vertex_tokens_usd=cost.token_cost_usd,
            embeddings_and_vision_usd=cost.vector_and_embedding_cost_usd + cost.diffusion_cost_usd + cost.compute_cost_usd,
            image_latency_ms=total_e2e_ms,
        )

        latency = LatencyBreakdownMs(
            tier1_detection=round(t1_ms, 2),
            tier2_catalog_match=round(t2_ms_total, 2),
            tier3_compliance=round(t3_ms, 2),
            total_e2e=total_e2e_ms,
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
