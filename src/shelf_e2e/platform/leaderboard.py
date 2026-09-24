"""Kaggle-style Public & Private Split Evaluation & Pareto Leaderboard Engine (`SPEC-006`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.download_open_datasets import build_sku110k_rpc_benchmark_slice
from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.mt_gondola_analytics import evaluate_full_mt_gondola_audit
from shelf_e2e.platform.registry import ExperimentRunRecord, MLflowRunRegistry
from shelf_e2e.schemas import InputContract, PlanogramContract, PromoRules, StoreMetadata
from shelf_e2e.tracks import (
    BaseTrackPipeline,
    TrackACascadingViTPipeline,
    TrackB2TwoStageCropVLMPipeline,
    TrackBEndToEndVLMPipeline,
    TrackCTieredHybridPipeline,
    TrackDJevRoutingPipeline,
    TrackEIJEPALatentWorldModelPipeline,
    TrackFPaliGemma2LoRAPipeline,
)
from tests.benchmark_harness import (
    calculate_json_schema_adherence,
    calculate_map50_and_map50_95,
    calculate_retrieval_and_classification_metrics,
    run_concurrency_stress_test,
)


class KaggleLeaderboardEngine:
    """Evaluates tracks across Public & Private SKU-110k + Smart-Retail splits and logs to MLflowRunRegistry."""

    PUBLIC_IMAGES = ("sku110k_val_000.jpg", "sku110k_val_002.jpg", "smart_retail_val_000.jpg")
    PRIVATE_IMAGES = ("sku110k_val_001.jpg", "sku110k_val_003.jpg", "smart_retail_val_001.jpg")

    def __init__(self, registry: Optional[MLflowRunRegistry] = None):
        self.repo_root = Path(__file__).resolve().parent.parent.parent.parent
        self.slice_path = build_sku110k_rpc_benchmark_slice()
        self.slice_data = json.loads(self.slice_path.read_text(encoding="utf-8"))
        self.images_map = self.slice_data.get("images_by_name") or self.slice_data.get("images", {})
        self.catalog = RPCCatalogAdapter.from_json(
            self.repo_root / "configs" / "mock_rpc_catalog.json"
        )
        self.registry = registry or MLflowRunRegistry()

    def _evaluate_image_split(
        self, track: BaseTrackPipeline, image_names: tuple[str, ...]
    ) -> tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]], Dict[str, float], float]:
        map50_vals: List[float] = []
        map50_95_vals: List[float] = []
        top1_vals: List[float] = []
        top5_vals: List[float] = []
        mrr_vals: List[float] = []
        f1_vals: List[float] = []
        halluc_vals: List[float] = []
        schema_vals: List[float] = []
        cost_vals: List[float] = []
        sos_vals: List[float] = []
        linear_sos_vals: List[float] = []
        area_sos_vals: List[float] = []
        seq_comp_vals: List[float] = []
        purity_vals: List[float] = []
        oos_void_counts: List[int] = []

        tier_latencies = {
            "tier1_detection": 0.0,
            "tier2_catalog_match": 0.0,
            "tier3_compliance": 0.0,
            "total_e2e": 0.0,
        }
        preds_by_img: Dict[str, List[Dict[str, Any]]] = {}

        for img_name in image_names:
            img_path = self.repo_root / "data" / "sku110k" / "images" / img_name
            target_skus = ["BP-DOVE-BW-750", "BP-TRES-SH-750"]
            contract = InputContract(
                image_path=str(img_path),
                store_metadata=StoreMetadata(
                    store_id="MT-MUMBAI-042",
                    channel="MODERN_TRADE",
                    planogram_id="PLANO-SKIN-HAIR-Q3",
                ),
                planogram_contract=PlanogramContract(
                    target_skus=target_skus,
                    promo_rules=PromoRules(toker_text="20% Extra", min_display_count=2),
                ),
            )
            out = track.run(contract)
            gt_records = self.images_map[img_name]
            gt_boxes = [r["box_xyxy"] for r in gt_records]
            gt_ids = [r["gt_base_pack_id"] for r in gt_records]
            gt_cats = [r["category"] for r in gt_records]

            det = calculate_map50_and_map50_95(
                pred_boxes=[s.box_xyxy for s in out.marketshare.resolved_skus],
                pred_scores=[s.confidence for s in out.marketshare.resolved_skus],
                gt_boxes=gt_boxes,
            )
            ret = calculate_retrieval_and_classification_metrics(
                resolved_skus=out.marketshare.resolved_skus,
                gt_boxes=gt_boxes,
                gt_base_pack_ids=gt_ids,
                gt_categories=gt_cats,
                master_catalog_ids=self.catalog.valid_base_pack_ids(),
            )
            mt_audit = evaluate_full_mt_gondola_audit(out.marketshare.resolved_skus, target_skus)

            map50_vals.append(det.map_50)
            map50_95_vals.append(det.map_50_95)
            top1_vals.append(ret.top1_accuracy)
            top5_vals.append(ret.top5_recall)
            mrr_vals.append(ret.mrr)
            f1_vals.append(ret.f1_score)
            halluc_vals.append(ret.hallucination_rate)
            schema_vals.append(calculate_json_schema_adherence([out.model_dump()]))
            cost_vals.append(out.metrics.estimated_cost_inr)
            sos_vals.append(out.metrics.share_of_shelf_pct)
            linear_sos_vals.append(mt_audit.linear_width_sos_pct)
            area_sos_vals.append(mt_audit.area_2d_sos_pct)
            seq_comp_vals.append(mt_audit.planogram_sequence_compliance_pct)
            purity_vals.append(mt_audit.brand_block_purity_pct)
            oos_void_counts.append(len(mt_audit.oos_void_gaps))

            tier_latencies["tier1_detection"] = out.metrics.latency_ms.tier1_detection
            tier_latencies["tier2_catalog_match"] = out.metrics.latency_ms.tier2_catalog_match
            tier_latencies["tier3_compliance"] = out.metrics.latency_ms.tier3_compliance
            tier_latencies["total_e2e"] = out.metrics.latency_ms.total_e2e

            preds_by_img[img_name] = [
                {
                    "box_xyxy": s.box_xyxy,
                    "base_pack_id": s.base_pack_id,
                    "confidence": s.confidence,
                    "candidates": s.candidate_ranking or [s.base_pack_id],
                    "category": s.category or "Unknown",
                }
                for s in out.marketshare.resolved_skus
            ]

        n = float(len(image_names))
        avg_cost = round(sum(cost_vals) / n, 4)
        metrics = {
            "map_50": round(sum(map50_vals) / n, 4),
            "map_50_95": round(sum(map50_95_vals) / n, 4),
            "top1_acc": round(sum(top1_vals) / n, 4),
            "top5_recall": round(sum(top5_vals) / n, 4),
            "mrr": round(sum(mrr_vals) / n, 4),
            "f1_score": round(sum(f1_vals) / n, 4),
            "hallucination_rate": round(sum(halluc_vals) / n, 4),
            "schema_adherence": round(sum(schema_vals) / n, 4),
            "sos_pct": round(sum(sos_vals) / n, 2),
            "linear_width_sos_pct": round(sum(linear_sos_vals) / n, 2),
            "area_2d_sos_pct": round(sum(area_sos_vals) / n, 2),
            "planogram_seq_compliance_pct": round(sum(seq_comp_vals) / n, 2),
            "brand_block_purity_pct": round(sum(purity_vals) / n, 2),
            "avg_oos_void_gaps": round(sum(oos_void_counts) / n, 2),
        }
        return metrics, preds_by_img, tier_latencies, avg_cost

    def evaluate_and_log_track(
        self,
        track: BaseTrackPipeline,
        engineer_ldap: str = "jjuneja",
        experiment_name: str = "unilever-mt-500k-daily-arena",
        custom_hyperparams: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRunRecord:
        pub_metrics, pub_preds, tier_lat, pub_cost_inr = self._evaluate_image_split(
            track, self.PUBLIC_IMAGES
        )
        priv_metrics, priv_preds, _, priv_cost_inr = self._evaluate_image_split(
            track, self.PRIVATE_IMAGES
        )

        sample_contract = InputContract(
            image_path=str(self.repo_root / "data" / "sku110k" / "images" / "sku110k_val_001.jpg"),
            store_metadata=StoreMetadata(
                store_id="MT-MUMBAI-042",
                channel="MODERN_TRADE",
                planogram_id="PLANO-SKIN-HAIR-Q3",
            ),
            planogram_contract=PlanogramContract(
                target_skus=["BP-DOVE-BW-750", "BP-TRES-SH-750"],
                promo_rules=PromoRules(toker_text="20% Extra", min_display_count=2),
            ),
        )
        stress = run_concurrency_stress_test(track.run, sample_contract, worker_count=48)
        p95_ms = stress["p95_ms"]

        within_cost_sla = pub_cost_inr <= 0.22
        within_latency_sla = p95_ms <= 20000.0
        meets_all_slas = (
            within_cost_sla
            and within_latency_sla
            and pub_metrics["schema_adherence"] == 1.0
            and pub_metrics["hallucination_rate"] == 0.0
        )

        pub_q = 100.0 * (
            0.35 * pub_metrics["map_50_95"]
            + 0.35 * pub_metrics["top1_acc"]
            + 0.15 * pub_metrics["mrr"]
            + 0.15 * (1.0 - pub_metrics["hallucination_rate"])
        )
        priv_q = 100.0 * (
            0.35 * priv_metrics["map_50_95"]
            + 0.35 * priv_metrics["top1_acc"]
            + 0.15 * priv_metrics["mrr"]
            + 0.15 * (1.0 - priv_metrics["hallucination_rate"])
        )
        blended_quality = 0.40 * pub_q + 0.60 * priv_q
        penalty = (0.0 if within_cost_sla else 25.0) + (0.0 if within_latency_sla else 25.0)
        pareto_score = round(max(0.0, blended_quality - penalty), 2)

        if not meets_all_slas:
            medal_tier = "SLA_VIOLATION"
        elif pareto_score >= 98.0:
            medal_tier = "GOLD"
        elif pareto_score >= 90.0:
            medal_tier = "SILVER"
        else:
            medal_tier = "BRONZE"

        all_preds = {**pub_preds, **priv_preds}
        hparams = {
            "concurrency_workers": 525,
            "iou_threshold": 0.50,
            "vector_top_k": 5,
            "cost_ceiling_inr": 0.22,
            "usd_to_inr_rate": 84.0,
            **(custom_hyperparams or {}),
        }

        record = ExperimentRunRecord(
            run_id=self.registry.new_run_id(track.track_id),
            experiment_name=experiment_name,
            track_id=track.track_id,
            track_name=track.track_name,
            engineer_ldap=engineer_ldap,
            timestamp_utc=self.registry.now_iso(),
            hyperparameters=hparams,
            public_metrics=pub_metrics,
            private_metrics=priv_metrics,
            latency_tiers_ms={**tier_lat, "p95_525_workers_ms": p95_ms},
            cost_breakdown={
                "cost_per_image_inr": pub_cost_inr,
                "cost_per_image_usd": round(pub_cost_inr / 84.0, 6),
                "daily_500k_cost_inr": round(pub_cost_inr * 500_000, 2),
                "private_dense_cost_inr": priv_cost_inr,
            },
            pareto_score=pareto_score,
            medal_tier=medal_tier,
            meets_all_slas=meets_all_slas,
            predictions_by_image=all_preds,
        )
        return self.registry.log_run(record)

    def seed_default_arena_runs(self) -> List[ExperimentRunRecord]:
        """Ensure all 8 neural architecture tracks (Tracks A, B1, B2, C, D1, D2 [djev], E [I-JEPA], F [PaliGemma-2]) are registered."""
        required_ids = {
            "track_a_cascading_vit",
            "track_b_e2e_vlm",
            "track_b2_two_stage_crop_vlm",
            "track_c_tiered_hybrid",
            "track_d_jev_routing",
            "track_d_gemini_diffusion_as_jev",
            "track_e_ijepa_world_model",
            "track_f_paligemma2_lora",
        }
        existing = self.registry.list_runs()
        if required_ids.issubset({r.track_id for r in existing}):
            return existing

        self.registry.clear_runs()
        candidates = [
            (
                TrackACascadingViTPipeline(self.catalog),
                {
                    "architecture": "Legacy 8-Model Supervised Cascade (RT-DETR -> ViT-B/16 -> 6x InceptionNet)",
                    "detector": "RT-DETR",
                    "glare_solver": "None",
                },
            ),
            (
                TrackBEndToEndVLMPipeline(self.catalog),
                {
                    "architecture": "Riley Path 1/2/3/6: 1-Pass Full-Shelf Gemini 2.5 Flash VLM",
                    "model": "gemini-2.5-flash",
                    "glare_solver": "Raw Full-Image Prompt",
                },
            ),
            (
                TrackB2TwoStageCropVLMPipeline(self.catalog, sku110k_slice_path=self.slice_path),
                {
                    "architecture": "Riley Path 5 (two_stage_detection.py): Stage 1 Detector + Stage 2 Parallel PIL Crop Gemini 2.5 Flash-Lite",
                    "model": "gemini-2.5-flash-lite (per-crop)",
                    "glare_solver": "High-Res PIL Crop OCR",
                },
            ),
            (
                TrackCTieredHybridPipeline(self.catalog, sku110k_slice_path=self.slice_path),
                {
                    "architecture": "Riley Path 4/7: DINOv2-ViT-L/14-reg4 + ScaNN Vector Search + Promo VLM",
                    "detector": "SKU-110k/RT-DETR + Depth-NMS",
                    "glare_solver": "None",
                },
            ),
            (
                TrackDJevRoutingPipeline(
                    self.catalog, use_gemini_diffusion_as_jev=False, sku110k_slice_path=self.slice_path
                ),
                {
                    "architecture": "Track D1: ScaNN + Jev Deterministic 7-Dim State Machine",
                    "detector": "SKU-110k/RT-DETR + Depth-NMS",
                    "glare_solver": "Jev Spatial Width Lock",
                },
            ),
            (
                TrackDJevRoutingPipeline(
                    self.catalog, use_gemini_diffusion_as_jev=True, sku110k_slice_path=self.slice_path
                ),
                {
                    "architecture": "Track D2: mmastrac/djev (/v1/systemone DiffusionGemma-26B-A4B-it 64-Token Seeded/Pinned/Constrained Canvas)",
                    "detector": "SKU-110k/RT-DETR + Depth-NMS",
                    "glare_solver": "djev /v1/systemone + vllm#58216 Constrained Top-5 + DAG",
                },
            ),
            (
                TrackEIJEPALatentWorldModelPipeline(self.catalog, sku110k_slice_path=self.slice_path),
                {
                    "architecture": "Track E: I-JEPA / V-JEPA Latent World-Model Predictor + DINOv2-ViT-L/14-reg4 + ScaNN",
                    "detector": "SKU-110k/RT-DETR + Depth-NMS",
                    "glare_solver": "I-JEPA Masked Glare Patch Predictor in Latent Space (0 Token Cost)",
                },
            ),
            (
                TrackFPaliGemma2LoRAPipeline(self.catalog, sku110k_slice_path=self.slice_path),
                {
                    "architecture": "Track F (Riley Path 8 fine_tuning.py): PaliGemma-2-3B-LoRA / Florence-2 Specialist on L4 GPU",
                    "detector": "SKU-110k/RT-DETR + Depth-NMS",
                    "glare_solver": "LoRA Fine-Tuned <OD>+<OCR>+<7-Dim> Specialist (0 API Token Cost)",
                },
            ),
        ]
        records = []
        for track, hp in candidates:
            records.append(self.evaluate_and_log_track(track, engineer_ldap="jjuneja", custom_hyperparams=hp))
        return self.registry.list_runs()
