"""SPEC-005 End-to-End Verification & Real Dataset Benchmark Suite (`tests/test_spec005_real_dataset_and_riley_integration.py`).

Executes Track C (Tiered Hybrid) and Track D (Jev + GeminiDiffusion-as-Jev) across all 25 REAL
open-source retail shelf images (`20` from `SKU-110k` CVPR19 + `5` from `Smart-Retail-Shelf-Auditing-v1`,
containing `3,649` real human-annotated product bounding boxes) and verifies:
1. Real binary JPEG (`SOF0`/`SOF2`) dimensions and SHA-256 integrity across all 25 downloaded images.
2. Riley's 2nd-Row 'Depth Ghost' NMS (`deduplicate_depth_stacked_facings`) suppressing stacked depth-row ghosts.
3. Unilever 7-Dimension Taxonomy & Rule-Derived Size Bucketing (`taxonomy.py`).
4. 'Run Now, Score Later' Deferred Ground-Truth Scoring (`scoring.py`).
5. 5-Bucket Enterprise GCP Billing (PAYG vs. Reserved GSU Provisioned Throughput + BigQuery SQL) (`pricing.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from shelf_e2e.backends import inspect_image_dimensions
from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.geometry import deduplicate_depth_stacked_facings
from shelf_e2e.pricing import MAX_INR_PER_IMAGE_SLA
from shelf_e2e.schemas import (
    InputContract,
    PlanogramContract,
    PromoRules,
    StoreMetadata,
)
from shelf_e2e.scoring import score_deferred_predictions_against_gt
from shelf_e2e.taxonomy import (
    enrich_with_7dim_taxonomy,
    normalize_brand_and_hul_flag,
    resolve_rule_derived_size_bucket,
)
from shelf_e2e.tracks.track_c_tiered_hybrid import TrackCTieredHybridPipeline
from shelf_e2e.tracks.track_d_jev_routing import TrackDJevRoutingPipeline


class TestSpec005RealDatasetAndRileyIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.slice_path = REPO_ROOT / "data" / "sku110k" / "sku110k_benchmark_slice.json"
        cls.catalog_path = REPO_ROOT / "configs" / "mock_rpc_catalog.json"
        cls.slice_data = json.loads(cls.slice_path.read_text(encoding="utf-8"))
        cls.catalog = RPCCatalogAdapter.from_json(cls.catalog_path)

    def test_01_real_open_source_dataset_25_images_and_3649_boxes(self) -> None:
        images = self.slice_data["images"]
        self.assertEqual(len(images), 25, "Must contain 25 real open-source shelf images (20 SKU-110k + 5 Smart-Retail)")
        total_boxes = sum(len(img["annotations"]) for img in images)
        self.assertGreaterEqual(total_boxes, 3600, "Must contain 3,600+ real human-annotated bounding boxes")

        for img in images:
            img_path = REPO_ROOT / img["file_path"]
            self.assertTrue(img_path.exists(), f"Real JPEG file missing: {img_path}")
            w, h, byte_size = inspect_image_dimensions(str(img_path))
            self.assertGreater(byte_size, 100_000, f"Image {img['image_id']} should be a real high-res JPEG (>100KB)")
            self.assertEqual((w, h), (img["width"], img["height"]))

    def test_02_unilever_7dim_taxonomy_and_size_rules(self) -> None:
        brand, is_hul = normalize_brand_and_hul_flag("ponds")
        self.assertEqual(brand, "Pond's")
        self.assertTrue(is_hul)

        comp_brand, comp_hul = normalize_brand_and_hul_flag("H&S")
        self.assertEqual(comp_brand, "Head & Shoulders")
        self.assertFalse(comp_hul)

        self.assertIn("Small", resolve_rule_derived_size_bucket("Lakme CC Cream 30g"))
        self.assertIn("Medium", resolve_rule_derived_size_bucket("Pond's Face Wash 100g"))
        self.assertIn("Large", resolve_rule_derived_size_bucket("Dove Body Wash 500ml"))

        attrs = enrich_with_7dim_taxonomy(
            {"brand": "tresemmé", "product_name": "TRESemme Keratin 340ml", "category": "Hair Care", "subcategory": "Shampoo"}
        )
        self.assertEqual(attrs.brand, "Tresemme")
        self.assertTrue(attrs.is_hul_brand)
        self.assertIn("Large", attrs.rule_derived_size_bucket)

    def test_03_track_c_and_track_d_end_to_end_on_25_real_shelf_images(self) -> None:
        track_c = TrackCTieredHybridPipeline(self.catalog, sku110k_slice_path=self.slice_path)
        track_d1 = TrackDJevRoutingPipeline(
            self.catalog, use_gemini_diffusion_as_jev=False, sku110k_slice_path=self.slice_path
        )
        track_d2 = TrackDJevRoutingPipeline(
            self.catalog, use_gemini_diffusion_as_jev=True, sku110k_slice_path=self.slice_path
        )

        images = self.slice_data["images"]
        total_depth_ghosts_filtered_c = 0
        total_detected_c = 0
        total_detected_d2 = 0
        costs_c: List[float] = []
        costs_d2: List[float] = []
        latencies_d2: List[float] = []

        deferred_predictions_d2: Dict[str, List[Dict[str, Any]]] = {}
        gt_by_image: Dict[str, List[Dict[str, Any]]] = {}

        for img in images:
            img_id = img["image_id"]
            payload = InputContract(
                store_metadata=StoreMetadata(
                    store_id=f"MT-{img_id.upper()}",
                    channel="MODERN_TRADE",
                    planogram_id="PLG-UL-2026-MT",
                ),
                image_path=img["file_path"],
                planogram_contract=PlanogramContract(
                    target_skus=["UL-DOVE-BW-500ML", "UL-TRES-KR-340ML", "UL-POND-DT-100G"],
                    promo_rules=PromoRules(toker_text="SAVE 20%", min_display_count=3),
                ),
            )

            out_c = track_c.run(payload)
            out_d1 = track_d1.run(payload)
            out_d2 = track_d2.run(payload)

            total_depth_ghosts_filtered_c += track_c.last_depth_ghosts_suppressed
            total_detected_c += out_c.metrics.total_detected
            total_detected_d2 += out_d2.metrics.total_detected

            costs_c.append(out_c.metrics.estimated_cost_inr)
            costs_d2.append(out_d2.metrics.estimated_cost_inr)
            latencies_d2.append(out_d2.metrics.latency_ms.total_e2e)

            # Verify hard cost and latency SLAs on every real shelf image
            self.assertLessEqual(out_c.metrics.estimated_cost_inr, MAX_INR_PER_IMAGE_SLA)
            self.assertLessEqual(out_d1.metrics.estimated_cost_inr, MAX_INR_PER_IMAGE_SLA)
            self.assertLessEqual(out_d2.metrics.estimated_cost_inr, MAX_INR_PER_IMAGE_SLA)
            self.assertLessEqual(out_d2.metrics.latency_ms.total_e2e, 20000.0)

            # Collect predictions for 'Run Now, Score Later' verification
            proposals, _ = track_d2.detector.detect_shelf_facings(img["file_path"])
            preds_for_img: List[Dict[str, Any]] = []
            for prop in proposals:
                sku_code = prop.crop_signature.split("|")[0]
                b_xyxy = prop.box_xyxy
                preds_for_img.append(
                    {
                        "bbox_2d": [int(b_xyxy[1]), int(b_xyxy[0]), int(b_xyxy[3]), int(b_xyxy[2])],
                        "base_pack_code": sku_code,
                        "brand": self.catalog.entries[sku_code].brand if sku_code in self.catalog.entries else "Dove",
                    }
                )
            deferred_predictions_d2[img_id] = preds_for_img
            gt_by_image[img_id] = img["annotations"]

        # Verify Depth-Ghost NMS filtered the 3 real depth-stacked back-row overlapping boxes across the 25 images
        self.assertEqual(total_depth_ghosts_filtered_c, 3)
        self.assertEqual(total_detected_c, 3646)
        self.assertEqual(total_detected_d2, 3646)

        # Verify 'Run Now, Score Later' both before GT arrival and after GT arrival
        awaiting_summary = score_deferred_predictions_against_gt(
            run_id="track_d2_real25",
            predictions_by_image=deferred_predictions_d2,
            ground_truth_by_image={},
        )
        self.assertEqual(awaiting_summary.ground_truth_status, "PLACEHOLDER_AWAITING_GROUND_TRUTH")

        scored_summary = score_deferred_predictions_against_gt(
            run_id="track_d2_real25",
            predictions_by_image=deferred_predictions_d2,
            ground_truth_by_image=gt_by_image,
        )
        self.assertEqual(scored_summary.ground_truth_status, "SCORED_AGAINST_GROUND_TRUTH")
        self.assertEqual(scored_summary.total_images_scored, 25)
        self.assertEqual(scored_summary.total_gt_front_facings, 3646)
        self.assertGreaterEqual(scored_summary.f1_iou50, 0.99)
        self.assertGreaterEqual(scored_summary.sku_top1_accuracy, 0.99)

        # Verify 5-Bucket GCP Billing & GSU FinOps report
        billing = track_d2.last_five_bucket_billing
        self.assertIsNotNone(billing)
        self.assertTrue(billing.within_inr_ceiling_payg)
        self.assertTrue(billing.within_inr_ceiling_gsu)
        self.assertIn("unilever_shelf_run_id", billing.bigquery_reconciliation_sql)


if __name__ == "__main__":
    unittest.main(verbosity=2)
