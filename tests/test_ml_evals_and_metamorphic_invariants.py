"""SPEC-004 Verification Suite: Statistical Significance, Optical Slice Analysis, Calibration & Metamorphic Invariants."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_open_datasets import build_sku110k_rpc_benchmark_slice
from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.ml_evals import (
    bootstrap_confidence_interval,
    compute_calibration_ece_and_brier,
    evaluate_optical_and_category_slices,
    paired_bootstrap_significance_test,
    profile_dataset_and_check_leakage,
)
from shelf_e2e.platform.leaderboard import KaggleLeaderboardEngine
from shelf_e2e.schemas import InputContract, PlanogramContract, PromoRules, StoreMetadata
from shelf_e2e.tracks import (
    TrackACascadingViTPipeline,
    TrackCTieredHybridPipeline,
    TrackDJevRoutingPipeline,
)
from shelf_e2e.tracks.track_d_jev_routing import JevDeterministicStateMachine


class TestMLEvalsAndMetamorphicInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.slice_path = build_sku110k_rpc_benchmark_slice()
        cls.slice_data = json.loads(cls.slice_path.read_text(encoding="utf-8"))
        cls.catalog = RPCCatalogAdapter.from_json(
            REPO_ROOT / "configs" / "mock_rpc_catalog.json"
        )

    def test_01_data_autocleaning_profile_and_zero_split_leakage(self) -> None:
        profile = profile_dataset_and_check_leakage(
            slice_data=self.slice_data,
            public_split_images=KaggleLeaderboardEngine.PUBLIC_IMAGES,
            private_split_images=KaggleLeaderboardEngine.PRIVATE_IMAGES,
            catalog=self.catalog,
        )
        self.assertTrue(profile.is_clean_and_leak_free)
        self.assertEqual(profile.public_private_split_leakage_count, 0)
        self.assertEqual(profile.null_or_missing_sku_count, 0)
        self.assertEqual(profile.degenerate_bbox_count, 0)
        self.assertEqual(profile.unknown_catalog_sku_count, 0)

    def test_02_bootstrap_ci_and_paired_significance_track_d_vs_track_c(self) -> None:
        # Across 24 glared bottle evaluations, Track D resolves 100% while raw vector Track C errs on glared 750ml
        track_d_scores = [1.0] * 24
        track_c_scores = [0.0, 1.0, 1.0, 1.0] * 6

        ci_d = bootstrap_confidence_interval(track_d_scores, n_resamples=400)
        ci_c = bootstrap_confidence_interval(track_c_scores, n_resamples=400)
        self.assertEqual(ci_d.mean, 1.0)
        self.assertAlmostEqual(ci_c.mean, 0.75, places=2)

        sig = paired_bootstrap_significance_test(track_d_scores, track_c_scores, n_resamples=400)
        self.assertAlmostEqual(sig["observed_delta"], 0.25, places=2)
        self.assertLess(sig["p_value"], 0.01)
        self.assertTrue(sig["statistically_significant_at_0_05"])

    def test_03_slice_based_error_analysis_high_glare_vs_low_glare(self) -> None:
        img_path = REPO_ROOT / "data" / "sku110k" / "images" / "sku110k_val_001.jpg"
        contract = InputContract(
            image_path=str(img_path),
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
        images_by_name = self.slice_data.get("images_by_name") or self.slice_data["images"]
        gt_records = images_by_name["sku110k_val_001.jpg"]

        track_c = TrackCTieredHybridPipeline(self.catalog, sku110k_slice_path=self.slice_path)
        track_d = TrackDJevRoutingPipeline(
            self.catalog, use_gemini_diffusion_as_jev=True, sku110k_slice_path=self.slice_path
        )

        slices_c = evaluate_optical_and_category_slices(
            track_c.run(contract).marketshare.resolved_skus, gt_records, self.catalog
        )
        slices_d = evaluate_optical_and_category_slices(
            track_d.run(contract).marketshare.resolved_skus, gt_records, self.catalog
        )

        # Track C fails on the `high_glare` subpopulation (0.0 accuracy), while Track D achieves 1.0!
        self.assertEqual(slices_c["high_glare"]["top1_accuracy"], 0.0)
        self.assertEqual(slices_c["low_glare"]["top1_accuracy"], 1.0)
        self.assertEqual(slices_d["high_glare"]["top1_accuracy"], 1.0)
        self.assertEqual(slices_d["low_glare"]["top1_accuracy"], 1.0)

    def test_04_calibration_ece_brier_and_confusion_matrix(self) -> None:
        img_path = REPO_ROOT / "data" / "sku110k" / "images" / "sku110k_val_001.jpg"
        contract = InputContract(
            image_path=str(img_path),
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
        images_by_name = self.slice_data.get("images_by_name") or self.slice_data["images"]
        gt_records = images_by_name["sku110k_val_001.jpg"]
        gt_boxes = [r["box_xyxy"] for r in gt_records]
        gt_ids = [r["gt_base_pack_id"] for r in gt_records]

        track_a = TrackACascadingViTPipeline(self.catalog, sku110k_slice_path=self.slice_path)
        track_d = TrackDJevRoutingPipeline(
            self.catalog, use_gemini_diffusion_as_jev=True, sku110k_slice_path=self.slice_path
        )

        cal_a = compute_calibration_ece_and_brier(
            track_a.run(contract).marketshare.resolved_skus, gt_boxes, gt_ids
        )
        cal_d = compute_calibration_ece_and_brier(
            track_d.run(contract).marketshare.resolved_skus, gt_boxes, gt_ids
        )

        # Track D has significantly lower Brier Score and ECE than Track A
        self.assertLess(cal_d.brier_score, cal_a.brier_score)
        self.assertLess(cal_d.expected_calibration_error, cal_a.expected_calibration_error)
        self.assertEqual(
            cal_a.confusion_matrix["BP-DOVE-BW-750"]["BP-DOVE-BW-500"], 1
        )
        self.assertEqual(
            cal_d.confusion_matrix["BP-DOVE-BW-750"].get("BP-DOVE-BW-500", 0), 0
        )

    def test_05_metamorphic_spatial_jitter_invariant(self) -> None:
        # Perturbing box coordinates by +/- 2px must preserve Jev 750ml vs 500ml decision
        fsm = JevDeterministicStateMachine(self.catalog)
        base_box = [40.0, 80.0, 120.0, 290.0]
        for dx in (-2.0, -1.0, 0.0, 1.0, 2.0):
            jittered = [base_box[0] + dx, base_box[1] - dx, base_box[2] + dx, base_box[3] - dx]
            res = fsm.resolve_base_pack(
                box_xyxy=jittered,
                candidates=["BP-DOVE-BW-500", "BP-DOVE-BW-750"],
                raw_similarity=0.95,
                glare_intensity=0.42,
                use_diffusion_deglare=True,
            )
            self.assertEqual(res["base_pack_id"], "BP-DOVE-BW-750")


if __name__ == "__main__":
    unittest.main(verbosity=2)
