"""Verification Suite for All 9 Real-World Production Defenses Across Layers 1, 2, and 3.

Tests:
  - Layer 1 (Physical Geometry & Capture):
      1. Oblique narrow-aisle (`40 deg`) & wide-angle local rail-spacing rectification
      2. Panorama seam deduplication anchored on price-rail strips across 12 repeating bottles
      3. Depth + shadow-boosted CLAHE disambiguation (`DEEP_RECESSED_NEEDS_PULL_FORWARD` vs `BRANDED_BACKBOARD_TRUE_OOS`)
  - Layer 2 (Coarse-to-Fine 3-Task + Pre-Filtered Embedding Cascade):
      4. Entropy-gated soft vs. hard `ScaNN` pre-filtering (`H3 > 0.030` rescues refill pouch misread as bottle)
      5. Bottom-15% plastic shelf-rail lip occlusion fallback to below-box price-tag OCR & rail-rectified height
      6. Horizontal Markov neighbor smoothing for `90-180 deg` rotated back-label bottles
      7. Multi-prototype centroid matching for quarterly festive/promo artwork refreshes
  - Layer 3 (`GT / Shikkar` Chaos & Field Anti-Fraud):
      8. Oriented PCA centerline + heat-seal notch counting for diagonal (`38 deg`) hanging `"Ladi"` sachet strips
      9. Stage 0 liveness & anti-spoofing gate (FFT screen Moire + regional cross-store `pHash` dedup)
"""

from __future__ import annotations

import unittest

from shelf_e2e.djev_client import DjevSystemOneClient
from shelf_e2e.real_world_defenses import (
    disambiguate_oos_void_vs_recessed_or_backboard,
    entropy_gated_3task_scann_prefilter,
    match_multi_prototype_sku_centroids,
    normalize_boxes_by_local_rail_spacing,
    resolve_size_with_rail_lip_and_pricetag_fallback,
    run_all_9_real_world_defense_benchmarks,
    slice_oriented_ladi_sachet_strip,
    smooth_rotated_or_srp_boxes_with_rail_neighbors,
    stitch_panorama_with_structural_rail_anchors,
    verify_stage0_image_liveness_and_dedup,
)


class TestRealWorldDefenseLayers(unittest.TestCase):
    """Verify all 9 Senior Staff MLE / Principal FDE real-world edge-case defenses."""

    def test_layer1_oblique_rail_rectification_and_panorama_and_depth_voids(self) -> None:
        # 1. Near bottle (160px wide @ 320px rail) and far identical bottle (80px wide @ 160px rail)
        rect = normalize_boxes_by_local_rail_spacing(
            boxes_xyxy=[[0.0, 100.0, 160.0, 420.0], [1840.0, 150.0, 1920.0, 310.0]],
            image_width_px=1920.0,
            near_rail_spacing_px=320.0,
            far_rail_spacing_px=160.0,
            standard_rail_height_cm=28.0,
        )
        self.assertAlmostEqual(rect[0].rectified_width_cm, rect[1].rectified_width_cm, delta=0.8)

        # 2. Structural panorama seam stitching on 12 repeating bottles
        seam = stitch_panorama_with_structural_rail_anchors(
            raw_rois_per_image=40, image_count=6, repeating_identical_run_length=12
        )
        self.assertTrue(seam.aliasing_prevented_on_repeating_runs)
        self.assertGreater(seam.structural_anchor_keypoints_matched, 50)

        # 3. Recessed stock in shadow vs branded backboard OOS
        recessed = disambiguate_oos_void_vs_recessed_or_backboard(
            [100, 100, 200, 280],
            depth_jump_cm=9.0,
            raw_rgb_texture_energy=0.15,
            shadow_boosted_clahe_product_score=0.85,
        )
        self.assertEqual(recessed.verdict, "DEEP_RECESSED_NEEDS_PULL_FORWARD")
        self.assertFalse(recessed.counts_as_oos_in_osa)

        backboard = disambiguate_oos_void_vs_recessed_or_backboard(
            [300, 100, 420, 280],
            depth_jump_cm=18.5,
            raw_rgb_texture_energy=0.80,
            shadow_boosted_clahe_product_score=0.10,
        )
        self.assertEqual(backboard.verdict, "BRANDED_BACKBOARD_TRUE_OOS")
        self.assertTrue(backboard.counts_as_oos_in_osa)

    def test_layer2_entropy_gated_soft_prefilter_rail_lip_rotated_and_multiprototype(self) -> None:
        catalog = [
            {"sku_id": "BP-HUL-DOVE-BW-750-BOTTLE", "brand": "Dove", "packaging_type": "bottle"},
            {"sku_id": "BP-HUL-DOVE-HW-500-POUCH", "brand": "Dove", "packaging_type": "pouch"},
        ]
        # 4. Hard filter locks out pouch when misread as bottle; soft filter (H3=0.058 > 0.030) rescues it!
        hard = entropy_gated_3task_scann_prefilter(
            catalog, "Dove", "bottle", h3_packaging_entropy=0.012, ground_truth_sku="BP-HUL-DOVE-HW-500-POUCH"
        )
        soft = entropy_gated_3task_scann_prefilter(
            catalog, "Dove", "bottle", h3_packaging_entropy=0.058, ground_truth_sku="BP-HUL-DOVE-HW-500-POUCH"
        )
        self.assertFalse(hard.true_sku_retained_in_pool)
        self.assertTrue(soft.true_sku_retained_in_pool)
        self.assertEqual(soft.filter_mode, "SOFT_EQUIVALENCE_GROUP_EXPANSION")

        # Also verify DjevSystemOneClient.classify_3task_and_prefilter_scann uses entropy gating
        client = DjevSystemOneClient()
        res_soft = client.classify_3task_and_prefilter_scann(
            box_xyxy=[10.0, 20.0, 80.0, 180.0],
            hint_brand="Dove",
            hint_packaging="bottle",
            h3_packaging_entropy=0.062,
        )
        self.assertIn("BP-HUL-DOVE-HW-500-POUCH", res_soft.filtered_candidate_skus)

        # 5. Rail-lip occlusion fallback to shelf strip price tag OCR
        sz = resolve_size_with_rail_lip_and_pricetag_fallback(
            pack_ocr_snippet="Dove Intense Repair",
            below_box_shelf_strip_ocr="DOVE SHMP 340ML MRP 299",
            rectified_height_cm=18.0,
        )
        self.assertEqual(sz.resolved_size_str, "340ml")
        self.assertEqual(sz.resolution_source, "BELOW_BOX_SHELF_PRICETAG_OCR")

        # 6. Rotated 180-deg bottle recovered by flanking neighbors
        smoothed = smooth_rotated_or_srp_boxes_with_rail_neighbors(
            [
                {"sku_id": "BP-HUL-DOVE-IR-340ML", "confidence": 0.96, "height_cm": 18.2, "cap_color": "gold"},
                {"sku_id": "UNKNOWN", "confidence": 0.25, "height_cm": 18.1, "cap_color": "gold", "is_rotated_back_label": True},
                {"sku_id": "BP-HUL-DOVE-IR-340ML", "confidence": 0.95, "height_cm": 18.3, "cap_color": "gold"},
            ]
        )
        self.assertEqual(smoothed[1].resolved_sku_id, "BP-HUL-DOVE-IR-340ML")
        self.assertEqual(smoothed[1].resolution_method, "RAIL_MARKOV_NEIGHBOR_CONSENSUS")

        # 7. Multi-prototype festive pack matching
        sku, sim, ptype = match_multi_prototype_sku_centroids(
            crop_embedding=[0.10, 0.90, 0.40, 0.20],
            sku_prototypes={
                "BP-HUL-LUX-150G": [
                    [0.80, 0.15, 0.30, 0.10],
                    [0.11, 0.89, 0.41, 0.19],
                ]
            },
        )
        self.assertEqual(sku, "BP-HUL-LUX-150G")
        self.assertGreater(sim, 0.99)
        self.assertTrue(ptype.startswith("IN_STORE_PROMO_PROTOTYPE"))

    def test_layer3_oriented_ladi_slicer_and_stage0_liveness(self) -> None:
        # 8. Oriented Ladi sachet strip slicer at 38 deg tilt
        ladi = slice_oriented_ladi_sachet_strip([40.0, 20.0, 95.0, 420.0], tilt_angle_deg=38.0)
        self.assertGreater(ladi.individual_sachet_count, ladi.naive_vertical_slice_count)
        self.assertGreaterEqual(ladi.sachet_recall_gain, 3)

        # 9. Stage 0 liveness & anti-fraud
        moire = verify_stage0_image_liveness_and_dedup(fft_moire_peak_score=0.82, screen_bezel_detected=True)
        self.assertFalse(moire.passed_liveness)
        self.assertEqual(moire.verdict, "REJECTED_SCREEN_RECAPTURE_MOIRE")

        dup = verify_stage0_image_liveness_and_dedup(scene_phash_similarity_to_recent=0.95)
        self.assertFalse(dup.passed_liveness)
        self.assertEqual(dup.verdict, "REJECTED_DUPLICATE_STORE_PHOTO")

        summary = run_all_9_real_world_defense_benchmarks()
        self.assertTrue(summary["aggregate_stress_benchmark_summary"]["all_9_defenses_verified"])
        self.assertGreater(
            summary["aggregate_stress_benchmark_summary"]["defended_pipeline_stress_slice_f2"],
            0.95,
        )


if __name__ == "__main__":
    unittest.main()
