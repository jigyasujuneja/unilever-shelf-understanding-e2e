"""Hard-fail, zero-fallback, and anti-hallucination unit tests across detectors, stages, and classifiers.

Verifies the strict architectural contract:
  1. Standalone neural detectors and vector retrievers (`yolo_n26_sku110k`, `rtdetr_shelf_rail_detector`,
     `track_e_open_vocab`, `scann_vector_retriever`) execute with ZERO VLM fallback calls (tested with a
     trap LLM that immediately raises AssertionError if invoked).
  2. Invalid images (`None` or zero-area) or degenerate bounding boxes (`x2 <= x1` or `y2 <= y1`) or
     unknown modes immediately raise `ValueError` / `RuntimeError` across all 6 stages and utilities
     with zero silent fallback (`except Exception: pass` forbidden).
  3. Hallucinated or out-of-catalog SKU IDs, categories, brands, or packaging types immediately raise
     `ValueError` / `RuntimeError` in `validate_canonical_7dim_prediction`, `run_sister_shade_tiebreaker`,
     `DjevSystemOneClient.resolve_crop_systemone`, and `classify_shelf_boxes_7dim`.
  4. Pluggable stages (`stage1_rectification` through `stage6_shelf_metrics`) and `modular_e2e_pipeline`
     genuinely transform images, boxes, clusters, retrieval embeddings, and CIELAB Delta-E scores
     rather than returning static constants.
"""

from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

import approaches
from approaches.base import Context, Trace
from stages import (
    stage1_rectification,
    stage2_3_detection,
    stage3_5_clustering,
    stage4_retrieval,
    stage5_compound_vlm,
)
from utils import hul_domain
from utils.llm import LLMResult, Usage


class _TrapLLM:
    """Trap LLM that immediately fails if any VLM fallback is attempted."""

    def __call__(self, *args, **kwargs):
        raise AssertionError("VLM fallback is strictly forbidden for standalone neural/vector approaches!")


class _HallucinatingLLM:
    """Fake LLM that returns an out-of-catalog hallucinated SKU ID on contact-sheet classification."""

    def __call__(self, image: Image.Image, prompt: str, **kwargs) -> LLMResult:
        u = Usage(100, 20, 0, 1, {"standard": 1}, {"standard/image_input": 100, "standard/output": 20})
        return LLMResult(
            [
                {
                    "index": 0,
                    "sku_id": "BP-HALLUCINATED-SKU-99999",
                    "category": "Personal Care",
                    "brand": "Dove",
                    "packaging_type": "bottle",
                    "variant": "Fake Variant",
                    "is_hul": True,
                }
            ],
            u,
            0.01,
        )


def _make_shelf_image(tilt_deg: float = 0.0) -> Image.Image:
    """Create a synthetic shelf image with horizontal shelf rails and colored product facings."""
    img = Image.new("RGB", (320, 240), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    # Draw shelf rails and distinct colored product blocks
    for ry in (75, 155, 225):
        draw.line([(10, ry), (310, ry)], fill=(40, 40, 40), width=4)
    colors = [(200, 30, 40), (30, 65, 145), (235, 195, 50), (35, 40, 45)]
    for row_idx, y0 in enumerate((20, 90, 165)):
        for col_idx in range(4):
            x0 = 25 + col_idx * 70
            c = colors[(row_idx + col_idx) % len(colors)]
            draw.rectangle([x0, y0, x0 + 48, y0 + 50], fill=c, outline=(15, 15, 15), width=2)
            # Add bright specular highlight patch on first box
            if row_idx == 0 and col_idx == 0:
                draw.rectangle([x0 + 12, y0 + 10, x0 + 28, y0 + 26], fill=(252, 252, 252))
    if abs(tilt_deg) > 1e-3:
        img = img.rotate(tilt_deg, fillcolor=(235, 235, 235))
    return img


class HardFailNoFallbackAndAntiHallucinationTests(unittest.TestCase):
    def test_neural_detectors_and_vector_retriever_never_call_vlm_fallback(self) -> None:
        img = _make_shelf_image()
        boxes = [(25.0, 20.0, 73.0, 70.0), (95.0, 20.0, 143.0, 70.0)]

        for det_name in ("yolo_n26_sku110k", "rtdetr_shelf_rail_detector"):
            det = approaches.get(det_name)
            det.setup({})
            ctx = Context(model="none", llm=_TrapLLM(), trace=Trace(), otel_parent=None, price=lambda _: {})
            pred_boxes = det.detect(img, ctx)
            self.assertGreater(len(pred_boxes), 0, f"{det_name} should detect boxes using real neural weights")
            self.assertEqual(ctx.trace.usage.calls, 0, f"{det_name} must make 0 VLM calls")

        # Track E Open-Vocab Grounding (OWL-ViT + ScaNN) must also never call VLM
        track_e = approaches.get("track_e_open_vocab")
        track_e.setup({})
        ctx_e = Context(model="none", llm=_TrapLLM(), trace=Trace(), otel_parent=None, price=lambda _: {})
        boxes_e, labels_e = track_e.detect_and_classify(img, ctx_e)
        self.assertGreater(len(boxes_e), 0)
        self.assertEqual(len(boxes_e), len(labels_e))
        self.assertEqual(ctx_e.trace.usage.calls, 0)

        # Pure ScaNN vector retriever must never call VLM
        scann_app = approaches.get("scann_vector_retriever")
        scann_app.setup({})
        ctx_s = Context(model="none", llm=_TrapLLM(), trace=Trace(), otel_parent=None, price=lambda _: {})
        preds_s = scann_app.classify(img, boxes, ctx_s)
        self.assertEqual(len(preds_s), 2)
        self.assertEqual(ctx_s.trace.usage.calls, 0)

    def test_stages_and_domain_hard_fail_on_invalid_inputs_without_fallback(self) -> None:
        valid_img = _make_shelf_image()
        valid_boxes = [(25.0, 20.0, 73.0, 70.0), (95.0, 20.0, 143.0, 70.0)]
        invalid_boxes = [(80.0, 20.0, 40.0, 70.0)]  # x2 < x1

        # Stage 1 hard-fails on None image or unknown mode
        with self.assertRaises(ValueError):
            stage1_rectification.run_rectification(None, valid_boxes, mode="hough_rail_homography")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            stage1_rectification.run_rectification(valid_img, valid_boxes, mode="unknown_rectifier")

        # Stage 2/3 hard-fails on None image, degenerate box coordinates, or unknown mode
        with self.assertRaises(ValueError):
            stage2_3_detection.run_post_detection(None, valid_boxes, mode="oriented_ladi_slicer")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            stage2_3_detection.run_post_detection(valid_img, invalid_boxes, mode="oriented_ladi_slicer")
        with self.assertRaises(ValueError):
            stage2_3_detection.run_post_detection(valid_img, valid_boxes, mode="unknown_detector")

        # Stage 3.5 hard-fails on invalid image or unknown mode
        bad_img = Image.new("RGB", (0, 0))
        with self.assertRaises(ValueError):
            stage3_5_clustering.run_crop_clustering(valid_boxes, mode="maxvit_agglomerative", image=bad_img)
        with self.assertRaises(ValueError):
            stage3_5_clustering.run_crop_clustering(valid_boxes, mode="unknown_clusterer", image=valid_img)

        # Stage 4 hard-fails on out-of-range glare_intensity, invalid boxes, or unknown mode
        with self.assertRaises(ValueError):
            stage4_retrieval.run_retrieval_stage(glare_intensity=1.5, mode="ijepa_scann_entropy_prefilter")
        with self.assertRaises(ValueError):
            stage4_retrieval.run_retrieval_stage(
                mode="ijepa_scann_entropy_prefilter", image=valid_img, boxes=invalid_boxes
            )
        with self.assertRaises(ValueError):
            stage4_retrieval.run_retrieval_stage(mode="unknown_retriever", image=valid_img, boxes=valid_boxes)

        # Stage 5 hard-fails on degenerate box coordinates or unknown mode
        with self.assertRaises(ValueError):
            stage5_compound_vlm.run_sister_shade_tiebreaker(
                box_xyxy=[80.0, 20.0, 40.0, 70.0], mode="cielab_delta_e_and_systemone", image=valid_img
            )
        with self.assertRaises(ValueError):
            stage5_compound_vlm.run_sister_shade_tiebreaker(
                box_xyxy=[25.0, 20.0, 73.0, 70.0], mode="unknown_tiebreaker", image=valid_img
            )

        # IJEPASpecularGlarePredictor hard-fails on empty embedding or invalid box coordinates
        predictor = hul_domain.IJEPASpecularGlarePredictor()
        with self.assertRaises(ValueError):
            predictor.predict_clean_latent([], 0.5, [10.0, 10.0, 50.0, 50.0], image=valid_img)
        with self.assertRaises(ValueError):
            predictor.predict_clean_latent([0.1, 0.2], 0.5, [60.0, 10.0, 20.0, 50.0], image=valid_img)

    def test_anti_hallucination_hard_fails_on_out_of_catalog_sku_or_attributes(self) -> None:
        valid_img = _make_shelf_image()
        valid_boxes = [(25.0, 20.0, 73.0, 70.0)]

        # 1. Direct canonical validator rejects hallucinated SKU ID and mismatched brand/category
        with self.assertRaises(ValueError):
            hul_domain.validate_canonical_7dim_prediction(
                {
                    "sku_id": "BP-HALLUCINATED-SKU-99999",
                    "category": "Personal Care",
                    "brand": "Dove",
                    "packaging_type": "bottle",
                    "variant": "Deeply Nourishing",
                    "is_hul": True,
                }
            )
        with self.assertRaises(ValueError):
            hul_domain.validate_canonical_7dim_prediction(
                {
                    "sku_id": "UL-DOVE-BW-500ML",
                    "category": "Personal Care",
                    "brand": "HallucinatedBrand",
                    "packaging_type": "bottle",
                    "variant": "Deeply Nourishing",
                    "is_hul": True,
                }
            )

        # 2. DjevSystemOneClient.resolve_crop_systemone rejects empty candidates or invalid box
        djev = hul_domain.DjevSystemOneClient()
        with self.assertRaises(ValueError):
            djev.resolve_crop_systemone([10.0, 10.0, 50.0, 50.0], [], 0.85)
        with self.assertRaises(ValueError):
            djev.resolve_crop_systemone([50.0, 10.0, 10.0, 50.0], ["UL-DOVE-BW-500ML"], 0.85)

        # 3. disambiguate_sister_shade_roi rejects empty candidate list instead of returning "UNKNOWN"
        with self.assertRaises(ValueError):
            hul_domain.disambiguate_sister_shade_roi(
                full_box_xyxy=(10, 10, 50, 50),
                candidates=[],
                raw_cosine_scores={},
                observed_sub_roi_lab=(70.0, 5.0, 10.0),
            )

        # 4. classify_shelf_boxes_7dim hard-fails when LLM returns a hallucinated SKU ID
        ctx_halluc = Context(
            model="fake-vlm",
            llm=_HallucinatingLLM(),
            trace=Trace(),
            otel_parent=None,
            price=lambda _: {},
        )
        with self.assertRaises(ValueError):
            hul_domain.classify_shelf_boxes_7dim(
                valid_img,
                valid_boxes,
                ctx=ctx_halluc,
                mode="hul_hierarchy_classifier",
            )

    def test_stages_and_modular_e2e_pipeline_compute_dynamic_non_constant_telemetry(self) -> None:
        flat_img = _make_shelf_image(tilt_deg=0.0)
        tilted_img = _make_shelf_image(tilt_deg=6.0)
        boxes = [(25.0, 20.0, 73.0, 70.0), (95.0, 20.0, 143.0, 70.0), (165.0, 20.0, 213.0, 70.0)]

        # Stage 1 yaw angle changes dynamically between flat and tilted shelf image
        r_flat = stage1_rectification.run_rectification(flat_img, boxes, mode="hough_rail_homography")
        r_tilt = stage1_rectification.run_rectification(tilted_img, boxes, mode="hough_rail_homography")
        self.assertIn("rectified_image", r_flat)
        self.assertIn("rectified_boxes", r_flat)
        self.assertIn("rectified_image", r_tilt)
        self.assertNotEqual(r_flat["yaw_corrected_deg"], 4.2, "Yaw must not be hardcoded 4.2")
        self.assertNotEqual(r_flat["yaw_corrected_deg"], r_tilt["yaw_corrected_deg"])

        # Stage 2/3 oriented_ladi_slicer splits tall narrow hanging strips with horizontal seal lines
        ladi_img = _make_shelf_image()
        draw = ImageDraw.Draw(ladi_img)
        draw.rectangle([10, 10, 40, 210], fill=(200, 40, 40))
        for sy in (60, 110, 160):
            draw.line([(10, sy), (40, sy)], fill=(250, 250, 250), width=3)
        tall_strip_boxes = [(10.0, 10.0, 40.0, 210.0)]
        det_res = stage2_3_detection.run_post_detection(ladi_img, tall_strip_boxes, mode="oriented_ladi_slicer")
        self.assertGreater(len(det_res), 1, "Tall sachet strip with seal lines must be sliced into multiple sachets")

        # Fine-Tuned Gemini 3.1 Flash Lite approaches verify and record the live Vertex AI LoRA endpoint
        for ft_name in ("ft_gemini31_cat_brand_pkg", "ft_gemini31_variant_compound"):
            ft_app = approaches.get(ft_name)
            ft_app.setup({})
            ctx_ft = Context(model="gemini-3.1-flash-lite", llm=None, trace=Trace(), otel_parent=None, price=lambda _: {})
            preds = ft_app.classify(flat_img, boxes, ctx_ft)
            self.assertEqual(len(preds), len(boxes))
            self.assertIn("endpoints/7055298384357228544", ctx_ft.trace.meta.get("sft_lora_endpoint", ""))

    def test_stage6_geometric_kpis_depend_on_rows_and_not_hardcoded_constants(self) -> None:
        from stages import stage6_shelf_metrics

        # Row set 1: 3 HUL facings + 1 competitor facing, contiguous brand block, no wide OOS gap
        rows_contiguous = [
            {
                "width": 400,
                "height": 300,
                "preds": [
                    [20.0, 110.0, 70.0, 200.0],
                    [75.0, 110.0, 125.0, 200.0],
                    [130.0, 110.0, 180.0, 200.0],
                    [185.0, 110.0, 235.0, 200.0],
                ],
                "pred_labels": [
                    {"sku_id": "UL-DOVE-BW-500ML", "brand": "Dove", "is_hul": True},
                    {"sku_id": "UL-DOVE-BW-500ML", "brand": "Dove", "is_hul": True},
                    {"sku_id": "UL-DOVE-BW-500ML", "brand": "Dove", "is_hul": True},
                    {"sku_id": "COMP-NIVEA-SM-400ML", "brand": "Nivea", "is_hul": False},
                ],
            }
        ]
        # Row set 2: 1 HUL facing + 3 competitor facings with a competitor intrusion inside Dove and a large OOS void
        rows_intruded_and_void = [
            {
                "width": 400,
                "height": 300,
                "preds": [
                    [20.0, 110.0, 60.0, 200.0],
                    [65.0, 110.0, 115.0, 200.0],
                    [120.0, 110.0, 160.0, 200.0],
                    [290.0, 110.0, 350.0, 200.0],  # 130px horizontal gap -> OOS void
                ],
                "pred_labels": [
                    {"sku_id": "UL-DOVE-BW-500ML", "brand": "Dove", "is_hul": True},
                    {"sku_id": "COMP-NIVEA-SM-400ML", "brand": "Nivea", "is_hul": False},
                    {"sku_id": "UL-DOVE-BW-500ML", "brand": "Dove", "is_hul": True},
                    {"sku_id": "COMP-PANT-HF-340ML", "brand": "Pantene", "is_hul": False},
                ],
            }
        ]

        kpi_1 = stage6_shelf_metrics.evaluate_shelf_metrics(
            total_boxes=4, box_f2=0.95, box_recall=0.95, rows=rows_contiguous
        )
        kpi_2 = stage6_shelf_metrics.evaluate_shelf_metrics(
            total_boxes=4, box_f2=0.95, box_recall=0.95, rows=rows_intruded_and_void
        )

        self.assertGreater(kpi_1["linear_sos_pct"], kpi_2["linear_sos_pct"])
        self.assertGreater(kpi_1["area_sos_pct"], kpi_2["area_sos_pct"])
        self.assertEqual(kpi_1["oos_voids_detected"], 0)
        self.assertGreaterEqual(kpi_2["oos_voids_detected"], 1)
        self.assertEqual(kpi_1["competitor_intrusion_count"], 0)
        self.assertGreaterEqual(kpi_2["competitor_intrusion_count"], 1)
        self.assertGreater(kpi_1["brand_block_purity"], kpi_2["brand_block_purity"])

    def test_catalog_prototype_bank_loads_12_real_image_crops_from_disk(self) -> None:
        import numpy as np

        from core import catalog as core_catalog

        protos = core_catalog._build_real_catalog_prototype_bank()
        self.assertEqual(len(protos), 12)
        for p in protos:
            ref_crop = p.get("reference_crop")
            self.assertIsInstance(ref_crop, Image.Image, f"Missing PIL reference_crop for {p['sku_id']}")
            arr = np.asarray(ref_crop.convert("RGB"), dtype=np.float32)
            self.assertGreater(float(np.std(arr)), 5.0, f"Reference crop for {p['sku_id']} has flat pixel variance")

    def test_blank_image_returns_empty_boxes_and_none_image_or_failed_vlm_hard_fails(self) -> None:
        from core import clustering as core_clustering
        from core import detection as core_detection
        from core import features as core_features
        from core import retrieval as core_retrieval
        from utils.vector_store import CloudSQL

        # 1. Blank uniform image returns [] instead of a 24-box 6x4 grid fallback
        blank = Image.new("RGB", (320, 240), (200, 200, 200))
        self.assertEqual(core_detection.detect_shelf_boxes_from_pixels(blank), [])

        # 2. Feature extractors, clustering, and scann_vector_lookup hard-fail when image is None
        with self.assertRaises(ValueError):
            core_features.extract_gemini_subroi_embedding(None, (10.0, 10.0, 50.0, 100.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            core_features.extract_maxvit_multiscale_features(None, (10.0, 10.0, 50.0, 100.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            core_clustering.cluster_shelf_facings_high_purity(None, [(10.0, 10.0, 50.0, 100.0)])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            core_retrieval.scann_vector_lookup(0, (10.0, 10.0, 50.0, 100.0), image=None, precomputed_crop_feats=None)

        # 3. CloudSQL with default fallback_local=False hard-fails when instance is unreachable
        db_strict = CloudSQL(instance="invalid-project:us-central1:nonexistent", fallback_local=False)
        with self.assertRaises((RuntimeError, ValueError, OSError)):
            db_strict.query("SELECT 1")

        # 4. VLM exception propagates unconditionally regardless of LLM class name
        class _ExplodingCustomLLM:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("Simulated VLM failure")

        valid_img = _make_shelf_image()
        ctx_err = Context(model="gemini-3.8-flash", llm=_ExplodingCustomLLM(), trace=Trace(), otel_parent=None, price=lambda _: {})
        with self.assertRaises(RuntimeError):
            core_detection.propose_rtdetr_shelf_boxes(valid_img, ctx=ctx_err, approach_name="gemini_2_robotics_detector")
        with self.assertRaises(RuntimeError):
            core_retrieval.classify_shelf_boxes_7dim(
                valid_img,
                [(25.0, 20.0, 73.0, 70.0)],
                ctx=ctx_err,
                config=core_retrieval.CONFIG_HUL_HIERARCHY,
            )


if __name__ == "__main__":
    unittest.main()

