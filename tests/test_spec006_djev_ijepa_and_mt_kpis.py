"""SPEC-006 Verification Suite:
1. `mmastrac/djev` (`/v1/systemone` `DiffusionGemma-26B-A4B-it`) 64-token `diffusion_seed_canvas`, `diffusion_pinned`, `diffusion_constrained` (`vllm#58216`), and `depends_on`/`ask_if` DAGs.
2. All 8 Neural Architecture Tracks (`Track A`, `Track B1`, `Track B2`, `Track C`, `Track D1`, `Track D2 [djev]`, `Track E [I-JEPA]`, `Track F [PaliGemma-2-LoRA]`).
3. All 8 Unilever Modern Trade (`MT`) Gondola-Level KPIs (`Linear SOS`, `2D Area SOS`, `OOS Void Gap Detector`, `Planogram Sequence Levenshtein Compliance`, `Brand-Block Purity`).
4. Open-Source Retail Benchmark Manifest (`7` datasets, `298,821` streamable images).
"""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.stream_open_retail_benchmarks import (
    OPEN_RETAIL_DATASETS_CATALOG,
    export_open_retail_datasets_manifest,
)
from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.djev_client import DjevSystemOneClient
from shelf_e2e.ijepa_predictor import IJEPALatentGlarePredictor
from shelf_e2e.mt_gondola_analytics import (
    compute_brand_block_purity,
    compute_linear_and_area_sos,
    compute_planogram_sequence_compliance,
    detect_shelf_oos_void_gaps,
    evaluate_full_mt_gondola_audit,
)
from shelf_e2e.schemas import (
    InputContract,
    PlanogramContract,
    PromoRules,
    ResolvedSKU,
    StoreMetadata,
)
from shelf_e2e.tracks import (
    TrackACascadingViTPipeline,
    TrackB2TwoStageCropVLMPipeline,
    TrackBEndToEndVLMPipeline,
    TrackCTieredHybridPipeline,
    TrackDJevRoutingPipeline,
    TrackEIJEPALatentWorldModelPipeline,
    TrackFPaliGemma2LoRAPipeline,
)


class TestSpec006DjevIjepaAndMTKPIs(unittest.TestCase):
    """End-to-end verification of SPEC-006 neural architectures, djev /v1/systemone, I-JEPA, and MT KPIs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.catalog = RPCCatalogAdapter.from_json(
            cls.repo_root / "configs" / "mock_rpc_catalog.json"
        )
        cls.slice_path = cls.repo_root / "data" / "sku110k" / "sku110k_benchmark_slice.json"
        cls.sample_contract = InputContract(
            image_path=str(cls.repo_root / "data" / "sku110k" / "images" / "sku110k_val_000.jpg"),
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

    def test_djev_64token_canvas_and_vllm58216_constrained_vocab(self) -> None:
        client = DjevSystemOneClient()
        top5 = ["BP-DOVE-BW-500", "BP-DOVE-BW-750", "BP-TRES-SH-750"]
        canvas = client.build_djev_64token_canvas(
            box_xyxy=[100.0, 200.0, 148.0, 340.0],
            scann_top5=top5,
            ocr_snippet="Dove Deep Moisture 750ml",
            glare_intensity=0.42,
        )
        self.assertEqual(len(canvas.diffusion_seed_canvas), 64)
        self.assertEqual(len(canvas.diffusion_pinned), 64)
        self.assertEqual(canvas.diffusion_constrained["slot_15_sku"], top5)

        resp = client.resolve_crop_systemone(
            box_xyxy=[100.0, 200.0, 148.0, 340.0],
            scann_top5=top5,
            raw_similarity=0.94,
            glare_intensity=0.42,
            ocr_snippet="Dove Deep Moisture 750ml",
        )
        self.assertEqual(resp.resolved_base_pack_id, "BP-DOVE-BW-750")
        self.assertIn(resp.resolved_base_pack_id, top5)
        self.assertGreater(resp.pinned_ratio, 0.85)
        self.assertNotIn("<|pad|>", resp.denoised_canvas_tokens)

    def test_ijepa_latent_glare_predictor_unit_norm_and_gain(self) -> None:
        predictor = IJEPALatentGlarePredictor()
        emb = [0.25] * 16
        res = predictor.predict_clean_latent(
            corrupted_embedding=emb,
            glare_intensity=0.45,
            box_xyxy=[10.0, 20.0, 58.0, 160.0],
        )
        norm_sq = sum(v * v for v in res.reconstructed_embedding)
        self.assertAlmostEqual(norm_sq, 1.0, places=3)
        self.assertGreater(res.latent_cosine_gain, 0.02)
        self.assertLess(res.predictor_latency_ms, 2.0)

    def test_all_eight_neural_tracks_execute_and_respect_contracts(self) -> None:
        tracks = [
            TrackACascadingViTPipeline(self.catalog),
            TrackBEndToEndVLMPipeline(self.catalog),
            TrackB2TwoStageCropVLMPipeline(self.catalog, sku110k_slice_path=self.slice_path),
            TrackCTieredHybridPipeline(self.catalog, sku110k_slice_path=self.slice_path),
            TrackDJevRoutingPipeline(
                self.catalog, use_gemini_diffusion_as_jev=False, sku110k_slice_path=self.slice_path
            ),
            TrackDJevRoutingPipeline(
                self.catalog, use_gemini_diffusion_as_jev=True, sku110k_slice_path=self.slice_path
            ),
            TrackEIJEPALatentWorldModelPipeline(self.catalog, sku110k_slice_path=self.slice_path),
            TrackFPaliGemma2LoRAPipeline(self.catalog, sku110k_slice_path=self.slice_path),
        ]
        for track in tracks:
            out = track.run(self.sample_contract)
            self.assertGreater(out.metrics.total_detected, 50, msg=f"Failed on {track.track_id}")
            if track.track_id in (
                "track_c_tiered_hybrid",
                "track_d_jev_routing",
                "track_d_gemini_diffusion_as_jev",
                "track_e_ijepa_world_model",
                "track_f_paligemma2_lora",
            ):
                self.assertLessEqual(
                    out.metrics.estimated_cost_inr,
                    0.22,
                    msg=f"{track.track_id} exceeded ₹0.22 SLA ceiling",
                )

    def test_mt_gondola_8_kpis_linear_area_sos_oos_and_brand_block(self) -> None:
        skus = [
            ResolvedSKU(box_xyxy=[10.0, 100.0, 50.0, 200.0], base_pack_id="BP-DOVE-BW-750", confidence=0.98),
            ResolvedSKU(box_xyxy=[55.0, 100.0, 95.0, 200.0], base_pack_id="BP-COMP-LOREAL-500", confidence=0.95),
            ResolvedSKU(box_xyxy=[100.0, 100.0, 140.0, 200.0], base_pack_id="BP-DOVE-BW-750", confidence=0.97),
            # Physical empty shelf OOS gap between x=140 and x=240 (100px gap >= 1.25 * 40px median width)
            ResolvedSKU(box_xyxy=[240.0, 100.0, 280.0, 200.0], base_pack_id="BP-TRES-SH-750", confidence=0.96),
        ]
        sos = compute_linear_and_area_sos(skus)
        self.assertEqual(sos["facing_count_sos_pct"], 75.0)
        self.assertEqual(sos["linear_width_sos_pct"], 75.0)

        voids = detect_shelf_oos_void_gaps(skus)
        self.assertEqual(len(voids), 1)
        self.assertEqual(voids[0].void_width_px, 100.0)
        self.assertGreaterEqual(voids[0].estimated_missing_facings, 2)

        seq_pct = compute_planogram_sequence_compliance(skus, ["BP-DOVE-BW-750", "BP-TRES-SH-750"])
        self.assertEqual(seq_pct, 100.0)

        purity_pct, intrusions = compute_brand_block_purity(skus)
        self.assertEqual(intrusions, 1)
        self.assertLess(purity_pct, 100.0)

        full_audit = evaluate_full_mt_gondola_audit(skus, ["BP-DOVE-BW-750", "BP-TRES-SH-750"])
        self.assertEqual(len(full_audit.oos_void_gaps), 1)
        self.assertEqual(full_audit.competitor_intrusions_count, 1)

    def test_open_retail_datasets_manifest_exports_seven_datasets(self) -> None:
        manifest_path = export_open_retail_datasets_manifest()
        self.assertTrue(manifest_path.exists())
        self.assertEqual(len(OPEN_RETAIL_DATASETS_CATALOG), 7)

    def test_hul_8stage_e2e_and_dual_slas_marketshare_30s_merchandizing_10s(self) -> None:
        from shelf_e2e.hul_e2e_pipeline import HULEndToEndShelfProcessor

        processor = HULEndToEndShelfProcessor(self.repo_root)
        # 1. HUL Marketshare Workflow: 5-7 images per request, Response time <= 30,000 ms (30s)
        ms_res = processor.execute_workflow("MARKETSHARE", image_count=6)
        self.assertEqual(ms_res.image_count, 6)
        self.assertEqual(ms_res.sla_limit_ms, 30000.0)
        self.assertTrue(ms_res.within_sla)
        self.assertLessEqual(ms_res.actual_total_ms, 30000.0)
        self.assertGreater(ms_res.hul_skus_identified_count, 0)
        self.assertGreater(ms_res.non_hul_competitor_skus_count, 0)
        self.assertGreaterEqual(len(ms_res.recommendations), 3)
        # Verify Classify (5 dims) + Derive (Pack type, Size, Base Pack code)
        first_roi = ms_res.sample_resolved_rois[0]
        self.assertTrue(bool(first_roi.category and first_roi.subcategory and first_roi.brand and first_roi.variant and first_roi.packaging_type))
        self.assertTrue(bool(first_roi.pack_type and first_roi.size and first_roi.base_pack_code))

        # 2. HUL Merchandizing Workflow: 1 image per request, Response time <= 10,000 ms (10s)
        merch_res = processor.execute_workflow("MERCHANDIZING", image_count=1)
        self.assertEqual(merch_res.image_count, 1)
        self.assertEqual(merch_res.sla_limit_ms, 10000.0)
        self.assertTrue(merch_res.within_sla)
        self.assertLessEqual(merch_res.actual_total_ms, 10000.0)

    def test_sister_shade_disambiguator_rescues_lakme_cc_almond_and_honey_from_bronze_collapse(
        self,
    ) -> None:
        from shelf_e2e.sister_shade_disambiguator import (
            SisterCandidateProfile,
            resolve_sister_shade_and_low_f2,
        )

        candidates = [
            SisterCandidateProfile(
                canonical_variant_id="HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Bronze",
                brand="Lakme",
                product_line_cluster="Lakme_9to5_CC",
                shade_or_active_token="Bronze",
                discriminative_sub_roi_rel=(0.15, 0.62, 0.85, 0.88),
                reference_cielab_swatch=(48.0, 14.5, 22.0),
                training_prior_count=314,
                cap_orientation="CAP_DOWN_TUBE",
            ),
            SisterCandidateProfile(
                canonical_variant_id="HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Almond",
                brand="Lakme",
                product_line_cluster="Lakme_9to5_CC",
                shade_or_active_token="Almond",
                discriminative_sub_roi_rel=(0.15, 0.62, 0.85, 0.88),
                reference_cielab_swatch=(72.0, 7.2, 16.5),
                training_prior_count=68,
                cap_orientation="CAP_DOWN_TUBE",
            ),
            SisterCandidateProfile(
                canonical_variant_id="HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Honey",
                brand="Lakme",
                product_line_cluster="Lakme_9to5_CC",
                shade_or_active_token="Honey",
                discriminative_sub_roi_rel=(0.15, 0.62, 0.85, 0.88),
                reference_cielab_swatch=(64.0, 10.5, 26.0),
                training_prior_count=77,
                cap_orientation="CAP_DOWN_TUBE",
            ),
        ]
        # Raw global ViT embeddings are 99.4% identical across sister tubes
        raw_scores = {
            "HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Bronze": 0.942,
            "HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Almond": 0.940,
            "HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Honey": 0.939,
        }
        res = resolve_sister_shade_and_low_f2(
            full_box_xyxy=(100, 200, 240, 520),
            candidates=candidates,
            raw_cosine_scores=raw_scores,
            observed_sub_roi_lab=(71.5, 7.5, 16.2),
            observed_ocr_shade_hint="Almond",
            observed_cap_orientation="CAP_DOWN_TUBE",
        )
        # Baseline collapses into CC_Bronze due to majority prior + global embedding tie
        self.assertEqual(
            res.baseline_winner_id,
            "HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Bronze",
        )
        # 5-Stage Sister-Shade Disambiguator correctly recovers CC_Almond
        self.assertEqual(
            res.resolved_variant_id,
            "HUL__Skin__Skin_Lightening__Lakme__9_To_5_CC_Almond",
        )
        self.assertEqual(res.systemone_pinned_tokens_pct, 89.1)

    def test_coarse_to_fine_3task_and_prefiltered_scann_cascade(self) -> None:
        client = DjevSystemOneClient()
        # 1. HUL SKU: 3-Task (Category | Brand | Packaging) pre-filters 50,000 SKUs -> ~11 sister variants
        hul_res = client.classify_3task_and_prefilter_scann(
            box_xyxy=[20.0, 40.0, 95.0, 220.0],
            hint_category="Hair Care",
            hint_brand="Dove",
            hint_packaging="bottle",
            ocr_snippet="340ml",
        )
        self.assertTrue(hul_res.is_hul_brand)
        self.assertEqual(hul_res.category, "Hair Care")
        self.assertEqual(hul_res.brand, "Dove")
        self.assertEqual(hul_res.packaging_type, "bottle")
        self.assertEqual(hul_res.scann_pool_before_filter, 50000)
        self.assertLessEqual(hul_res.scann_pool_after_3task_filter, 15)
        self.assertGreater(hul_res.scann_pool_after_3task_filter, 0)
        self.assertEqual(hul_res.crop_embedding_dim, 768)

        # 2. Competitor SKU (Pantene): 3-Task completes immediately with 0 catalog cardinality
        comp_res = client.classify_3task_and_prefilter_scann(
            box_xyxy=[110.0, 40.0, 185.0, 220.0],
            hint_category="Hair Care",
            hint_brand="Pantene",
            hint_packaging="bottle",
            ocr_snippet="340ml",
        )
        self.assertFalse(comp_res.is_hul_brand)
        self.assertEqual(comp_res.routing_decision, "COMPETITOR_3TASK_COMPLETE")
        self.assertEqual(comp_res.scann_pool_after_3task_filter, 0)
        self.assertTrue(comp_res.resolved_base_pack_id.startswith("NON-HUL-"))


if __name__ == "__main__":
    unittest.main()


