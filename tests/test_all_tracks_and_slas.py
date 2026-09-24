"""End-to-end verification across Tracks A, B, C, and D (+ GeminiDiffusion-as-Jev)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.schemas import (
    ComplianceStatus,
    InputContract,
    PlanogramContract,
    PromoRules,
    StoreMetadata,
)
from shelf_e2e.tracks import (
    TrackACascadingViTPipeline,
    TrackBEndToEndVLMPipeline,
    TrackCTieredHybridPipeline,
    TrackDJevRoutingPipeline,
)
from tests.benchmark_harness import (
    calculate_json_schema_adherence,
    calculate_map50_and_map50_95,
    calculate_retrieval_and_classification_metrics,
)


def _load_catalog() -> RPCCatalogAdapter:
    return RPCCatalogAdapter.from_json(REPO_ROOT / "configs" / "mock_rpc_catalog.json")


def _sample_contract() -> InputContract:
    return InputContract(
        image_path="gs://unilever-shelf-images/mt_shelf_4k_01.png",
        store_metadata=StoreMetadata(
            store_id="MT-BLR-108",
            channel="MODERN_TRADE",
            planogram_id="PLANO-DOVE-TRES-2026",
        ),
        planogram_contract=PlanogramContract(
            target_skus=["BP-DOVE-BW-750", "BP-TRES-SH-750"],
            promo_rules=PromoRules(toker_text="20% Extra", min_display_count=2),
        ),
    )


def test_all_tracks_adhere_to_universal_schema_and_pareto_tradeoffs() -> None:
    catalog = _load_catalog()
    contract = _sample_contract()

    tracks = [
        TrackACascadingViTPipeline(catalog),
        TrackBEndToEndVLMPipeline(catalog),
        TrackCTieredHybridPipeline(catalog),
        TrackDJevRoutingPipeline(catalog, use_gemini_diffusion_as_jev=False),
        TrackDJevRoutingPipeline(catalog, use_gemini_diffusion_as_jev=True),
    ]

    gt_boxes = [
        [40.0, 80.0, 120.0, 290.0],
        [125.0, 80.0, 205.0, 290.0],
        [215.0, 75.0, 295.0, 295.0],
        [310.0, 90.0, 385.0, 290.0],
    ]
    gt_ids = ["BP-DOVE-BW-750", "BP-DOVE-BW-750", "BP-TRES-SH-750", "BP-COMP-SH-650"]
    gt_cats = ["Skin Cleansing", "Skin Cleansing", "Hair Care", "Hair Care"]

    outputs_by_id = {}
    for track in tracks:
        out = track.run(contract)
        outputs_by_id[track.track_id] = out
        assert calculate_json_schema_adherence([out.model_dump()]) == 1.0
        assert out.metrics.latency_ms.total_e2e <= 20000.0
        assert out.compliance.toker_status == ComplianceStatus.COMPLIANT

        det_kpis = calculate_map50_and_map50_95(
            pred_boxes=[s.box_xyxy for s in out.marketshare.resolved_skus],
            pred_scores=[s.confidence for s in out.marketshare.resolved_skus],
            gt_boxes=gt_boxes,
        )
        assert det_kpis.map_50 >= 0.90

        ret_kpis = calculate_retrieval_and_classification_metrics(
            resolved_skus=out.marketshare.resolved_skus,
            gt_boxes=gt_boxes,
            gt_base_pack_ids=gt_ids,
            gt_categories=gt_cats,
            master_catalog_ids=catalog.valid_base_pack_ids(),
        )
        assert ret_kpis.hallucination_rate == 0.0

    # Verify Track C (Tiered Hybrid) and Track D (Jev Routing) meet the <= ₹0.22 hard cost SLA
    assert outputs_by_id["track_c_tiered_hybrid"].metrics.estimated_cost_inr <= 0.22
    assert outputs_by_id["track_d_jev_routing"].metrics.estimated_cost_inr <= 0.22
    assert outputs_by_id["track_d_gemini_diffusion_as_jev"].metrics.estimated_cost_inr <= 0.22


class TestAllTracksAndSLAs(unittest.TestCase):
    def test_all_tracks(self) -> None:
        test_all_tracks_adhere_to_universal_schema_and_pareto_tradeoffs()


if __name__ == "__main__":
    unittest.main(verbosity=2)
