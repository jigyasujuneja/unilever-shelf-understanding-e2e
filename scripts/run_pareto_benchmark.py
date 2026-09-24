#!/usr/bin/env python3
"""Generate the Goldfish 3 Pareto Matrix comparing Tracks A, B, C, and D (+ GeminiDiffusion-as-Jev)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.schemas import InputContract, PlanogramContract, PromoRules, StoreMetadata
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
    run_concurrency_stress_test,
)


def main() -> None:
    catalog = RPCCatalogAdapter.from_json(REPO_ROOT / "configs" / "mock_rpc_catalog.json")
    sample_input = InputContract(
        image_path="gs://unilever-shelf-images/mt_shelf_4k_01.png",
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

    gt_boxes = [
        [40.0, 80.0, 120.0, 290.0],
        [125.0, 80.0, 205.0, 290.0],
        [215.0, 75.0, 295.0, 295.0],
        [310.0, 90.0, 385.0, 290.0],
    ]
    gt_ids = ["BP-DOVE-BW-750", "BP-DOVE-BW-750", "BP-TRES-SH-750", "BP-COMP-SH-650"]
    gt_cats = ["Skin Cleansing", "Skin Cleansing", "Hair Care", "Hair Care"]

    tracks = [
        TrackACascadingViTPipeline(catalog),
        TrackBEndToEndVLMPipeline(catalog),
        TrackCTieredHybridPipeline(catalog),
        TrackDJevRoutingPipeline(catalog, use_gemini_diffusion_as_jev=False),
        TrackDJevRoutingPipeline(catalog, use_gemini_diffusion_as_jev=True),
    ]

    rows = []
    for track in tracks:
        out = track.run(sample_input)
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
            master_catalog_ids=catalog.valid_base_pack_ids(),
        )
        stress = run_concurrency_stress_test(track.run, sample_input, worker_count=525)
        rows.append(
            {
                "track_id": track.track_id,
                "track_name": track.track_name,
                "map_50": det.map_50,
                "map_50_95": det.map_50_95,
                "top1_acc": ret.top1_accuracy,
                "top5_recall": ret.top5_recall,
                "mrr": ret.mrr,
                "f1_score": ret.f1_score,
                "hallucination_rate": ret.hallucination_rate,
                "schema_adherence": calculate_json_schema_adherence([out.model_dump()]),
                "sos_pct": out.metrics.share_of_shelf_pct,
                "p95_latency_ms": stress["p95_ms"],
                "cost_inr": out.metrics.estimated_cost_inr,
                "meets_cost_sla_0_22_inr": out.metrics.estimated_cost_inr <= 0.22,
                "meets_latency_sla_20s": stress["p95_ms"] <= 20000.0,
            }
        )

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_json = reports_dir / "pareto_matrix.json"
    report_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== SPEC-001 & SPEC-002 PARETO BENCHMARK MATRIX (500K IMAGES/DAY) ===")
    header = (
        f"{'Track ID':<33} | {'mAP@50:95':<9} | {'Top-1':<6} | {'MRR':<6} | "
        f"{'P95 (ms)':<8} | {'Cost (₹)':<10} | {'SLA <= ₹0.22'}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sla_badge = "PASS" if r["meets_cost_sla_0_22_inr"] else "EXCEEDS"
        print(
            f"{r['track_id']:<33} | {r['map_50_95']:<9.4f} | {r['top1_acc']:<6.2f} | "
            f"{r['mrr']:<6.2f} | {r['p95_latency_ms']:<8.1f} | ₹{r['cost_inr']:<9.4f} | {sla_badge}"
        )
    print(f"\nSaved JSON Pareto report to: {report_json}")


if __name__ == "__main__":
    main()
