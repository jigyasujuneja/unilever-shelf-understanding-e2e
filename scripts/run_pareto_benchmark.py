#!/usr/bin/env python3
"""Generate the Goldfish 3 Pareto Matrix across SKU-110k + RPC Open-Source Benchmark Images."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_open_datasets import build_sku110k_rpc_benchmark_slice
from shelf_e2e.datasets import RPCCatalogAdapter
from shelf_e2e.real_world_defenses import run_all_9_real_world_defense_benchmarks
from shelf_e2e.schemas import InputContract, PlanogramContract, PromoRules, StoreMetadata
from shelf_e2e.tracks import (
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


def main() -> None:
    slice_path = build_sku110k_rpc_benchmark_slice()
    slice_data = json.loads(slice_path.read_text(encoding="utf-8"))
    catalog = RPCCatalogAdapter.from_json(REPO_ROOT / "configs" / "mock_rpc_catalog.json")

    images_node = slice_data.get("images", {})
    if isinstance(images_node, list):
        first_img_entry = images_node[0]
        raw_id = first_img_entry.get("image_id", "sku110k_val_000")
        img_name = raw_id if raw_id.endswith((".jpg", ".png")) else f"{raw_id}.jpg"
        gt_records = first_img_entry.get("annotations", [])
    else:
        img_name = "sku110k_val_001.png"
        gt_records = images_node[img_name]

    image_file = REPO_ROOT / "data" / "sku110k" / "images" / img_name
    sample_input = InputContract(
        image_path=str(image_file),
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
        r["box_xyxy"]
        if "box_xyxy" in r
        else [float(r["bbox_2d"][1]), float(r["bbox_2d"][0]), float(r["bbox_2d"][3]), float(r["bbox_2d"][2])]
        for r in gt_records
    ]
    gt_ids = [
        r.get("gt_base_pack_id") or r.get("base_pack_code") or r.get("base_pack_id", "BP-DOVE-BW-750")
        for r in gt_records
    ]
    gt_cats = [r.get("category", "Personal Care") for r in gt_records]

    tracks = [
        TrackACascadingViTPipeline(catalog),
        TrackBEndToEndVLMPipeline(catalog),
        TrackB2TwoStageCropVLMPipeline(catalog, sku110k_slice_path=slice_path),
        TrackCTieredHybridPipeline(catalog, sku110k_slice_path=slice_path),
        TrackDJevRoutingPipeline(
            catalog, use_gemini_diffusion_as_jev=False, sku110k_slice_path=slice_path
        ),
        TrackDJevRoutingPipeline(
            catalog, use_gemini_diffusion_as_jev=True, sku110k_slice_path=slice_path
        ),
        TrackEIJEPALatentWorldModelPipeline(catalog, sku110k_slice_path=slice_path),
        TrackFPaliGemma2LoRAPipeline(catalog, sku110k_slice_path=slice_path),
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
        stress = run_concurrency_stress_test(track.run, sample_input, worker_count=125)
        rows.append(
            {
                "track_id": track.track_id,
                "track_name": track.track_name,
                "dataset_image": image_file.name,
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

    defenses_res = run_all_9_real_world_defense_benchmarks()
    defenses_json = REPO_ROOT / "data" / "results" / "real_world_9defenses_benchmark.json"
    defenses_json.parent.mkdir(parents=True, exist_ok=True)
    defenses_json.write_text(json.dumps(defenses_res, indent=2), encoding="utf-8")

    print("\n=== ALL 8 NEURAL TRACKS PARETO MATRIX ON REAL SKU-110K + RPC BENCHMARK SLICE ===")
    header = (
        f"{'Track ID':<33} | {'mAP@50:95':<9} | {'Top-1':<6} | {'Top-5':<6} | {'MRR':<6} | "
        f"{'P95 (ms)':<8} | {'Cost (₹)':<10} | {'SLA <= ₹0.22'}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sla_badge = "PASS" if r["meets_cost_sla_0_22_inr"] else "EXCEEDS"
        print(
            f"{r['track_id']:<33} | {r['map_50_95']:<9.4f} | {r['top1_acc']:<6.2f} | "
            f"{r['top5_recall']:<6.2f} | {r['mrr']:<6.2f} | {r['p95_latency_ms']:<8.1f} | "
            f"₹{r['cost_inr']:<9.4f} | {sla_badge}"
        )
    print(f"\nSaved JSON Pareto report to: {report_json}")
    print("\n=== 9 REAL-WORLD DEFENSE LAYERS STRESS BENCHMARK SUMMARY ===")
    print(json.dumps(defenses_res["aggregate_stress_benchmark_summary"], indent=2))
    print(f"Saved 9-Defense Stress Benchmark to: {defenses_json}")


if __name__ == "__main__":
    main()

