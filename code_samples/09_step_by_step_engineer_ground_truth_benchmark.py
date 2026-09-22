#!/usr/bin/env python3
"""Sample 09: Connect ground truth to your own approach and read the scores.

Run it (no GCP project, no credentials, no network):

    .venv/bin/python code_samples/09_step_by_step_engineer_ground_truth_benchmark.py

It uses the annotation file that ships with the repo, `configs/sample_ground_truth.json`, which
is the canonical worked example of the suite-native schema (see `docs/GROUND_TRUTH_CONTRACT.md`).
That file describes four annotated units on one shelf image: three front-row facings and one
`back_row: true` unit, which the loader excludes, so the ground-truth facing count is 3.

The predictions below are deliberately imperfect so the scores are worth reading:

  facing 1  Dove      box exactly right, labels exactly right
  facing 2  Pond's    box shifted right by 10/1000 of the image width, labels right
  facing 3  Vaseline  box exactly right, brand wrong ("Nivea")

Expected result: detection precision, recall and F1 all 1.0 (three predictions, three matches),
while brand accuracy is 2/3. That separation is the point of the pairing algorithm: boxes are
matched on geometry only, so a naming mistake can never be disguised as a detection mistake.
See `docs/EVALUATION_PROTOCOL.md`.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

from shelf_benchmark import register_approach_function
from shelf_benchmark.approaches import CommonLayerContext
from shelf_benchmark.testing import OFFLINE_IMAGE_URI, make_offline_sdk

SAMPLE_GT_FILE = "configs/sample_ground_truth.json"


def fmt(value: Optional[float]) -> str:
    """Render a metric so `None` (not measured) can never be misread as 0.0 (measured, wrong)."""
    return "None (not measured)" if value is None else f"{value:.4f}"


# STEP 1: register your approach. One decorated function is enough.
@register_approach_function(
    approach_id="engineer_onboarding_demo_approach",
    display_name="Engineer Onboarding Demo: detector + depth NMS",
    category="two_stage_vlm",
)
def run_engineer_demo_approach(
    ctx: CommonLayerContext,
    model_name: str,
    shelf_image_uri: str,
) -> List[Dict[str, Any]]:
    raw_candidates = [
        {
            "bbox_2d": [125, 83, 625, 250],
            "brand": "Dove",
            "product_name": "Dove Deeply Nourishing Body Wash",
            "category": "Personal Care",
            "subcategory": "Skin Cleansing",
            "variant": "Deeply Nourishing",
            "packaging_type": "bottle",
            "pack_type": "Single",
            "size": "500ml",
            "matched_sku_id": "UL-DV-BW-500",
            "shelf_row": "top",
            "confidence": 0.98,
            "_input_tokens": 210,
            "_thinking_tokens": 20,
            "_output_tokens": 80,
        },
        {
            # Same product, box 10 units to the right. IoU 0.887, so still a match at 0.50.
            "bbox_2d": [125, 393, 625, 560],
            "brand": "Pond's",
            "product_name": "Pond's Bright Beauty Face Cream",
            "category": "Personal Care",
            "subcategory": "Skin Care",
            "variant": "Bright Beauty",
            "packaging_type": "jar",
            "pack_type": "Single",
            "size": "50g",
            "matched_sku_id": "UL-PD-FC-050",
            "shelf_row": "top",
            "confidence": 0.91,
        },
        {
            # Box perfect, brand wrong. Counts as a detection true positive and a brand error.
            "bbox_2d": [125, 683, 625, 850],
            "brand": "Nivea",
            "product_name": "Nivea Body Lotion",
            "category": "Personal Care",
            "subcategory": "Skin Care",
            "variant": "Intensive Care",
            "packaging_type": "bottle",
            "pack_type": "Single",
            "size": "400ml",
            "shelf_row": "top",
            "confidence": 0.74,
        },
    ]
    front_facings, depth_filtered = ctx.deduplicate_depth_stacked_facings(raw_candidates)
    if front_facings:
        front_facings[0]["_depth_filtered"] = depth_filtered
    return front_facings


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="shelf-engineer-gt-"))
    sdk = make_offline_sdk(work)

    # STEP 2: connect the annotations. One call, no code change anywhere else.
    # `strict=True` makes a broken join raise instead of quietly scoring nothing.
    summary_gt = sdk.connect_ground_truth(
        provider_type="json",
        source_uri=SAMPLE_GT_FILE,
        gt_version="sample-v1",
        strict=True,
    )
    print("=== ground truth connected ===")
    print(f"  file                : {SAMPLE_GT_FILE}")
    print(f"  images parsed       : {summary_gt['images']}")
    print(f"  items parsed        : {summary_gt['items']}")
    print(f"  back-row excluded   : {summary_gt['excluded_back_row']}")
    print(f"  gt_version          : {summary_gt['gt_version']}")

    # STEP 3: run. The image key in the annotation file ends in `shelf_sample_01.png`, and so does
    # the bundled fixture path, so the provider's basename alias lookup joins them.
    summary = sdk.run_suite(
        models=["gemini-3.8-flash"],  # Recorded on the row; this approach does not call a model.
        tasks=["classification"],
        approaches=["engineer_onboarding_demo_approach"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )

    res = summary["results"][0]
    acc = res.accuracy
    print("\n=== scored against ground truth ===")
    print(f"  approach            : {res.separation_approach}")
    print(f"  accuracy_status     : {acc.accuracy_status}")
    print(f"  gt_version          : {acc.gt_version}")
    print(f"  iou_threshold       : {acc.iou_threshold}")
    print(f"  predicted / gt      : {acc.predicted_count} / {acc.ground_truth_count}")
    print(f"  TP / FP / FN        : {acc.true_positives} / {acc.false_positives} / {acc.false_negatives}")
    print(f"  detection_precision : {fmt(acc.detection_precision)}")
    print(f"  detection_recall    : {fmt(acc.detection_recall)}")
    print(f"  detection_f1        : {fmt(acc.detection_f1)}")
    print(f"  mean_iou_matched    : {fmt(acc.mean_iou_matched)}")
    print(f"  count_accuracy      : {fmt(acc.count_accuracy)}")
    print(f"  brand accuracy      : {fmt(acc.brand_classification_accuracy)}")
    print(f"  product accuracy    : {fmt(acc.product_classification_accuracy)}")
    print(f"  brand_set_recall    : {fmt(acc.brand_set_recall)}")
    print(f"  depth filtered      : {acc.depth_duplicates_filtered}")
    print(f"  cost / shelf image  : ${res.cost.cost_per_shelf_image_usd:.6f}")
    print(f"  markdown report     : {summary['artifacts']['markdown_report']}")
    print(
        "\nDetection is perfect and brand accuracy is not, because the third facing was found in\n"
        "the right place and given the wrong name. Metrics that were never measured print as None,\n"
        "never as 0.0."
    )


if __name__ == "__main__":
    main()
