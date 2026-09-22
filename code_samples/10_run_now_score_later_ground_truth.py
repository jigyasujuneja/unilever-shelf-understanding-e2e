#!/usr/bin/env python3
"""Sample 10: Run now, score later - the ground-truth workflow end to end.

Run it:

    .venv/bin/python code_samples/10_run_now_score_later_ground_truth.py

This is the workflow the whole suite is built around, and the one every engineer on this project
needs first, because **ground truth does not exist yet**.

    Step 1  Run the benchmark today. Costs inference. Produces predictions, latency, tokens and
            cost. Accuracy is reported as `None` with
            `accuracy_status="PLACEHOLDER_AWAITING_GROUND_TRUTH"` -- NOT as 0.0.
    Step 2  Annotations arrive, weeks later.
    Step 3  Re-score the run you already paid for. Costs nothing. Every accuracy metric populates.
    Step 4  Annotations get corrected. Re-score again under a new `gt_version`, so the two result
            sets can never be silently compared.

Everything here runs offline against the bundled fixture image, so it needs no GCP project, no
credentials and no network. Swap `make_offline_sdk` for a real `ShelfBenchmarkSDK` and the fixture
payload for a real model, and the rest of the script is unchanged.

Modelled on `tests/test_ground_truth_workflow.py`.
See `docs/GROUND_TRUTH_CONTRACT.md` and `docs/EVALUATION_PROTOCOL.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.data.ground_truth import create_ground_truth_provider
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.models import AccuracyMetrics
from shelf_benchmark.scoring import score_predictions
from shelf_benchmark.sdk import UniversalModelSpec
from shelf_benchmark.testing import (
    OFFLINE_IMAGE_URI,
    make_offline_sdk,
    perfect_prediction_payload,
    write_sample_ground_truth_file,
)


def _fmt(value: float | None) -> str:
    """Render a metric so that `None` (not measured) never looks like 0.0 (measured, wrong)."""
    return "None (not measured)" if value is None else f"{value:.4f}"


def show(label: str, acc: AccuracyMetrics) -> None:
    print(f"\n  --- {label} ---")
    print(f"    accuracy_status                 : {acc.accuracy_status}")
    print(f"    gt_version                      : {acc.gt_version}")
    print(f"    iou_threshold                   : {acc.iou_threshold}")
    print(f"    predicted / ground truth facings: {acc.predicted_count} / {acc.ground_truth_count}")
    print(f"    TP / FP / FN                    : "
          f"{acc.true_positives} / {acc.false_positives} / {acc.false_negatives}")
    print(f"    detection_precision             : {_fmt(acc.detection_precision)}")
    print(f"    detection_recall                : {_fmt(acc.detection_recall)}")
    print(f"    detection_f1                    : {_fmt(acc.detection_f1)}")
    print(f"    mean_iou_matched                : {_fmt(acc.mean_iou_matched)}")
    print(f"    brand_classification_accuracy   : {_fmt(acc.brand_classification_accuracy)}")
    print(f"    product_classification_accuracy : {_fmt(acc.product_classification_accuracy)}")
    print(f"    count_accuracy                  : {_fmt(acc.count_accuracy)}")


def ground_truth_provider(config: BenchmarkConfig, gt_file: Path, version: str):
    """Build a provider for an annotation file, exactly as `shelf-benchmark score` does."""
    config.ground_truth.provider_type = "json"
    config.ground_truth.source_uri = str(gt_file)
    config.ground_truth.gt_version = version
    storage = StorageManager(
        project_id=config.gcp.project_id,
        bucket_config=config.buckets,
        offline=config.offline.enabled,
    )
    return create_ground_truth_provider(
        gt_config=config.ground_truth,
        storage_manager=storage,
        project_id=config.gcp.project_id,
    )


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="shelf-gt-workflow-"))
    print(f"Working directory: {work}")

    # -----------------------------------------------------------------------
    # STEP 1: Run today, with no annotations in existence.
    # -----------------------------------------------------------------------
    print("\n=== STEP 1: benchmark run with NO ground truth ===")
    sdk = make_offline_sdk(work)
    sdk.register_model(
        UniversalModelSpec(
            model_id="demo-shelf-model",
            display_name="demo-shelf-model",
            provider_family="custom_callable",
            # Stands in for a real Vertex AI call. Returns the fixture's correct answer so the
            # numbers below are checkable by hand.
            custom_handler=lambda prompt, image_uri, schema: perfect_prediction_payload(),
        )
    )
    summary = sdk.run_suite(
        models=["demo-shelf-model"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )

    result = summary["results"][0]
    print(f"  run status        : {result.status}")
    print(f"  facings predicted : {len(result.row_level_items)}")
    print(f"  latency           : {result.latency_ms:.1f} ms")
    print(f"  tokens            : {result.tokens.total_tokens}")
    print(f"  cost / shelf image: ${result.cost.cost_per_shelf_image_usd:.8f} "
          f"(billing_source={result.cost.billing_source}, "
          f"includes_modelled_infrastructure={result.cost.includes_modelled_infrastructure})")
    show("accuracy before annotations exist", result.accuracy)
    print("\n  Note every accuracy metric is None, not 0.0. 'Not measured' and 'measured and")
    print("  wrong' are different claims, and the suite never conflates them.")

    predictions_path = Path(summary["artifacts"]["predictions_json"])
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    print(f"\n  predictions file  : {predictions_path}")
    print(f"  schema_version    : {payload['schema_version']}")
    print(f"  runs stored       : {len(payload['runs'])}")
    print(f"  rows in first run : {len(payload['runs'][0]['rows'])}")
    print("  The file holds raw model output only. No accuracy is baked into it, so a stale")
    print("  score can never leak into a later re-scoring.")

    # -----------------------------------------------------------------------
    # STEP 2: Annotations arrive.
    # -----------------------------------------------------------------------
    print("\n=== STEP 2: annotations arrive (v1) ===")
    gt_v1 = write_sample_ground_truth_file(work / "annotations_v1.json")
    print(f"  wrote {gt_v1}")
    provider_v1 = ground_truth_provider(sdk.config, gt_v1, version="v1")
    print(f"  {provider_v1.describe()}")

    # -----------------------------------------------------------------------
    # STEP 3: Re-score the run you already paid for. Zero inference cost.
    # -----------------------------------------------------------------------
    print("\n=== STEP 3: re-score the existing run against v1 (no inference) ===")
    rescored_v1 = score_predictions(
        predictions_path=predictions_path,
        gt_provider=provider_v1,
        evaluation_config=sdk.config.evaluation,
        gt_version="v1",
    )
    show("accuracy against annotations v1", rescored_v1[0].accuracy)
    print(f"\n  tokens carried through unchanged: "
          f"{rescored_v1[0].tokens.total_tokens} == {result.tokens.total_tokens}")
    print("  Re-scoring does not re-measure latency, tokens or cost. Those were measured at run")
    print("  time and are copied through verbatim.")

    # -----------------------------------------------------------------------
    # STEP 4: A corrected annotation batch. Same predictions, new version label.
    # -----------------------------------------------------------------------
    print("\n=== STEP 4: annotations corrected (v2), same predictions re-scored ===")
    v2_payload = json.loads(gt_v1.read_text(encoding="utf-8"))
    entry = v2_payload[OFFLINE_IMAGE_URI]
    entry["gt_version"] = "v2"
    # The vendor decides the third facing was mislabelled: it is Lakme, not "Lakme".
    entry["items"][2]["brand"] = "Nivea"
    entry["items"][2]["product_name"] = "Nivea Soft Face Wash"
    entry["expected_brands"] = ["Pond's", "Himalaya", "Nivea"]
    gt_v2 = work / "annotations_v2.json"
    gt_v2.write_text(json.dumps(v2_payload, indent=2), encoding="utf-8")

    provider_v2 = ground_truth_provider(sdk.config, gt_v2, version="v2")
    print(f"  {provider_v2.describe()}")
    rescored_v2 = score_predictions(
        predictions_path=predictions_path,
        gt_provider=provider_v2,
        evaluation_config=sdk.config.evaluation,
        gt_version="v2",
    )
    show("accuracy against annotations v2", rescored_v2[0].accuracy)

    acc1, acc2 = rescored_v1[0].accuracy, rescored_v2[0].accuracy
    print(f"\n  Detection is identical ({_fmt(acc1.detection_f1)} vs {_fmt(acc2.detection_f1)}):")
    print("  only the labels changed, not the boxes. Brand accuracy moved from")
    print(f"  {_fmt(acc1.brand_classification_accuracy)} to "
          f"{_fmt(acc2.brand_classification_accuracy)} without the model changing at all.")
    print(f"  That is why every row is stamped with gt_version "
          f"('{acc1.gt_version}' vs '{acc2.gt_version}').")

    # -----------------------------------------------------------------------
    # STEP 5: A key mismatch is a configuration error, and says so.
    # -----------------------------------------------------------------------
    print("\n=== STEP 5: what a broken join looks like ===")
    gt_wrong = write_sample_ground_truth_file(
        work / "annotations_wrong_keys.json",
        image_key="gs://some-other-bucket/a-shelf-nobody-benchmarked.png",
    )
    provider_wrong = ground_truth_provider(sdk.config, gt_wrong, version="v1")
    try:
        score_predictions(predictions_path, provider_wrong, sdk.config.evaluation)
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}")
    print("\n  Scoring aborted rather than emitting a table of zeros that would have been read")
    print("  as a model failure. Fix `image_key_field`, then re-score. Still free.")

    # -----------------------------------------------------------------------
    print("\n=== The same thing from the command line ===")
    print(f"""
  .venv/bin/shelf-benchmark run --offline --approaches single_pass_full_shelf

  .venv/bin/shelf-benchmark validate-gt --offline \\
      --gt-provider json --ground-truth-uri {gt_v1} --gt-version v1

  .venv/bin/shelf-benchmark score --offline \\
      --predictions {predictions_path} \\
      --gt-provider json --ground-truth-uri {gt_v1} --gt-version v1
""")


if __name__ == "__main__":
    main()
