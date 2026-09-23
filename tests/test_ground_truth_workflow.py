"""Tests for the ground-truth plug-in workflow and the run/score decoupling.

The scenario these encode is the one this project is actually in: the benchmark must be usable
*before* annotations exist, and the day they arrive it must be possible to score every historical
run without paying for inference again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.data.ground_truth import GroundTruthError, create_ground_truth_provider
from shelf_benchmark.data.storage import StorageManager
from shelf_benchmark.scoring import ScoringError, score_predictions
from shelf_benchmark.sdk import UniversalModelSpec
from shelf_benchmark.testing import (
    OFFLINE_IMAGE_URI,
    make_offline_sdk,
    perfect_prediction_payload,
    write_sample_ground_truth_file,
)

pytestmark = pytest.mark.offline


def _provider_for(cfg: BenchmarkConfig, gt_file: Path, version: str = "v1"):
    cfg.ground_truth.provider_type = "json"
    cfg.ground_truth.source_uri = str(gt_file)
    cfg.ground_truth.gt_version = version
    storage = StorageManager(cfg.gcp.project_id, cfg.buckets, offline=True)
    return create_ground_truth_provider(
        gt_config=cfg.ground_truth, storage_manager=storage, project_id=cfg.gcp.project_id
    )


def _run_without_ground_truth(tmp_path: Path):
    """Run the benchmark with no annotations available, writing a predictions file."""
    sdk = make_offline_sdk(tmp_path)
    sdk.report_generator.write_predictions_file = True
    sdk.register_model(
        UniversalModelSpec(
            model_id="gt-workflow-model",
            display_name="gt-workflow-model",
            provider_family="custom_callable",
            custom_handler=lambda p, u, s: perfect_prediction_payload(),
        )
    )
    summary = sdk.run_suite(
        models=["gt-workflow-model"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )
    return sdk, summary


def test_absent_ground_truth_yields_placeholders_not_zeros(tmp_path):
    """Before annotations exist, metrics must be None with an explicit placeholder status.

    Reporting 0.0 here would be indistinguishable from a model that got everything wrong, which
    is how an unscored benchmark comes to look like a failing one.
    """
    _, summary = _run_without_ground_truth(tmp_path)
    acc = summary["results"][0].accuracy
    assert acc.ground_truth_available is False
    assert acc.detection_f1 is None
    assert acc.brand_classification_accuracy is None
    assert "PLACEHOLDER" in acc.accuracy_status or "NO_GROUND_TRUTH" in acc.accuracy_status


def test_ground_truth_arrives_and_old_run_is_rescored_for_free(tmp_path):
    """The headline workflow: run today, score later, no re-inference."""
    sdk, summary = _run_without_ground_truth(tmp_path)
    predictions_path = summary["artifacts"]["predictions_json"]
    assert Path(predictions_path).exists()

    # Annotations land.
    gt_file = tmp_path / "annotations_v1.json"
    write_sample_ground_truth_file(gt_file)
    provider = _provider_for(sdk.config, gt_file, version="v1")

    rescored = score_predictions(
        predictions_path=predictions_path,
        gt_provider=provider,
        evaluation_config=sdk.config.evaluation,
        gt_version="v1",
    )
    assert len(rescored) == len(summary["results"])
    acc = rescored[0].accuracy
    assert acc.detection_f1 == pytest.approx(1.0)
    assert acc.brand_classification_accuracy == pytest.approx(1.0)
    assert acc.gt_version == "v1"
    # Cost and tokens are carried through unchanged: re-scoring does not re-measure them.
    assert rescored[0].tokens.total_tokens == summary["results"][0].tokens.total_tokens


def test_rescoring_with_mismatched_keys_raises_instead_of_reporting_zeros(tmp_path):
    """A key mismatch is a configuration error, not a model failure, and must not look like one."""
    sdk, summary = _run_without_ground_truth(tmp_path)
    predictions_path = summary["artifacts"]["predictions_json"]

    # Ground truth keyed on an image nobody benchmarked.
    gt_file = tmp_path / "annotations_wrong_keys.json"
    write_sample_ground_truth_file(gt_file, image_key="gs://some-other-bucket/unrelated.png")
    provider = _provider_for(sdk.config, gt_file)

    with pytest.raises(ScoringError, match="matched a ground-truth entry"):
        score_predictions(predictions_path, provider, sdk.config.evaluation)


def test_unreadable_ground_truth_source_raises(tmp_path):
    """A typo in the path must fail loudly rather than degrading to 'no ground truth'."""
    cfg = BenchmarkConfig()
    cfg.ground_truth.provider_type = "json"
    cfg.ground_truth.source_uri = str(tmp_path / "does_not_exist.json")
    cfg.ground_truth.strict = True
    storage = StorageManager(cfg.gcp.project_id, cfg.buckets, offline=True)
    with pytest.raises(GroundTruthError):
        create_ground_truth_provider(
            gt_config=cfg.ground_truth, storage_manager=storage, project_id=cfg.gcp.project_id
        )


def test_empty_ground_truth_file_raises(tmp_path):
    """Zero parsed images means the schema mapping is wrong; say so instead of scoring nothing."""
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    cfg = BenchmarkConfig()
    cfg.ground_truth.provider_type = "json"
    cfg.ground_truth.source_uri = str(empty)
    cfg.ground_truth.strict = True
    storage = StorageManager(cfg.gcp.project_id, cfg.buckets, offline=True)
    with pytest.raises(GroundTruthError):
        create_ground_truth_provider(
            gt_config=cfg.ground_truth, storage_manager=storage, project_id=cfg.gcp.project_id
        )


def test_coco_bbox_conversion(tmp_path):
    """COCO ships [x, y, w, h] in pixels; the suite works in [ymin, xmin, ymax, xmax] over 0-1000.

    Getting this backwards is the most common ground-truth integration bug, so the conversion is
    pinned with an example that can be checked by hand:
    a 100x50 box at (200, 100) in a 1000x500 image becomes
      xmin = 200/1000*1000 = 200,  xmax = 300/1000*1000 = 300
      ymin = 100/500*1000  = 200,  ymax = 150/500*1000  = 300
    """
    coco = {
        "images": [{"id": 1, "file_name": "shelf_a.png", "width": 1000, "height": 500}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [200, 100, 100, 50]}],
        "categories": [{"id": 1, "name": "Brand_A Bright Beauty"}],
    }
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco), encoding="utf-8")

    cfg = BenchmarkConfig()
    cfg.ground_truth.provider_type = "coco"
    cfg.ground_truth.source_uri = str(path)
    cfg.ground_truth.schema_mapping.bbox_format = "coco_xywh_px"
    storage = StorageManager(cfg.gcp.project_id, cfg.buckets, offline=True)
    provider = create_ground_truth_provider(
        gt_config=cfg.ground_truth, storage_manager=storage, project_id=cfg.gcp.project_id
    )

    gt = provider.get_ground_truth(shelf_image_uri="shelf_a.png", ground_truth_id="shelf_a.png")
    assert gt is not None, "COCO image key should resolve by file_name"
    assert gt.items[0].bbox_2d == [200, 200, 300, 300]
