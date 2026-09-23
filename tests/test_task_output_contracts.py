"""Contract tests for what each task is allowed to claim it predicted.

The defects these lock down share one shape: a task filled a field it had no information for with
a plausible-looking constant, and something downstream -- the scorer, a report, a fine-tuning
dataset -- then treated that constant as a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.models import GroundTruthProductItem, ImageGroundTruth, RowLevelReportItem

# --------------------------------------------------------------------------
# Detection must not fabricate taxonomy attributes, and must not be scored on them.
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_detection_does_not_emit_placeholder_taxonomy_values() -> None:
    """Detection localises facings; it must leave classification fields empty.

    These fields used to carry the literals "Detected Facing", "Shelf Facing", "tube" and
    "Single". They are written into `predicted_category` / `predicted_subcategory` /
    `predicted_packaging` / `predicted_pack_type`, which the scorer compares against real ground
    truth -- so every detection run scored ~0% on four attributes it never predicted.
    """
    import inspect

    from shelf_benchmark.tasks import detection

    # Strip comments so the explanatory note about the old behaviour does not trip this.
    code = "\n".join(
        line for line in inspect.getsource(detection).splitlines()
        if not line.lstrip().startswith("#")
    )
    for literal in ('"Detected Facing"', '"Shelf Facing"'):
        assert literal not in code, (
            f"{literal} is a placeholder written into a scored prediction field. "
            "Leave the field empty instead."
        )


@pytest.mark.offline
def test_detection_is_not_scored_on_attributes_it_never_predicts() -> None:
    """`evaluate_task_accuracy` must respect `task_type` when computing per-attribute accuracy."""
    from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy

    gt = ImageGroundTruth(
        image_id="y.png",
        total_main_shelf_facings=1,
        items=[
            GroundTruthProductItem(
                item_id=1,
                brand="Dove",
                product_name="Dove Beauty Bar",
                category="Skin Cleansing",
                subcategory="Bath Soap",
                packaging_type="bar",
                pack_type="Single",
                bbox_2d=[100, 100, 300, 200],
            )
        ],
    )
    row = RowLevelReportItem(
        task_type="detection",
        bbox_ymin=100,
        bbox_xmin=100,
        bbox_ymax=300,
        bbox_xmax=200,
        predicted_brand="Dove",
    )

    detection_metrics = evaluate_task_accuracy("detection", [row], gt)
    scored = set((detection_metrics.per_attribute_accuracy or {}).keys())

    forbidden = {"category", "subcategory", "packaging_type", "pack_type", "variant", "product_name"}
    assert not (scored & forbidden), (
        f"detection was scored on attributes it does not predict: {sorted(scored & forbidden)}"
    )

    # The same row under a task that *does* classify is still scored on them, so the filter is
    # task-specific rather than a blanket removal.
    classification_row = row.model_copy(update={"task_type": "classification"})
    classification_metrics = evaluate_task_accuracy("classification", [classification_row], gt)
    assert set((classification_metrics.per_attribute_accuracy or {}).keys()) & forbidden


# --------------------------------------------------------------------------
# Every task must declare its own approach id.
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_every_task_declares_a_distinct_separation_approach() -> None:
    """A task that omits `default_separation_approach` must fail loudly, not inherit a wrong one.

    `execute()` used to fall back to the literal "single_pass_full_shelf", a *classification*
    approach id. The fine-tuning task never set one, so its rows, cost records and GCP billing
    labels all claimed it had run classification.
    """
    from shelf_benchmark.tasks.base import BaseBenchmarkTask
    from shelf_benchmark.tasks.classification import ProductClassificationTask
    from shelf_benchmark.tasks.detection import ProductDetectionTask
    from shelf_benchmark.tasks.fine_tuning import GeminiFineTuningTask
    from shelf_benchmark.tasks.matching import ProductMatchingTask

    tasks = [ProductClassificationTask, ProductDetectionTask, ProductMatchingTask, GeminiFineTuningTask]
    approaches = {t.task_type: t.default_separation_approach for t in tasks}

    assert all(approaches.values()), f"a task declares no approach id: {approaches}"
    assert len(set(approaches.values())) == len(approaches), (
        f"two tasks share an approach id, so reports cannot tell them apart: {approaches}"
    )
    assert approaches["fine_tuning"] != approaches["classification"]

    with pytest.raises(TypeError, match="default_separation_approach"):

        class _Forgetful(BaseBenchmarkTask):
            task_type = "forgetful"


# --------------------------------------------------------------------------
# The SFT dataset must reflect the annotations, not hardcoded constants.
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_sft_example_uses_real_ground_truth_attributes(tmp_path) -> None:
    """The fine-tuning target payload must come from the annotation, not from literals.

    It used to write `"variant": it.sku_id`, `"category": "Face Wash"` (a subcategory, identical on
    every row) and `"packaging_type": "Tube"` (the taxonomy uses lowercase "tube") -- so a model
    fine-tuned on this dataset was supervised to emit those two constants for every product. The
    real fields existed on `GroundTruthProductItem` and were simply never read.
    """
    from shelf_benchmark.tasks.fine_tuning import GeminiFineTuningTask

    task = GeminiFineTuningTask.__new__(GeminiFineTuningTask)
    task.config = BenchmarkConfig()

    class _Storage:
        @staticmethod
        def guess_mime_type(uri: str) -> str:
            return "image/png"

    task.storage = _Storage()

    gt = ImageGroundTruth(
        image_id="y.png",
        total_main_shelf_facings=1,
        items=[
            GroundTruthProductItem(
                item_id=1,
                brand="Pepsodent",
                product_name="Pepsodent Germicheck",
                category="Oral Care",
                subcategory="Toothpaste",
                variant="Germicheck+",
                packaging_type="box",
                pack_type="Single",
                size="150g",
                sku_id="SKU-001",
                bbox_2d=[10, 10, 50, 40],
            )
        ],
    )

    example = task.build_sft_jsonl_example(shelf_image_uri="gs://x/y.png", ground_truth=gt)
    target = json.loads(example["contents"][1]["parts"][0]["text"])
    product = target["classified_products"][0]

    assert product["category"] == "Oral Care", "category must come from the annotation"
    assert product["subcategory"] == "Toothpaste"
    assert product["variant"] == "Germicheck+", "variant must not be the SKU id"
    assert product["packaging_type"] == "box"
    assert product["size"] == "150g"
    assert product["sku_id"] == "SKU-001"
    assert "Face Wash" not in json.dumps(target)
    assert "Tube" not in json.dumps(target)


@pytest.mark.offline
def test_sft_dataset_does_not_duplicate_a_single_example_by_default() -> None:
    """One annotated image must produce one JSONL row unless duplication is explicitly requested."""
    from shelf_benchmark.config import FineTuningConfig

    assert FineTuningConfig().duplicate_single_example_times == 1, (
        "The default used to be 16, which wrote the same example 16 times and reported the result "
        "as a 16-example training dataset."
    )


# ---------------------------------------------------------------------------
# Single pipeline (Stage 2 item 21 / E1): task path == plugin path, no fabrications
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_task_path_and_plugin_path_produce_identical_metrics_for_identical_predictions(
    tmp_path: Path,
) -> None:
    """A built-in task and a plugin fed the same predictions must agree on cost, tokens, and accuracy.

    This is the E1 regression gate: before `pipeline.py` replaced the two assemblers, the plugin
    path divided per-row token columns by the row count, defaulted embeddings cost to $0.00,
    recomputed `cost_per_product_usd`, and omitted `container_cpu_active_ms` (pushing Cloud Run
    billing onto the estimate branch). The two paths reported an 11% cost difference and a 2.8x
    token difference on identical inputs.
    """
    from datetime import datetime, timedelta, timezone

    from shelf_benchmark.approaches.base import CommonLayerContext
    from shelf_benchmark.data.storage import StorageManager
    from shelf_benchmark.models import ShelfAssociationRecord, TokenUsageMetrics
    from shelf_benchmark.pipeline import InvocationContext, PipelineExecutor, RawInvocation
    from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

    cfg = BenchmarkConfig()
    storage = StorageManager(project_id=cfg.gcp.project_id, bucket_config=cfg.buckets, offline=True)
    telemetry = OpenTelemetryBenchmarkLogger(config=cfg.telemetry, project_id=cfg.gcp.project_id)
    ctx = CommonLayerContext(
        config=cfg, storage=storage, telemetry=telemetry, reports_dir=tmp_path / "reports"
    )
    record = ShelfAssociationRecord(
        association_id="assoc-1",
        shelf_image_uri="gs://shelf-bucket/aisle1.png",
        store_id="store-01",
    )
    gt = ImageGroundTruth(
        image_id="aisle1.png",
        total_main_shelf_facings=2,
        items=[
            GroundTruthProductItem(
                item_id=1,
                brand="Dove",
                product_name="Dove Beauty Bar",
                category="Skin Cleansing",
                subcategory="Bar Soap",
                variant="Beauty Bar",
                packaging_type="bar",
                pack_type="Single",
                size="100g",
                bbox_2d=[100, 100, 300, 250],
            ),
            GroundTruthProductItem(
                item_id=2,
                brand="Lux",
                product_name="Lux Soft Touch",
                category="Skin Cleansing",
                subcategory="Bar Soap",
                variant="Soft Touch",
                packaging_type="bar",
                pack_type="Single",
                size="100g",
                bbox_2d=[100, 300, 300, 450],
            ),
        ],
    )
    facing_dicts = [
        {
            "product_index": 1,
            "bbox_2d": [100, 100, 300, 250],
            "brand": "Dove",
            "product_name": "Dove Beauty Bar",
            "category": "Skin Cleansing",
            "subcategory": "Bar Soap",
            "variant": "Beauty Bar",
            "packaging_type": "bar",
            "pack_type": "Single",
            "size": "100g",
            "confidence": 0.91,
        },
        {
            "product_index": 2,
            "bbox_2d": [100, 300, 300, 450],
            "brand": "Lux",
            "product_name": "Lux Soft Touch",
            "category": "Skin Cleansing",
            "subcategory": "Bar Soap",
            "variant": "Soft Touch",
            "packaging_type": "bar",
            "pack_type": "Single",
            "size": "100g",
            "confidence": 0.88,
        },
    ]
    tokens = TokenUsageMetrics(input_tokens=1200, thinking_tokens=150, output_tokens=320, total_tokens=1670)
    t0 = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(milliseconds=840)

    # Path A: a plugin calling `ctx.finalize(...)`.
    plugin_res = ctx.finalize(
        approach_id="two_stage_physical_crop_per_facing",
        model_name="gemini-3-flash-preview",
        record=record,
        raw_outputs=facing_dicts,
        start_dt=t0,
        end_dt=t1,
        gt_record=gt,
        run_id="parity-check",
        tokens=tokens,
        cpu_active_ms=25.0,
    )

    # Path B: a built-in task handing the same rows to `PipelineExecutor.finalize(...)`.
    task_rows = ctx.rows_from_facing_dicts(facing_dicts, task_type="classification")
    task_res = PipelineExecutor(config=cfg, telemetry=telemetry).finalize(
        RawInvocation(rows=task_rows, tokens=tokens),
        ctx=InvocationContext(
            task_type="classification",
            approach_id="two_stage_physical_crop_per_facing",
            model_name="gemini-3-flash-preview",
            shelf_image_uri=record.shelf_image_uri,
            run_id="parity-check",
            store_id=record.store_id,
            ground_truth=gt,
        ),
        start_dt=t0,
        end_dt=t1,
        cpu_active_ms=25.0,
    )

    assert plugin_res.cost.model_dump() == task_res.cost.model_dump(), (
        "A plugin and a built-in task fed identical predictions and timings must report "
        "identical 5-bucket GCP costs ( including embeddings/vision per-facing cost)."
    )
    assert plugin_res.tokens.model_dump() == task_res.tokens.model_dump()
    assert plugin_res.accuracy.model_dump() == task_res.accuracy.model_dump()
    for pr, tr in zip(plugin_res.row_level_items, task_res.row_level_items):
        assert pr.input_tokens == tr.input_tokens == 1200, (
            "Per-row token columns must carry the image total on both paths, not be divided by "
            "the row count on the plugin path."
        )
        assert pr.cost_per_product_usd == tr.cost_per_product_usd


@pytest.mark.offline
def test_plugin_omitting_tokens_and_bbox_is_not_given_fabricated_values(tmp_path: Path) -> None:
    """When a plugin reports neither token usage nor geometry, the pipeline must record 0 and flag it.

    `CommonLayerContext.finalize()` previously invented `180 * n or 350` input tokens, `90 * n or
    160` output tokens, a bounding box of `[500, 100, 800, 200]`, `"Personal Care"`, `"General"`,
    `"Standard"`, `"Single"`, `"box"`, and `confidence=0.95`. The `[500, 100, 800, 200]` box could
    overlap a real shelf item and earn a true positive on geometry the plugin never predicted.
    """
    from datetime import datetime, timezone

    from shelf_benchmark.approaches.base import CommonLayerContext
    from shelf_benchmark.data.storage import StorageManager
    from shelf_benchmark.models import ShelfAssociationRecord
    from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

    cfg = BenchmarkConfig()
    ctx = CommonLayerContext(
        config=cfg,
        storage=StorageManager(project_id=cfg.gcp.project_id, bucket_config=cfg.buckets, offline=True),
        telemetry=OpenTelemetryBenchmarkLogger(config=cfg.telemetry, project_id=cfg.gcp.project_id),
        reports_dir=tmp_path / "r",
    )
    now = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
    res = ctx.finalize(
        approach_id="custom_brand_only_plugin",
        model_name="gemini-3-flash-preview",
        record=ShelfAssociationRecord(association_id="a", shelf_image_uri="gs://b/i.png"),
        raw_outputs=[{"brand": "Dove"}],  # No tokens, no bbox, no category/packaging/variant/confidence
        start_dt=now,
        end_dt=now,
        tokens=None,
    )

    assert res.tokens.total_tokens == 0, "Unreported tokens must be recorded as 0, not 350+160"
    assert res.raw_output["token_usage_reported"] is False
    row = res.row_level_items[0]
    assert (row.bbox_ymin, row.bbox_xmin, row.bbox_ymax, row.bbox_xmax) == (0, 0, 0, 0), (
        "A plugin that returned no bbox must not be given [500, 100, 800, 200]."
    )
    assert row.predicted_category == ""
    assert row.predicted_subcategory == ""
    assert row.predicted_variant == ""
    assert row.predicted_packaging == ""
    assert row.predicted_pack_type == ""
    assert row.shelf_row == ""
    assert row.confidence == 0.0



# ---------------------------------------------------------------------------
# Stage 2 & Stage 3 contracts: atomic writes, associations validation,
# O(1) telemetry flush, optimal Hungarian pairing, AP@50/mAP, pricing staleness
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_atomic_text_writer_preserves_previous_file_on_mid_write_crash(tmp_path: Path) -> None:
    """A failure mid-write must leave the existing file intact, never truncated."""
    from shelf_benchmark.artifacts import atomic_text_writer, atomic_write_text

    target = tmp_path / "benchmark_summary.json"
    atomic_write_text(target, '{"status": "complete"}')

    with pytest.raises(RuntimeError, match="disk full"), atomic_text_writer(target) as handle:
        handle.write('{"status": "truncated')
        raise RuntimeError("disk full")

    assert target.read_text(encoding="utf-8") == '{"status": "complete"}'
    assert list(tmp_path.glob(".*.part")) == [], "Temporary staging file must be cleaned up on error"


@pytest.mark.offline
def test_unknown_associations_provider_type_raises_instead_of_bucket_scan() -> None:
    """A typo in `associations.provider_type` must raise `AssociationConfigError`, not scan a bucket."""
    from shelf_benchmark.config import AssociationConfig, BucketConfig
    from shelf_benchmark.data.associations import (
        AssociationConfigError,
        create_association_provider,
    )
    from shelf_benchmark.data.storage import StorageManager

    storage = StorageManager(project_id="test-proj", bucket_config=BucketConfig(), offline=True)
    bad_cfg = AssociationConfig(provider_type="bigqeury", source_uri="proj.ds.table")
    with pytest.raises(AssociationConfigError, match="bigqeury"):
        create_association_provider(assoc_config=bad_cfg, bucket_config=BucketConfig(), storage_manager=storage)


@pytest.mark.offline
def test_optimal_bipartite_pairing_beats_greedy_on_overlapping_shelf_boxes() -> None:
    """Hungarian (`pairing_strategy='optimal'`) finds 2 TPs where greedy IoU finds only 1 TP."""
    from shelf_benchmark.config import EvaluationConfig
    from shelf_benchmark.evaluation.metrics import evaluate_task_accuracy
    from shelf_benchmark.models import RowLevelReportItem

    # GT1=[0, 0, 100, 100], GT2=[0, 20, 100, 120]
    # Pred1=[0, 18, 100, 118] -> IoU(Pred1, GT2)=0.9608, IoU(Pred1, GT1)=0.6949
    # Pred2=[0, 20, 100, 120] -> IoU(Pred2, GT2)=1.0000, IoU(Pred2, GT1)=0.6667
    # Wait: if Pred1=[0, 20, 100, 120] (IoU with GT2=1.0, IoU with GT1=0.6667) and
    # Pred2=[0, 55, 100, 155] (IoU with GT2=0.4815 < 0.50, IoU with GT1=0.2903 < 0.50) -> flip:
    # Pred1=[0, 20, 100, 120] matches GT2 (1.0) AND GT1 (0.6667 >= 0.50).
    # Pred2=[0, 30, 100, 130] matches ONLY GT2 (IoU=80/110=0.7273 >= 0.50), while IoU(Pred2, GT1)=70/130=0.5385...
    # Make IoU(Pred2, GT1) < 0.50: Pred2=[0, 38, 100, 138] -> IoU(Pred2, GT1)=62/138=0.449 < 0.50,
    # while IoU(Pred2, GT2)=82/118=0.6949 >= 0.50!
    # Greedy picks (Pred1, GT2) first because IoU=1.00 is highest, leaving Pred2 with NO valid GT above 0.50 (1 TP)!
    # Optimal picks (Pred1 -> GT1 at 0.6667, Pred2 -> GT2 at 0.6949), achieving 2 TPs (total IoU 1.3616 > 1.00)!
    gt = ImageGroundTruth(
        image_id="overlap.png",
        total_main_shelf_facings=2,
        items=[
            GroundTruthProductItem(item_id=1, brand="Dove", product_name="A", bbox_2d=[0, 0, 100, 100]),
            GroundTruthProductItem(item_id=2, brand="Dove", product_name="B", bbox_2d=[0, 20, 100, 120]),
        ],
    )
    rows_greedy = [
        RowLevelReportItem(product_index=1, predicted_brand="Dove", bbox_ymin=0, bbox_xmin=20, bbox_ymax=100, bbox_xmax=120, confidence=0.95),
        RowLevelReportItem(product_index=2, predicted_brand="Dove", bbox_ymin=0, bbox_xmin=38, bbox_ymax=100, bbox_xmax=138, confidence=0.85),
    ]
    rows_optimal = [r.model_copy(deep=True) for r in rows_greedy]

    acc_greedy = evaluate_task_accuracy("detection", rows_greedy, gt, EvaluationConfig(pairing_strategy="greedy_iou", iou_threshold=0.50))
    acc_opt = evaluate_task_accuracy("detection", rows_optimal, gt, EvaluationConfig(pairing_strategy="optimal", iou_threshold=0.50))

    assert acc_greedy.true_positives == 1
    assert acc_opt.true_positives == 2
    assert acc_opt.detection_f1 == 1.0
    assert acc_opt.average_precision_at_50 is not None and acc_opt.average_precision_at_50 > 0.0
    assert len(acc_opt.pr_curve_points) == 2


@pytest.mark.offline
def test_pricing_staleness_and_invoice_reconciliation() -> None:
    """Rate table staleness check raises when older than `max_pricing_age_days` and invoice reconciliation computes drift."""
    from shelf_benchmark.config import GCPBillingConfig
    from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine

    cfg = GCPBillingConfig(pricing_last_verified_date="2026-01-01", max_pricing_age_days=30)
    with pytest.raises(ValueError, match="exceeding max_pricing_age_days=30"):
        GCPBillingAndCostEngine.check_pricing_staleness(cfg, reference_date="2026-03-01")

    recon = GCPBillingAndCostEngine.reconcile_modelled_vs_billed(0.0105, 0.0100, tolerance_pct=10.0)
    assert recon["within_tolerance"] is True
    assert recon["drift_pct"] == 5.0
