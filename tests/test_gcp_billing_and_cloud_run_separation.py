"""Tests for GCP cost separation, billing-source honesty, and OTel cost export.

These tests exist to pin down three claims the suite makes about money, each of which was
previously wrong in a way that produced plausible-looking numbers:

1.  Modelled infrastructure (Cloud Run, GCS, Cloud Logging) is opt-in. It used to switch on merely
    because a billing config object had been passed, so laptop runs were billed for containers
    that never existed.
2.  On-demand and provisioned-throughput serving costs are mutually exclusive. The GSU bucket used
    to be populated even for on-demand traffic.
3.  `billing_source` distinguishes a metered rate from an estimate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.evaluation.cost import compute_cost_metrics
from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
from shelf_benchmark.models import TokenUsageMetrics
from shelf_benchmark.telemetry import OpenTelemetryBenchmarkLogger

# Everything in this module runs without network or GCP credentials.
pytestmark = pytest.mark.offline


def _tokens() -> TokenUsageMetrics:
    return TokenUsageMetrics(
        input_tokens=2000,
        thinking_tokens=500,
        output_tokens=1000,
        total_tokens=3500,
        traffic_type="ON_DEMAND",
    )


def test_infrastructure_costs_are_opt_in():
    """Modelled Cloud Run / GCS costs stay out of the total unless explicitly enabled."""
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    pricing = cfg.get_pricing("gemini-3.8-flash")

    off = cfg.billing.model_copy(deep=True)
    off.include_infrastructure_costs = False
    cost_off = compute_cost_metrics(
        tokens=_tokens(), pricing=pricing, product_count=20, latency_ms=3000.0,
        extra_embedding_or_vision_cost_usd=0.0020, billing_cfg=off,
    )
    assert cost_off.cloud_run_compute_usd == 0.0
    assert cost_off.gcs_and_observability_usd == 0.0
    assert cost_off.includes_modelled_infrastructure is False
    # The embedding charge is a real API call, so it survives with infra disabled.
    assert cost_off.vertex_ai_embeddings_and_vision_usd == 0.0020

    on = cfg.billing.model_copy(deep=True)
    on.include_infrastructure_costs = True
    cost_on = compute_cost_metrics(
        tokens=_tokens(), pricing=pricing, product_count=20, latency_ms=3000.0,
        extra_embedding_or_vision_cost_usd=0.0020, billing_cfg=on,
    )
    assert cost_on.cloud_run_compute_usd > 0.0
    assert cost_on.gcs_and_observability_usd > 0.0
    assert cost_on.includes_modelled_infrastructure is True
    assert cost_on.cost_per_shelf_image_usd > cost_off.cost_per_shelf_image_usd


def test_payg_and_provisioned_throughput_are_mutually_exclusive():
    """Exactly one serving-cost bucket may be non-zero: you are billed one way or the other."""
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    pricing = cfg.get_pricing("gemini-3.8-flash")
    engine = GCPBillingAndCostEngine(project_id=cfg.gcp.project_id, billing_cfg=cfg.billing)
    labels = engine.build_gcp_billing_labels(
        run_id="test-run-payg-001",
        approach_id="class_agnostic_visual_embedding",
        task_type="classification",
        model_name="gemini-3.8-flash",
    )

    infra_on = cfg.billing.model_copy(deep=True)
    infra_on.include_infrastructure_costs = True

    # 1. ON_DEMAND: PAYG token cost is charged, provisioned throughput is not.
    payg_cost = compute_cost_metrics(
        tokens=_tokens(), pricing=pricing, product_count=20, latency_ms=3000.0,
        extra_embedding_or_vision_cost_usd=0.0020, billing_cfg=infra_on, gcp_labels=labels,
    )
    assert payg_cost.traffic_type == "ON_DEMAND"
    assert payg_cost.vertex_ai_payg_tokens_usd > 0.0
    assert payg_cost.vertex_ai_provisioned_throughput_usd == 0.0
    assert payg_cost.vertex_ai_embeddings_and_vision_usd == 0.0020
    assert payg_cost.cloud_run_compute_usd > 0.0
    assert payg_cost.gcs_and_observability_usd > 0.0

    expected_total_payg = round(
        payg_cost.vertex_ai_payg_tokens_usd
        + payg_cost.vertex_ai_embeddings_and_vision_usd
        + payg_cost.cloud_run_compute_usd
        + payg_cost.gcs_and_observability_usd,
        8,
    )
    assert abs(payg_cost.cost_per_shelf_image_usd - expected_total_payg) < 1e-7

    # 2. PROVISIONED_THROUGHPUT: the GSU bucket carries the serving cost instead.
    pt_billing = infra_on.model_copy(deep=True)
    pt_billing.provisioned_throughput.enabled = True
    pt_billing.provisioned_throughput.gsu_count = 1
    pt_billing.provisioned_throughput.hourly_rate_per_gsu_usd = 22.00
    pt_billing.provisioned_throughput.monthly_commitment_discount_pct = 0.20
    pt_billing.provisioned_throughput.target_images_per_hour_per_gsu = 1800

    pt_cost = compute_cost_metrics(
        tokens=_tokens().model_copy(update={"traffic_type": "PROVISIONED_THROUGHPUT"}),
        pricing=pricing, product_count=20, latency_ms=3000.0,
        extra_embedding_or_vision_cost_usd=0.0020, billing_cfg=pt_billing, gcp_labels=labels,
    )
    assert pt_cost.traffic_type == "PROVISIONED_THROUGHPUT"
    assert pt_cost.vertex_ai_provisioned_throughput_usd > 0.0
    assert pt_cost.cloud_run_compute_usd == payg_cost.cloud_run_compute_usd
    # PAYG tokens stay populated for comparison, but must not be added to the total.
    assert pt_cost.vertex_ai_payg_tokens_usd > 0.0

    expected_total_pt = round(
        pt_cost.vertex_ai_provisioned_throughput_usd
        + pt_cost.vertex_ai_embeddings_and_vision_usd
        + pt_cost.cloud_run_compute_usd
        + pt_cost.gcs_and_observability_usd,
        8,
    )
    assert abs(pt_cost.cost_per_shelf_image_usd - expected_total_pt) < 1e-7


def test_billing_source_reports_estimates_honestly():
    """Costs derived from the YAML rate table must never claim to be live catalog rates."""
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    billing = cfg.billing.model_copy(deep=True)
    billing.use_live_cloud_billing_catalog_api = False

    cost = compute_cost_metrics(
        tokens=TokenUsageMetrics(input_tokens=100, output_tokens=50, total_tokens=150),
        pricing=cfg.get_pricing("gemini-3.8-flash"),
        product_count=1, latency_ms=100.0, billing_cfg=billing,
    )
    assert cost.billing_source == "yaml_rate_table"
    assert cost.rates_from_live_catalog is False


def test_otel_export_and_bigquery_reconciliation(tmp_path: Path):
    """OTel spans carry every separated cost bucket, and reconciliation SQL filters by run id."""
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    engine = GCPBillingAndCostEngine(project_id=cfg.gcp.project_id, billing_cfg=cfg.billing)
    labels = engine.build_gcp_billing_labels(
        run_id="test-run-payg-001", task_type="classification", model_name="gemini-3.8-flash",
    )
    infra_on = cfg.billing.model_copy(deep=True)
    infra_on.include_infrastructure_costs = True
    payg_cost = compute_cost_metrics(
        tokens=_tokens(), pricing=cfg.get_pricing("gemini-3.8-flash"), product_count=20,
        latency_ms=3000.0, extra_embedding_or_vision_cost_usd=0.0020,
        billing_cfg=infra_on, gcp_labels=labels,
    )

    cfg.telemetry.otel_log_path = str(tmp_path / "otel_cost_test.jsonl")
    cfg.telemetry.export_to_gcp_cloud_logging = False
    cfg.telemetry.sync_otel_logs_to_gcs = False
    otel = OpenTelemetryBenchmarkLogger(cfg.telemetry, project_id=cfg.gcp.project_id)
    now = otel.now_utc()
    _, _, record = otel.log_task_execution(
        run_id="test-run-payg-001",
        task_type="classification",
        model_name="gemini-3.8-flash",
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
        start_dt=now,
        end_dt=now,
        tokens=_tokens(),
        cost=payg_cost,
    )
    attrs = record["Attributes"]
    assert attrs["shelf_benchmark.cost.vertex_ai_payg_tokens_usd"] == payg_cost.vertex_ai_payg_tokens_usd
    assert (
        attrs["shelf_benchmark.cost.vertex_ai_provisioned_throughput_usd"]
        == payg_cost.vertex_ai_provisioned_throughput_usd
    )
    assert attrs["shelf_benchmark.cost.vertex_ai_embeddings_and_vision_usd"] == 0.0020
    assert attrs["shelf_benchmark.cost.cloud_run_compute_usd"] == payg_cost.cloud_run_compute_usd
    assert attrs["gcp.billing.label.benchmark_run_id"] == "test-run-payg-001"

    sql = engine.build_bigquery_true_cost_reconciliation_sql(
        run_id="test-run-payg-001",
        export_table="unilever-shelf-understanding.billing_export.gcp_billing_export_resource_v1_TEST",
    )
    assert "benchmark_run_id" in sql
    assert "test-run-payg-001" in sql
    assert "net_true_gcp_cost_usd" in sql
