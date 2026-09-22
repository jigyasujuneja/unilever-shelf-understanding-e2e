#!/usr/bin/env python3
"""Code Sample 07: Live GCP Cloud Billing API, Provisioned Throughput (GSU), Cloud Run Cost Separation & GCP OTel Export.

Answers the core architectural questions around GCP Cost & Observability:
1. Why `pricing_per_million_tokens` exists alongside Live GCP Billing APIs:
   - Vertex AI `generate_content` returns `usage_metadata` (exact input, thinking, output, cached token
     counts + `traffic_type`: `ON_DEMAND` vs `PROVISIONED_THROUGHPUT`), NOT a dollar string in the payload.
   - Our `GCPBillingAndCostEngine` queries Google Cloud's official **Cloud Billing Catalog API**
     (`cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus`) live for SKU rates and uses YAML only as
     an offline/preview fallback.
2. How **Provisioned Throughput (GSU — Generative AI Scale Units)** is calculated:
   - Automatically detects `response.usage_metadata.traffic_type == "PROVISIONED_THROUGHPUT"` (or config override),
     sets PAYG token cost to `$0.00` (avoiding double-counting!), and computes the exact amortized GSU reservation
     cost (`gsu_count * hourly_rate * (1 - discount) / target_images_per_hour`), while also supporting
     `ON_DEMAND_SPILLOVER` when burst traffic exceeds reserved GSUs.
3. How **100% of Every GCP Cost Component is Separated**:
   - `vertex_ai_payg_tokens_usd` (On-Demand Input + Thinking + Output tokens)
   - `vertex_ai_provisioned_throughput_usd` (Reserved GSU amortized cost)
   - `vertex_ai_embeddings_and_vision_usd` (`multimodalembedding@001` 1408-D & `gemini-embedding-001` 3072-D)
   - `cloud_run_compute_usd` (Exact Cloud Run `vCPU-seconds` + `GiB-seconds` + request fee for `latency_ms`)
   - `gcs_and_observability_usd` (GCS Class A/B ops + Cloud Logging/Trace ingestion)
4. How **True Invoice Reconciliation in GCP BigQuery** works:
   - Attaches GCP Billing `labels` (`unilever_shelf_run_id`, `unilever_shelf_approach`, `unilever_shelf_model`)
     to Vertex AI calls and Cloud Run revisions, and generates the parameterized BigQuery SQL query against
     `gcp_billing_export_resource_v1_*`.
5. How **OpenTelemetry Logs & Reports are Saved Directly to GCP**:
   - Exports structured OTel records directly to **Google Cloud Logging**
     (`projects/unilever-shelf-understanding/logs/unilever-shelf-benchmark-otel`) AND
     **Google Cloud Storage** (`gs://unilever-shelf-understanding-shelf-images/otel/otel_logs.jsonl`).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.evaluation.cost import compute_cost_metrics
from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
from shelf_benchmark.models import TokenUsageMetrics
from shelf_benchmark.sdk import ShelfBenchmarkSDK


def main() -> None:
    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    engine = GCPBillingAndCostEngine(project_id=cfg.gcp.project_id, billing_cfg=cfg.billing)

    print("=" * 95)
    print("1. LIVE GCP CLOUD BILLING CATALOG API SKU RATE RESOLUTION")
    print("=" * 95)
    fallback = cfg.get_pricing("gemini-3.8-flash")
    live_rates, rate_source = engine.resolve_live_or_configured_token_rates("gemini-3.8-flash", fallback)
    print(f"  * Active Rate Source : {rate_source}")
    print(
        f"  * Resolved Rates     : Input=${live_rates.input:.4f}/1M | "
        f"Thinking=${live_rates.thinking:.4f}/1M | Output=${live_rates.output:.4f}/1M"
    )

    print("\n" + "=" * 95)
    print("2. 100% SEPARATED GCP COST BREAKDOWN ACROSS 3 BILLING MODES (FOR 18 FRONT FACINGS, 4.2s LATENCY)")
    print("=" * 95)

    sample_tokens = TokenUsageMetrics(
        input_tokens=1520,
        thinking_tokens=340,
        output_tokens=1180,
        total_tokens=3040,
        traffic_type="ON_DEMAND",
    )

    # Mode A: Standard On-Demand (PAYG) + Cloud Run + Embeddings
    cost_payg = compute_cost_metrics(
        tokens=sample_tokens,
        pricing=live_rates,
        product_count=18,
        latency_ms=4200.0,
        extra_embedding_or_vision_cost_usd=0.001800,  # 18 crops * $0.000100/crop (multimodalembedding@001)
        billing_cfg=cfg.billing,
        project_id=cfg.gcp.project_id,
        model_name="gemini-3.8-flash",
        gcp_labels=engine.build_gcp_billing_labels(
            run_id="run-payg-demo",
            approach_id="class_agnostic_visual_embedding",
            task_type="classification",
            model_name="gemini-3.8-flash",
        ),
    )

    # Mode B: Vertex AI Provisioned Throughput (1 GSU @ $22/hr, 20% 1-month commitment discount) + Cloud Run
    pt_billing_cfg = cfg.billing.model_copy(deep=True)
    pt_billing_cfg.provisioned_throughput.enabled = True
    pt_billing_cfg.provisioned_throughput.gsu_count = 1
    pt_billing_cfg.provisioned_throughput.hourly_rate_per_gsu_usd = 22.00
    pt_billing_cfg.provisioned_throughput.monthly_commitment_discount_pct = 0.20
    pt_billing_cfg.provisioned_throughput.target_images_per_hour_per_gsu = 1800

    pt_tokens = sample_tokens.model_copy(update={"traffic_type": "PROVISIONED_THROUGHPUT"})
    cost_pt = compute_cost_metrics(
        tokens=pt_tokens,
        pricing=live_rates,
        product_count=18,
        latency_ms=4200.0,
        extra_embedding_or_vision_cost_usd=0.001800,
        billing_cfg=pt_billing_cfg,
        project_id=cfg.gcp.project_id,
        model_name="gemini-3.8-flash",
    )

    for title, cm in [
        ("MODE A: ON-DEMAND PAYG + CLOUD RUN + 1408-D EMBEDDINGS", cost_payg),
        ("MODE B: PROVISIONED THROUGHPUT (1 GSU RESERVED) + CLOUD RUN + 1408-D EMBEDDINGS", cost_pt),
    ]:
        print(f"\n  [{title}]")
        print(f"    - Traffic Type                          : {cm.traffic_type}")
        print(f"    - 1. Vertex AI PAYG Tokens ($)          : ${cm.vertex_ai_payg_tokens_usd:.6f}")
        print(f"    - 2. Vertex AI Provisioned GSU ($)      : ${cm.vertex_ai_provisioned_throughput_usd:.6f}")
        print(f"    - 3. Vertex AI Embeddings & Vision ($)  : ${cm.vertex_ai_embeddings_and_vision_usd:.6f}")
        print(f"    - 4. Cloud Run Compute (vCPU + RAM) ($) : ${cm.cloud_run_compute_usd:.6f}")
        print(f"    - 5. GCS + Cloud Logging OTel ($)       : ${cm.gcs_and_observability_usd:.6f}")
        print(f"    ------------------------------------------------------------------------")
        print(f"    = ALL-IN COST PER SHELF IMAGE ($)       : ${cm.cost_per_shelf_image_usd:.6f}")
        print(f"    = ALL-IN COST PER FRONT FACING ($)      : ${cm.cost_per_product_usd:.6f}")

    print("\n" + "=" * 95)
    print("3. BIGQUERY CLOUD BILLING EXPORT SQL (100% TRUE GCP INVOICE RECONCILIATION BY RUN_ID)")
    print("=" * 95)
    sql = engine.build_bigquery_true_cost_reconciliation_sql(
        run_id="run-payg-demo",
        export_table="unilever-shelf-understanding.billing_export.gcp_billing_export_resource_v1_01ABCD",
    )
    print(sql)

    print("\n" + "=" * 95)
    print("4. CLOUD RUN DEPLOYMENT SPEC & LIVE BENCHMARK + GCP OTEL LOGGING VERIFICATION")
    print("=" * 95)
    cloud_run_spec = engine.generate_cloud_run_deploy_command(
        service_name="unilever-shelf-benchmark-service",
        region="us-central1",
    )
    print(f"  * Cloud Run Deploy Command:\n    {cloud_run_spec['gcloud_deploy_command']}\n")

    sdk = ShelfBenchmarkSDK(
        config_path="configs/default_config.yaml",
        output_dir="reports/sample_07_gcp_billing_and_otel",
    )
    summary = sdk.run_suite(
        models=["gemini-2.5-flash"],
        approaches=["single_pass_full_shelf"],
        tasks=["detection"],
    )
    run_res = summary["results"][0]
    print(f"  * Live Detection Run ID                : {run_res.run_id}")
    print(f"  * Trace ID (Linked in Cloud Trace)     : {run_res.trace_id}")
    print(f"  * Cloud Logging Status                 : {sdk.telemetry._last_cloud_logging_status}")
    print(f"  * GCS OpenTelemetry JSONL URI          : {sdk.telemetry._last_gcs_otel_uri}")
    print(
        f"  * Live Run Separated Costs             : "
        f"Total=${run_res.cost.cost_per_shelf_image_usd:.6f} "
        f"(PAYG=${run_res.cost.vertex_ai_payg_tokens_usd:.6f}, "
        f"PT_GSU=${run_res.cost.vertex_ai_provisioned_throughput_usd:.6f}, "
        f"Embed=${run_res.cost.vertex_ai_embeddings_and_vision_usd:.6f}, "
        f"CloudRun=${run_res.cost.cloud_run_compute_usd:.6f}, "
        f"GCS+Obs=${run_res.cost.gcs_and_observability_usd:.6f})"
    )
    print(
        f"  * GCS Synced Reports                   : "
        f"{json.dumps({k: v for k, v in summary.get('reports', {}).items() if k.startswith('gcs_')}, indent=4)}"
    )


if __name__ == "__main__":
    main()
