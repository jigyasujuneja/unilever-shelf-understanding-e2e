#!/usr/bin/env python3
"""Sample 07: The GCP cost model, bucket by bucket.

Run it (no GCP project, no credentials, no network):

    .venv/bin/python code_samples/07_gcp_billing_provisioned_throughput_and_cloud_run.py

Run it with live Cloud Billing Catalog SKU lookups and a live benchmark:

    .venv/bin/python code_samples/07_gcp_billing_provisioned_throughput_and_cloud_run.py --live

What a reported cost is, and is not:

  * Vertex AI returns token counts, not dollars. Every dollar figure in this suite is
    counts multiplied by a rate. `CostMetrics.billing_source` tells you which rates were used:
    `yaml_rate_table` for the configured rate card, and a live-catalog value only when SKU rates
    were genuinely fetched and parsed. It is never guessed.
  * Costs are split into five buckets. Two are derived from measured quantities (tokens, and
    the embedding or vision calls actually made). Three are modelled from latency and assumed
    machine shape, so they are excluded from the total unless you opt in with
    `billing.include_infrastructure_costs = True`. Reporting a modelled Cloud Run cost inside a
    headline per-image figure would make two approaches look different because of an assumption
    rather than because of a measurement.

    | # | Bucket                                | Basis                                   | In total by default |
    |---|---------------------------------------|-----------------------------------------|---------------------|
    | 1 | vertex_ai_payg_tokens_usd             | measured tokens x rate card             | yes                 |
    | 2 | vertex_ai_provisioned_throughput_usd  | reserved GSU amortised over throughput  | yes, when enabled   |
    | 3 | vertex_ai_embeddings_and_vision_usd   | measured embedding / vision calls       | yes                 |
    | 4 | cloud_run_compute_usd                 | modelled from latency and machine shape | no                  |
    | 5 | gcs_and_observability_usd             | modelled from op counts and log volume  | no                  |

  * Provisioned Throughput is driven by `billing.provisioned_throughput.enabled`, not by the
    token payload. When it is on, pay-as-you-go tokens are excluded from the total so a reserved
    GSU is not billed twice; bucket 1 is still reported on its own so you can see what the same
    traffic would have cost on demand.
"""

from __future__ import annotations

import argparse

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.evaluation.cost import compute_cost_metrics
from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
from shelf_benchmark.models import TokenUsageMetrics

FACINGS = 18
LATENCY_MS = 4200.0
EMBEDDING_COST_USD = 0.001800  # 18 crops x $0.000100 (multimodalembedding@001).


def show(title: str, cm) -> None:
    print(f"\n  [{title}]")
    print(f"    traffic_type                        : {cm.traffic_type}")
    print(f"    billing_source                      : {cm.billing_source}")
    print(f"    rates_from_live_catalog             : {cm.rates_from_live_catalog}")
    print(f"    includes_modelled_infrastructure    : {cm.includes_modelled_infrastructure}")
    print(f"    1. vertex_ai_payg_tokens_usd        : ${cm.vertex_ai_payg_tokens_usd:.8f}")
    print(f"    2. vertex_ai_provisioned_throughput : ${cm.vertex_ai_provisioned_throughput_usd:.8f}")
    print(f"    3. vertex_ai_embeddings_and_vision  : ${cm.vertex_ai_embeddings_and_vision_usd:.8f}")
    print(f"    4. cloud_run_compute_usd (modelled) : ${cm.cloud_run_compute_usd:.8f}")
    print(f"    5. gcs_and_observability_usd (model): ${cm.gcs_and_observability_usd:.8f}")
    print(f"    = cost_per_shelf_image_usd          : ${cm.cost_per_shelf_image_usd:.8f}")
    print(f"    = cost_per_product_usd              : ${cm.cost_per_product_usd:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Query the Cloud Billing Catalog API for SKU rates and run a live benchmark.",
    )
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    if not args.live:
        cfg.billing.use_live_cloud_billing_catalog_api = False
    engine = GCPBillingAndCostEngine(project_id=cfg.gcp.project_id, billing_cfg=cfg.billing)

    print("=" * 95)
    print("1. WHERE THE TOKEN RATES COME FROM")
    print("=" * 95)
    fallback = cfg.get_pricing("gemini-3.8-flash")
    live_rates, rate_source = engine.resolve_live_or_configured_token_rates(
        "gemini-3.8-flash", fallback
    )
    print(f"  rate source : {rate_source}")
    print(
        f"  rates       : input=${live_rates.input:.4f}/1M  "
        f"thinking=${live_rates.thinking:.4f}/1M  output=${live_rates.output:.4f}/1M"
    )
    print(
        "  A source of 'yaml_rate_table' means the Cloud Billing Catalog API was not consulted,\n"
        "  or was consulted and did not yield a usable SKU. The configured rate card was used."
    )

    tokens = TokenUsageMetrics(
        input_tokens=1520,
        thinking_tokens=340,
        output_tokens=1180,
        total_tokens=3040,
    )

    print("\n" + "=" * 95)
    print(f"2. THE FIVE BUCKETS, FOR {FACINGS} FACINGS AT {LATENCY_MS:.0f} MS")
    print("=" * 95)

    labels = engine.build_gcp_billing_labels(
        run_id="run-payg-demo",
        approach_id="class_agnostic_visual_embedding",
        task_type="classification",
        model_name="gemini-3.8-flash",
    )
    print(f"  GCP billing labels attached to this run: {labels}")

    # Mode A: on-demand, infrastructure excluded. This is the default the leaderboard uses.
    cost_payg = compute_cost_metrics(
        tokens=tokens,
        pricing=live_rates,
        product_count=FACINGS,
        latency_ms=LATENCY_MS,
        extra_embedding_or_vision_cost_usd=EMBEDDING_COST_USD,
        billing_cfg=cfg.billing,
        project_id=cfg.gcp.project_id,
        model_name="gemini-3.8-flash",
        gcp_labels=labels,
    )
    show("MODE A: on-demand, include_infrastructure_costs = False (default)", cost_payg)

    # Mode B: same run, infrastructure included. Note which numbers move.
    infra_cfg = cfg.billing.model_copy(deep=True)
    infra_cfg.include_infrastructure_costs = True
    cost_infra = compute_cost_metrics(
        tokens=tokens,
        pricing=live_rates,
        product_count=FACINGS,
        latency_ms=LATENCY_MS,
        extra_embedding_or_vision_cost_usd=EMBEDDING_COST_USD,
        billing_cfg=infra_cfg,
        project_id=cfg.gcp.project_id,
        model_name="gemini-3.8-flash",
    )
    show("MODE B: on-demand, include_infrastructure_costs = True", cost_infra)

    # Mode C: 1 reserved GSU at $22/hr with a 20% one-month commitment discount.
    pt_cfg = cfg.billing.model_copy(deep=True)
    pt_cfg.provisioned_throughput.enabled = True
    pt_cfg.provisioned_throughput.gsu_count = 1
    pt_cfg.provisioned_throughput.hourly_rate_per_gsu_usd = 22.00
    pt_cfg.provisioned_throughput.monthly_commitment_discount_pct = 0.20
    pt_cfg.provisioned_throughput.target_images_per_hour_per_gsu = 1800
    cost_pt = compute_cost_metrics(
        tokens=tokens,
        pricing=live_rates,
        product_count=FACINGS,
        latency_ms=LATENCY_MS,
        extra_embedding_or_vision_cost_usd=EMBEDDING_COST_USD,
        billing_cfg=pt_cfg,
        project_id=cfg.gcp.project_id,
        model_name="gemini-3.8-flash",
    )
    show("MODE C: provisioned throughput, 1 GSU reserved", cost_pt)
    print(
        "\n  In mode C bucket 1 is still reported but excluded from the total: those tokens were\n"
        "  served by the reservation you already paid for."
    )

    print("\n" + "=" * 95)
    print("3. RECONCILING MODELLED COST AGAINST THE REAL INVOICE")
    print("=" * 95)
    print(
        "  Every run attaches the labels above to its Vertex AI calls and Cloud Run revisions,\n"
        "  so the billing export can be grouped by run_id. Generated SQL:\n"
    )
    print(
        engine.build_bigquery_true_cost_reconciliation_sql(
            run_id="run-payg-demo",
            export_table=(
                "unilever-shelf-understanding.billing_export."
                "gcp_billing_export_resource_v1_01ABCD"
            ),
        )
    )

    print("\n" + "=" * 95)
    print("4. CLOUD RUN DEPLOYMENT SPEC")
    print("=" * 95)
    spec = engine.generate_cloud_run_deploy_command(
        service_name="unilever-shelf-benchmark-service",
        region="us-central1",
    )
    print(f"  {spec['gcloud_deploy_command']}")

    if not args.live:
        print(
            "\nStopping here. Add --live to resolve live SKU rates and run a real benchmark,\n"
            "which needs GCP credentials and will spend money."
        )
        return

    from shelf_benchmark.sdk import ShelfBenchmarkSDK

    sdk = ShelfBenchmarkSDK(
        config_path="configs/default_config.yaml",
        output_dir="reports/sample_07_gcp_billing_and_otel",
    )
    summary = sdk.run_suite(
        models=["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"],
        approaches=["single_pass_full_shelf"],
        tasks=["detection"],
    )
    for run_res in summary["results"]:
        print(f"\n  [{run_res.model_name}] run_id={run_res.run_id} status={run_res.status}")
        print(f"    facings={run_res.cost.product_count} latency={run_res.latency_ms:.1f}ms")
        print(
            f"    tokens in/think/out/total = {run_res.tokens.input_tokens}/"
            f"{run_res.tokens.thinking_tokens}/{run_res.tokens.output_tokens}/"
            f"{run_res.tokens.total_tokens}"
        )
        show(f"live cost for {run_res.model_name}", run_res.cost)


if __name__ == "__main__":
    main()
