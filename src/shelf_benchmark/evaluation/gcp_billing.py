"""GCP Cloud Billing Catalog API, Provisioned Throughput (GSU) Calculator, Cloud Run Cost Tracker & BigQuery Billing Export Reconciliation.

Ensures 100% of pricing calculations are grounded in GCP and strictly separated into 5 distinct GCP cost buckets:
1. `vertex_ai_payg_tokens_usd`: On-Demand Pay-As-You-Go token cost (Input + Thinking + Output tokens via Live Cloud Billing SKU Catalog API).
2. `vertex_ai_provisioned_throughput_usd`: Amortized Vertex AI Provisioned Throughput (GSU — Generative AI Scale Units) cost when running on dedicated GSU capacity (`traffic_type="PROVISIONED_THROUGHPUT"`).
3. `vertex_ai_embeddings_and_vision_usd`: Dense text (`gemini-embedding-001` 3072-D), visual crop (`multimodalembedding@001` 1408-D), and Cloud Vision API costs.
4. `cloud_run_compute_usd`: Exact Google Cloud Run container execution cost (`vCPU-seconds` + `GiB-seconds` + per-request invocation fee) for the benchmark run's duration.
5. `gcs_and_observability_usd`: Google Cloud Storage Class A/B operations + Google Cloud Logging / Cloud Trace ingestion cost.

Also provides:
- `build_gcp_billing_labels(...)`: Attaches GCP Billing `labels` (`benchmark_run_id`, `task_type`, `separation_approach`, `model`) to Vertex AI API calls and Cloud Run workloads so every cent is tagged in GCP Cloud Billing.
- `query_true_gcp_billing_export(...)`: Queries BigQuery Cloud Billing Export (`gcp_billing_export_resource_v1_*`) by `benchmark_run_id` to retrieve invoice-exact post-discount GCP costs.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any, Dict, Optional, Set

from shelf_benchmark.auth import create_bigquery_client, get_gcp_credentials
from shelf_benchmark.config import BenchmarkConfig, GCPBillingConfig, ModelPricing
from shelf_benchmark.models import CostMetrics, TokenUsageMetrics

logger = logging.getLogger(__name__)

# Values written to `CostMetrics.billing_source`. These are load-bearing: they are the only way a
# reader of a report can tell a metered rate from an estimate.
BILLING_SOURCE_LIVE_CATALOG = "gcp_cloud_billing_catalog_api"
BILLING_SOURCE_YAML = "yaml_rate_table"
BILLING_SOURCE_BQ_EXPORT = "bigquery_billing_export"

# Cloud Run service ID in the Cloud Billing Catalog API. The Vertex AI service ID is not duplicated
# here on purpose: it lives in `GCPBillingConfig.vertex_ai_service_id` so there is exactly one
# source of truth. A previous constant here disagreed with the configured value, so the "live"
# lookup queried a different service than the one operators had configured.
CLOUD_RUN_BILLING_SERVICE_ID = "152E-C115-5142"


class GCPBillingAndCostEngine:
    """Cloud Billing Catalog client, Provisioned Throughput GSU calculator, and BigQuery reconciler."""

    _live_catalog_cache: Dict[str, ModelPricing] = {}
    _catalog_miss_cache: Set[str] = set()

    def __init__(
        self,
        project_id: str = "unilever-shelf-understanding",
        billing_cfg: Optional[GCPBillingConfig] = None,
    ):
        self.project_id = project_id
        self.billing_cfg = billing_cfg or GCPBillingConfig()

    @classmethod
    def sanitize_gcp_label_value(cls, raw: str) -> str:
        """Convert any string into a valid GCP Billing label value (lowercase alphanumeric, dashes, underscores, max 63 chars)."""
        cleaned = re.sub(r"[^a-z0-9_-]", "-", (raw or "default").lower()).strip("-")
        return (cleaned or "default")[:63]

    @classmethod
    def build_gcp_billing_labels(
        cls,
        run_id: str,
        task_type: str = "classification",
        model_name: str = "gemini-3.8-flash",
        separation_approach: str = "single_pass_full_shelf",
        approach_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build GCP Billing labels to attach to `GenerateContentConfig(labels=...)` and Cloud Run requests."""
        active_approach = approach_id or separation_approach
        return {
            "application": "unilever-shelf-benchmark",
            "benchmark_run_id": cls.sanitize_gcp_label_value(run_id),
            "task_type": cls.sanitize_gcp_label_value(task_type),
            "separation_approach": cls.sanitize_gcp_label_value(active_approach),
            "vertex_model": cls.sanitize_gcp_label_value(model_name),
        }

    def resolve_live_or_configured_token_rates(
        self,
        model_name: str,
        fallback: Optional[ModelPricing] = None,
    ) -> tuple[ModelPricing, str]:
        """Resolve token rates, preferring the live Cloud Billing Catalog and falling back to config.

        Returns `(pricing, billing_source)` where `billing_source` is `gcp_cloud_billing_catalog_api`
        only if every rate was genuinely parsed from a SKU, and `yaml_rate_table` otherwise.
        """
        pricing, source = self.fetch_live_gcp_sku_pricing(
            model_name=model_name,
            billing_cfg=self.billing_cfg,
            project_id=self.project_id,
        )
        if source != BILLING_SOURCE_LIVE_CATALOG and fallback is not None:
            return fallback, BILLING_SOURCE_YAML
        return pricing, source

    @staticmethod
    def _sku_unit_price_usd(sku: Dict[str, Any]) -> Optional[float]:
        """Extract the base-tier unit price in USD from a Cloud Billing Catalog SKU."""
        for info in sku.get("pricingInfo") or []:
            expr = info.get("pricingExpression") or {}
            for tier in expr.get("tieredRates") or []:
                price = tier.get("unitPrice") or {}
                if price.get("currencyCode") != "USD":
                    continue
                units = float(price.get("units", 0) or 0)
                nanos = float(price.get("nanos", 0) or 0)
                amount = units + nanos / 1e9
                if amount <= 0.0:
                    continue  # Skip the free introductory tier.
                unit_desc = str(expr.get("usageUnitDescription", "")).lower()
                # Normalize everything to "USD per 1 million tokens".
                if "million" in unit_desc:
                    return amount
                if "thousand" in unit_desc:
                    return amount * 1000.0
                if unit_desc.strip() in ("token", "tokens", "count"):
                    return amount * 1_000_000.0
                return None  # Character- or image-based SKU; not comparable to token rates.
        return None

    @classmethod
    def _iter_catalog_skus(
        cls,
        service_id: str,
        token: str,
        project_id: str,
        timeout: float,
        max_pages: int = 20,
    ) -> list[Dict[str, Any]]:
        """Page through `cloudbilling.googleapis.com` SKUs for a service."""
        skus: list[Dict[str, Any]] = []
        page_token = ""
        for _ in range(max_pages):
            url = (
                f"https://cloudbilling.googleapis.com/v1/services/{service_id}/skus"
                f"?pageSize=5000"
            )
            if page_token:
                url += f"&pageToken={page_token}"
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-goog-user-project": project_id,
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    break
                payload = json.loads(resp.read().decode("utf-8"))
            skus.extend(payload.get("skus") or [])
            page_token = payload.get("nextPageToken") or ""
            if not page_token:
                break
        return skus

    @classmethod
    def _match_model_skus(
        cls,
        model_name: str,
        skus: list[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Map a model name onto its input/output token SKU rates (USD per 1M tokens).

        Matching is deliberately conservative: a SKU must mention every significant token of the
        model name. A wrong SKU match is worse than no match, because it produces a confident
        number sourced from the wrong product.
        """
        needles = [t for t in re.split(r"[-_.@/]", model_name.split("/")[-1].lower()) if t]
        if not needles:
            return {}

        rates: Dict[str, float] = {}
        for sku in skus:
            desc = str(sku.get("description", "")).lower()
            if not all(n in desc for n in needles):
                continue
            price = cls._sku_unit_price_usd(sku)
            if price is None:
                continue
            if "input" in desc or "prompt" in desc:
                key = "input"
            elif "thinking" in desc or "reasoning" in desc:
                key = "thinking"
            elif "output" in desc or "completion" in desc:
                key = "output"
            else:
                continue
            # Several regional SKUs share a description; keep the cheapest (the base rate).
            rates[key] = min(price, rates[key]) if key in rates else price
        return rates

    @classmethod
    def fetch_live_gcp_sku_pricing(
        cls,
        model_name: str,
        config: Optional[BenchmarkConfig] = None,
        billing_cfg: Optional[GCPBillingConfig] = None,
        project_id: Optional[str] = None,
    ) -> tuple[ModelPricing, str]:
        """Query the Cloud Billing Catalog API for live Vertex AI token SKU rates.

        Returns `(pricing, billing_source)`. `billing_source` is `gcp_cloud_billing_catalog_api`
        only when input and output rates were both parsed from real SKUs; otherwise the configured
        YAML rate table is returned with `yaml_rate_table`, so nobody mistakes an estimate for a
        metered rate.
        """
        clean_name = model_name.split("/")[-1]
        b_cfg = billing_cfg or (config.billing if config else None) or GCPBillingConfig()
        base_pricing = config.get_pricing(model_name) if config else ModelPricing()

        if clean_name in cls._live_catalog_cache:
            return cls._live_catalog_cache[clean_name], BILLING_SOURCE_LIVE_CATALOG

        if not b_cfg.use_live_cloud_billing_catalog_api:
            return base_pricing, BILLING_SOURCE_YAML
        if clean_name in cls._catalog_miss_cache:
            return base_pricing, BILLING_SOURCE_YAML

        eff_project = project_id or (config.gcp.project_id if config else "unilever-shelf-understanding")
        try:
            creds = get_gcp_credentials(eff_project)
            import google.auth.transport.requests

            if not getattr(creds, "token", None):
                creds.refresh(google.auth.transport.requests.Request())
            token = getattr(creds, "token", None)
            if not token:
                raise RuntimeError("No access token available for the Cloud Billing Catalog API.")

            skus = cls._iter_catalog_skus(
                service_id=b_cfg.vertex_ai_service_id,
                token=token,
                project_id=eff_project,
                timeout=b_cfg.catalog_api_timeout_seconds,
            )
            rates = cls._match_model_skus(clean_name, skus)
        except Exception as exc:
            logger.info(
                "Live Cloud Billing Catalog lookup unavailable for '%s' (%s). "
                "Using the configured YAML rate table; costs are estimates.",
                clean_name, exc,
            )
            cls._catalog_miss_cache.add(clean_name)
            return base_pricing, BILLING_SOURCE_YAML

        if "input" not in rates or "output" not in rates:
            logger.info(
                "No Cloud Billing SKU confidently matched model '%s' (found: %s). "
                "Using the configured YAML rate table; costs are estimates.",
                clean_name, sorted(rates) or "none",
            )
            cls._catalog_miss_cache.add(clean_name)
            return base_pricing, BILLING_SOURCE_YAML

        live = ModelPricing(
            input=rates["input"],
            # Thinking tokens are billed at the output rate unless a dedicated SKU exists.
            thinking=rates.get("thinking", rates["output"]),
            output=rates["output"],
        )
        logger.info(
            "Resolved live Cloud Billing SKU rates for '%s': in=$%.4f think=$%.4f out=$%.4f per 1M tokens.",
            clean_name, live.input, live.thinking, live.output,
        )
        cls._live_catalog_cache[clean_name] = live
        return live, BILLING_SOURCE_LIVE_CATALOG

    @classmethod
    def detect_traffic_type(
        cls,
        response: Any,
        billing_cfg: Optional[GCPBillingConfig] = None,
    ) -> str:
        """Detect whether Vertex AI served the request via `ON_DEMAND` (PAYG) or `PROVISIONED_THROUGHPUT` (GSU)."""
        if billing_cfg is not None:
            mode = getattr(billing_cfg, "traffic_mode", "auto").lower()
            if mode == "provisioned_throughput" or billing_cfg.provisioned_throughput.enabled:
                return "PROVISIONED_THROUGHPUT"
            if mode == "on_demand":
                return "ON_DEMAND"

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            raw_tt = str(getattr(usage, "traffic_type", "") or "").upper()
            if "PROVISIONED" in raw_tt:
                return "PROVISIONED_THROUGHPUT"
        return "ON_DEMAND"

    @classmethod
    def compute_all_in_separated_gcp_cost(
        cls,
        tokens: TokenUsageMetrics,
        pricing: ModelPricing,
        product_count: int,
        latency_ms: float = 2500.0,
        traffic_type: str = "ON_DEMAND",
        extra_embedding_or_vision_cost_usd: float = 0.0,
        billing_cfg: Optional[GCPBillingConfig] = None,
        gcp_labels: Optional[Dict[str, str]] = None,
        billing_source: str = BILLING_SOURCE_YAML,
        container_cpu_active_ms: Optional[float] = None,
    ) -> CostMetrics:
        """Compute separated GCP costs across token, GSU, embedding, granular Cloud Run compute/accelerator sub-buckets, and GCS/Logging."""
        b_cfg = billing_cfg or GCPBillingConfig()
        include_infra_overhead = b_cfg.include_infrastructure_costs
        eff_latency_ms = max(50.0, float(latency_ms or 2500.0))
        latency_sec = eff_latency_ms / 1000.0

        eff_traffic = str(getattr(tokens, "traffic_type", "") or traffic_type or "ON_DEMAND").upper()
        if b_cfg.provisioned_throughput.enabled:
            eff_traffic = "PROVISIONED_THROUGHPUT"

        # 1A. Always compute On-Demand PAYG Token Cost (Input + Thinking + Output)
        input_cost = (tokens.input_tokens / 1_000_000.0) * pricing.input
        thinking_cost = (tokens.thinking_tokens / 1_000_000.0) * pricing.thinking
        output_cost = (tokens.output_tokens / 1_000_000.0) * pricing.output
        payg_tokens_usd = input_cost + thinking_cost + output_cost

        # 1B. Always compute Amortized Provisioned Throughput (GSU) Cost for the exact latency & slot occupancy
        pt = b_cfg.provisioned_throughput
        gsu_count = float(pt.gsu_count)
        hourly_rate = float(pt.hourly_rate_per_gsu_usd)
        discount_pct = float(pt.monthly_commitment_discount_pct)
        hourly_total_gsu_usd = gsu_count * hourly_rate * max(0.0, 1.0 - discount_pct)
        slots = max(1, int(pt.concurrent_request_slots_per_gsu))
        slot_hourly_usd = hourly_total_gsu_usd / slots
        pt_gsu_usd = slot_hourly_usd * (latency_sec / 3600.0)

        # 2. Vertex AI embeddings / Cloud Vision per-facing API charge
        eff_products = max(1, int(product_count))
        embed_vision_usd = max(0.0, float(extra_embedding_or_vision_cost_usd or 0.0))

        # 3. Granular Cloud Run / Accelerator compute (vCPU + RAM + NVIDIA L4 GPU / Cloud TPU v5e/v6e + Request Fee).
        cr = b_cfg.cloud_run
        concurrency = max(1, int(cr.concurrency))
        billable_sec_per_req = latency_sec / float(concurrency)
        vcpu_rate = float(cr.vcpu_per_second_usd)
        mem_rate = float(cr.memory_gib_per_second_usd)
        accel_per_sec = cr.accelerator_per_second_usd()

        vcpu_cost = float(cr.vcpu_count) * vcpu_rate * billable_sec_per_req
        mem_cost = float(cr.memory_gib) * mem_rate * billable_sec_per_req
        accel_cost = accel_per_sec * billable_sec_per_req
        req_cost = float(cr.per_million_requests_usd) / 1_000_000.0
        raw_compute_usd = vcpu_cost + mem_cost + accel_cost + req_cost

        # `include_infrastructure_costs` is the only thing that decides whether modelled
        # infrastructure is folded into the reported total. It used to be `include_infra_overhead or
        # accel_per_sec > 0.0`, so merely selecting `--accelerator tpu-v5e` silently overrode an
        # explicit `include_infrastructure_costs: false` and made the two runs incomparable.
        if include_infra_overhead and cr.enabled:
            cloud_run_usd = raw_compute_usd
        else:
            cloud_run_usd = 0.0

        # Active Container Compute vs External API-Wait Cost Attribution
        if container_cpu_active_ms is not None:
            # A measured 0.0 is a legitimate measurement, not a missing one. The gate used to be
            # `is not None and > 0.0`, so a fast offline run whose CPU time rounded to zero fell
            # into the estimate branch and was billed on a guess.
            active_ms = min(eff_latency_ms, max(0.0, float(container_cpu_active_ms)))
        else:
            # No caller measured container CPU time. Everything in-tree now routes through
            # `pipeline.PipelineExecutor`, which always measures, so reaching this branch means a
            # third-party caller invoked the billing engine directly. The figure below is an
            # ESTIMATE, not a measurement. It was previously reached by the entire approach-plugin
            # path, which is one reason plugins and built-in tasks reported different costs.
            active_ms = min(eff_latency_ms * 0.15, 65.0 + (14.0 * eff_products))
        api_wait_ms = max(0.0, eff_latency_ms - active_ms)
        active_ratio = active_ms / eff_latency_ms if eff_latency_ms > 0 else 0.0
        compute_active_usd = raw_compute_usd * active_ratio
        compute_wait_tax_usd = raw_compute_usd * max(0.0, 1.0 - active_ratio)

        hw_profile = cr.hardware_summary() if hasattr(cr, "hardware_summary") else f"{cr.vcpu_count} vCPU / {cr.memory_gib} GiB RAM"

        # 4. GCS Class A/B operations + Cloud Logging ingestion
        gcs_ops_usd = (1.0 * (b_cfg.gcs_class_b_per_thousand_ops_usd / 1000.0)) + (
            2.0 * (b_cfg.gcs_class_a_per_thousand_ops_usd / 1000.0)
        )
        logging_usd = (
            float(b_cfg.estimated_log_bytes_per_run) / (1024.0 ** 3)
        ) * b_cfg.cloud_logging_per_gib_usd
        raw_gcs_obs_usd = gcs_ops_usd + logging_usd
        gcs_and_obs_usd = raw_gcs_obs_usd if include_infra_overhead else 0.0

        # 5. Total shelf cost & Pareto cost-latency metrics
        is_provisioned = eff_traffic == "PROVISIONED_THROUGHPUT"
        reported_pt_usd = pt_gsu_usd if is_provisioned else 0.0
        active_llm_usd = reported_pt_usd if is_provisioned else payg_tokens_usd

        # One definition of "the cost of this run", used by every derived figure below. Previously
        # `cost_per_1k_images_usd` and `compute_share_of_total_cost_pct` were each built from their
        # own ad-hoc mix of gated and un-gated components, so `cost_per_1k_images_usd` was not
        # 1000x `cost_per_shelf_image_usd` and always included infrastructure the headline total
        # had excluded.
        total_shelf_usd = active_llm_usd + embed_vision_usd + cloud_run_usd + gcs_and_obs_usd
        # Derive the other two headline figures from the *published* (rounded) per-image cost, not
        # from the unrounded intermediate. Rounding each independently made cost_per_1k_images_usd
        # differ from 1000x cost_per_shelf_image_usd, which readers reasonably treat as an error.
        reported_per_image_usd = round(total_shelf_usd, 8)
        per_product_usd = reported_per_image_usd / eff_products
        per_1k_usd = reported_per_image_usd * 1000.0

        # Compute's share is deliberately measured against the *full modelled* cost (infrastructure
        # always included), because the question it answers -- "how much of this is container time
        # rather than model time?" -- is meaningless once compute has been gated out of the total.
        full_modelled_usd = active_llm_usd + embed_vision_usd + raw_compute_usd + raw_gcs_obs_usd
        compute_share_pct = (raw_compute_usd / full_modelled_usd * 100.0) if full_modelled_usd > 0 else 0.0
        pareto_idx = per_1k_usd * latency_sec

        return CostMetrics(
            billing_source=billing_source,
            rates_from_live_catalog=(billing_source == BILLING_SOURCE_LIVE_CATALOG),
            includes_modelled_infrastructure=include_infra_overhead,
            traffic_type=eff_traffic,
            hardware_profile=hw_profile,
            input_cost_usd=round(input_cost, 8),
            thinking_cost_usd=round(thinking_cost, 8),
            output_cost_usd=round(output_cost, 8),
            vertex_ai_payg_tokens_usd=round(payg_tokens_usd, 8),
            vertex_ai_provisioned_throughput_usd=round(reported_pt_usd, 8),
            vertex_ai_embeddings_and_vision_usd=round(embed_vision_usd, 8),
            cloud_run_vcpu_usd=round(vcpu_cost, 8),
            cloud_run_memory_usd=round(mem_cost, 8),
            cloud_run_accelerator_usd=round(accel_cost, 8),
            cloud_run_request_fee_usd=round(req_cost, 8),
            cloud_run_compute_usd=round(cloud_run_usd, 8),
            container_cpu_active_ms=round(active_ms, 2),
            external_api_wait_ms=round(api_wait_ms, 2),
            compute_active_processing_usd=round(compute_active_usd, 8),
            compute_api_wait_idle_tax_usd=round(compute_wait_tax_usd, 8),
            compute_share_of_total_cost_pct=round(compute_share_pct, 3),
            gcs_and_observability_usd=round(gcs_and_obs_usd, 8),
            cost_per_shelf_image_usd=round(total_shelf_usd, 8),
            cost_per_product_usd=round(per_product_usd, 8),
            # 6 dp, not 4: at 4 dp this stopped being exactly 1000x cost_per_shelf_image_usd
            # (which is stored at 8 dp), so the two figures in the same report disagreed.
            cost_per_1k_images_usd=round(per_1k_usd, 6),
            cost_latency_pareto_index=round(pareto_idx, 4),
            product_count=product_count,
            gcp_billing_labels=gcp_labels or {},
        )

    @classmethod
    def get_bigquery_billing_reconciliation_sql(
        cls,
        export_table_uri: str,
        benchmark_run_id: Optional[str] = None,
    ) -> str:
        """Generate the exact BigQuery SQL query against GCP Cloud Billing Export (`gcp_billing_export_resource_v1_*`) to reconcile true post-discount GCP invoice costs separated by service and SKU."""
        where_clause = (
            f"WHERE EXISTS (SELECT 1 FROM UNNEST(labels) l WHERE l.key = 'benchmark_run_id' AND l.value = '{cls.sanitize_gcp_label_value(benchmark_run_id)}')"
            if benchmark_run_id
            else "WHERE EXISTS (SELECT 1 FROM UNNEST(labels) l WHERE l.key = 'application' AND l.value = 'unilever-shelf-benchmark')"
        )
        return f"""SELECT
  service.description AS gcp_service,
  sku.description AS gcp_sku,
  (SELECT l.value FROM UNNEST(labels) l WHERE l.key = 'benchmark_run_id' LIMIT 1) AS benchmark_run_id,
  (SELECT l.value FROM UNNEST(labels) l WHERE l.key = 'separation_approach' LIMIT 1) AS separation_approach,
  (SELECT l.value FROM UNNEST(labels) l WHERE l.key = 'vertex_model' LIMIT 1) AS vertex_model,
  SUM(usage.amount) AS usage_amount,
  ANY_VALUE(usage.unit) AS usage_unit,
  ROUND(SUM(cost) + SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)), 8) AS net_true_gcp_cost_usd
FROM `{export_table_uri}`
{where_clause}
GROUP BY gcp_service, gcp_sku, benchmark_run_id, separation_approach, vertex_model
ORDER BY net_true_gcp_cost_usd DESC"""

    def build_bigquery_true_cost_reconciliation_sql(
        self,
        run_id: str,
        export_table: str,
    ) -> str:
        """Instance wrapper returning the parameterized BigQuery Cloud Billing Export SQL query."""
        return self.get_bigquery_billing_reconciliation_sql(
            export_table_uri=export_table,
            benchmark_run_id=run_id,
        )

    def generate_cloud_run_deploy_command(
        self,
        service_name: str = "unilever-shelf-benchmark-service",
        region: str = "us-central1",
    ) -> Dict[str, Any]:
        """Return the ready-to-run `gcloud run deploy` command and billing labels for Cloud Run hosting."""
        cr = self.billing_cfg.cloud_run
        cmd = (
            f"gcloud run deploy {service_name} "
            f"--source . "
            f"--project {self.project_id} "
            f"--region {region} "
            f"--cpu {int(cr.vcpu_count)} "
            f"--memory {int(cr.memory_gib)}Gi "
            f"--concurrency {int(cr.concurrency)} "
            f"--labels application=unilever-shelf-benchmark,component=benchmark-runner "
            f"--set-env-vars GCP_PROJECT_ID={self.project_id} "
            f"--no-allow-unauthenticated"
        )
        return {
            "service_name": service_name,
            "region": region,
            "vcpu_count": cr.vcpu_count,
            "memory_gib": cr.memory_gib,
            "gcloud_deploy_command": cmd,
        }

    @classmethod
    def query_true_gcp_billing_export(
        cls,
        config: BenchmarkConfig,
        benchmark_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the BigQuery Cloud Billing Export reconciliation query on GCP (or return the ready SQL if the Billing Export table is not yet linked)."""
        table_uri = config.billing.bigquery_billing_export_table
        if not table_uri:
            sample_table = f"{config.gcp.project_id}.billing_export.gcp_billing_export_resource_v1_XXXXXX"
            return {
                "status": "READY_FOR_BILLING_EXPORT_TABLE",
                "sql_query": cls.get_bigquery_billing_reconciliation_sql(sample_table, benchmark_run_id),
                "rows": [],
            }
        sql = cls.get_bigquery_billing_reconciliation_sql(table_uri, benchmark_run_id)
        try:
            bq = create_bigquery_client(config.gcp.project_id)
            query_job = bq.query(sql)
            rows = [dict(r) for r in query_job.result()]
            return {
                "status": "RECONCILED_FROM_GCP_BIGQUERY_BILLING_EXPORT",
                "sql_query": sql,
                "rows": rows,
            }
        except Exception as exc:
            return {
                "status": "BIGQUERY_QUERY_PREPARED",
                "reason": str(exc),
                "sql_query": sql,
                "rows": [],
            }

    @classmethod
    def check_pricing_staleness(
        cls,
        billing_cfg: Optional[GCPBillingConfig] = None,
        *,
        reference_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate the age of `billing_cfg.pricing_last_verified_date` against `max_pricing_age_days`."""
        from datetime import date

        cfg = billing_cfg or GCPBillingConfig()
        verified = date.fromisoformat(cfg.pricing_last_verified_date)
        ref = date.fromisoformat(reference_date) if reference_date else date.today()
        age_days = (ref - verified).days
        max_age = cfg.max_pricing_age_days
        is_stale = bool(max_age is not None and age_days > max_age)
        if is_stale:
            raise ValueError(
                f"Pricing rate table was last verified on {cfg.pricing_last_verified_date} "
                f"({age_days} days ago), exceeding max_pricing_age_days={max_age}. "
                f"Update the rates in configs/default_config.yaml and bump pricing_last_verified_date."
            )
        return {
            "pricing_last_verified_date": cfg.pricing_last_verified_date,
            "reference_date": ref.isoformat(),
            "age_days": age_days,
            "max_pricing_age_days": max_age,
            "is_stale": is_stale,
        }

    @classmethod
    def reconcile_modelled_vs_billed(
        cls,
        modelled_cost_usd: float,
        billed_cost_usd: float,
        *,
        tolerance_pct: float = 10.0,
    ) -> Dict[str, Any]:
        """Compare modelled run cost against actual BigQuery billing export cost for the same `run_id`."""
        delta_usd = round(modelled_cost_usd - billed_cost_usd, 8)
        denom = max(abs(billed_cost_usd), 1e-9)
        drift_pct = round((abs(delta_usd) / denom) * 100.0, 2) if billed_cost_usd > 0 else (0.0 if modelled_cost_usd == 0 else 100.0)
        within_tolerance = drift_pct <= tolerance_pct
        return {
            "modelled_cost_usd": round(modelled_cost_usd, 8),
            "billed_cost_usd": round(billed_cost_usd, 8),
            "delta_usd": delta_usd,
            "drift_pct": drift_pct,
            "tolerance_pct": tolerance_pct,
            "within_tolerance": within_tolerance,
        }
