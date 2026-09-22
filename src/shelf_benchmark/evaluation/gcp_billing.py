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
    ) -> CostMetrics:
        """Compute separated GCP costs across token, GSU, embedding, Cloud Run, and GCS buckets.

        `include_infrastructure_costs` gates the modelled Cloud Run / GCS / Cloud Logging buckets.
        It defaults to off because those are simulations of what a serving deployment *would* cost,
        and folding them into a laptop benchmark run both inflates the totals and compresses the
        measured differences between models into noise.
        """
        b_cfg = billing_cfg or GCPBillingConfig()
        include_infra_overhead = b_cfg.include_infrastructure_costs
        latency_sec = max(0.05, float(latency_ms or 2500.0) / 1000.0)

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
        gsu_count = float(getattr(pt, "gsu_count", None) or getattr(pt, "reserved_gsus", 1) or 1)
        hourly_rate = float(getattr(pt, "hourly_rate_per_gsu_usd", None) or getattr(pt, "gsu_hourly_rate_usd", 22.0) or 22.0)
        discount_pct = float(getattr(pt, "monthly_commitment_discount_pct", 0.0) or 0.0)
        hourly_total_gsu_usd = gsu_count * hourly_rate * max(0.0, 1.0 - discount_pct)
        slots = max(1, int(getattr(pt, "concurrent_request_slots_per_gsu", 4)))
        slot_hourly_usd = hourly_total_gsu_usd / slots
        pt_gsu_usd = slot_hourly_usd * (latency_sec / 3600.0)

        # 2. Vertex AI embeddings / Cloud Vision. This is a genuine per-facing API charge incurred
        #    by the approach itself, so it is NOT gated on include_infrastructure_costs. The caller
        #    supplies the amount from `billing.embeddings_and_vision`; a zero here means the
        #    approach performed no embedding calls, which must not be silently back-filled.
        eff_products = max(1, int(product_count))
        embed_vision_usd = max(0.0, float(extra_embedding_or_vision_cost_usd or 0.0))

        # 3. Cloud Run compute (modelled, not measured).
        if include_infra_overhead and b_cfg.cloud_run.enabled:
            cr = b_cfg.cloud_run
            vcpu_rate = float(getattr(cr, "vcpu_per_second_usd", None) or getattr(cr, "vcpu_second_rate_usd", 0.000024))
            mem_rate = float(getattr(cr, "memory_gib_per_second_usd", None) or getattr(cr, "gib_second_rate_usd", 0.0000025))
            vcpu_cost = float(cr.vcpu_count) * vcpu_rate * latency_sec
            mem_cost = float(cr.memory_gib) * mem_rate * latency_sec
            req_cost = float(cr.per_million_requests_usd) / 1_000_000.0
            cloud_run_usd = vcpu_cost + mem_cost + req_cost
        else:
            cloud_run_usd = 0.0

        # 4. GCS Class A/B operations + Cloud Logging ingestion (modelled, not measured).
        if include_infra_overhead:
            gcs_ops_usd = (1.0 * (b_cfg.gcs_class_b_per_thousand_ops_usd / 1000.0)) + (
                2.0 * (b_cfg.gcs_class_a_per_thousand_ops_usd / 1000.0)
            )
            logging_usd = (
                float(b_cfg.estimated_log_bytes_per_run) / (1024.0 ** 3)
            ) * b_cfg.cloud_logging_per_gib_usd
            gcs_and_obs_usd = gcs_ops_usd + logging_usd
        else:
            gcs_and_obs_usd = 0.0

        # 5. Active inference cost depends on how Vertex served the request. Provisioned throughput
        #    is a model-serving cost, not infrastructure overhead, so it is reported whenever the
        #    traffic was actually served that way.
        is_provisioned = eff_traffic == "PROVISIONED_THROUGHPUT"
        reported_pt_usd = pt_gsu_usd if is_provisioned else 0.0
        active_llm_usd = reported_pt_usd if is_provisioned else payg_tokens_usd
        total_shelf_usd = active_llm_usd + embed_vision_usd + cloud_run_usd + gcs_and_obs_usd
        per_product_usd = total_shelf_usd / eff_products

        return CostMetrics(
            billing_source=billing_source,
            rates_from_live_catalog=(billing_source == BILLING_SOURCE_LIVE_CATALOG),
            includes_modelled_infrastructure=include_infra_overhead,
            traffic_type=eff_traffic,
            input_cost_usd=round(input_cost, 8),
            thinking_cost_usd=round(thinking_cost, 8),
            output_cost_usd=round(output_cost, 8),
            vertex_ai_payg_tokens_usd=round(payg_tokens_usd, 8),
            vertex_ai_provisioned_throughput_usd=round(reported_pt_usd, 8),
            vertex_ai_embeddings_and_vision_usd=round(embed_vision_usd, 8),
            cloud_run_compute_usd=round(cloud_run_usd, 8),
            gcs_and_observability_usd=round(gcs_and_obs_usd, 8),
            cost_per_shelf_image_usd=round(total_shelf_usd, 8),
            cost_per_product_usd=round(per_product_usd, 8),
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
            f"--allow-unauthenticated"
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
