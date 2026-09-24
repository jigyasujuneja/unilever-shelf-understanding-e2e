"""Pricing, FinOps SLA Calculator, and Riley's 5-Bucket Enterprise GCP Billing (PAYG vs. GSU Provisioned Throughput + Cloud Run + BigQuery SQL).

Enforces Unilever's hard ceiling: <= ₹0.22 (~$0.002619 USD at 1 USD = 84 INR) per image at 500,000 images/day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


USD_TO_INR_RATE: float = 84.0
MAX_INR_PER_IMAGE_SLA: float = 0.22
MAX_USD_PER_IMAGE_SLA: float = MAX_INR_PER_IMAGE_SLA / USD_TO_INR_RATE  # ~$0.002619 USD
MAX_P95_LATENCY_MS_SLA: float = 20000.0


@dataclass(frozen=True)
class CostRateConfig:
    """Standard GCP / Vertex AI unit cost rates in USD."""

    usd_to_inr: float = USD_TO_INR_RATE
    usd_to_inr_rate: float = USD_TO_INR_RATE
    hard_ceiling_inr: float = MAX_INR_PER_IMAGE_SLA
    input_token_per_1k_usd: float = 0.000075
    output_token_per_1k_usd: float = 0.000300
    l4_gpu_per_second_usd: float = 0.000220
    vcpu_per_second_usd: float = 0.000024
    vector_query_per_1k_usd: float = 0.000400
    crop_embedding_per_1k_usd: float = 0.000800
    diffusion_deglare_per_crop_usd: float = 0.000025


@dataclass(frozen=True)
class BlendedCostBreakdown:
    """Detailed per-image cost breakdown across tokens, GPU/CPU compute, and embeddings."""

    token_cost_usd: float
    compute_cost_usd: float
    vector_and_embedding_cost_usd: float
    diffusion_cost_usd: float
    total_cost_usd: float
    total_cost_inr: float
    within_sla_ceiling: bool
    within_hard_ceiling: bool


def calculate_blended_cost(
    input_tokens: int,
    output_tokens: int,
    gpu_seconds: float = 0.0,
    vcpu_seconds: float = 0.0,
    vector_queries: int = 0,
    embedded_crops: int = 0,
    diffusion_deglare_crops: int = 0,
    config: CostRateConfig = CostRateConfig(),
    rates: CostRateConfig | None = None,
) -> BlendedCostBreakdown:
    """Compute blended USD and INR cost per shelf image and verify against the <= ₹0.22 ceiling."""
    active_cfg = rates if rates is not None else config
    fx = active_cfg.usd_to_inr_rate if active_cfg.usd_to_inr_rate else active_cfg.usd_to_inr
    ceiling = active_cfg.hard_ceiling_inr if active_cfg.hard_ceiling_inr else MAX_INR_PER_IMAGE_SLA

    token_usd = (input_tokens / 1000.0) * active_cfg.input_token_per_1k_usd + (
        output_tokens / 1000.0
    ) * active_cfg.output_token_per_1k_usd
    compute_usd = (gpu_seconds * active_cfg.l4_gpu_per_second_usd) + (
        vcpu_seconds * active_cfg.vcpu_per_second_usd
    )
    vec_usd = (vector_queries / 1000.0) * active_cfg.vector_query_per_1k_usd + (
        embedded_crops / 1000.0
    ) * active_cfg.crop_embedding_per_1k_usd
    diff_usd = diffusion_deglare_crops * active_cfg.diffusion_deglare_per_crop_usd

    total_usd = token_usd + compute_usd + vec_usd + diff_usd
    total_inr = round(total_usd * fx, 4)
    is_within = total_inr <= ceiling
    return BlendedCostBreakdown(
        token_cost_usd=round(token_usd, 6),
        compute_cost_usd=round(compute_usd, 6),
        vector_and_embedding_cost_usd=round(vec_usd, 6),
        diffusion_cost_usd=round(diff_usd, 6),
        total_cost_usd=round(total_usd, 6),
        total_cost_inr=total_inr,
        within_sla_ceiling=is_within,
        within_hard_ceiling=is_within,
    )


@dataclass(frozen=True)
class FiveBucketGCPBillingReport:
    """Riley's 5-Bucket Enterprise GCP Billing Separation + Provisioned Throughput (GSU) Capacity Model."""

    vertex_ai_payg_tokens_usd: float
    vertex_ai_provisioned_throughput_gsu_usd: float
    vertex_ai_embeddings_and_vision_usd: float
    cloud_run_compute_usd: float
    gcs_and_observability_usd: float
    total_payg_cost_per_image_usd: float
    total_payg_cost_per_image_inr: float
    total_gsu_committed_cost_per_image_usd: float
    total_gsu_committed_cost_per_image_inr: float
    recommended_gsu_units_for_525_workers: int
    within_inr_ceiling_payg: bool
    within_inr_ceiling_gsu: bool
    bigquery_reconciliation_sql: str


def compute_five_bucket_gcp_billing(
    run_id: str,
    vertex_tokens_usd: float,
    embeddings_and_vision_usd: float,
    image_latency_ms: float,
    daily_volume_images: int = 500_000,
    target_concurrent_workers: int = 525,
) -> FiveBucketGCPBillingReport:
    """Compute the 5 separated GCP cost buckets (PAYG vs. Reserved GSU Provisioned Throughput + Cloud Run + GCS)."""
    latency_sec = max(0.05, image_latency_ms / 1000.0)
    cloud_run_usd = ((2.0 * 0.000024 + 4.0 * 0.0000025) * latency_sec) / 8.0
    gcs_obs_usd = 0.000012

    gsu_tokens_usd = vertex_tokens_usd * 0.62
    total_payg_usd = vertex_tokens_usd + embeddings_and_vision_usd + cloud_run_usd + gcs_obs_usd
    total_gsu_usd = gsu_tokens_usd + embeddings_and_vision_usd + cloud_run_usd + gcs_obs_usd

    total_payg_inr = total_payg_usd * USD_TO_INR_RATE
    total_gsu_inr = total_gsu_usd * USD_TO_INR_RATE

    gsu_units = max(2, min(32, int(round((vertex_tokens_usd / 0.0004) * 2.0))))

    bq_sql = (
        "SELECT labels.value AS run_id, service.description AS gcp_service, sku.description AS gcp_sku, "
        "ROUND(SUM(cost), 6) AS gross_cost_usd, ROUND(SUM(cost) * 84.0, 4) AS gross_cost_inr "
        "FROM `unilever-shelf-understanding.billing_export.gcp_billing_export_resource_v1` "
        f"LEFT JOIN UNNEST(labels) AS labels ON labels.key = 'unilever_shelf_run_id' "
        f"WHERE labels.value = '{run_id}' GROUP BY 1, 2, 3 ORDER BY gross_cost_usd DESC;"
    )

    return FiveBucketGCPBillingReport(
        vertex_ai_payg_tokens_usd=round(vertex_tokens_usd, 6),
        vertex_ai_provisioned_throughput_gsu_usd=round(gsu_tokens_usd, 6),
        vertex_ai_embeddings_and_vision_usd=round(embeddings_and_vision_usd, 6),
        cloud_run_compute_usd=round(cloud_run_usd, 6),
        gcs_and_observability_usd=round(gcs_obs_usd, 6),
        total_payg_cost_per_image_usd=round(total_payg_usd, 6),
        total_payg_cost_per_image_inr=round(total_payg_inr, 4),
        total_gsu_committed_cost_per_image_usd=round(total_gsu_usd, 6),
        total_gsu_committed_cost_per_image_inr=round(total_gsu_inr, 4),
        recommended_gsu_units_for_525_workers=gsu_units,
        within_inr_ceiling_payg=(total_payg_inr <= MAX_INR_PER_IMAGE_SLA),
        within_inr_ceiling_gsu=(total_gsu_inr <= MAX_INR_PER_IMAGE_SLA),
        bigquery_reconciliation_sql=bq_sql,
    )
