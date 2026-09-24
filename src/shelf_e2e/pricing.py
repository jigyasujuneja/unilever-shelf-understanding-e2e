"""Blended USD and INR (₹) Unit Cost Calculator (SPEC-001 & SPEC-002)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostRateConfig:
    """Pricing rates defined in SPEC-002 Section 1.B."""

    input_token_usd_per_1m: float = 0.075  # Gemini 2.5 Flash Lite ($0.075 / 1M input tokens)
    output_token_usd_per_1m: float = 0.300  # Gemini 2.5 Flash Lite ($0.30 / 1M output tokens)
    cloud_run_l4_gpu_hourly_usd: float = 0.85  # Cloud Run L4 GPU hourly rate
    cloud_run_vcpu_hourly_usd: float = 0.086  # Cloud Run vCPU hourly rate
    scann_query_usd_per_1k: float = 0.00015  # Vertex AI Vector Search QPS rate
    embedding_usd_per_1k_crops: float = 0.00020  # Multimodal embedding rate
    usd_to_inr_rate: float = 84.0  # Fixed conversion rate: 1 USD = 84 INR
    hard_ceiling_inr: float = 0.22  # SPEC-001 hard ceiling: <= ₹0.22 / image


@dataclass(frozen=True)
class BlendedCostBreakdown:
    """Itemized cost per shelf image in USD and Indian Rupees (₹)."""

    token_cost_usd: float
    compute_cost_usd: float
    vector_search_cost_usd: float
    total_cost_usd: float
    total_cost_inr: float
    within_hard_ceiling: bool


def calculate_blended_cost(
    input_tokens: int,
    output_tokens: int,
    gpu_seconds: float = 0.0,
    vcpu_seconds: float = 0.0,
    vector_queries: int = 0,
    embedded_crops: int = 0,
    rates: CostRateConfig = CostRateConfig(),
) -> BlendedCostBreakdown:
    """Compute exact blended unit cost in USD and INR (₹ at 1 USD = 84 INR)."""
    token_cost_usd = (
        (max(0, input_tokens) / 1_000_000.0) * rates.input_token_usd_per_1m
        + (max(0, output_tokens) / 1_000_000.0) * rates.output_token_usd_per_1m
    )
    compute_cost_usd = (
        (max(0.0, gpu_seconds) / 3600.0) * rates.cloud_run_l4_gpu_hourly_usd
        + (max(0.0, vcpu_seconds) / 3600.0) * rates.cloud_run_vcpu_hourly_usd
    )
    vector_search_cost_usd = (
        (max(0, vector_queries) / 1_000.0) * rates.scann_query_usd_per_1k
        + (max(0, embedded_crops) / 1_000.0) * rates.embedding_usd_per_1k_crops
    )

    total_cost_usd = token_cost_usd + compute_cost_usd + vector_search_cost_usd
    total_cost_inr = total_cost_usd * rates.usd_to_inr_rate

    return BlendedCostBreakdown(
        token_cost_usd=round(token_cost_usd, 8),
        compute_cost_usd=round(compute_cost_usd, 8),
        vector_search_cost_usd=round(vector_search_cost_usd, 8),
        total_cost_usd=round(total_cost_usd, 8),
        total_cost_inr=round(total_cost_inr, 6),
        within_hard_ceiling=(total_cost_inr <= rates.hard_ceiling_inr),
    )
