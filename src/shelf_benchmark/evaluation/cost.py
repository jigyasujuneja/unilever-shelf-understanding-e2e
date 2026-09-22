"""Token usage extraction & All-In Separated GCP Cost Calculators."""

from __future__ import annotations

from typing import Any, Dict, Optional

from shelf_benchmark.config import GCPBillingConfig, ModelPricing
from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
from shelf_benchmark.models import CostMetrics, TokenUsageMetrics


def extract_token_usage(response: Any) -> TokenUsageMetrics:
    """Extract input, thinking (thoughts), output, total, and cached tokens from a Vertex AI GenAI response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return TokenUsageMetrics()

    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    cached_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_token_count", 0)
        or (input_tokens + thinking_tokens + output_tokens)
    )
    return TokenUsageMetrics(
        input_tokens=input_tokens,
        thinking_tokens=thinking_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
    )


def compute_cost_metrics(
    tokens: TokenUsageMetrics,
    pricing: ModelPricing,
    product_count: int,
    latency_ms: float = 2500.0,
    traffic_type: str = "ON_DEMAND",
    extra_embedding_or_vision_cost_usd: float = 0.0,
    billing_cfg: Optional[GCPBillingConfig] = None,
    project_id: Optional[str] = None,
    model_name: Optional[str] = None,
    gcp_labels: Optional[Dict[str, str]] = None,
) -> CostMetrics:
    """Compute 100% separated GCP costs (Vertex AI PAYG Tokens, Provisioned Throughput GSUs, Embeddings/Vision, Cloud Run Compute, and GCS/Observability)."""
    return GCPBillingAndCostEngine.compute_all_in_separated_gcp_cost(
        tokens=tokens,
        pricing=pricing,
        product_count=product_count,
        latency_ms=latency_ms,
        traffic_type=traffic_type,
        extra_embedding_or_vision_cost_usd=extra_embedding_or_vision_cost_usd,
        billing_cfg=billing_cfg,
        gcp_labels=gcp_labels,
    )
