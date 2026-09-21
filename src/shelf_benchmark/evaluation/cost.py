"""Token usage extraction & Cost per Shelf Image / Cost per Product calculators."""

from __future__ import annotations

from typing import Any

from shelf_benchmark.config import ModelPricing
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
) -> CostMetrics:
    """Compute cost per shelf image and cost per product (USD)."""
    input_cost = (tokens.input_tokens / 1_000_000.0) * pricing.input
    thinking_cost = (tokens.thinking_tokens / 1_000_000.0) * pricing.thinking
    output_cost = (tokens.output_tokens / 1_000_000.0) * pricing.output
    total_image_cost = input_cost + thinking_cost + output_cost
    effective_products = max(product_count, 1)
    cost_per_product = total_image_cost / effective_products

    return CostMetrics(
        input_cost_usd=round(input_cost, 8),
        thinking_cost_usd=round(thinking_cost, 8),
        output_cost_usd=round(output_cost, 8),
        cost_per_shelf_image_usd=round(total_image_cost, 8),
        cost_per_product_usd=round(cost_per_product, 8),
        product_count=product_count,
    )
