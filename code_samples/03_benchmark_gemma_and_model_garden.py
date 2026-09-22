"""Sample 03: Benchmarking Gemma (Gemma 3 / Gemma 2 / PaliGemma) & Model Garden Endpoints.

Demonstrates two simple ways to benchmark open-weights Gemma models with full OpenTelemetry
token logging, cost calculation, and 7-dimension taxonomy validation:
  Option A: Vertex AI Model Garden / MaaS endpoint (`provider_family="vertex_gemma"`)
  Option B: Any custom Gemma server (Cloud Run, vLLM, HuggingFace TGI, Ollama) via a simple Python function (`provider_family="custom_callable"`)
"""

from typing import Any, Dict, Optional

from shelf_benchmark import ModelPricing, ShelfBenchmarkSDK, UniversalModelSpec


def custom_gemma_vllm_adapter(
    prompt: str,
    image_uri: str,
    response_schema: Optional[Any] = None,
) -> Dict[str, Any]:
    """Example adapter for a Gemma 3 / PaliGemma model hosted on vLLM, Cloud Run, or Vertex AI Custom Prediction.

    Replace the HTTP call inside this function with your `requests.post("https://your-gemma-endpoint/v1/chat/completions", ...)`
    call. Return a dict matching the requested task schema plus optional `_token_usage`.
    """
    return {
        "total_classified_products": 2,
        "distinct_brands_found": ["Pepsodent", "Pond's"],
        "classified_products": [
            {
                "product_index": 1,
                "bbox_2d": [171, 676, 487, 813],
                "shelf_row": "top",
                "position_on_shelf": 1,
                "category": "Oral Care",
                "subcategory": "Toothpaste",
                "brand": "Pepsodent",
                "is_hul_brand": True,
                "variant": "Germi Check",
                "packaging_type": "box",
                "pack_type": "Single",
                "size": "150g",
                "product_name": "Pepsodent Germi Check Toothpaste",
                "confidence": 0.96,
            },
            {
                "product_index": 2,
                "bbox_2d": [535, 386, 795, 440],
                "shelf_row": "middle",
                "position_on_shelf": 2,
                "category": "Skin Care",
                "subcategory": "Face Wash",
                "brand": "Pond's",
                "is_hul_brand": True,
                "variant": "Bright Beauty Spot-less Glow",
                "packaging_type": "tube",
                "pack_type": "Single",
                "size": "100g",
                "product_name": "Pond's Bright Beauty Face Wash",
                "confidence": 0.95,
            },
        ],
        # Optional: pass exact token counts from your vLLM / Gemma endpoint response for OTel logging
        "_token_usage": {
            "input_tokens": 640,
            "thinking_tokens": 0,
            "output_tokens": 210,
        },
    }


def main() -> None:
    sdk = ShelfBenchmarkSDK(output_dir="reports/sample_03_gemma")

    # Option A: Vertex AI Model Garden Endpoint (uncomment and set your deployed endpoint_uri when live)
    gemma_vertex_endpoint_spec = UniversalModelSpec(
        model_id="gemma-3-27b-it",
        display_name="gemma-3-27b-it-vertex-endpoint",
        provider_family="vertex_gemma",
        endpoint_uri="projects/unilever-shelf-understanding/locations/us-central1/endpoints/YOUR_GEMMA_ENDPOINT_ID",
        pricing=ModelPricing(input=0.08, thinking=0.0, output=0.24),
    )

    # Option B: Custom Gemma Callable (runs out-of-the-box for testing adapter integration + OTel logging)
    gemma_custom_spec = UniversalModelSpec(
        model_id="gemma-3-27b-it-vllm",
        display_name="gemma-3-27b-it-vllm",
        provider_family="custom_callable",
        custom_handler=custom_gemma_vllm_adapter,
        pricing=ModelPricing(input=0.08, thinking=0.0, output=0.24),
    )
    sdk.register_model(gemma_custom_spec)

    summary = sdk.run_suite(
        models=["gemma-3-27b-it-vllm"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    )

    for res in summary["results"]:
        print(
            f"[Gemma Benchmark] Model={res.model_name} | Facings={len(res.row_level_items)} | "
            f"Tokens={res.tokens.total_tokens} | Cost/Image=${res.cost.cost_per_shelf_image_usd:.6f} | "
            f"OTel Log={summary['otel_log_path']}"
        )


if __name__ == "__main__":
    main()
