#!/usr/bin/env python3
"""Sample 03: Benchmark Gemma / Model Garden / any self-hosted endpoint.

Run it (no GCP project, no credentials, no network):

    .venv/bin/python code_samples/03_benchmark_gemma_and_model_garden.py

Two ways to plug a non-Gemini model into the suite:

  Option A  A Vertex AI Model Garden / MaaS endpoint, via `provider_family="vertex_gemma"`
            and an `endpoint_uri`. Needs a deployed endpoint, so it is only registered here,
            not called.
  Option B  Any HTTP endpoint you control (Cloud Run, vLLM, TGI, Ollama) wrapped in a Python
            function, via `provider_family="custom_callable"`. This one runs for real below,
            offline, against the bundled fixture image.

The adapter contract for Option B is:

    handler(prompt: str, image_uri: str, response_schema: Any | None) -> dict

Return a dict matching the task schema. Optionally include `_token_usage` so the suite can log
and cost real token counts from your server instead of guessing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from shelf_benchmark import ModelPricing, UniversalModelSpec
from shelf_benchmark.testing import OFFLINE_IMAGE_URI, make_offline_sdk


def custom_gemma_vllm_adapter(
    prompt: str,
    image_uri: str,
    response_schema: Optional[Any] = None,
) -> Dict[str, Any]:
    """Stand-in for a Gemma 3 / PaliGemma server.

    Replace the body with your own call, for example:

        resp = requests.post(
            "https://your-gemma-endpoint/v1/chat/completions",
            json={"model": "gemma-3-27b-it", "messages": messages},
            timeout=120,
        ).json()
        return json.loads(resp["choices"][0]["message"]["content"])

    `prompt` is the fully rendered task prompt (taxonomy included) and `image_uri` is the shelf
    image the suite wants classified.
    """
    return {
        "total_classified_products": 2,
        "distinct_brands_found": ["Brand_A", "Brand_A"],
        "classified_products": [
            {
                "product_index": 1,
                "bbox_2d": [171, 676, 487, 813],
                "shelf_row": "top",
                "position_on_shelf": 1,
                "category": "Oral Care",
                "subcategory": "Toothpaste",
                "brand": "Brand_A",
                "is_hul_brand": True,
                "variant": "Germi Check",
                "packaging_type": "box",
                "pack_type": "Single",
                "size": "150g",
                "product_name": "Brand_A Germi Check Toothpaste",
                "confidence": 0.96,
            },
            {
                "product_index": 2,
                "bbox_2d": [535, 386, 795, 440],
                "shelf_row": "middle",
                "position_on_shelf": 2,
                "category": "Skin Care",
                "subcategory": "Face Wash",
                "brand": "Brand_A",
                "is_hul_brand": True,
                "variant": "Bright Beauty Spot-less Glow",
                "packaging_type": "tube",
                "pack_type": "Single",
                "size": "100g",
                "product_name": "Brand_A Radiance Daily Cleanser",
                "confidence": 0.95,
            },
        ],
        # Optional. Exact counts from your server, used for OTel logging and cost.
        "_token_usage": {
            "input_tokens": 640,
            "thinking_tokens": 0,
            "output_tokens": 210,
        },
    }


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="shelf-gemma-"))
    sdk = make_offline_sdk(work)

    # Option A: Vertex AI Model Garden endpoint. Registered only; calling it needs a live endpoint.
    gemma_vertex_endpoint_spec = UniversalModelSpec(
        model_id="gemma-3-27b-it",
        display_name="gemma-3-27b-it-vertex-endpoint",
        provider_family="vertex_gemma",
        endpoint_uri=(
            "projects/unilever-shelf-understanding/locations/us-central1/"
            "endpoints/YOUR_GEMMA_ENDPOINT_ID"
        ),
        pricing=ModelPricing(input=0.08, thinking=0.0, output=0.24),
    )
    sdk.register_model(gemma_vertex_endpoint_spec)

    # Option B: your own server behind a Python function. Runs offline, right now.
    gemma_custom_spec = UniversalModelSpec(
        model_id="gemma-3-27b-it-vllm",
        display_name="gemma-3-27b-it-vllm",
        provider_family="custom_callable",
        custom_handler=custom_gemma_vllm_adapter,
        pricing=ModelPricing(input=0.08, thinking=0.0, output=0.24),
    )
    sdk.register_model(gemma_custom_spec)
    print(f"Registered models: {sdk.config.models}")

    summary = sdk.run_suite(
        models=["gemma-3-27b-it-vllm"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri=OFFLINE_IMAGE_URI,
    )

    for res in summary["results"]:
        print(
            f"[Gemma benchmark] model={res.model_name} "
            f"facings={len(res.row_level_items)} "
            f"tokens={res.tokens.total_tokens} "
            f"(in={res.tokens.input_tokens} out={res.tokens.output_tokens}) "
            f"cost/image=${res.cost.cost_per_shelf_image_usd:.6f} "
            f"billing_source={res.cost.billing_source}"
        )
        print(f"  accuracy_status={res.accuracy.accuracy_status} (no annotations connected)")
    print(f"OTel log: {summary['otel_log_path']}")
    print(
        "\nTo run this against the real endpoint, build the SDK with\n"
        "  ShelfBenchmarkSDK(config_path='configs/default_config.yaml')\n"
        "and pass a gs:// shelf image instead of the bundled fixture."
    )


if __name__ == "__main__":
    main()
