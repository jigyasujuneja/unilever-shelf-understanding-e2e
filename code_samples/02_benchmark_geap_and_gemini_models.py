"""Sample 02: Benchmarking GEAP (Google Early Access Program) & Preview Gemini Models.

Demonstrates how developers with access to GEAP / Early-Access / Experimental Gemini models
can register them with custom `api_version` (`v1beta1` or `v1alpha`), custom regional endpoints,
and custom token pricing, and run the entire shelf benchmark suite with OpenTelemetry logging.
"""

from shelf_benchmark import ModelPricing, ShelfBenchmarkSDK, UniversalModelSpec


def main() -> None:
    sdk = ShelfBenchmarkSDK(output_dir="reports/sample_02_geap")

    # 1. Register a GEAP / Early-Access Gemini model using UniversalModelSpec
    geap_flash_spec = UniversalModelSpec(
        model_id="gemini-3.8-flash",  # Replace with any allowlisted GEAP model ID (e.g. "gemini-exp-geap-001")
        display_name="geap-gemini-3.8-flash-preview",
        provider_family="vertex_geap",
        project_id="unilever-shelf-understanding",
        location="global",            # Or "us-central1" for regional GEAP endpoints
        api_version="v1beta1",        # Enables early-access / preview Vertex AI features
        pricing=ModelPricing(
            input=0.30,               # USD per 1M input tokens
            thinking=0.30,            # USD per 1M thinking tokens
            output=2.50,              # USD per 1M output tokens
        ),
    )
    sdk.register_model(geap_flash_spec)

    # 2. Run any task or separation approach against the registered GEAP model
    summary = sdk.run_suite(
        models=["geap-gemini-3.8-flash-preview"],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    )

    for res in summary["results"]:
        print(
            f"[GEAP Run] Model={res.model_name} | Approach={res.separation_approach} | "
            f"Facings={len(res.row_level_items)} | Latency={res.latency_ms:.1f}ms | "
            f"Tokens={res.tokens.total_tokens} | Cost/Image=${res.cost.cost_per_shelf_image_usd:.6f} | "
            f"OTel Trace={res.trace_id}"
        )


if __name__ == "__main__":
    main()
