#!/usr/bin/env python3
"""Sample 02: Register a GEAP / early-access / preview Gemini model.

Run it (registration only, no network):

    .venv/bin/python code_samples/02_benchmark_geap_and_gemini_models.py

Run the benchmark against the real allowlisted model:

    .venv/bin/python code_samples/02_benchmark_geap_and_gemini_models.py --live

`UniversalModelSpec` is how any model that is not one of the config's default Gemini ids gets
into the suite: an allowlisted GEAP model id, a preview `api_version`, a regional endpoint and
its own rate card. Everything downstream (prompts, approaches, scoring, cost, OTel) is unchanged.

Why the rate card is part of the spec: Vertex AI returns token counts, not dollars, so the suite
multiplies counts by rates you supply. A preview model with no published SKU has no live rate to
look up, which is exactly when `pricing=` matters. The resulting `CostMetrics.billing_source`
reports `yaml_rate_table` so nobody mistakes a modelled cost for an invoice.
"""

from __future__ import annotations

import argparse

from shelf_benchmark import ModelPricing, ShelfBenchmarkSDK, UniversalModelSpec

GEAP_SPEC = UniversalModelSpec(
    # Replace with any allowlisted GEAP model id, for example "gemini-exp-geap-001".
    model_id="gemini-3.8-flash",
    display_name="geap-gemini-3.8-flash-preview",
    provider_family="vertex_geap",
    project_id="unilever-shelf-understanding",
    location="global",       # Or a regional endpoint such as "us-central1".
    api_version="v1beta1",   # Enables early-access / preview Vertex AI features.
    pricing=ModelPricing(
        input=0.30,          # USD per 1M input tokens.
        thinking=0.30,       # USD per 1M thinking tokens.
        output=2.50,         # USD per 1M output tokens.
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the benchmark against the registered GEAP model (needs GCP credentials).",
    )
    args = parser.parse_args()

    sdk = ShelfBenchmarkSDK(output_dir="reports/sample_02_geap")
    sdk.register_model(GEAP_SPEC)

    print("Registered GEAP model spec:")
    print(f"  display_name    : {GEAP_SPEC.display_name}")
    print(f"  model_id        : {GEAP_SPEC.model_id}")
    print(f"  provider_family : {GEAP_SPEC.provider_family}")
    print(f"  api_version     : {GEAP_SPEC.api_version}")
    print(f"  location        : {GEAP_SPEC.location}")
    print(f"  rate card       : {sdk.config.get_pricing(GEAP_SPEC.display_name)}")
    print(f"  models now known: {sdk.config.models}")

    if not args.live:
        print(
            "\nStopping here. Add --live to call the model, which needs\n"
            "  gcloud auth application-default login\n"
            "and read access to gs://unilever-shelf-understanding-shelf-images.\n"
            "For a run that needs neither, see code_samples/01_quickstart_run_full_suite.py."
        )
        return

    summary = sdk.run_suite(
        models=[GEAP_SPEC.display_name],
        tasks=["classification"],
        approaches=["single_pass_full_shelf"],
        shelf_image_uri="gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
    )

    for res in summary["results"]:
        print(
            f"[GEAP run] model={res.model_name} approach={res.separation_approach} "
            f"facings={len(res.row_level_items)} latency={res.latency_ms:.1f}ms "
            f"tokens={res.tokens.total_tokens} "
            f"cost/image=${res.cost.cost_per_shelf_image_usd:.6f} "
            f"billing_source={res.cost.billing_source} trace={res.trace_id}"
        )
        print(f"  accuracy_status={res.accuracy.accuracy_status}")


if __name__ == "__main__":
    main()
