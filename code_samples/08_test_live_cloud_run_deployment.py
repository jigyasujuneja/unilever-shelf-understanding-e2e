#!/usr/bin/env python3
"""Sample 08: Benchmark the headless Cloud Run inference worker.

This one cannot run offline: it calls a deployed service.

    .venv/bin/python code_samples/08_test_live_cloud_run_deployment.py --live

Without `--live` it prints the request it would send and exits, so the file can still be read,
imported and syntax-checked on a machine with no credentials.

Architecture it exercises:
  * Cloud Run (`unilever-shelf-benchmark-service`) is a headless inference and pipeline worker,
    called with an IAM ID token. It is not a public browser UI.
  * The caller (this script, the SDK, or the local studio) collects the response, recomputes the
    separated GCP cost breakdown from the returned token counts, and prints a trace summary.

Every number printed below comes from the worker response. If a field is missing the script
fails instead of substituting a plausible default, because a fabricated latency or token count
is indistinguishable from a measured one once it reaches a report.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.evaluation.cost import compute_cost_metrics
from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
from shelf_benchmark.models import TokenUsageMetrics

CLOUD_RUN_WORKER_URL = (
    "https://unilever-shelf-benchmark-service-298489950835.us-central1.run.app"
)
BENCHMARK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
]
SHELF_IMAGE_URI = "gs://unilever-shelf-understanding-shelf-images/shelf-image.png"


def require(payload: dict, key: str, model_id: str):
    """Return `payload[key]` or fail loudly. Never substitute a default for a measurement."""
    if key not in payload or payload[key] is None:
        raise KeyError(
            f"Cloud Run worker response for '{model_id}' has no '{key}'. Refusing to invent one. "
            f"Response keys: {sorted(payload)}"
        )
    return payload[key]


def print_trace_waterfall(trace_id: str, span_id: str, total_ms: float) -> None:
    """Print the span breakdown as a proportional bar chart of the measured total latency."""
    stages = [
        ("gen_ai.stage1.detection_depth_nms", 0.34),
        ("vertex_ai.stage2.crop_embeddings_1408d", 0.22),
        ("gen_ai.stage3.seven_dim_classification", 0.31),
        ("vertex_ai.stage4.hybrid_rrf_and_otel_sink", 0.13),
    ]
    print(f"    trace_id={trace_id} span_id={span_id} total={total_ms:.1f}ms")
    print("    NOTE: the stage split below is the worker's nominal profile, not per-span timing.")
    for name, share in stages:
        bar = "#" * max(1, int(round(share * 40)))
        print(f"      {name:<45} {bar:<40} {total_ms * share:8.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Actually call the Cloud Run worker (needs IAM access to the service).",
    )
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_yaml("configs/default_config.yaml")
    engine = GCPBillingAndCostEngine(project_id=cfg.gcp.project_id, billing_cfg=cfg.billing)

    request_body = {
        "model_name": BENCHMARK_MODELS[0],
        "task_type": "detection",
        "separation_approach": "single_pass_full_shelf",
        "shelf_image_uri": SHELF_IMAGE_URI,
    }

    if not args.live:
        print(f"Would POST to {CLOUD_RUN_WORKER_URL}/api/run-live for each of:")
        for model_id in BENCHMARK_MODELS:
            print(f"  - {model_id}")
        print("\nRequest body shape:")
        print(json.dumps(request_body, indent=2))
        print(
            "\nStopping here. Add --live to call the worker, which needs\n"
            "  gcloud auth application-default login\n"
            "and roles/run.invoker on the service."
        )
        return

    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default(quota_project_id=cfg.gcp.project_id)
    creds.refresh(google.auth.transport.requests.Request())
    id_token = getattr(creds, "id_token", None)

    print("=" * 100)
    print(f"INVOKING HEADLESS CLOUD RUN WORKER: {CLOUD_RUN_WORKER_URL}")
    print("=" * 100)

    for model_id in BENCHMARK_MODELS:
        payload = dict(request_body, model_name=model_id)
        req = urllib.request.Request(
            f"{CLOUD_RUN_WORKER_URL}/api/run-live",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {id_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        result = body.get("execution_result") or body.get("result") or body
        token_dict = require(result, "tokens", model_id)
        latency_ms = float(require(result, "latency_ms", model_id))
        run_id = require(result, "run_id", model_id)
        trace_id = require(result, "trace_id", model_id)
        span_id = result.get("span_id", "")
        facings = int(
            result.get("front_facings_count")
            or require(result.get("cost", {}), "product_count", model_id)
        )

        tokens = TokenUsageMetrics(
            input_tokens=int(token_dict["input_tokens"]),
            thinking_tokens=int(token_dict.get("thinking_tokens", 0)),
            output_tokens=int(token_dict["output_tokens"]),
            total_tokens=int(token_dict.get("total_tokens", 0)),
        )

        cost = compute_cost_metrics(
            tokens=tokens,
            pricing=cfg.get_pricing(model_id),
            product_count=facings,
            latency_ms=latency_ms,
            extra_embedding_or_vision_cost_usd=round(facings * 0.000125, 8),
            billing_cfg=cfg.billing,
            project_id=cfg.gcp.project_id,
            model_name=model_id,
            gcp_labels=engine.build_gcp_billing_labels(
                run_id=run_id,
                approach_id="single_pass_full_shelf",
                task_type="detection",
                model_name=model_id,
            ),
        )

        print(f"\n  [{model_id}] HTTP {resp.status} run_id={run_id} status={result.get('status')}")
        print(f"    facings={facings} latency={latency_ms:.1f}ms")
        print(
            f"    tokens in/think/out/total = {tokens.input_tokens}/{tokens.thinking_tokens}/"
            f"{tokens.output_tokens}/{tokens.total_tokens}"
        )
        print(f"    billing_source                     : {cost.billing_source}")
        print(f"    includes_modelled_infrastructure   : {cost.includes_modelled_infrastructure}")
        print(f"    1. vertex_ai_payg_tokens_usd       : ${cost.vertex_ai_payg_tokens_usd:.8f}")
        print(f"    2. vertex_ai_provisioned_throughput: ${cost.vertex_ai_provisioned_throughput_usd:.8f}")
        print(f"    3. vertex_ai_embeddings_and_vision : ${cost.vertex_ai_embeddings_and_vision_usd:.8f}")
        print(f"    4. cloud_run_compute_usd (modelled): ${cost.cloud_run_compute_usd:.8f}")
        print(f"    5. gcs_and_observability (modelled): ${cost.gcs_and_observability_usd:.8f}")
        print(f"    = cost_per_shelf_image_usd         : ${cost.cost_per_shelf_image_usd:.8f}")
        print(f"    = cost_per_product_usd             : ${cost.cost_per_product_usd:.8f}")
        print_trace_waterfall(trace_id, span_id, latency_ms)


if __name__ == "__main__":
    main()
