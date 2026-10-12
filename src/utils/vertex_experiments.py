"""GCP Vertex AI Experiments & Vertex AI Model Registry integration.

Records every benchmark run as a structured ``system.ExperimentRun`` context (with full
hyperparameters, accuracy/F2/latency/cost metrics, and GCS artifact lineage) inside the project's
Vertex AI Metadata Store (``metadataStores/default``), and registers exported/distilled ONNX
detectors and calibration artifacts in **Vertex AI Model Registry**.
"""

from __future__ import annotations

import re
import time
from typing import Any

import google.auth
from google.auth.transport.requests import AuthorizedSession

from utils.llm import load_config

DEFAULT_EXPERIMENT = "shelf-bench-l8-experiments"
DEFAULT_ONNX_CONTAINER = "us-central1-docker.pkg.dev/ul-fs-s-sandbx-16-prj/cloud-run-source-deploy/shelf-bench:latest"


def _session() -> AuthorizedSession:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return s[:60] if s else "run"


def ensure_experiment(
    project: str,
    region: str = "us-central1",
    experiment_id: str = DEFAULT_EXPERIMENT,
    session: AuthorizedSession | None = None,
) -> str:
    """Idempotently create a ``system.Experiment`` context in Vertex AI Metadata Store."""
    s = session or _session()
    base = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/metadataStores/default"
    ctx_url = f"{base}/contexts/{experiment_id}"
    r = s.get(ctx_url)
    if r.status_code == 200:
        return r.json()["name"]
    body = {
        "displayName": experiment_id,
        "schemaTitle": "system.Experiment",
        "schemaVersion": "0.0.1",
        "description": "Unilever Shelf Understanding — Cloud Run & Vertex AI Benchmark Experiments",
        "labels": {"app": "shelf-bench"},
    }
    r = s.post(f"{base}/contexts", params={"contextId": experiment_id}, json=body)
    if r.status_code >= 400 and r.status_code != 409:
        raise RuntimeError(f"Failed to create Vertex AI Experiment {experiment_id}: {r.status_code} {r.text[:400]}")
    return f"projects/{project}/locations/{region}/metadataStores/default/contexts/{experiment_id}"


def log_experiment_run(
    summary: dict[str, Any],
    results_uri: str = "",
    experiment_id: str = DEFAULT_EXPERIMENT,
    session: AuthorizedSession | None = None,
) -> dict[str, str]:
    """Log a completed ``shelf-bench`` run to Vertex AI Experiments (Metadata Store)."""
    cfg = load_config()
    project = cfg["gcp"]["project"]
    region = cfg["gcp"].get("region", "us-central1")
    s = session or _session()

    exp_name = ensure_experiment(project, region, experiment_id, session=s)
    run_id = summary.get("run_id", "unknown-run")
    run_slug = _slug(run_id)
    base = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/metadataStores/default"

    rate = summary.get("usd_to_inr") or summary.get("pricing", {}).get("usd_to_inr") or 95.8
    cost_usd = float(summary.get("cost_per_image_usd", 0.0))
    cost_inr = round(cost_usd * float(rate), 5)
    tokens = summary.get("tokens", {})
    cost_breakdown = summary.get("cost", {})

    params = {
        "approach": str(summary.get("approach", "")),
        "task": str(summary.get("task", "")),
        "use_case": str(summary.get("use_case", "")),
        "dataset": str(summary.get("dataset", "")),
        "model": str(summary.get("model", "")),
        "tier": str(summary.get("tier", "standard")),
        "split": str(summary.get("split", "test")),
        "limit": int(summary.get("limit", 50)),
        "seed": int(summary.get("seed", 0)),
        "platform": str(summary.get("environment", {}).get("platform", "local")),
        "architecture": str(summary.get("architecture", ""))[:200],
    }
    metrics = {
        "f2": float(summary.get("f2", 0.0)),
        "precision": float(summary.get("precision", 0.0)),
        "recall": float(summary.get("recall", 0.0)),
        "accuracy": float(summary.get("accuracy", 0.0)),
        "found_recall": float(summary.get("found_recall", 0.0)),
        "p50_latency_s": float(summary.get("p50_latency_s", 0.0)),
        "p95_latency_s": float(summary.get("p95_latency_s", 0.0)),
        "p99_latency_s": float(summary.get("p99_latency_s", 0.0)),
        "cost_per_image_usd": cost_usd,
        "cost_per_image_inr": cost_inr,
        "token_cost_per_image_usd": float(summary.get("token_cost_per_image_usd", 0.0)),
        "compute_cost_per_image_usd": float(summary.get("compute_cost_per_image_usd", 0.0)),
        "services_usd_per_image": float(cost_breakdown.get("services_usd_per_image", 0.0)),
        "wall_time_s": float(summary.get("wall_time_s", 0.0)),
        "input_tokens": int(tokens.get("input_tokens", 0)),
        "output_tokens": int(tokens.get("output_tokens", 0)),
        "calls": int(tokens.get("calls", 0)),
        "images": int(summary.get("images", 0)),
        "errors": int(summary.get("errors", 0)),
    }
    if "field_accuracy" in summary and isinstance(summary["field_accuracy"], dict):
        for k, v in summary["field_accuracy"].items():
            if isinstance(v, (int, float)):
                metrics[f"field_acc_{k}"] = float(v)
    if "macro_accuracy" in summary:
        if isinstance(summary["macro_accuracy"], dict):
            for k, v in summary["macro_accuracy"].items():
                if isinstance(v, (int, float)):
                    metrics[f"macro_{k}"] = float(v)
        elif isinstance(summary["macro_accuracy"], (int, float)):
            metrics["macro_accuracy"] = float(summary["macro_accuracy"])
    if isinstance(summary.get("hierarchical_accuracy"), dict):
        for k, v in summary["hierarchical_accuracy"].items():
            if isinstance(v, (int, float)):
                metrics[f"hier_{k}"] = float(v)
    if isinstance(summary.get("hul_vs_competitor_breakdown"), dict):
        for k, v in summary["hul_vs_competitor_breakdown"].items():
            if isinstance(v, (int, float)):
                metrics[f"hul_comp_{k}"] = float(v)
    if isinstance(summary.get("sos_metrics"), dict):
        for k, v in summary["sos_metrics"].items():
            if isinstance(v, (int, float)):
                metrics[f"sos_{k}"] = float(v)
    if isinstance(summary.get("stage_metrics"), dict):
        for k, v in summary["stage_metrics"].items():
            if isinstance(v, (int, float)):
                metrics[f"stage_{k}"] = float(v)

    ctx_body = {
        "displayName": run_id,
        "schemaTitle": "system.ExperimentRun",
        "schemaVersion": "0.0.1",
        "parentContexts": [exp_name],
        "metadata": {
            "params": params,
            "metrics": metrics,
            "sos_metrics": summary.get("sos_metrics", {}),
            "stage_metrics": summary.get("stage_metrics", {}),
            "hierarchical_accuracy": summary.get("hierarchical_accuracy", {}),
            "hul_vs_competitor_breakdown": summary.get("hul_vs_competitor_breakdown", {}),
            "ci95": summary.get("ci95", {}),
            "state": "COMPLETE",
            "gcs_results_uri": results_uri,
            "trace_url": (summary.get("telemetry") or {}).get("trace_url", ""),
        },
        "labels": {
            "approach": _slug(str(summary.get("approach", "unknown")))[:60],
            "task": _slug(str(summary.get("task", "unknown")))[:60],
            "dataset": _slug(str(summary.get("dataset", "unknown")))[:60],
        },
    }

    ctx_resp = s.post(f"{base}/contexts", params={"contextId": run_slug}, json=ctx_body)
    if ctx_resp.status_code == 409:
        ctx_resp = s.patch(f"{base}/contexts/{run_slug}", json=ctx_body)

    # Record Execution & Artifact in Metadata Store for full lineage
    exec_body = {
        "displayName": f"exec-{run_id}",
        "schemaTitle": "system.Run",
        "schemaVersion": "0.0.1",
        "state": "COMPLETE",
        "metadata": {**params, **metrics},
    }
    exec_resp = s.post(f"{base}/executions", params={"executionId": run_slug}, json=exec_body)
    if exec_resp.status_code == 409:
        exec_resp = s.patch(f"{base}/executions/{run_slug}", json=exec_body)

    art_names = []
    if results_uri:
        art_body = {
            "displayName": f"results-{run_id}",
            "schemaTitle": "system.Dataset",
            "schemaVersion": "0.0.1",
            "uri": f"{results_uri.rstrip('/')}/summary.json",
            "metadata": {"images_jsonl": f"{results_uri.rstrip('/')}/images.jsonl", **metrics},
        }
        art_resp = s.post(f"{base}/artifacts", params={"artifactId": run_slug}, json=art_body)
        if art_resp.status_code in (200, 409):
            art_names.append(f"projects/{project}/locations/{region}/metadataStores/default/artifacts/{run_slug}")

    s.post(
        f"{base}/contexts/{run_slug}:addContextArtifactsAndExecutions",
        json={
            "artifacts": art_names,
            "executions": [f"projects/{project}/locations/{region}/metadataStores/default/executions/{run_slug}"],
        },
    )

    console_url = (
        f"https://console.cloud.google.com/vertex-ai/experiments/locations/{region}/"
        f"experiments/{experiment_id}?project={project}"
    )
    return {
        "experiment": experiment_id,
        "run_context": f"projects/{project}/locations/{region}/metadataStores/default/contexts/{run_slug}",
        "console_url": console_url,
    }


def register_model(
    display_name: str,
    artifact_uri: str,
    description: str,
    metrics_metadata: dict[str, Any] | None = None,
    version_aliases: list[str] | None = None,
    container_image: str = DEFAULT_ONNX_CONTAINER,
    session: AuthorizedSession | None = None,
) -> dict[str, str]:
    """Upload or version a model artifact (e.g. ONNX detector) in Vertex AI Model Registry."""
    cfg = load_config()
    project = cfg["gcp"]["project"]
    region = cfg["gcp"].get("region", "us-central1")
    s = session or _session()
    base = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}"

    # Check if display_name already exists so we version under parentModel
    existing = s.get(f"{base}/models").json().get("models", [])
    parent_model = next((m["name"] for m in existing if m.get("displayName") == display_name), None)

    labels = {"app": "shelf-bench", "framework": "onnxruntime"}
    if metrics_metadata:
        for k, v in metrics_metadata.items():
            labels[_slug(k)[:60]] = _slug(str(v))[:60]

    payload: dict[str, Any] = {
        "model": {
            "displayName": display_name,
            "description": description,
            "artifactUri": artifact_uri.rstrip("/"),
            "containerSpec": {"imageUri": container_image},
            "versionAliases": version_aliases or ["default"],
            "labels": labels,
        }
    }
    if parent_model:
        payload["parentModel"] = parent_model

    r = s.post(f"{base}/models:upload", json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"Model Registry upload failed ({r.status_code}): {r.text[:600]}")
    op_name = r.json()["name"]
    while True:
        op = s.get(f"https://{region}-aiplatform.googleapis.com/v1/{op_name}").json()
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"Model Registry upload error: {op['error']}")
            model_res = op.get("response", {}).get("model", "")
            model_id = model_res.rsplit("/", 1)[-1]
            return {
                "model_resource": model_res,
                "artifact_uri": artifact_uri,
                "console_url": f"https://console.cloud.google.com/vertex-ai/models/locations/{region}/models/{model_id}?project={project}",
            }
        time.sleep(3)
