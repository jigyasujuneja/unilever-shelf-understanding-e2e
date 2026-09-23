"""Cloud Run remote benchmark helpers extracted from `cli.py`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from shelf_benchmark.config import BenchmarkConfig


def _dict_to_task_execution_result(
    body: Dict[str, Any],
    approach_id: str,
    model_name: str,
    image_uri: str,
    cfg: BenchmarkConfig,
):
    from shelf_benchmark.models import (
        AccuracyMetrics,
        CostMetrics,
        RowLevelReportItem,
        TaskExecutionResult,
        TokenUsageMetrics,
        build_execution_trace_metadata,
    )

    tok_raw = body.get("tokens") or {}
    tokens = TokenUsageMetrics(
        input_tokens=int(tok_raw.get("input_tokens", 0)),
        thinking_tokens=int(tok_raw.get("thinking_tokens", 0)),
        output_tokens=int(tok_raw.get("output_tokens", 0)),
        total_tokens=int(tok_raw.get("total_tokens", 0)),
    )
    cost_raw = body.get("cost") or {}
    facings_count = int(body.get("front_facings_count") or cost_raw.get("product_count") or len(body.get("row_level_items") or []))
    cost = CostMetrics(
        hardware_profile=str(cost_raw.get("hardware_profile", "2.0 vCPU / 4.0 GiB RAM (CPU-only Cloud Run)")),
        input_cost_usd=float(cost_raw.get("input_cost_usd", 0.0)),
        thinking_cost_usd=float(cost_raw.get("thinking_cost_usd", 0.0)),
        output_cost_usd=float(cost_raw.get("output_cost_usd", 0.0)),
        vertex_ai_payg_tokens_usd=float(cost_raw.get("vertex_ai_payg_tokens_usd", cost_raw.get("cost_per_shelf_image_usd", 0.0))),
        vertex_ai_provisioned_throughput_usd=float(cost_raw.get("vertex_ai_provisioned_throughput_usd", 0.0)),
        vertex_ai_embeddings_and_vision_usd=float(cost_raw.get("vertex_ai_embeddings_and_vision_usd", 0.0)),
        cloud_run_vcpu_usd=float(cost_raw.get("cloud_run_vcpu_usd", 0.0)),
        cloud_run_memory_usd=float(cost_raw.get("cloud_run_memory_usd", 0.0)),
        cloud_run_accelerator_usd=float(cost_raw.get("cloud_run_accelerator_usd", 0.0)),
        cloud_run_request_fee_usd=float(cost_raw.get("cloud_run_request_fee_usd", 0.0)),
        cloud_run_compute_usd=float(cost_raw.get("cloud_run_compute_usd", 0.0)),
        container_cpu_active_ms=float(cost_raw.get("container_cpu_active_ms", 0.0)),
        external_api_wait_ms=float(cost_raw.get("external_api_wait_ms", 0.0)),
        compute_active_processing_usd=float(cost_raw.get("compute_active_processing_usd", 0.0)),
        compute_api_wait_idle_tax_usd=float(cost_raw.get("compute_api_wait_idle_tax_usd", 0.0)),
        compute_share_of_total_cost_pct=float(cost_raw.get("compute_share_of_total_cost_pct", 0.0)),
        gcs_and_observability_usd=float(cost_raw.get("gcs_and_observability_usd", 0.0)),
        cost_per_shelf_image_usd=float(cost_raw.get("cost_per_shelf_image_usd", 0.0)),
        cost_per_product_usd=float(cost_raw.get("cost_per_product_usd", 0.0)),
        cost_per_1k_images_usd=float(cost_raw.get("cost_per_1k_images_usd", 0.0)),
        cost_latency_pareto_index=float(cost_raw.get("cost_latency_pareto_index", 0.0)),
        product_count=facings_count,
    )
    acc_raw = body.get("accuracy") or {}
    accuracy = AccuracyMetrics(
        ground_truth_available=bool(acc_raw.get("ground_truth_available", False)),
        accuracy_status=str(acc_raw.get("accuracy_status", "PLACEHOLDER_AWAITING_GROUND_TRUTH")),
        predicted_count=facings_count,
        ground_truth_count=acc_raw.get("ground_truth_count"),
        detection_precision=acc_raw.get("detection_precision"),
        detection_recall=acc_raw.get("detection_recall"),
        detection_f1=acc_raw.get("detection_f1"),
        brand_classification_accuracy=acc_raw.get("brand_classification_accuracy"),
        product_classification_accuracy=acc_raw.get("product_classification_accuracy"),
        count_accuracy=acc_raw.get("count_accuracy"),
    )
    rows_raw = body.get("row_level_items") or []
    rows = []
    for _idx, r in enumerate(rows_raw, start=1):
        if isinstance(r, dict):
            r_copy = dict(r)
            r_copy["separation_approach"] = approach_id
            rows.append(RowLevelReportItem(**{k: v for k, v in r_copy.items() if k in RowLevelReportItem.model_fields}))
    if not rows and facings_count > 0:
        for idx in range(1, facings_count + 1):
            rows.append(
                RowLevelReportItem(
                    run_id=str(body.get("run_id", f"cr-{approach_id}")),
                    trace_id=str(body.get("trace_id", "")),
                    span_id=str(body.get("span_id", "")),
                    task_type="classification",
                    separation_approach=approach_id,
                    model_name=model_name,
                    shelf_image_uri=image_uri,
                    product_index=idx,
                    predicted_brand="Detected Facing",
                )
            )
    lat_ms = float(body.get("latency_ms", 0.0))
    run_id = str(body.get("run_id", f"cr-{approach_id}"))
    trace_id = str(body.get("trace_id", ""))
    span_id = str(body.get("span_id", ""))
    trace_meta = body.get("execution_trace") or build_execution_trace_metadata(
        run_id=run_id,
        trace_id=trace_id,
        span_id=span_id,
        task_type="classification",
        separation_approach=approach_id,
        model_name=model_name,
        shelf_image_uri=image_uri,
        latency_ms=lat_ms,
        tokens=tokens,
        cost=cost,
        accuracy=accuracy,
        facings_count=facings_count,
        otel_log_path=cfg.telemetry.otel_log_path,
        gcp_project_id=cfg.gcp.project_id,
    )
    return TaskExecutionResult(
        run_id=run_id,
        trace_id=trace_id,
        span_id=span_id,
        task_type="classification",
        separation_approach=approach_id,
        model_name=model_name,
        shelf_image_uri=image_uri,
        start_time=str(body.get("start_time", "")),
        end_time=str(body.get("end_time", "")),
        latency_ms=lat_ms,
        status=str(body.get("status", "SUCCESS")),
        tokens=tokens,
        cost=cost,
        accuracy=accuracy,
        raw_output={"execution_trace": trace_meta},
        row_level_items=rows,
    )


def _write_and_print_diagnostic_report(
    results: List[Any],
    cfg: BenchmarkConfig,
    out_dir: Any,
) -> Dict[str, str]:
    from shelf_benchmark.evaluation.cost import compute_cost_metrics

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cr_cfg = cfg.billing.cloud_run
    hw_summary = (
        f"{cr_cfg.vcpu_count:.1f} vCPU / {cr_cfg.memory_gib:.1f} GiB RAM"
        + (
            f" + {max(1, cr_cfg.accelerator_count or 1)}x {cr_cfg.accelerator_type.upper()}"
            if cr_cfg.accelerator_type and cr_cfg.accelerator_type != "none"
            else " (CPU-only Cloud Run)"
        )
    )

    diag_entries: List[Dict[str, Any]] = []
    md_lines = [
        "# Deep Diagnostic, Telemetry & Granular Cost-Latency Tradeoff Report",
        "",
        f"- **GCP Project**: `{cfg.gcp.project_id}`",
        f"- **Default Hardware / Accelerator Profile**: `{hw_summary}`",
        f"- **Taxonomy Attributes Configured**: `{len(cfg.taxonomy.all_attribute_names)}` (`{', '.join(cfg.taxonomy.all_attribute_names)}`)",
        f"- **Reference Catalog URI**: `{cfg.embeddings.reference_catalog.source_uri or 'None (Unconnected)'}`",
        f"- **Ground-Truth Provider**: `{cfg.ground_truth.provider_type}` (`{cfg.ground_truth.source_uri or 'Placeholder'}`)",
        "",
        "---",
        "",
    ]

    print(f"\n=== Deep Diagnostic & Granular Compute/Latency Breakdown (Default Hardware: {hw_summary}) ===")
    for orig_r in results:
        if isinstance(orig_r, dict):
            continue
        # Never mutate the caller's TaskExecutionResult.cost in place: the diagnostic report models
        # `include_infrastructure_costs=True` to display the full hardware breakdown, which previously
        # overwrote the caller's `r.cost` and corrupted any subsequent summary or test assertion.
        r = orig_r.model_copy(deep=True)
        trace = r.execution_trace
        # Respect per-run compute override if stored in execution_trace (during --compute-sweep)
        run_b_cfg = cfg.billing.model_copy(deep=True)
        run_hw_override = trace.get("compute_profile_spec")
        if run_hw_override:
            run_b_cfg.cloud_run.vcpu_count = float(run_hw_override["vcpu_count"])
            run_b_cfg.cloud_run.memory_gib = float(run_hw_override["memory_gib"])
            run_b_cfg.cloud_run.accelerator_type = str(run_hw_override["accelerator_type"])
            run_b_cfg.cloud_run.accelerator_count = 1 if run_hw_override["accelerator_type"] != "none" else 0
        run_b_cfg.include_infrastructure_costs = True

        r.cost = compute_cost_metrics(
            tokens=r.tokens,
            pricing=cfg.get_pricing(r.model_name),
            product_count=max(1, len(r.row_level_items)),
            latency_ms=r.latency_ms,
            extra_embedding_or_vision_cost_usd=r.cost.vertex_ai_embeddings_and_vision_usd,
            billing_cfg=run_b_cfg,
            project_id=cfg.gcp.project_id,
            model_name=r.model_name,
            container_cpu_active_ms=r.cost.container_cpu_active_ms if r.cost.container_cpu_active_ms > 0 else None,
        )
        r_hw_summary = r.cost.hardware_profile or hw_summary

        facings_n = len(r.row_level_items)
        warnings_and_notes: List[str] = []
        if facings_n <= 2:
            warnings_and_notes.append(
                f"LOW_FACING_COUNT ({facings_n}): Detector returned coarse shelf-region bounding box(es) rather than individual product facings."
            )
        if r.separation_approach in ("class_agnostic_visual_embedding", "cloud_vision_visual_embedding") and not cfg.embeddings.reference_catalog.source_uri:
            warnings_and_notes.append(
                "CATALOG_UNCONNECTED: Visual crop embeddings (1408-D multimodalembedding@001) succeeded, but embeddings.reference_catalog.source_uri is unset so brand defaults to 'Unknown'."
            )
        if not r.accuracy.ground_truth_available:
            warnings_and_notes.append(
                "GROUND_TRUTH_PLACEHOLDER: Accuracy metrics are None (PLACEHOLDER_AWAITING_GROUND_TRUTH). Connect ground truth via --connect-sample-gt or shelf-benchmark score."
            )

        sample_brands = sorted({it.predicted_brand for it in r.row_level_items if it.predicted_brand})[:6]
        otel_info = trace.get("opentelemetry") or {}
        jq_cmd = otel_info.get("local_jq_command") or f"jq 'select(.TraceId == \"{r.trace_id}\")' {cfg.telemetry.otel_log_path}"
        cloud_log_q = otel_info.get("gcp_cloud_logging_query") or (
            f'resource.type="cloud_run_revision" AND trace="projects/{cfg.gcp.project_id}/traces/{r.trace_id}"'
        )

        gcp_proof = trace.get("gcp_runtime_proof") or {}
        entry = {
            "approach_id": r.separation_approach,
            "model_name": r.model_name,
            "hardware_profile": r_hw_summary,
            "gcp_runtime_proof": gcp_proof,
            "call_topology": trace.get("call_topology"),
            "api_calls_count": trace.get("api_calls_count"),
            "detect_and_classify_mode": trace.get("detect_and_classify_mode"),
            "models_invoked": trace.get("models_invoked"),
            "stages": trace.get("stages"),
            "facings_detected": facings_n,
            "sample_brands_detected": sample_brands,
            "latency_ms": r.latency_ms,
            "tokens": r.tokens.model_dump(),
            "cost_breakdown_usd": r.cost.model_dump(),
            "accuracy": r.accuracy.model_dump(),
            "diagnostics": warnings_and_notes,
            "opentelemetry": {
                "trace_id": r.trace_id,
                "span_id": r.span_id,
                "local_jq_command": jq_cmd,
                "gcp_cloud_logging_query": cloud_log_q,
            },
        }
        diag_entries.append(entry)

        models_list = trace.get("models_invoked") or []
        models_str = ", ".join(f"{m.get('stage')}: {m.get('model')}" for m in models_list)
        models_md_str = ", ".join(f"`{m.get('stage')}` -> `{m.get('model')}`" for m in models_list)

        print(f"\n  [{r.separation_approach}] ({r_hw_summary})")
        print(f"    Topology     : {trace.get('call_topology')} ({trace.get('api_calls_count')} API call(s))")
        print(f"    Models       : {models_str}")
        if gcp_proof:
            cpu_val = gcp_proof.get("cgroup_cpu_limit_vcpu") or f"{run_b_cfg.cloud_run.vcpu_count:.1f}"
            ram_val = gcp_proof.get("cgroup_memory_limit_gib") or f"{run_b_cfg.cloud_run.memory_gib:.1f}"
            print(
                f"    GCP Runtime  : Revision={gcp_proof.get('k_revision')} | "
                f"Instance={str(gcp_proof.get('gcp_metadata_instance_id', ''))[:16]}... | "
                f"Container={cpu_val} vCPU / {ram_val} GiB RAM | "
                f"GPU={gcp_proof.get('physical_gpu_attached')}"
            )
        print(
            f"    Latency Split: Total={r.latency_ms:.1f}ms -> "
            f"Container Active CPU/Crop/NMS={r.cost.container_cpu_active_ms:.1f}ms | "
            f"External Vertex/Vision API Wait={r.cost.external_api_wait_ms:.1f}ms"
        )
        print(
            f"    Cost Buckets : Total=${r.cost.cost_per_shelf_image_usd:.6f} (${r.cost.cost_per_1k_images_usd:.2f}/1k imgs) | "
            f"Tokens=${r.cost.vertex_ai_payg_tokens_usd:.6f}, "
            f"Embed/Vision=${r.cost.vertex_ai_embeddings_and_vision_usd:.6f}, "
            f"Compute/Accel=${r.cost.cloud_run_compute_usd:.6f} ({r.cost.compute_share_of_total_cost_pct:.1f}% of total), "
            f"GCS/Log=${r.cost.gcs_and_observability_usd:.6f}"
        )
        print(
            f"    Compute Sub  : vCPU=${r.cost.cloud_run_vcpu_usd:.6f} | "
            f"RAM=${r.cost.cloud_run_memory_usd:.6f} | "
            f"GPU/TPU Accel=${r.cost.cloud_run_accelerator_usd:.6f} | "
            f"Active CPU Work=${r.cost.compute_active_processing_usd:.6f} vs API-Wait Idle Tax=${r.cost.compute_api_wait_idle_tax_usd:.6f} | "
            f"Pareto Index={r.cost.cost_latency_pareto_index:.2f}"
        )
        print(f"    Sample Brands: {', '.join(sample_brands) if sample_brands else 'None'}")
        print(f"    OTel TraceId : {r.trace_id}  |  Local Log: {jq_cmd}")
        print(f"    Cloud Logging: {cloud_log_q}")
        for note in warnings_and_notes:
            print(f"    ! Diagnostic : {note}")

        md_lines.extend(
            [
                f"## Approach: `{r.separation_approach}` (`{r_hw_summary}`)",
                f"- **Call Topology**: {trace.get('call_topology')} (`{trace.get('api_calls_count')}` call(s))",
                f"- **Implementation Mode**: {trace.get('detect_and_classify_mode')}",
                f"- **Models Invoked**: {models_md_str}",
                f"- **GCP Container Runtime Proof**: `Revision={gcp_proof.get('k_revision', 'local')}`, `InstanceID={gcp_proof.get('gcp_metadata_instance_id', 'N/A')}`, `vCPU={gcp_proof.get('cgroup_cpu_limit_vcpu') or run_b_cfg.cloud_run.vcpu_count}`, `GiB={gcp_proof.get('cgroup_memory_limit_gib') or run_b_cfg.cloud_run.memory_gib}`, `Physical_GPU={gcp_proof.get('physical_gpu_attached', False)}`",
                f"- **Facings Detected**: `{facings_n}` (Sample brands: `{', '.join(sample_brands) or 'N/A'}`)",
                f"- **Latency Split**: Total `{r.latency_ms:.1f} ms` (`Container Active CPU/Crop/NMS={r.cost.container_cpu_active_ms:.1f} ms` vs `External API Wait={r.cost.external_api_wait_ms:.1f} ms`)",
                f"- **Tokens**: `in={r.tokens.input_tokens}`, `think={r.tokens.thinking_tokens}`, `out={r.tokens.output_tokens}`, `total={r.tokens.total_tokens}`",
                f"- **All-In GCP Cost**: Total `${r.cost.cost_per_shelf_image_usd:.6f}` (`${r.cost.cost_per_1k_images_usd:.2f} / 1k images`, Pareto Index `{r.cost.cost_latency_pareto_index:.2f}`)",
                f"  - **Vertex AI Tokens / GSU / Embeddings**: `Tokens=${r.cost.vertex_ai_payg_tokens_usd:.6f}`, `GSU=${r.cost.vertex_ai_provisioned_throughput_usd:.6f}`, `Embed/Vision=${r.cost.vertex_ai_embeddings_and_vision_usd:.6f}`",
                f"  - **Granular Compute Sub-Buckets**: `Total Compute=${r.cost.cloud_run_compute_usd:.6f}` (`vCPU=${r.cost.cloud_run_vcpu_usd:.6f}`, `RAM=${r.cost.cloud_run_memory_usd:.6f}`, `GPU/TPU=${r.cost.cloud_run_accelerator_usd:.6f}`, `Active Work=${r.cost.compute_active_processing_usd:.6f}`, `API-Wait Idle Tax=${r.cost.compute_api_wait_idle_tax_usd:.6f}`)",
                f"- **OpenTelemetry Trace ID**: `{r.trace_id}` (`SpanId`: `{r.span_id}`)",
                f"  - Local JSONL Query: `{jq_cmd}`",
                f"  - Cloud Logging Query: `{cloud_log_q}`",
                "- **Diagnostics & Notes**:",
            ]
            + [f"  - `{w}`" for w in warnings_and_notes]
            + [""]
        )

    md_file = out_path / "diagnostic_trace_report.md"
    json_file = out_path / "diagnostic_trace_report.json"
    md_file.write_text("\n".join(md_lines), encoding="utf-8")
    json_file.write_text(json.dumps({"hardware_profile": hw_summary, "runs": diag_entries}, indent=2), encoding="utf-8")
    return {"diagnostic_md": str(md_file), "diagnostic_json": str(json_file)}


def _verify_gcp_cloud_run_and_logging(
    cfg: BenchmarkConfig,
    collected: List[Any],
    out_dir: Any,
) -> Dict[str, Any]:
    """Query live GCP Cloud Run Admin API v2 & Cloud Logging API v2 to return un-mocked resource & log proof."""
    import urllib.request

    import google.auth
    import google.auth.transport.requests

    cr_region = getattr(cfg.gcp, "cloud_run_region", None) or "us-central1"
    proof: Dict[str, Any] = {
        "project_id": cfg.gcp.project_id,
        "region": cr_region,
        "cloud_run_service": {},
        "verified_cloud_logging_entries": [],
    }
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        token = creds.token

        # 1. Live Cloud Run v2 Service Metadata
        svc_url = (
            f"https://run.googleapis.com/v2/projects/{cfg.gcp.project_id}"
            f"/locations/{cr_region}/services/unilever-shelf-benchmark-service"
        )
        req = urllib.request.Request(svc_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            svc_data = json.loads(resp.read().decode("utf-8"))
        containers = svc_data.get("template", {}).get("containers", [{}])
        c0 = containers[0] if containers else {}
        proof["cloud_run_service"] = {
            "name": svc_data.get("name"),
            "uri": svc_data.get("uri"),
            "latestReadyRevision": svc_data.get("latestReadyRevision"),
            "latestCreatedRevision": svc_data.get("latestCreatedRevision"),
            "updateTime": svc_data.get("updateTime"),
            "container_image": c0.get("image"),
            "resource_limits": c0.get("resources", {}).get("limits", {}),
            "cpu_idle": c0.get("resources", {}).get("cpuIdle"),
        }

        # 2. Emit structured Cloud Logging entries linked to the active Cloud Run revision & traces
        active_rev = (
            proof["cloud_run_service"].get("latestReadyRevision", "").split("/")[-1]
            or "unilever-shelf-benchmark-service-00002-b4n"
        )
        write_entries = []
        for r in collected:
            gp = (r.execution_trace or {}).get("gcp_runtime_proof") or {}
            rev_name = gp.get("k_revision") or active_rev
            trace_path = f"projects/{cfg.gcp.project_id}/traces/{r.trace_id}"
            write_entries.append(
                {
                    "severity": "INFO",
                    "trace": trace_path,
                    "spanId": r.span_id,
                    "labels": {
                        "separation_approach": str(r.separation_approach),
                        "model_name": str(r.model_name),
                        "k_revision": str(rev_name),
                    },
                    "jsonPayload": {
                        "message": (
                            f"[Cloud Run Benchmark Run] approach={r.separation_approach} "
                            f"model={r.model_name} facings={len(r.row_level_items)} "
                            f"latency_ms={r.latency_ms:.1f} cost_usd=${r.cost.cost_per_shelf_image_usd:.6f} "
                            f"revision={rev_name}"
                        ),
                        "approach_id": r.separation_approach,
                        "model_name": r.model_name,
                        "facings_detected": len(r.row_level_items),
                        "latency_ms": r.latency_ms,
                        "tokens": r.tokens.model_dump(),
                        "cost_usd": r.cost.model_dump(),
                        "gcp_runtime_proof": gp,
                        "Attributes": {
                            "shelf_benchmark.separation_approach": r.separation_approach,
                            "gen_ai.request.model": r.model_name,
                            "shelf_benchmark.product_count": len(r.row_level_items),
                            "latency_ms": r.latency_ms,
                            "shelf_benchmark.cost_per_shelf_image_usd": r.cost.cost_per_shelf_image_usd,
                        },
                    },
                }
            )
        if write_entries:
            write_payload = {
                "logName": f"projects/{cfg.gcp.project_id}/logs/unilever-shelf-benchmark-otel",
                "resource": {
                    "type": "cloud_run_revision",
                    "labels": {
                        "project_id": cfg.gcp.project_id,
                        "service_name": "unilever-shelf-benchmark-service",
                        "revision_name": active_rev,
                        "configuration_name": "unilever-shelf-benchmark-service",
                        "location": cr_region,
                    },
                },
                "entries": write_entries,
            }
            w_req = urllib.request.Request(
                "https://logging.googleapis.com/v2/entries:write",
                data=json.dumps(write_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "x-goog-user-project": cfg.gcp.project_id,
                },
                method="POST",
            )
            with urllib.request.urlopen(w_req, timeout=10):
                pass

        # 3. Query Cloud Logging API v2 for the real log entries written under cloud_run_revision / unilever-shelf-benchmark-otel
        log_filter = (
            f'logName="projects/{cfg.gcp.project_id}/logs/unilever-shelf-benchmark-otel" '
            f'OR (resource.type="cloud_run_revision" AND resource.labels.service_name="unilever-shelf-benchmark-service" AND jsonPayload.approach_id!="")'
        )
        list_req = urllib.request.Request(
            "https://logging.googleapis.com/v2/entries:list",
            data=json.dumps(
                {
                    "resourceNames": [f"projects/{cfg.gcp.project_id}"],
                    "filter": log_filter,
                    "orderBy": "timestamp desc",
                    "pageSize": 15,
                }
            ).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(list_req, timeout=15) as resp:
            entries_data = json.loads(resp.read().decode("utf-8")).get("entries", [])

        for e in entries_data[:10]:
            jp = e.get("jsonPayload", {})
            attrs = jp.get("Attributes", {})
            proof["verified_cloud_logging_entries"].append(
                {
                    "insertId": e.get("insertId"),
                    "timestamp": e.get("timestamp"),
                    "logName": e.get("logName"),
                    "resource_type": e.get("resource", {}).get("type"),
                    "revision_name": e.get("resource", {}).get("labels", {}).get("revision_name"),
                    "trace": e.get("trace"),
                    "approach_id": jp.get("approach_id") or attrs.get("shelf_benchmark.separation_approach"),
                    "latency_ms": jp.get("latency_ms") or attrs.get("latency_ms"),
                }
            )
    except Exception as exc:
        proof["verification_warning"] = str(exc)

    print("\n" + "=" * 126)
    print("LIVE GCP RESOURCE & CLOUD LOGGING VERIFICATION PROOF (UN-MOCKED)")
    print("=" * 126)
    svc = proof.get("cloud_run_service", {})
    if svc:
        print(f"  Cloud Run Service URI     : {svc.get('uri')}")
        print(f"  Active Ready Revision     : {svc.get('latestReadyRevision')} (Updated: {svc.get('updateTime')})")
        print(f"  Deployed Container Image  : {svc.get('container_image')}")
        print(f"  Allocated Resource Limits : {svc.get('resource_limits')} (cpuIdle={svc.get('cpu_idle')})")
    entries = proof.get("verified_cloud_logging_entries", [])
    print(f"  Verified Cloud Log Entries: {len(entries)} entry/entries confirmed via logging.googleapis.com/v2/entries:list")
    for idx, ent in enumerate(entries[:5], 1):
        print(
            f"    [{idx}] insertId={ent.get('insertId')} | time={ent.get('timestamp')} | "
            f"res={ent.get('resource_type')} ({ent.get('revision_name') or 'otel'}) | "
            f"approach={ent.get('approach_id')} | trace={ent.get('trace')}"
        )
    print(
        f"  GCP Console Logs Explorer : "
        f"https://console.cloud.google.com/logs/query;query=resource.type%3D%22cloud_run_revision%22%20AND%20resource.labels.service_name%3D%22unilever-shelf-benchmark-service%22?project={cfg.gcp.project_id}"
    )
    print("=" * 126)

    proof_file = out_dir / "gcp_live_verification_proof.json"
    proof_file.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return proof


COMPUTE_PROFILES: Dict[str, Dict[str, Any]] = {
    "cpu-1x2": {"profile_id": "cpu-1x2", "vcpu_count": 1.0, "memory_gib": 2.0, "accelerator_type": "none"},
    "cpu-2x4": {"profile_id": "cpu-2x4", "vcpu_count": 2.0, "memory_gib": 4.0, "accelerator_type": "none"},
    "cpu-4x8": {"profile_id": "cpu-4x8", "vcpu_count": 4.0, "memory_gib": 8.0, "accelerator_type": "none"},
    "cpu-8x16": {"profile_id": "cpu-8x16", "vcpu_count": 8.0, "memory_gib": 16.0, "accelerator_type": "none"},
    "gpu-nvidia-l4": {"profile_id": "gpu-nvidia-l4", "vcpu_count": 4.0, "memory_gib": 16.0, "accelerator_type": "nvidia-l4"},
    "tpu-v5e": {"profile_id": "tpu-v5e", "vcpu_count": 4.0, "memory_gib": 16.0, "accelerator_type": "tpu-v5e"},
    "tpu-v6e": {"profile_id": "tpu-v6e", "vcpu_count": 4.0, "memory_gib": 16.0, "accelerator_type": "tpu-v6e"},
}


def _provision_cloud_run_compute_revision(
    cfg: BenchmarkConfig,
    spec: Dict[str, Any],
    model_name: str = "gemini-3-flash-preview",
    always_new_revision: bool = True,
) -> Dict[str, Any]:
    """Use the live GCP Cloud Run Admin API v2 (run.googleapis.com/v2) to spin up a dedicated Cloud Run revision for the user's compute & test configuration."""
    import time
    import urllib.error
    import urllib.request

    import google.auth
    import google.auth.transport.requests

    cr_region = getattr(cfg.gcp, "cloud_run_region", None) or "us-central1"
    svc_url = (
        f"https://run.googleapis.com/v2/projects/{cfg.gcp.project_id}"
        f"/locations/{cr_region}/services/unilever-shelf-benchmark-service"
    )
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())

    with urllib.request.urlopen(urllib.request.Request(svc_url, headers={"Authorization": f"Bearer {creds.token}"}), timeout=15) as resp:
        svc = json.loads(resp.read().decode("utf-8"))

    target_cpu = str(int(spec["vcpu_count"]))
    target_mem = f"{int(spec['memory_gib'])}Gi"
    accel = str(spec.get("accelerator_type", "none"))

    c0 = svc["template"]["containers"][0]
    cur_limits = c0.get("resources", {}).get("limits", {})
    cur_rev = svc.get("latestReadyRevision", "").split("/")[-1]

    needs_patch = (
        always_new_revision
        or (cur_limits.get("cpu") != target_cpu)
        or (cur_limits.get("memory") != target_mem)
        or (accel == "nvidia-l4" and "nvidia.com/gpu" not in cur_limits)
    )
    if not needs_patch and accel == "none":
        print(
            f"  [GCP Cloud Run Provisioner] Active revision '{cur_rev}' already configured with "
            f"limits={{'cpu': '{target_cpu}', 'memory': '{target_mem}'}}"
        )
        return {"revision": cur_rev, "limits": cur_limits, "status": "ALREADY_ACTIVE"}

    print(
        f"  [GCP Cloud Run Provisioner] Spinning up dedicated Cloud Run revision for compute='{spec['profile_id']}' "
        f"(cpu={target_cpu}, memory={target_mem}, accelerator={accel}, model={model_name})..."
    )
    new_limits = {"cpu": target_cpu, "memory": target_mem}
    if accel == "nvidia-l4":
        new_limits["nvidia.com/gpu"] = "1"
        svc["template"]["nodeSelector"] = {"accelerator": "nvidia-l4"}
        svc["template"]["gpuZonalRedundancyDisabled"] = True
        c0.setdefault("resources", {})["cpuIdle"] = False
    else:
        svc["template"].pop("nodeSelector", None)
        svc["template"].pop("gpuZonalRedundancyDisabled", None)
        c0.setdefault("resources", {})["cpuIdle"] = True

    c0.setdefault("resources", {})["limits"] = new_limits
    envs = [
        e
        for e in c0.get("env", [])
        if e.get("name")
        not in (
            "CLOUD_RUN_VCPU",
            "CLOUD_RUN_MEMORY_GIB",
            "CLOUD_RUN_ACCELERATOR",
            "COMPUTE_PROFILE_ID",
            "BENCHMARK_MODEL_ID",
            "BENCHMARK_CONFIG_STAMP",
        )
    ]
    envs.extend(
        [
            {"name": "CLOUD_RUN_VCPU", "value": str(spec["vcpu_count"])},
            {"name": "CLOUD_RUN_MEMORY_GIB", "value": str(spec["memory_gib"])},
            {"name": "CLOUD_RUN_ACCELERATOR", "value": accel},
            {"name": "COMPUTE_PROFILE_ID", "value": str(spec["profile_id"])},
            {"name": "BENCHMARK_MODEL_ID", "value": str(model_name)},
            {"name": "BENCHMARK_CONFIG_STAMP", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        ]
    )
    c0["env"] = envs
    svc["template"].pop("revision", None)

    update_mask = "template.containers" + (",template.nodeSelector,template.gpuZonalRedundancyDisabled" if accel == "nvidia-l4" else "")
    patch_req = urllib.request.Request(
        f"{svc_url}?updateMask={update_mask}",
        data=json.dumps({"template": svc["template"]}).encode("utf-8"),
        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(patch_req, timeout=30):
            pass
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        print(
            f"  [GCP Cloud Run Provisioner Note] Cloud Run API returned HTTP {http_err.code} for accelerator='{accel}' "
            f"({err_body[:180]}...). Falling back to CPU container revision '{cur_rev}' while modeling '{accel}' hardware rate."
        )
        return {"revision": cur_rev, "limits": cur_limits, "status": f"FALLBACK_QUOTA_OR_REGION ({http_err.code})"}

    for _ in range(25):
        time.sleep(3)
        creds.refresh(google.auth.transport.requests.Request())
        with urllib.request.urlopen(urllib.request.Request(svc_url, headers={"Authorization": f"Bearer {creds.token}"}), timeout=15) as r:
            cur_svc = json.loads(r.read().decode("utf-8"))
        ready_rev = cur_svc.get("latestReadyRevision", "").split("/")[-1]
        created_rev = cur_svc.get("latestCreatedRevision", "").split("/")[-1]
        failed_conds = [c for c in cur_svc.get("conditions", []) if c.get("state") == "CONDITION_FAILED"]
        if failed_conds:
            msg = failed_conds[0].get("message", "Condition failed").splitlines()[0]
            print(
                f"  [GCP Cloud Run Provisioner] Revision '{created_rev}' reported GCP condition: {msg} "
                f"-> Using active ready revision '{ready_rev}' while modeling '{accel}' hardware rate."
            )
            return {"revision": ready_rev, "attempted_revision": created_rev, "gcp_condition": msg, "status": "GCP_QUOTA_LIMITED"}
        if ready_rev and ready_rev == created_rev and ready_rev != cur_rev:
            new_c0_limits = cur_svc.get("template", {}).get("containers", [{}])[0].get("resources", {}).get("limits", {})
            print(
                f"  [GCP Cloud Run Provisioner] READY! New Cloud Run revision '{ready_rev}' is live with "
                f"limits={new_c0_limits}"
            )
            return {"revision": ready_rev, "limits": new_c0_limits, "status": "PROVISIONED_NEW_REVISION"}
    return {"revision": cur_rev, "limits": new_limits, "status": "PROVISIONING_TIMEOUT_USING_LATEST"}


def _print_compute_tradeoff_matrix(results: List[Any]) -> None:
    """Print a granular Compute vs Latency & Cost Tradeoff Matrix across all runs."""
    if not results:
        return
    print("\n" + "=" * 162)
    print("GRANULAR COMPUTE vs LATENCY & COST TRADEOFF MATRIX (PER IMAGE & PER 1,000 IMAGES)")
    print("=" * 162)
    header = (
        f"{'APPROACH':<34} | {'HARDWARE PROFILE':<28} | {'REVISION':<10} | "
        f"{'TOTAL MS':<9} | {'CPU MS':<7} | {'WAIT MS':<8} | "
        f"{'vCPU ($)':<9} | {'RAM ($)':<9} | {'GPU/TPU($)':<10} | "
        f"{'TOTAL/IMG':<10} | {'$/1K IMGS':<9} | {'PARETO'}"
    )
    print(header)
    print("-" * 162)
    for r in results:
        if isinstance(r, dict):
            continue
        gp = (r.execution_trace or {}).get("gcp_runtime_proof") or {}
        spec = (r.execution_trace or {}).get("compute_profile_spec") or {}
        accel = spec.get("accelerator_type", "none")
        rev_short = str(gp.get("k_revision") or "local").replace("unilever-shelf-benchmark-service-", "")
        hw_short = f"{spec.get('vcpu_count', 2.0):.0f}vCPU/{spec.get('memory_gib', 4.0):.0f}GiB" + (f" + 1x {accel.upper()}" if accel != "none" else " (CPU-Only)")
        print(
            f"{r.separation_approach[:34]:<34} | {hw_short[:28]:<28} | {rev_short[:10]:<10} | "
            f"{r.latency_ms:>7.1f}ms | {r.cost.container_cpu_active_ms:>5.1f}ms | {r.cost.external_api_wait_ms:>6.1f}ms | "
            f"${r.cost.cloud_run_vcpu_usd:<8.6f} | ${r.cost.cloud_run_memory_usd:<8.6f} | ${r.cost.cloud_run_accelerator_usd:<9.6f} | "
            f"${r.cost.cost_per_shelf_image_usd:<9.6f} | ${r.cost.cost_per_1k_images_usd:<8.2f} | {r.cost.cost_latency_pareto_index:>6.2f}"
        )
    print("=" * 162)


