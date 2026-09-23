"""Interactive Executive & Engineering UI Server for Unilever Shelf Understanding Benchmark.

This module acts purely as a visual and REST API presentation layer on top of the
existing `shelf_benchmark` package (`src/shelf_benchmark/`) without modifying any
core library code.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

# Ensure `src/` is importable without modifying anything in `src/`
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shelf_benchmark.auth import create_genai_client
from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.models import TaskExecutionResult
from shelf_benchmark.runner import BenchmarkRunner
from shelf_benchmark.tasks.facing_utils import (
    check_is_hul_brand,
    deduplicate_depth_stacked_facings,
    derive_size_bucket_from_bbox,
)

UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_PATH = REPO_ROOT / "configs" / "default_config.yaml"


def _build_real_depth_candidates(
    kept_products: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Returns ONLY the real, model-produced front-facing boxes from the row-level report.

    Each box is re-run through `deduplicate_depth_stacked_facings` from
    `shelf_benchmark.tasks.facing_utils` so the UI can show the genuine Depth NMS decision
    for the boxes that were actually persisted.

    The boxes that Depth NMS suppressed are NOT reconstructed here: the reports persist only
    the suppressed *count* (`depth_duplicates_filtered`), not their coordinates. Fabricating
    plausible-looking back-row boxes for the visualization would put boxes on screen that no
    model ever emitted, so this function never invents geometry.
    """
    candidates: List[Dict[str, Any]] = []
    for p in kept_products:
        candidates.append(
            {
                "product_index": p.get("product_index"),
                "bbox_2d": [
                    p.get("bbox_ymin", 0),
                    p.get("bbox_xmin", 0),
                    p.get("bbox_ymax", 0),
                    p.get("bbox_xmax", 0),
                ],
                "shelf_row": p.get("shelf_row", "middle"),
                "position_on_shelf": p.get("position_on_shelf", 1),
                "is_front_facing": True,
                "brand": p.get("predicted_brand", ""),
                "variant": p.get("predicted_variant", ""),
                "confidence": p.get("confidence", 0.95),
                "source": "model_output",
                "synthetic_demo_data": False,
            }
        )

    if not candidates:
        return []

    kept_after_nms, _filtered_count = deduplicate_depth_stacked_facings(
        candidates, x_overlap_threshold=0.45
    )
    kept_coords = {tuple(item["bbox_2d"]) for item in kept_after_nms}

    annotated: List[Dict[str, Any]] = []
    for item in candidates:
        is_kept = tuple(item["bbox_2d"]) in kept_coords
        annotated.append(
            {
                **item,
                "kept_by_depth_nms": is_kept,
                "nms_reason": (
                    "Front-most unit in horizontal shelf slot (highest y_max base)"
                    if is_kept
                    else "Suppressed by Depth NMS: 1D X-overlap >= 0.45 with a nearer facing"
                ),
            }
        )
    return annotated


def load_dashboard_payload() -> Dict[str, Any]:
    """Loads all benchmark reports, crops, SFT datasets, and taxonomy metadata."""
    summary_path = REPORTS_DIR / "benchmark_summary.json"
    rows_path = REPORTS_DIR / "row_level_report.json"
    otel_path = REPORTS_DIR / "otel_logs.jsonl"
    sft_path = REPORTS_DIR / "tuning_data" / "shelf_sft_train.jsonl"

    summary_data = {"summary": [], "results": []}
    if summary_path.exists():
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    rows_data: List[Dict[str, Any]] = []
    if rows_path.exists():
        rows_data = json.loads(rows_path.read_text(encoding="utf-8"))

    otel_spans: List[Dict[str, Any]] = []
    if otel_path.exists():
        for line in otel_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    otel_spans.append(json.loads(line))
                except Exception:
                    pass

    sft_examples: List[Dict[str, Any]] = []
    if sft_path.exists():
        for line in sft_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    sft_examples.append(json.loads(line))
                except Exception:
                    pass

    # Build per-model detection depth NMS visualization data.
    # Only real, model-produced boxes are exposed here. The coordinates of boxes suppressed by
    # Depth NMS are not persisted in the reports (only their count is), so they cannot be drawn.
    depth_demos: Dict[str, Any] = {}
    for s in summary_data.get("summary", []):
        if s.get("task_type") == "detection":
            model_name = s.get("model_name")
            det_rows = [
                r
                for r in rows_data
                if r.get("task_type") == "detection" and r.get("model_name") == model_name
            ]
            real_boxes = _build_real_depth_candidates(det_rows)
            depth_demos[model_name] = {
                "synthetic_demo_data": False,
                "data_source": "row_level_report.json (model output only)",
                "suppressed_box_coordinates_available": False,
                "suppressed_boxes_note": (
                    "Depth NMS suppressed "
                    f"{int(s.get('depth_duplicates_filtered', 0))} back-row candidate(s). "
                    "Only the count is persisted by the benchmark, so no back-row boxes are drawn."
                ),
                "front_facings_count": s.get("front_facings_count", len(det_rows)),
                "depth_duplicates_filtered": s.get("depth_duplicates_filtered", 0),
                "raw_detections_count": s.get("front_facings_count", len(det_rows))
                + s.get("depth_duplicates_filtered", 0),
                "raw_candidate_count": len(real_boxes),
                "raw_candidates": real_boxes,
                "kept_front_facings": [
                    b for b in real_boxes if b.get("kept_by_depth_nms")
                ],
                "boxes": real_boxes,
            }

    # Discover available physical crop images per model
    crops_dir = REPORTS_DIR / "crops"
    crops_manifest: Dict[str, Dict[str, Any]] = {}
    if crops_dir.exists():
        for model_folder in sorted(crops_dir.iterdir()):
            if model_folder.is_dir():
                facing_files = sorted(
                    [
                        f"/crops/{model_folder.name}/{p.name}"
                        for p in model_folder.glob("facing_*.png")
                    ]
                )
                montage_file = (
                    f"/crops/{model_folder.name}/montage_all_facings.png"
                    if (model_folder / "montage_all_facings.png").exists()
                    else None
                )
                crops_manifest[model_folder.name] = {
                    "montage_url": montage_file,
                    "facing_urls": facing_files,
                }

    cfg = BenchmarkConfig.from_yaml(CONFIG_PATH)

    from shelf_benchmark.approaches import GLOBAL_APPROACH_REGISTRY

    embedding_model_name = getattr(cfg, "embedding_model", "gemini-embedding-001")
    classification_approaches = [
        "single_pass_full_shelf",
        "two_stage_bbox_guided_nms",
        "two_stage_physical_crop_per_facing",
        "class_agnostic_visual_embedding",
    ]
    registered_plugins = [
        {
            "approach_id": p.approach_id,
            "display_name": p.display_name,
            "category": p.category,
            "stages_description": p.stages_description,
        }
        for p in GLOBAL_APPROACH_REGISTRY.list_all()
    ]

    from shelf_benchmark.models import (
        AccuracyMetrics,
        CostMetrics,
        TokenUsageMetrics,
        build_execution_trace_metadata,
    )

    summary_records_enriched = []
    for s in summary_data.get("summary", []):
        s_copy = dict(s)
        if "execution_trace" not in s_copy:
            tok = TokenUsageMetrics(
                input_tokens=int(s.get("input_tokens", 0) or 0),
                thinking_tokens=int(s.get("thinking_tokens", 0) or 0),
                output_tokens=int(s.get("output_tokens", 0) or 0),
                total_tokens=int(s.get("total_tokens", 0) or 0),
            )
            cst = CostMetrics(
                vertex_ai_payg_tokens_usd=float(s.get("vertex_ai_payg_tokens_usd", 0.0) or 0.0),
                vertex_ai_provisioned_throughput_usd=float(s.get("vertex_ai_provisioned_throughput_usd", 0.0) or 0.0),
                vertex_ai_embeddings_and_vision_usd=float(s.get("vertex_ai_embeddings_and_vision_usd", 0.0) or 0.0),
                cloud_run_compute_usd=float(s.get("cloud_run_compute_usd", 0.0) or 0.0),
                gcs_and_observability_usd=float(s.get("gcs_and_observability_usd", 0.0) or 0.0),
                cost_per_shelf_image_usd=float(s.get("cost_per_shelf_image_usd", 0.0) or 0.0),
                cost_per_product_usd=float(s.get("cost_per_product_usd", 0.0) or 0.0),
                billing_source=str(s.get("billing_source", "yaml_rate_table")),
            )
            acc = AccuracyMetrics(
                ground_truth_available=bool(s.get("ground_truth_available", False)),
                accuracy_status=str(s.get("accuracy_status", "PLACEHOLDER_AWAITING_GROUND_TRUTH")),
                gt_version=str(s.get("gt_version", "unversioned")),
                iou_threshold=float(s.get("iou_threshold", 0.50) or 0.50),
                detection_precision=s.get("detection_precision"),
                detection_recall=s.get("detection_recall"),
                detection_f1=s.get("detection_f1"),
                mean_iou_matched=s.get("mean_iou_matched"),
                brand_classification_accuracy=s.get("brand_classification_accuracy"),
                product_classification_accuracy=s.get("product_classification_accuracy"),
                count_accuracy=s.get("count_accuracy"),
                depth_duplicates_filtered=int(s.get("depth_duplicates_filtered", 0) or 0),
            )
            s_copy["execution_trace"] = build_execution_trace_metadata(
                run_id=str(s.get("run_id", "")),
                trace_id=str(s.get("trace_id", "")),
                span_id=str(s.get("span_id", "")),
                task_type=str(s.get("task_type", "classification")),
                separation_approach=str(s.get("separation_approach", "single_pass_full_shelf")),
                model_name=str(s.get("model_name", "gemini-3.8-flash")),
                shelf_image_uri=str(s.get("shelf_image_uri", "gs://unilever-shelf-understanding-shelf-images/shelf-image.png")),
                latency_ms=float(s.get("latency_ms", 0.0) or 0.0),
                tokens=tok,
                cost=cst,
                accuracy=acc,
                facings_count=int(s.get("front_facings_count", 0) or 0),
                otel_log_path=str(cfg.telemetry.otel_log_path),
                gcp_project_id=cfg.gcp.project_id,
                gcp_log_name=cfg.telemetry.gcp_log_name,
                taxonomy_source=cfg.taxonomy.taxonomy_file,
                ground_truth_provider=cfg.ground_truth.provider_type,
                reference_catalog_uri=cfg.embeddings.reference_catalog.source_uri,
            )
        summary_records_enriched.append(s_copy)

    return {
        "project_info": {
            "gcp_project_id": cfg.gcp.project_id,
            "location": cfg.gcp.location,
            "shelf_images_bucket": cfg.buckets.shelf_images_bucket,
            "catalog_images_bucket": cfg.buckets.catalog_images_bucket,
            "planograms_bucket": cfg.buckets.planograms_bucket,
            "active_shelf_image": "gs://unilever-shelf-understanding-shelf-images/shelf-image.png",
            "ground_truth_provider": cfg.ground_truth.provider_type,
            "associations_provider": cfg.associations.provider_type,
            "embedding_model": embedding_model_name,
            "visual_embedding_model": "multimodalembedding@001 (1408-D)",
            "models": cfg.models,
            "classification_approaches": classification_approaches,
            "registered_approaches": registered_plugins,
            "otel_log_path": cfg.telemetry.otel_log_path,
            "cloud_run_service": os.environ.get("K_SERVICE") or "local-workstation",
        },
        "taxonomy_reference": {
            "config_source": cfg.taxonomy.taxonomy_file,
            "categories": cfg.taxonomy.categories,
            "subcategories": cfg.taxonomy.subcategories,
            "hul_brands": cfg.taxonomy.hul_brands,
            "non_hul_brands": cfg.taxonomy.non_hul_brands,
            "packaging_types": cfg.taxonomy.packaging_types,
            "pack_types": cfg.taxonomy.pack_types,
            "size_buckets": cfg.taxonomy.size_bucket_labels,
        },
        "pricing_table": {
            k: v.model_dump() for k, v in cfg.pricing_per_million_tokens.items()
        },
        "summary": summary_records_enriched,
        "rows": rows_data,
        "depth_demos": depth_demos,
        "crops_manifest": crops_manifest,
        "otel_spans": otel_spans[-36:],
        "sft_examples": sft_examples,
    }


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_bm25_overlap(query: str, keywords: str, dense_text: str) -> float:
    q_tokens = {
        t.strip(".,()/+-'\"_").lower()
        for t in query.split()
        if len(t.strip(".,()/+-'\"_")) >= 2
    }
    if not q_tokens:
        return 0.0
    doc_text = f"{keywords} {dense_text}".lower()
    doc_tokens = {
        t.strip(".,()/+-'\"_").lower()
        for t in doc_text.split()
        if len(t.strip(".,()/+-'\"_")) >= 2
    }
    matched = sum(1 for qt in q_tokens if qt in doc_tokens or qt in doc_text)
    return min(1.0, matched / max(1, len(q_tokens)))


def execute_live_hybrid_search(
    query_text: str,
    model_name: str = "gemini-3.8-flash",
    sparse_weight: float = 0.35,
    dense_weight: float = 0.65,
) -> Dict[str, Any]:
    """Runs a real-time Hybrid Search (Lexical BM25 + Vertex AI `gemini-embedding-001` 3072-D vector cosine similarity)
    against all shelf facings extracted by the selected model.
    """
    t0 = time.perf_counter()
    cfg = BenchmarkConfig.from_yaml(CONFIG_PATH)
    embedding_model_name = getattr(cfg, "embedding_model", "gemini-embedding-001")
    client = create_genai_client(project_id=cfg.gcp.project_id, location=cfg.gcp.location)

    rows_path = REPORTS_DIR / "row_level_report.json"
    rows_data = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    matching_rows = [
        r
        for r in rows_data
        if r.get("task_type") == "matching" and r.get("model_name") == model_name
    ]
    if not matching_rows:
        matching_rows = [r for r in rows_data if r.get("task_type") == "matching"]

    texts_to_embed = [query_text] + [
        r.get("dense_embedding_text")
        or f"Brand: {r.get('predicted_brand')} | Variant: {r.get('predicted_variant')} | Packaging: {r.get('predicted_packaging')} | Size: {r.get('predicted_size')}"
        for r in matching_rows
    ]

    embed_resp = client.models.embed_content(
        model=embedding_model_name,
        contents=texts_to_embed,
    )
    vectors = [e.values for e in embed_resp.embeddings]
    query_vec = vectors[0]
    doc_vecs = vectors[1:]

    scored_results: List[Dict[str, Any]] = []
    for idx, (row, d_vec) in enumerate(zip(matching_rows, doc_vecs)):
        dense_sim = _cosine_similarity(query_vec, d_vec)
        lexical_sim = _lexical_bm25_overlap(
            query_text,
            row.get("lexical_search_keywords", ""),
            row.get("dense_embedding_text", ""),
        )
        hybrid_score = (dense_weight * dense_sim) + (sparse_weight * lexical_sim)
        scored_results.append(
            {
                "rank": 0,
                "product_index": row.get("product_index", idx + 1),
                "shelf_row": row.get("shelf_row", "middle"),
                "position_on_shelf": row.get("position_on_shelf", idx + 1),
                "bbox": [
                    row.get("bbox_ymin", 0),
                    row.get("bbox_xmin", 0),
                    row.get("bbox_ymax", 0),
                    row.get("bbox_xmax", 0),
                ],
                "brand": row.get("predicted_brand", ""),
                "is_hul_brand": row.get("is_hul_brand", False),
                "category": row.get("predicted_category", ""),
                "subcategory": row.get("predicted_subcategory", ""),
                "variant": row.get("predicted_variant", ""),
                "packaging": row.get("predicted_packaging", ""),
                "pack_type": row.get("predicted_pack_type", "Single"),
                "size": row.get("predicted_size", ""),
                "lexical_search_keywords": row.get("lexical_search_keywords", ""),
                "dense_embedding_text": row.get("dense_embedding_text", ""),
                "dense_cosine_similarity": round(dense_sim, 4),
                "sparse_lexical_score": round(lexical_sim, 4),
                "hybrid_rrf_score": round(hybrid_score, 4),
            }
        )

    scored_results.sort(key=lambda x: x["hybrid_rrf_score"], reverse=True)
    for r_idx, item in enumerate(scored_results, start=1):
        item["rank"] = r_idx

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "query": query_text,
        "embedding_model": embedding_model_name,
        "vector_dimension": len(query_vec),
        "model_name": model_name,
        "latency_ms": latency_ms,
        "sparse_weight": sparse_weight,
        "dense_weight": dense_weight,
        "results": scored_results,
    }


def collect_gcp_runtime_proof() -> Dict[str, Any]:
    """Read live, un-mocked GCP Cloud Run runtime metadata, cgroup CPU/RAM limits, and physical GPU status."""
    import subprocess
    import urllib.request

    k_service = os.environ.get("K_SERVICE")
    k_revision = os.environ.get("K_REVISION")
    k_config = os.environ.get("K_CONFIGURATION")
    instance_id = None
    region = None
    if k_service:
        for attr, path in [("instance_id", "id"), ("region", "region")]:
            try:
                req = urllib.request.Request(
                    f"http://metadata.google.internal/computeMetadata/v1/instance/{path}",
                    headers={"Metadata-Flavor": "Google"},
                )
                with urllib.request.urlopen(req, timeout=1.0) as r:
                    val = r.read().decode("utf-8").strip()
                    if attr == "instance_id":
                        instance_id = val
                    else:
                        region = val.split("/")[-1]
            except Exception:
                pass

    cpu_quota_vcpu: Optional[float] = None
    try:
        if Path("/sys/fs/cgroup/cpu.max").exists():
            cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip().split()
            if len(cpu_max) == 2 and cpu_max[0] != "max":
                cpu_quota_vcpu = round(float(cpu_max[0]) / float(cpu_max[1]), 2)
        if cpu_quota_vcpu is None and Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").exists():
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8").strip())
            if q > 0 and p > 0:
                cpu_quota_vcpu = round(q / p, 2)
        if cpu_quota_vcpu is None and os.environ.get("CLOUD_RUN_VCPU"):
            cpu_quota_vcpu = float(os.environ["CLOUD_RUN_VCPU"])
        if cpu_quota_vcpu is None and os.cpu_count():
            cpu_quota_vcpu = float(os.cpu_count())
    except Exception:
        pass

    mem_limit_gib: Optional[float] = None
    try:
        if Path("/sys/fs/cgroup/memory.max").exists():
            mem_max = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
            if mem_max.isdigit():
                mem_limit_gib = round(int(mem_max) / (1024 ** 3), 2)
        if mem_limit_gib is None and Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").exists():
            m = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text(encoding="utf-8").strip())
            if 0 < m < (1024 ** 4):
                mem_limit_gib = round(m / (1024 ** 3), 2)
        if mem_limit_gib is None and os.environ.get("CLOUD_RUN_MEMORY_GIB"):
            mem_limit_gib = float(os.environ["CLOUD_RUN_MEMORY_GIB"])
    except Exception:
        pass

    physical_gpu = "NONE (CPU Container)"
    try:
        res_gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=1.5, check=False)
        if res_gpu.returncode == 0 and res_gpu.stdout.strip():
            physical_gpu = res_gpu.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return {
        "running_on_cloud_run": bool(k_service),
        "k_service": k_service or "local-process",
        "k_revision": k_revision or "local-unversioned",
        "k_configuration": k_config or "local",
        "gcp_metadata_instance_id": instance_id,
        "gcp_metadata_region": region,
        "cgroup_cpu_limit_vcpu": cpu_quota_vcpu,
        "cgroup_memory_limit_gib": mem_limit_gib,
        "physical_gpu_attached": physical_gpu,
    }


def execute_live_benchmark_task(
    task_type: str,
    model_name: str,
    separation_approach: str = "single_pass_full_shelf",
    mode: str = "live",
    connect_sample_gt: bool = False,
    brand_mode: str = "open_vocabulary_generative",
    attribute_call_mode: str = "single_call",
    accelerator: str = "none",
    vcpu_count: Optional[float] = None,
    memory_gib: Optional[float] = None,
    concurrency: Optional[int] = None,
    shelf_image_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Invokes the `shelf_benchmark` task classes or `GLOBAL_APPROACH_REGISTRY` plugins.

    Supports both `mode="live"` (Vertex AI Gemini 3) and `mode="offline"`,
    plus `connect_sample_gt=True`, `brand_mode`, `attribute_call_mode`, `vcpu_count`, `memory_gib`, and `accelerator`.
    """
    from shelf_benchmark.approaches import CommonLayerContext, GLOBAL_APPROACH_REGISTRY
    from shelf_benchmark.approaches.registry import BUILTIN_VLM_CLASSIFICATION_APPROACHES
    from shelf_benchmark.config import normalize_vertex_gemini_model_id
    from shelf_benchmark.testing import sample_ground_truth

    model_name = normalize_vertex_gemini_model_id(model_name, for_live_vertex=(str(mode).lower().strip() != "offline"))
    cfg = BenchmarkConfig.from_yaml(CONFIG_PATH)
    cfg.billing.include_infrastructure_costs = True
    if brand_mode:
        cfg.taxonomy.brand_extraction_mode = str(brand_mode)
    if attribute_call_mode == "grouped_calls":
        cfg.taxonomy.attribute_call_groups = [
            ["category", "subcategory", "brand", "product_name", "variant"],
            ["packaging_type", "pack_type", "size"],
        ]
    if vcpu_count is not None:
        cfg.billing.cloud_run.vcpu_count = float(vcpu_count)
    elif os.environ.get("CLOUD_RUN_VCPU"):
        cfg.billing.cloud_run.vcpu_count = float(os.environ["CLOUD_RUN_VCPU"])
    if memory_gib is not None:
        cfg.billing.cloud_run.memory_gib = float(memory_gib)
    elif os.environ.get("CLOUD_RUN_MEMORY_GIB"):
        cfg.billing.cloud_run.memory_gib = float(os.environ["CLOUD_RUN_MEMORY_GIB"])
    if concurrency is not None:
        cfg.billing.cloud_run.concurrency = int(concurrency)
    if accelerator and accelerator != "none":
        cfg.billing.cloud_run.accelerator_type = str(accelerator)
        cfg.billing.cloud_run.accelerator_count = 1

    is_offline = str(mode).lower().strip() == "offline"
    if is_offline:
        cfg.offline.enabled = True
        cfg.telemetry.export_to_gcp_cloud_logging = False
        cfg.telemetry.sync_otel_logs_to_gcs = False
        cfg.reporting.sync_reports_to_gcs = False
        cfg.billing.use_live_cloud_billing_catalog_api = False

    runner = BenchmarkRunner(config=cfg)
    records = runner.resolve_records()
    record = records[0]
    if shelf_image_uri:
        record.shelf_image_uri = str(shelf_image_uri)
    gt_record = sample_ground_truth(record.shelf_image_uri) if connect_sample_gt else None

    if task_type == "detection":
        res: TaskExecutionResult = runner.detection_task.execute(
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            run_id=f"{mode}-det-{model_name}",
            store_id=record.store_id,
            ground_truth=gt_record,
        )
    elif task_type == "classification":
        plugin = (
            None
            if separation_approach in BUILTIN_VLM_CLASSIFICATION_APPROACHES
            else GLOBAL_APPROACH_REGISTRY.get(separation_approach)
        )
        if plugin is not None:
            ctx = CommonLayerContext(
                config=cfg,
                storage=runner.storage,
                telemetry=runner.telemetry,
                reports_dir=REPORTS_DIR,
                genai_client=runner._genai_client,
            )
            res = plugin.execute(
                ctx=ctx,
                model_name=model_name,
                record=record,
                gt_record=gt_record,
            )
        else:
            res = runner.classification_task.execute(
                model_name=model_name,
                shelf_image_uri=record.shelf_image_uri,
                run_id=f"{mode}-cls-{model_name}",
                store_id=record.store_id,
                ground_truth=gt_record,
                separation_approach=separation_approach,
            )
    elif task_type == "matching":
        res = runner.matching_task.execute(
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            run_id=f"{mode}-mat-{model_name}",
            store_id=record.store_id,
            ground_truth=gt_record,
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    dashboard = load_dashboard_payload()
    crops_key = f"{model_name}_{separation_approach}".replace("/", "_")
    crops_info = dashboard.get("crops_manifest", {}).get(
        crops_key,
        dashboard.get("crops_manifest", {}).get(model_name, {}),
    )
    depth_demo = dashboard.get("depth_demos", {}).get(model_name, {})

    gcp_proof = collect_gcp_runtime_proof()
    res.execution_trace["gcp_runtime_proof"] = gcp_proof
    summary_record = {
        "run_id": res.run_id,
        "trace_id": res.trace_id,
        "span_id": res.span_id,
        "task_type": res.task_type,
        "separation_approach": res.separation_approach,
        "model_name": res.model_name,
        "shelf_image_uri": res.shelf_image_uri,
        "status": res.status,
        "execution_mode": "offline_local_fixture" if is_offline else "live_vertex_ai",
        "start_time": res.start_time,
        "end_time": res.end_time,
        "latency_ms": round(res.latency_ms, 2),
        "front_facings_count": len(res.row_level_items),
        "depth_duplicates_filtered": res.accuracy.depth_duplicates_filtered,
        "latency_per_facing_ms": round(
            res.latency_ms / max(len(res.row_level_items), 1), 2
        ),
        "input_tokens": res.tokens.input_tokens,
        "thinking_tokens": res.tokens.thinking_tokens,
        "output_tokens": res.tokens.output_tokens,
        "total_tokens": res.tokens.total_tokens,
        "vertex_ai_payg_tokens_usd": res.cost.vertex_ai_payg_tokens_usd,
        "vertex_ai_provisioned_throughput_usd": res.cost.vertex_ai_provisioned_throughput_usd,
        "vertex_ai_embeddings_and_vision_usd": res.cost.vertex_ai_embeddings_and_vision_usd,
        "cloud_run_compute_usd": res.cost.cloud_run_compute_usd,
        "gcs_and_observability_usd": res.cost.gcs_and_observability_usd,
        "cost_per_shelf_image_usd": res.cost.cost_per_shelf_image_usd,
        "cost_per_product_usd": res.cost.cost_per_product_usd,
        "all_in_pt_gsu_total_usd": (
            round(
                res.cost.vertex_ai_provisioned_throughput_usd
                + res.cost.vertex_ai_embeddings_and_vision_usd
                + res.cost.cloud_run_compute_usd
                + res.cost.gcs_and_observability_usd,
                8,
            )
            if res.cost.vertex_ai_provisioned_throughput_usd > 0
            else None
        ),
        "cloud_run_worker_service": os.environ.get("K_SERVICE") or "local (not Cloud Run)",
        "gcp_runtime_proof": gcp_proof,
        "billing_source": res.cost.billing_source,
        "rates_from_live_catalog": res.cost.rates_from_live_catalog,
        "includes_modelled_infrastructure": res.cost.includes_modelled_infrastructure,
        "ground_truth_available": res.accuracy.ground_truth_available,
        "accuracy_status": res.accuracy.accuracy_status,
        "gt_version": res.accuracy.gt_version,
        "iou_threshold": res.accuracy.iou_threshold,
        "pairing_strategy": res.accuracy.pairing_strategy,
        "brand_matcher": res.accuracy.brand_matcher,
        "product_matcher": res.accuracy.product_matcher,
        "matched_pairs": res.accuracy.matched_pairs,
        "true_positives": res.accuracy.true_positives,
        "false_positives": res.accuracy.false_positives,
        "false_negatives": res.accuracy.false_negatives,
        "count_accuracy": res.accuracy.count_accuracy,
        "mean_iou_matched": res.accuracy.mean_iou_matched,
        "detection_precision": res.accuracy.detection_precision,
        "detection_recall": res.accuracy.detection_recall,
        "detection_f1": res.accuracy.detection_f1,
        "brand_classification_accuracy": res.accuracy.brand_classification_accuracy,
        "brand_set_recall": res.accuracy.brand_set_recall,
        "product_classification_accuracy": res.accuracy.product_classification_accuracy,
        "sku_matching_accuracy": res.accuracy.sku_matching_accuracy,
        "planogram_compliance_rate": res.accuracy.planogram_compliance_rate,
        "execution_trace": res.execution_trace,
    }

    return {
        "run_id": res.run_id,
        "trace_id": res.trace_id,
        "span_id": res.span_id,
        "start_time": res.start_time,
        "end_time": res.end_time,
        "task_type": res.task_type,
        "separation_approach": res.separation_approach,
        "model_name": res.model_name,
        "status": res.status,
        "execution_mode": summary_record["execution_mode"],
        "latency_ms": round(res.latency_ms, 2),
        "tokens": res.tokens.model_dump(),
        "cost": res.cost.model_dump(),
        "accuracy": res.accuracy.model_dump(),
        "execution_trace": res.execution_trace,
        "front_facings_count": len(res.row_level_items),
        "depth_duplicates_filtered": res.accuracy.depth_duplicates_filtered,
        "gcp_runtime_proof": gcp_proof,
        "summary_record": summary_record,
        "rows": [item.model_dump() for item in res.row_level_items],
        "row_level_items": [item.model_dump() for item in res.row_level_items],
        "items_preview": [item.model_dump() for item in res.row_level_items],
        "crops_info": crops_info,
        "depth_demo": depth_demo,
        "_task_execution_result_obj": res,
    }


def execute_cloud_run_benchmark_suite(body: Dict[str, Any]) -> Dict[str, Any]:
    """100% Cloud-Run-Native Full Suite Orchestrator (`POST /api/run-suite`).

    Executes inside the Cloud Run container on GCP:
    1. Downloads images & ground truth directly from GCS inside Cloud Run.
    2. Runs all requested approaches using the EXACT `model_name` requested by the user (zero substitution).
    3. Computes accuracy, live Cloud Billing Catalog rates, and granular container CPU/RAM/GPU costs inside Cloud Run.
    4. Emits OpenTelemetry traces & structured logs (`resource.type="cloud_run_revision"`) directly to GCP Cloud Logging.
    5. Generates all benchmark reports (`leaderboard.csv`, `benchmark_report.md`, `predictions.json`, `diagnostic_trace_report.json`)
       inside the Cloud Run container and syncs them to GCS, returning the complete suite bundle to the thin CLI client.
    """
    import tempfile
    from shelf_benchmark.approaches.registry import BUILTIN_VLM_CLASSIFICATION_APPROACHES, GLOBAL_APPROACH_REGISTRY
    from shelf_benchmark.config import normalize_vertex_gemini_model_id
    from shelf_benchmark.reporting.generator import BenchmarkReportGenerator

    model_name = normalize_vertex_gemini_model_id(body.get("model_name") or "gemini-3-flash-preview", for_live_vertex=True)
    raw_approaches = body.get("approaches") or ["all"]
    if isinstance(raw_approaches, str):
        raw_approaches = [x.strip() for x in raw_approaches.split(",") if x.strip()]

    GLOBAL_APPROACH_REGISTRY.ensure_discovered()
    if "all" in raw_approaches:
        approaches = sorted(set(GLOBAL_APPROACH_REGISTRY.list_ids()) | set(BUILTIN_VLM_CLASSIFICATION_APPROACHES))
    else:
        approaches = []
        for item in raw_approaches:
            for part in str(item).split(","):
                if part.strip():
                    approaches.append(part.strip())

    shelf_image_uri = body.get("shelf_image_uri") or "gs://unilever-shelf-understanding-shelf-images/shelf-image.png"
    connect_sample_gt = bool(body.get("connect_sample_gt", False))
    vcpu_count = float(body.get("vcpu_count") or os.environ.get("CLOUD_RUN_VCPU") or 2.0)
    memory_gib = float(body.get("memory_gib") or os.environ.get("CLOUD_RUN_MEMORY_GIB") or 4.0)
    accelerator = str(body.get("accelerator") or os.environ.get("CLOUD_RUN_ACCELERATOR") or "none")
    concurrency = int(body.get("concurrency") or 1)

    suite_out_dir = Path(tempfile.mkdtemp(prefix="cloudrun-suite-reports-"))
    cfg = BenchmarkConfig.from_yaml(CONFIG_PATH)
    cfg.billing.include_infrastructure_costs = True
    cfg.billing.cloud_run.vcpu_count = vcpu_count
    cfg.billing.cloud_run.memory_gib = memory_gib
    cfg.billing.cloud_run.concurrency = concurrency
    cfg.billing.cloud_run.accelerator_type = accelerator
    cfg.billing.cloud_run.accelerator_count = 1 if accelerator != "none" else 0
    cfg.reporting.output_dir = str(suite_out_dir)
    cfg.reporting.sync_reports_to_gcs = True
    cfg.telemetry.export_to_gcp_cloud_logging = True
    cfg.telemetry.otel_log_path = str(suite_out_dir / "otel_logs.jsonl")

    collected_objs = []
    serialized_runs = []
    for app_id in approaches:
        run_dict = execute_live_benchmark_task(
            task_type="classification",
            model_name=model_name,
            separation_approach=app_id,
            mode="live",
            connect_sample_gt=connect_sample_gt,
            accelerator=accelerator,
            vcpu_count=vcpu_count,
            memory_gib=memory_gib,
            concurrency=concurrency,
            shelf_image_uri=shelf_image_uri,
        )
        res_obj = run_dict.pop("_task_execution_result_obj")
        collected_objs.append(res_obj)
        serialized_runs.append(run_dict)

    # Generate all canonical reports INSIDE the Cloud Run container
    report_gen = BenchmarkReportGenerator.from_config(cfg)
    artifact_paths = report_gen.generate_all_reports(collected_objs)

    # Read generated report contents inside Cloud Run so the thin CLI can save them locally without re-computing anything
    report_files_content: Dict[str, str] = {}
    for k, p_str in artifact_paths.items():
        if k == "output_dir":
            continue
        p = Path(p_str)
        if p.exists() and p.is_file():
            report_files_content[p.name] = p.read_text(encoding="utf-8")

    gcp_proof = collect_gcp_runtime_proof()
    return {
        "orchestrated_ inside_cloud_run": True,
        "gcp_runtime_proof": gcp_proof,
        "model_name": model_name,
        "approaches_executed": approaches,
        "hardware_config": {
            "vcpu_count": vcpu_count,
            "memory_gib": memory_gib,
            "concurrency": concurrency,
            "accelerator": accelerator,
        },
        "runs": serialized_runs,
        "report_files_content": report_files_content,
    }


class BenchmarkUIRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving static UI assets and JSON API endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output concise
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        mime_type, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/dashboard":
            try:
                self._send_json(load_dashboard_payload())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        if route == "/shelf-image.png":
            self._serve_file(REPO_ROOT / "shelf-image.png")
            return

        if route.startswith("/crops/"):
            rel_path = route.lstrip("/")
            target = (REPORTS_DIR / rel_path).resolve()
            if str(target).startswith(str(REPORTS_DIR.resolve())):
                self._serve_file(target)
            else:
                self.send_error(403, "Forbidden")
            return

        if route == "/" or route == "/index.html":
            self._serve_file(STATIC_DIR / "index.html")
            return

        static_target = (STATIC_DIR / route.lstrip("/")).resolve()
        if str(static_target).startswith(str(STATIC_DIR.resolve())) and static_target.is_file():
            self._serve_file(static_target)
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(body_raw)
        except Exception:
            body = {}

        if route == "/api/hybrid-search":
            try:
                query = body.get(
                    "query",
                    "Brand_A Radiance Daily Cleanser Tube Single",
                )
                model_name = body.get("model_name", "gemini-3.8-flash")
                sparse_w = float(body.get("sparse_weight", 0.35))
                dense_w = float(body.get("dense_weight", 0.65))
                result = execute_live_hybrid_search(
                    query_text=query,
                    model_name=model_name,
                    sparse_weight=sparse_w,
                    dense_weight=dense_w,
                )
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        if route == "/api/run-suite":
            try:
                result = execute_cloud_run_benchmark_suite(body)
                self._send_json(result)
            except Exception as exc:
                import traceback
                self._send_json({"error": str(exc), "traceback": traceback.format_exc()}, status=500)
            return

        if route == "/api/run-live":
            try:
                task_type = body.get("task_type") or body.get("task") or "classification"
                model_name = body.get("model_name") or body.get("model") or "gemini-3-flash-preview"
                separation_approach = (
                    body.get("separation_approach")
                    or body.get("approach")
                    or "single_pass_full_shelf"
                )
                mode = str(body.get("mode", "live"))
                connect_sample_gt = bool(body.get("connect_sample_gt", False))
                brand_mode = str(body.get("brand_mode", "open_vocabulary_generative"))
                attribute_call_mode = str(body.get("attribute_call_mode", "single_call"))
                accelerator = str(body.get("accelerator", "none"))
                vcpu_count = body.get("vcpu_count")
                memory_gib = body.get("memory_gib")
                concurrency = body.get("concurrency")
                shelf_image_uri = body.get("shelf_image_uri")
                result = execute_live_benchmark_task(
                    task_type=task_type,
                    model_name=model_name,
                    separation_approach=separation_approach,
                    mode=mode,
                    connect_sample_gt=connect_sample_gt,
                    brand_mode=brand_mode,
                    attribute_call_mode=attribute_call_mode,
                    accelerator=accelerator,
                    vcpu_count=float(vcpu_count) if vcpu_count is not None else None,
                    memory_gib=float(memory_gib) if memory_gib is not None else None,
                    concurrency=int(concurrency) if concurrency is not None else None,
                    shelf_image_uri=shelf_image_uri,
                )
                result.pop("_task_execution_result_obj", None)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        self.send_error(404, "Unknown API endpoint")


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), BenchmarkUIRequestHandler)
    print(f"Unilever Shelf Understanding Benchmark UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port_val = int(os.environ.get("PORT", "8080"))
    run_server(host="0.0.0.0", port=port_val)
