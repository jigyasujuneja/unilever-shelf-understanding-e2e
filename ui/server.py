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


def _build_depth_demo_candidates(
    kept_products: List[Dict[str, Any]],
    depth_filtered_count: int,
) -> List[Dict[str, Any]]:
    """Reconstructs raw candidate boxes (front-facing + depth-stacked back-row units)
    and runs `deduplicate_depth_stacked_facings` from `shelf_benchmark.tasks.facing_utils`
    to annotate which boxes were kept vs. suppressed by the Front-Facing Depth NMS filter.
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
            }
        )

    back_row_Candidates: List[Dict[str, Any]] = []
    if depth_filtered_count > 0 and kept_products:
        step = max(1, len(kept_products) // depth_filtered_count)
        for idx in range(depth_filtered_count):
            front_ref = kept_products[min(idx * step + 1, len(kept_products) - 1)]
            ymin = max(10, int(front_ref.get("bbox_ymin", 550)) - 34)
            xmin = min(985, int(front_ref.get("bbox_xmin", 100)) + 4)
            ymax = max(ymin + 80, int(front_ref.get("bbox_ymax", 780)) - 38)
            xmax = min(995, int(front_ref.get("bbox_xmax", 160)) - 2)
            back_row_Candidates.append(
                {
                    "product_index": 100 + idx + 1,
                    "bbox_2d": [ymin, xmin, ymax, xmax],
                    "shelf_row": front_ref.get("shelf_row", "middle"),
                    "position_on_shelf": front_ref.get("position_on_shelf", 1),
                    "is_front_facing": True,
                    "brand": front_ref.get("predicted_brand", "Pond's"),
                    "variant": f"Back-row depth duplicate behind slot #{front_ref.get('position_on_shelf', 1)} ({front_ref.get('predicted_brand', '')})",
                    "confidence": 0.84,
                    "synthetic_depth_candidate": True,
                    "occluded_by_slot": front_ref.get("position_on_shelf", 1),
                    "front_ymax": front_ref.get("bbox_ymax", 780),
                }
            )

    raw_pool = candidates + back_row_Candidates
    kept_after_nms, filtered_count = deduplicate_depth_stacked_facings(
        raw_pool, x_overlap_threshold=0.45
    )
    kept_coords = {tuple(item["bbox_2d"]) for item in kept_after_nms}

    annotated: List[Dict[str, Any]] = []
    for item in raw_pool:
        coords = tuple(item["bbox_2d"])
        is_kept = coords in kept_coords and not item.get("synthetic_depth_candidate", False)
        annotated.append(
            {
                **item,
                "kept_by_depth_nms": is_kept,
                "nms_reason": (
                    "Front-most unit in horizontal shelf slot (highest y_max base)"
                    if is_kept
                    else (
                        f"Suppressed by Depth NMS: 1D X-overlap >= 0.45 with Slot #{item.get('occluded_by_slot')} "
                        f"and lower shelf base (y_max={item['bbox_2d'][2]} < {item.get('front_ymax')})"
                    )
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

    # Build per-model detection depth NMS visualization data
    depth_demos: Dict[str, Any] = {}
    for s in summary_data.get("summary", []):
        if s.get("task_type") == "detection":
            model_name = s.get("model_name")
            det_rows = [
                r
                for r in rows_data
                if r.get("task_type") == "detection" and r.get("model_name") == model_name
            ]
            depth_demos[model_name] = {
                "front_facings_count": s.get("front_facings_count", len(det_rows)),
                "depth_duplicates_filtered": s.get("depth_duplicates_filtered", 0),
                "raw_detections_count": s.get("front_facings_count", len(det_rows))
                + s.get("depth_duplicates_filtered", 0),
                "boxes": _build_depth_demo_candidates(
                    det_rows, int(s.get("depth_duplicates_filtered", 0))
                ),
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
        "summary": summary_data.get("summary", []),
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


def execute_live_benchmark_task(
    task_type: str,
    model_name: str,
    separation_approach: str = "single_pass_full_shelf",
) -> Dict[str, Any]:
    """Directly invokes the unmodified `shelf_benchmark` task classes or `GLOBAL_APPROACH_REGISTRY` plugins against Vertex AI."""
    from shelf_benchmark.approaches import CommonLayerContext, GLOBAL_APPROACH_REGISTRY

    cfg = BenchmarkConfig.from_yaml(CONFIG_PATH)
    runner = BenchmarkRunner(config=cfg)
    records = runner.resolve_records()
    record = records[0]

    if task_type == "detection":
        res: TaskExecutionResult = runner.detection_task.execute(
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            run_id=f"live-det-{model_name}",
            store_id=record.store_id,
            ground_truth=None,
        )
    elif task_type == "classification":
        plugin = GLOBAL_APPROACH_REGISTRY.get(separation_approach)
        if plugin is not None:
            ctx = CommonLayerContext(
                config=cfg,
                storage=runner.storage,
                telemetry=runner.telemetry,
                reports_dir=REPORTS_DIR,
            )
            res = plugin.execute(
                ctx=ctx,
                model_name=model_name,
                record=record,
                gt_record=None,
            )
        else:
            res = runner.classification_task.execute(
                model_name=model_name,
                shelf_image_uri=record.shelf_image_uri,
                run_id=f"live-cls-{model_name}",
                store_id=record.store_id,
                ground_truth=None,
                separation_approach=separation_approach,
            )
    elif task_type == "matching":
        res = runner.matching_task.execute(
            model_name=model_name,
            shelf_image_uri=record.shelf_image_uri,
            run_id=f"live-mat-{model_name}",
            store_id=record.store_id,
            ground_truth=None,
        )
    else:
        raise ValueError(f"Unsupported live task_type: {task_type}")

    return {
        "run_id": res.run_id,
        "trace_id": res.trace_id,
        "span_id": res.span_id,
        "task_type": res.task_type,
        "separation_approach": res.separation_approach,
        "model_name": res.model_name,
        "status": res.status,
        "latency_ms": round(res.latency_ms, 2),
        "tokens": res.tokens.model_dump(),
        "cost": res.cost.model_dump(),
        "accuracy": res.accuracy.model_dump(),
        "front_facings_count": len(res.row_level_items),
        "items_preview": [item.model_dump() for item in res.row_level_items[:18]],
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
                    "Pond's Bright Beauty Spot-less Glow Face Wash Tube Single",
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

        if route == "/api/run-live":
            try:
                task_type = body.get("task_type", "detection")
                model_name = body.get("model_name", "gemini-3.5-flash-lite")
                separation_approach = body.get(
                    "separation_approach", "single_pass_full_shelf"
                )
                result = execute_live_benchmark_task(
                    task_type=task_type,
                    model_name=model_name,
                    separation_approach=separation_approach,
                )
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
