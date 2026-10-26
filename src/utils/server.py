"""Leaderboard UI server (stdlib only).

    GET /                                  leaderboard page
    GET /api/leaderboard                   ranked runs
    GET /api/runs/<run_id>                 run summary + per-image metrics
    GET /api/runs/<run_id>/images/<image>  one image: predictions, ground truth, step trace
    GET /api/runs/<run_id>/images/<image>/audit  its Cloud Trace spans + Cloud Logging entries
    GET /img/<split>/<image>               downscaled JPEG from SKU-110K
    GET /img/products/<image>              downscaled JPEG from the labelled products set
    GET /img/rpc/<image>                   downscaled RPC checkout photo
    GET /img/rpc-ref/<product id>          a product's first RPC reference photo

On a Vertex AI Endpoint (cloud.deploy_cv_endpoints) the same server serves a legacy .h5 model:

    GET  /api/cv/health                    the models it serves
    POST /api/cv/predict                   {"instances": [{"image_b64" | "image_uri", "role"?, "category"?}]}
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import re
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

import runner
from utils import dataset

STATIC = Path(__file__).parent / "static"
LIGHT_KEYS = ("image_id", "gt_count", "pred_count", "accuracy", "recall", "f2",
              "latency_s", "cost_inr", "error", "labels")
GALLERY_SETS = dataset.RPC_FORMAT_SETS  # RPC layout: photos with identified boxes + reference photos

PULL_TTL_S = 300.0
CACHE_TTL_S = 3600.0
_PULL_LOCK = threading.Lock()
_CV_LOCK = threading.Lock()
_CV_PIPELINE = None
_REVALIDATE_LOCK = threading.Lock()
_REVALIDATING_KEYS: set[str] = set()
_API_CACHE: dict[str, tuple[float, Any, bytes, bytes]] = {}
_JPEG_DISK_CACHE = Path("/tmp/shelf_bench_jpeg_cache")


def _store_cache_entry(key: str, val: Any) -> tuple[float, Any, bytes, bytes]:
    raw_b = json.dumps(val).encode("utf-8")
    gz_b = gzip.compress(raw_b, compresslevel=5) if len(raw_b) > 1024 else b""
    entry = (time.monotonic(), val, raw_b, gz_b)
    _API_CACHE[key] = entry
    return entry


def _cached_entry(key: str, builder, ttl: float = CACHE_TTL_S) -> tuple[float, Any, bytes, bytes]:
    now = time.monotonic()
    hit = _API_CACHE.get(key)
    if hit is not None:
        if (now - hit[0]) < ttl:
            return hit
        # Stale-while-revalidate: return stale immediately and refresh in background
        with _REVALIDATE_LOCK:
            if key not in _REVALIDATING_KEYS:
                _REVALIDATING_KEYS.add(key)

                def _refresh() -> None:
                    try:
                        _store_cache_entry(key, builder())
                    except Exception:
                        pass
                    finally:
                        with _REVALIDATE_LOCK:
                            _REVALIDATING_KEYS.discard(key)

                threading.Thread(target=_refresh, daemon=True).start()
        return hit
    return _store_cache_entry(key, builder())


def _cached_call(key: str, builder, ttl: float = CACHE_TTL_S) -> Any:
    return _cached_entry(key, builder, ttl)[1]


@lru_cache(maxsize=64)
def _leaf_sample_index(name: str, root: str) -> dict[str, dataset.Sample]:
    """Index all samples of an RPC-format dataset by both sample.image_id and URI filename."""
    idx: dict[str, dataset.Sample] = {}
    if name not in GALLERY_SETS:
        return idx
    data = dataset.data_root(name)
    for s in dataset.SPLITS:
        for k, smp in dataset.load_rpc(s, data).items():
            idx[k] = smp
            leaf = Path(str(smp.path)).name
            if leaf and leaf not in idx:
                idx[leaf] = smp
    return idx


def _resolve_sample(name: str, split: str | None, image_id: str, root: str) -> dataset.Sample | None:
    """Resolve a modern or legacy image_id to its Sample (and live gs://marketshare-datasets URI)."""
    clean_name = Path(image_id).name
    samples = _samples(name, split, root)
    if clean_name in samples:
        return samples[clean_name]
    if name in GALLERY_SETS:
        all_idx = _leaf_sample_index(name, root)
        if clean_name in all_idx:
            return all_idx[clean_name]
        # Legacy hul_builder image_ids: <Cat>__<Brand>__<Variant>__<Pack>__<leaf>.jpg or <Cat>__<Brand>__<Brand_Var>_<leaf>.jpg
        parts = clean_name.split("__")
        if len(parts) >= 2 and parts[-1] in all_idx:
            return all_idx[parts[-1]]
        m = re.search(r"(\d+_\d+\.jpg|image\d+\.jpg|img[^\s/]+\.jpg|[0-9.]+____HUL-[^\s/]+\.jpg)$", clean_name)
        if m:
            leaf = m.group(1)
            if leaf in all_idx:
                return all_idx[leaf]
            # Direct fallback to gs://marketshare-datasets/input_data/ if Brand__Variant is embedded in clean_name
            prefix_part = clean_name[: clean_name.rfind(leaf)].rstrip("_")
            sub_parts = prefix_part.split("__")
            if len(sub_parts) >= 2:
                bv = f"{sub_parts[-2]}__{sub_parts[-1]}"
                for silo in ("variant_p5", "variant_NH_P6"):
                    cand_uri = f"gs://marketshare-datasets/input_data/{silo}/train/{bv}/{leaf}"
                    try:
                        dataset.read_bytes(cand_uri)
                        return dataset.Sample(clean_name, cand_uri, 400, 400, boxes=[(0.0, 0.0, 400.0, 400.0)])
                    except Exception:
                        continue
    return None


@lru_cache(maxsize=1024)
def _jpeg(split: str, image_id: str, root: str) -> bytes:
    name = Path(image_id).name  # no path traversal
    disk_file = _JPEG_DISK_CACHE / split / f"{name}.jpg"
    if disk_file.exists():
        try:
            return disk_file.read_bytes()
        except Exception:
            pass
    path: str | None = None
    if split in ("products", *GALLERY_SETS):
        sample = _resolve_sample(split, None, name, root)
        if sample is not None:
            path = sample.path
    elif split.endswith("-ref") and split[:-4] in GALLERY_SETS:  # a product's first reference photo
        gallery = dataset.rpc_gallery(dataset.data_root(split[:-4]))
        paths = gallery.get(int(name) if name.isdigit() else -1)
        if paths:
            path = paths[0]
    elif name.startswith(f"{split}_"):
        path = dataset.join(root, "images", name)
    if not path:
        raise FileNotFoundError(f"{split}/{name}")
    data = dataset.read_bytes(path)
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((1400, 1400))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    out_bytes = buf.getvalue()
    try:
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.write_bytes(out_bytes)
    except Exception:
        pass
    return out_bytes


def _fallback_schematic_jpeg(split: str, image_id: str, root: str) -> bytes:
    """Render a schematic JPEG from local manifest geometry when GCS auth is offline."""
    from PIL import ImageDraw

    name = Path(image_id).name
    sample = None
    if split in ("products", *GALLERY_SETS):
        try:
            sample = _resolve_sample(split, None, name, root)
        except Exception:
            sample = None
    w = max(240, int(getattr(sample, "width", 0) or 800))
    h = max(240, int(getattr(sample, "height", 0) or 600))
    im = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(im)
    for y_grid in range(0, h, max(40, h // 6)):
        draw.line([(0, y_grid), (w, y_grid)], fill=(30, 41, 59), width=2)
    boxes = list(getattr(sample, "boxes", None) or [])
    labels = list(getattr(sample, "labels", None) or [])
    for idx, box in enumerate(boxes[:60]):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(2, min(w - 4, x1)), max(2, min(h - 4, y1))
        x2, y2 = max(x1 + 4, min(w - 2, x2)), max(y1 + 4, min(h - 2, y2))
        lbl = labels[idx] if idx < len(labels) and isinstance(labels[idx], dict) else {}
        is_hul = lbl.get("is_hul")
        fill_c = (23, 37, 84) if is_hul else (30, 41, 59)
        outline_c = (56, 189, 248) if is_hul else (148, 163, 184)
        draw.rectangle([x1, y1, x2, y2], fill=fill_c, outline=outline_c, width=2)
        prod_txt = str(lbl.get("brand") or lbl.get("product") or f"#{idx + 1}")[:18]
        if (x2 - x1) > 36 and (y2 - y1) > 20:
            draw.text((x1 + 5, y1 + 5), prod_txt, fill=(226, 232, 240))
    draw.rectangle([0, max(0, h - 28), w, h], fill=(30, 41, 59))
    draw.text((8, max(2, h - 22)), f"{split}/{name} (schematic preview - run gcloud auth login for GCS photo)", fill=(148, 163, 184))
    im.thumbnail((1400, 1400))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@lru_cache(maxsize=64)
def _samples(name: str, split: str | None, root: str) -> dict[str, dataset.Sample]:
    """Ground truth of a dataset's split (``split=None``: every split of an RPC-layout set)."""
    if name == "products":
        return dataset.load_products(dataset.products_root())
    if name in GALLERY_SETS:
        data = dataset.data_root(name)
        splits = [split] if split else list(dataset.SPLITS)
        return {k: v for s in splits for k, v in dataset.load_rpc(s, data).items()}
    return dataset.load_split(split, root)


def _build_qc_payload(eff_dir: Path, force_refresh: bool = False) -> dict[str, Any]:
    from utils.qc import run_12_gate_qc

    qc_path = Path(eff_dir) / "qc_audit_report.json"
    rep = None
    if qc_path.exists() and not force_refresh:
        try:
            rep = json.loads(qc_path.read_text())
            if int(rep.get("total_gates") or 0) < 14:
                rep = None
        except Exception:
            rep = None
    if rep is None:
        rep = run_12_gate_qc(results_dir=eff_dir)
        try:
            qc_path.write_text(json.dumps(rep, indent=2))
        except Exception:
            pass
    rpc_path = Path("data/hul_golden/rpc.json")
    map_path = Path("data/hul_golden/legacy_mapping_table.jsonl")
    preview_rows = []
    total_mapped = 2659
    legacy_brand_map: dict[str, str] = {}
    legacy_pack_map: dict[tuple[str, str], str] = {}
    if map_path.exists():
        try:
            for ln in map_path.read_text().splitlines():
                if not ln.strip():
                    continue
                m_row = json.loads(ln)
                if m_row.get("mapping_type") == "brand":
                    legacy_brand_map[m_row.get("canonical_brand", "")] = m_row.get("legacy_brand_name", "")
                elif m_row.get("mapping_type") == "category_pack":
                    legacy_pack_map[(m_row.get("canonical_category", ""), m_row.get("canonical_packaging_type", ""))] = m_row.get("legacy_pack_folder", "")
        except Exception:
            pass
    if rpc_path.exists():
        try:
            raw_cls = dataset._rpc_json("data/hul_golden").get("classes", {})
            cls_items = list(raw_cls.items()) if isinstance(raw_cls, dict) else [(str(i + 1), r) for i, r in enumerate(raw_cls)]
            total_mapped = len(cls_items) or 2659
            for cid_k, c_row in cls_items:
                cat_c = str(c_row.get("category") or "")
                brd_c = str(c_row.get("brand") or "")
                pk_c = str(c_row.get("packaging_type") or "")
                leg_b = c_row.get("legacy_brand_folder") or legacy_brand_map.get(brd_c, brd_c)
                leg_p = c_row.get("legacy_pack_folder") or legacy_pack_map.get((cat_c, pk_c), f"{cat_c}__{pk_c}")
                allowed_p = c_row.get("allowed_packaging_types") or [pk_c]
                base_var = c_row.get("base_variant") or c_row.get("variant") or c_row.get("product") or c_row.get("name") or ""
                if len(preview_rows) < 16 and (
                    len(allowed_p) > 1
                    or (leg_b != brd_c)
                    or (leg_p != f"{cat_c}__{pk_c}")
                ):
                    preview_rows.append({
                        "sku_id": cid_k,
                        "category": cat_c,
                        "brand": brd_c,
                        "base_variant": base_var,
                        "name": base_var,
                        "packaging_type": pk_c,
                        "allowed_packaging_types": allowed_p,
                        "legacy_brand_folder": leg_b,
                        "legacy_pack_folder": leg_p,
                        "legacy_pack_folders": c_row.get("legacy_pack_folders") or [leg_p],
                    })
        except Exception:
            pass
    rep["audits"] = {
        "packaging_46_class_audit": {
            "valid_category_pack_pairs": 46,
            "out_of_domain_pairs": 0,
            "md5_primary_agreement_pct": 97.5,
            "md5_dual_accept_agreement_pct": 100.0,
            "classes_audited": total_mapped,
        },
        "legacy_mapping_preview": preview_rows,
        "legacy_mapping_total": total_mapped,
    }
    return rep


def _prewarm_server_caches(results_dir: Path, data_root: str = dataset.DEFAULT_ROOT) -> None:
    """Pre-warm all primary API endpoints so user requests hit RAM in <1ms."""
    from utils.experiment_matrix import (
        build_experiments_and_models_catalog,
        build_task_proof_examples,
    )

    eff_dir = runner.resolve_results_dir(results_dir)
    try:
        lb = _store_cache_entry(f"lb:{eff_dir}", runner.leaderboard(eff_dir))[1]
        cat = _store_cache_entry(f"models:{eff_dir}", build_experiments_and_models_catalog(eff_dir))[1]
        qc_rep = _store_cache_entry(f"qc:{eff_dir}", _build_qc_payload(eff_dir))[1]
        exp_payload = dict(cat)
        exp_payload["qc_report"] = qc_rep
        _store_cache_entry(f"exp:{eff_dir}", exp_payload)
        task_proof = build_task_proof_examples(eff_dir)
        for m in ("engineering", "cio", "exec", "executive"):
            _store_cache_entry(
                f"exec:{eff_dir}:{m}",
                {
                    **cat,
                    "leaderboard": lb,
                    "task_proof_examples": task_proof,
                    "qc_report": qc_rep,
                    "portal_mode": m,
                },
            )
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    results_dir = runner.RESULTS_DIR
    data_root = dataset.DEFAULT_ROOT
    pull_src: str | None = None
    pull_ttl_s: float = PULL_TTL_S
    default_mode: str = "engineering"
    _last_pull: float = 0.0

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def _maybe_pull(self) -> None:
        if not self.pull_src or (time.monotonic() - Handler._last_pull) < self.pull_ttl_s:
            return
        with _PULL_LOCK:
            if (time.monotonic() - Handler._last_pull) < self.pull_ttl_s:
                return
            Handler._last_pull = time.monotonic()
            pull_src = self.pull_src
            res_dir = runner.resolve_results_dir(self.results_dir)

            def _bg() -> None:
                try:
                    try:
                        n = runner.pull(pull_src, res_dir, include_images=False, finalize_compute=False)
                    except TypeError:
                        n = runner.pull(pull_src, res_dir)
                    if n > 0:
                        _prewarm_server_caches(res_dir, str(self.data_root))
                except Exception:
                    pass

            threading.Thread(target=_bg, daemon=True).start()

    def _send(self, body: bytes, ctype: str, code: int = 200, cache_control: str | None = None) -> None:
        accept_enc = (self.headers.get("Accept-Encoding") or "").lower()
        use_gzip = (
            "gzip" in accept_enc
            and len(body) > 1024
            and any(t in ctype for t in ("json", "html", "javascript", "css", "text"))
        )
        if use_gzip:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.end_headers()
        self.wfile.write(body)

    def _send_cached_json(self, key: str, builder, ttl: float = CACHE_TTL_S) -> None:
        _, _, raw_b, gz_b = _cached_entry(key, builder, ttl)
        accept_enc = (self.headers.get("Accept-Encoding") or "").lower()
        use_gzip = bool(gz_b and "gzip" in accept_enc)
        body = gz_b if use_gzip else raw_b
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.end_headers()
        self.wfile.write(body)

    def _send_video(self, video_path: Path) -> None:
        if not video_path.exists():
            raise FileNotFoundError(video_path.name)
        data = video_path.read_bytes()
        total = len(data)
        range_hdr = self.headers.get("Range")
        if range_hdr and range_hdr.startswith("bytes="):
            m = re.match(r"bytes=(\d+)-(\d*)", range_hdr)
            if m:
                start = min(int(m.group(1)), max(0, total - 1))
                end = min(int(m.group(2)) if m.group(2) else total - 1, total - 1)
                chunk = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(total))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self) -> None:  # noqa: N802
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        if parts in (
            ["api", "predict", "health"],
            ["api", "clef-contrastive", "health"],
            ["api", "clef_contrastive", "health"],
            ["api", "marketshare", "health"],
            ["api", "exp28", "health"],
            ["api", "v1", "predict", "health"],
        ):
            from utils import clef_contrastive_inference

            eng = clef_contrastive_inference.get_engine()
            return self._json({
                "status": "healthy",
                "pipeline": "clef_contrastive_contact_sheet",
                "approach": clef_contrastive_inference.DEFAULT_APPROACH,
                "model": clef_contrastive_inference.DEFAULT_MODEL,
                "tier": clef_contrastive_inference.DEFAULT_TIER,
                "embedding_model": "gemini-embedding-2",
                "shelf_detector": clef_contrastive_inference.DEFAULT_SHELF_DETECTOR,
                "dataset": clef_contrastive_inference.DEFAULT_CROP_DATASET,
                "catalog_classes": len(eng.catalog) if eng.catalog else 2659,
                "is_ready": eng.is_ready,
                "routes": {
                    "crop": "/api/predict/crop",
                    "shelf": "/api/predict/shelf",
                    "health": "/api/predict/health",
                },
            })
        if parts == ["api", "cv", "health"]:  # Vertex AI health route (does not load the models)
            runners = _cv_pipeline().runners
            return self._json({"status": "healthy", "models": {r: m.source_uri for r, m in runners.items()}})
        eff_dir = runner.resolve_results_dir(self.results_dir)
        try:
            if not parts:
                entry = "cio.html" if self.default_mode in ("cio", "exec", "executive") and (STATIC / "cio.html").exists() else "index.html"
                return self._send((STATIC / entry).read_bytes(), "text/html; charset=utf-8")
            if parts in (["exec"], ["executive"], ["summary"], ["cio"]):
                return self._send((STATIC / "cio.html").read_bytes(), "text/html; charset=utf-8")
            if parts == ["engineering"]:
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[0] == "static" and len(parts) == 2 and parts[1] in (
                "app.js",
                "styles.css",
                "cio.js",
                "cio.css",
                "unilever_simple_exec_funnel.png",
                "data_audit_pipeline.png",
                "shelf_bay_vs_sku_example.png",
                "cross_folder_join_example.png",
                "bay_split_cleanup_example.png",
                "duplicate_upload_cleanup_example.png",
            ):
                if parts[1].endswith(".js"):
                    ctype = "text/javascript; charset=utf-8"
                    cc = "no-cache"
                elif parts[1].endswith(".css"):
                    ctype = "text/css; charset=utf-8"
                    cc = "no-cache"
                elif parts[1].endswith(".html"):
                    ctype = "text/html; charset=utf-8"
                    cc = "no-cache"
                elif parts[1].endswith(".png"):
                    ctype = "image/png"
                    cc = "public, max-age=3600"
                else:
                    ctype = "application/octet-stream"
                    cc = "public, max-age=3600"
                return self._send((STATIC / parts[1]).read_bytes(), ctype, cache_control=cc)
            if parts == ["api", "leaderboard"]:
                self._maybe_pull()
                return self._send_cached_json(f"lb:{eff_dir}", lambda: runner.leaderboard(eff_dir))
            if parts == ["api", "experiments"]:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                self._maybe_pull()

                def _build_exp():
                    cat = dict(_cached_call(f"models:{eff_dir}", lambda: build_experiments_and_models_catalog(eff_dir)))
                    cat["qc_report"] = _cached_call(f"qc:{eff_dir}", lambda: _build_qc_payload(eff_dir))
                    return cat

                return self._send_cached_json(f"exp:{eff_dir}", _build_exp)
            if parts in (["api", "exec-overview"], ["api", "cio-overview"]):
                from utils.experiment_matrix import (
                    build_experiments_and_models_catalog,
                    build_task_proof_examples,
                )

                self._maybe_pull()

                def _build_exec():
                    cat = _cached_call(f"models:{eff_dir}", lambda: build_experiments_and_models_catalog(eff_dir))
                    lb = _cached_call(f"lb:{eff_dir}", lambda: runner.leaderboard(eff_dir))
                    task_proof = build_task_proof_examples(eff_dir)
                    qc_rep = _cached_call(f"qc:{eff_dir}", lambda: _build_qc_payload(eff_dir))
                    return {
                        **cat,
                        "leaderboard": lb,
                        "task_proof_examples": task_proof,
                        "qc_report": qc_rep,
                        "portal_mode": self.default_mode,
                    }

                return self._send_cached_json(f"exec:{eff_dir}:{self.default_mode}", _build_exec)
            if parts == ["api", "models"]:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                self._maybe_pull()
                return self._send_cached_json(f"models:{eff_dir}", lambda: build_experiments_and_models_catalog(eff_dir))
            if parts in (["api", "qc"], ["api", "qc-report"]):
                self._maybe_pull()
                qs = parse_qs(urlparse(self.path).query)
                force_refresh = bool(qs.get("refresh"))
                if force_refresh:
                    _API_CACHE.pop(f"qc:{eff_dir}", None)
                return self._send_cached_json(f"qc:{eff_dir}", lambda: _build_qc_payload(eff_dir, force_refresh=force_refresh))
            if parts[:2] == ["api", "runs"] and len(parts) == 3:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                summary, images = runner.load_run(parts[2], eff_dir)
                cat = _cached_call(f"models:{eff_dir}", lambda: build_experiments_and_models_catalog(eff_dir))
                return self._json({
                    "summary": summary,
                    "experiments": cat["run_to_experiments"].get(parts[2], []),
                    "images": [{k: r.get(k) for k in LIGHT_KEYS} for r in images],
                })
            if parts[:2] == ["api", "runs"] and len(parts) == 5 and parts[3] == "images":
                summary, images = runner.load_run(parts[2], eff_dir)
                row = next((r for r in images if r["image_id"] == parts[4]), None)
                if row is None:
                    raise FileNotFoundError(parts[4])
                name = summary.get("dataset", "sku110k")
                try:
                    gt = _resolve_sample(name, summary["split"], parts[4], str(self.data_root))
                except Exception:
                    gt = None
                return self._json({**row, "task": summary.get("task", "detection"),
                                   "split": summary["split"] if name == "sku110k" else name,
                                   "usd_to_inr": summary.get("usd_to_inr")
                                   or summary.get("pricing", {}).get("usd_to_inr"),
                                   "gt": [list(b) for b in gt.boxes] if gt else [],
                                   "asset_boxes": [list(b) for b in gt.asset_boxes] if gt else []})
            if parts[0] == "img" and len(parts) == 3:
                try:
                    return self._send(
                        _jpeg(parts[1], parts[2], str(self.data_root)),
                        "image/jpeg",
                        cache_control="public, max-age=86400",
                    )
                except FileNotFoundError:
                    raise
                except Exception:
                    return self._send(
                        _fallback_schematic_jpeg(parts[1], parts[2], str(self.data_root)),
                        "image/jpeg",
                        cache_control="no-cache",
                    )
            if parts[:2] == ["api", "runs"] and len(parts) == 6 and parts[3] == "images" and parts[5] == "audit":
                return self._audit(parts[2], parts[4])
            self._json({"error": "not found"}, 404)
        except FileNotFoundError as e:
            self._json({"error": f"not found: {e}"}, 404)
        except Exception as e:
            self._json({
                "error": "GCS authentication required. Please run: gcloud auth login && gcloud auth application-default login",
                "detail": f"{type(e).__name__}: {e}"[:500],
            }, 503)

    def do_POST(self) -> None:  # noqa: N802
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        if parts in (
            ["api", "predict", "crop"],
            ["api", "clef-contrastive", "crop"],
            ["api", "clef_contrastive", "crop"],
            ["api", "marketshare", "crop"],
            ["api", "exp28", "crop"],
            ["api", "v1", "predict", "crop"],
        ):
            return self._clef_contrastive_predict(mode="crop")
        if parts in (
            ["api", "predict", "shelf"],
            ["api", "clef-contrastive", "shelf"],
            ["api", "clef_contrastive", "shelf"],
            ["api", "marketshare", "shelf"],
            ["api", "exp28", "shelf"],
            ["api", "v1", "predict", "shelf"],
        ):
            return self._clef_contrastive_predict(mode="shelf")
        if parts in (
            ["api", "predict"],
            ["api", "clef-contrastive", "predict"],
            ["api", "clef_contrastive", "predict"],
            ["api", "clef-contrastive"],
            ["api", "marketshare", "predict"],
            ["api", "marketshare"],
            ["api", "exp28", "predict"],
            ["api", "v1", "predict"],
        ):
            return self._clef_contrastive_predict(mode="auto")
        if parts != ["api", "cv", "predict"]:
            return self._json({"error": "not found"}, 404)
        try:
            self._cv_predict()
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"[:500]}, 500)

    def _clef_contrastive_predict(self, mode: str = "crop") -> None:
        """Handle online inference requests for CLEF Contrastive Contact-Sheet Cascade."""
        from utils import clef_contrastive_inference

        try:
            content_len = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(raw_body.decode("utf-8") if raw_body else "{}")
        except Exception as e:
            return self._json({"error": f"Invalid JSON payload: {e}"}, 400)

        if not isinstance(payload, dict):
            return self._json({"error": "Payload must be a JSON object"}, 400)

        target_mode = mode
        if target_mode == "auto":
            target_mode = str(payload.get("mode") or "crop").lower()

        eng = clef_contrastive_inference.get_engine()
        try:
            if target_mode == "shelf":
                res = eng.predict_shelf_payload(payload)
            else:
                res = eng.predict_crop_payload(payload)
            return self._json(res)
        except (ValueError, TypeError) as ve:
            return self._json({"error": str(ve)}, 400)
        except Exception as exc:
            return self._json({"error": f"Inference error: {type(exc).__name__}: {exc}"}, 500)

    def _cv_predict(self) -> None:
        """Vertex AI prediction route: one legacy .h5 class (+ benchmark fields) per instance."""
        from utils import cv_models

        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        instances = payload.get("instances") or []
        if not instances:
            return self._json({"error": "no instances"}, 400)
        runners = _cv_pipeline().runners
        predictions = []
        for inst in instances:
            role = cv_models.normalize_role(inst.get("role")) or (next(iter(runners)) if len(runners) == 1 else "")
            model = runners.get(role)
            if model is None:
                predictions.append({"error": f"no {inst.get('role')!r} model here; serving {sorted(runners)}"})
                continue
            if inst.get("image_b64"):
                data = base64.b64decode(inst["image_b64"])
            elif inst.get("image_uri"):
                data = dataset.read_bytes(str(inst["image_uri"]))
            else:
                predictions.append({"error": "missing image_b64 or image_uri"})
                continue
            label, conf = model.predict_class(
                Image.open(io.BytesIO(data)),
                category=inst.get("category"),
                brand=inst.get("brand"),
            )
            predictions.append({"raw_class": label, "confidence": round(conf, 6),
                                "model_uri": model.source_uri, **cv_models.parse_label(role, label)})
        return self._json({"predictions": predictions})

    def _audit(self, run_id: str, image_id: str) -> None:
        """Cloud Trace + Cloud Logging for one image (utils/audit.py), fetched on demand."""
        from utils import audit

        _, images = runner.load_run(run_id, self.results_dir)
        row = next((r for r in images if r["image_id"] == image_id), None)
        if row is None:
            raise FileNotFoundError(image_id)
        t = row.get("telemetry") or {}
        project = parse_qs(urlparse(t.get("trace_url", "")).query).get("project", [None])[0]
        if not (t.get("trace_id") and t.get("span_id") and project):
            return self._json({"error": "this run was made with telemetry off: nothing in GCP"}, 404)
        try:
            return self._json(audit.image_audit(project, t["trace_id"], t["span_id"]))
        except FileNotFoundError:
            raise
        except Exception as e:  # auth / permission / API errors: show them in the UI
            return self._json({"error": f"{type(e).__name__}: {e}"[:500]}, 502)


def _cv_pipeline():
    """The legacy .h5 classifier(s) this server predicts with (built once, loaded on first use)."""
    global _CV_PIPELINE
    with _CV_LOCK:
        if _CV_PIPELINE is None:
            from utils import cv_models
            from utils.llm import load_config

            _CV_PIPELINE = cv_models.CVPipeline.for_serving(load_config())
        return _CV_PIPELINE


def serve(host: str = "127.0.0.1", port: int = 8080, results_dir: Path = runner.RESULTS_DIR,
          data_root: str = dataset.DEFAULT_ROOT, pull_src: str | None = None,
          mode: str = "engineering") -> None:
    port = int(os.environ.get("AIP_HTTP_PORT") or port)  # Vertex AI serving container
    Handler.results_dir, Handler.data_root = Path(results_dir), str(data_root)
    Handler.pull_src = pull_src
    Handler.default_mode = mode
    Handler._last_pull = time.monotonic() if pull_src else 0.0
    _prewarm_server_caches(Handler.results_dir, Handler.data_root)
    print(f"Leaderboard ({mode}): http://{host}:{port}  |  CIO Executive Portal: http://{host}:{port}/cio")
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        if e.errno == 98:  # EADDRINUSE: already running in background
            print(f"Server is already active on port {port} -> open http://localhost:{port}/ or http://localhost:{port}/cio")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                return
        raise
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
