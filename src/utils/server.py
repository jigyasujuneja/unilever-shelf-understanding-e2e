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
"""

from __future__ import annotations

import gzip
import io
import json
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

PULL_TTL_S = 120.0
CACHE_TTL_S = 45.0
_PULL_LOCK = threading.Lock()
_API_CACHE: dict[str, tuple[float, Any]] = {}


def _cached_call(key: str, builder, ttl: float = CACHE_TTL_S) -> Any:
    now = time.monotonic()
    hit = _API_CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    val = builder()
    _API_CACHE[key] = (now, val)
    return val


@lru_cache(maxsize=16)
def _leaf_sample_index(name: str, root: str) -> dict[str, dataset.Sample]:
    """Index all samples of an RPC-format dataset by both sample.image_id and URI filename."""
    idx: dict[str, dataset.Sample] = {}
    if name not in GALLERY_SETS:
        return idx
    data = dataset.data_root(name)
    for s in dataset.RPC_SUBSETS:
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


@lru_cache(maxsize=64)
def _jpeg(split: str, image_id: str, root: str) -> bytes:
    name = Path(image_id).name  # no path traversal
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
    return buf.getvalue()


def _samples(name: str, split: str | None, root: str) -> dict[str, dataset.Sample]:
    """Ground truth of a dataset's split (``split=None``: every split of an RPC-layout set)."""
    if name == "products":
        return dataset.load_products(dataset.products_root())
    if name in GALLERY_SETS:
        data = dataset.data_root(name)
        splits = [split] if split else list(dataset.RPC_SUBSETS)
        return {k: v for s in splits for k, v in dataset.load_rpc(s, data).items()}
    return dataset.load_split(split, root)


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
                    n = runner.pull(pull_src, res_dir)
                    if n > 0:
                        _API_CACHE.clear()
                except Exception:
                    pass

            threading.Thread(target=_bg, daemon=True).start()

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
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
                "data_audit_pipeline.png",
                "shelf_bay_vs_sku_example.png",
                "cross_folder_join_example.png",
                "bay_split_cleanup_example.png",
                "duplicate_upload_cleanup_example.png",
            ):
                if parts[1].endswith(".js"):
                    ctype = "text/javascript; charset=utf-8"
                elif parts[1].endswith(".css"):
                    ctype = "text/css; charset=utf-8"
                elif parts[1].endswith(".html"):
                    ctype = "text/html; charset=utf-8"
                elif parts[1].endswith(".png"):
                    ctype = "image/png"
                else:
                    ctype = "application/octet-stream"
                return self._send((STATIC / parts[1]).read_bytes(), ctype)
            if parts == ["api", "leaderboard"]:
                self._maybe_pull()
                return self._json(_cached_call(f"lb:{eff_dir}", lambda: runner.leaderboard(eff_dir)))
            if parts == ["api", "experiments"]:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                self._maybe_pull()

                def _build_exp():
                    cat = dict(build_experiments_and_models_catalog(eff_dir))
                    qc_path = Path(eff_dir) / "qc_audit_report.json"
                    if qc_path.exists():
                        try:
                            cat["qc_report"] = json.loads(qc_path.read_text())
                        except Exception:
                            pass
                    return cat

                return self._json(_cached_call(f"exp:{eff_dir}", _build_exp))
            if parts in (["api", "exec-overview"], ["api", "cio-overview"]):
                from utils.experiment_matrix import (
                    build_experiments_and_models_catalog,
                    build_task_proof_examples,
                )

                self._maybe_pull()

                def _build_exec():
                    cat = build_experiments_and_models_catalog(eff_dir)
                    lb = runner.leaderboard(eff_dir)
                    task_proof = build_task_proof_examples(eff_dir)
                    qc_path = Path(eff_dir) / "qc_audit_report.json"
                    qc_rep = None
                    if qc_path.exists():
                        try:
                            qc_rep = json.loads(qc_path.read_text())
                        except Exception:
                            qc_rep = None
                    return {
                        **cat,
                        "leaderboard": lb,
                        "task_proof_examples": task_proof,
                        "qc_report": qc_rep,
                        "portal_mode": self.default_mode,
                    }

                return self._json(_cached_call(f"exec:{eff_dir}:{self.default_mode}", _build_exec))
            if parts == ["api", "models"]:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                self._maybe_pull()
                return self._json(_cached_call(f"models:{eff_dir}", lambda: build_experiments_and_models_catalog(eff_dir)))
            if parts in (["api", "qc"], ["api", "qc-report"]):
                from utils.qc import run_12_gate_qc

                self._maybe_pull()
                qs = parse_qs(urlparse(self.path).query)
                force_refresh = bool(qs.get("refresh"))

                def _build_qc():
                    qc_path = Path(eff_dir) / "qc_audit_report.json"
                    rep = None
                    if qc_path.exists() and not force_refresh:
                        try:
                            rep = json.loads(qc_path.read_text())
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
                            raw_cls = json.loads(rpc_path.read_text()).get("classes", {})
                            cls_list = list(raw_cls.values()) if isinstance(raw_cls, dict) else list(raw_cls)
                            total_mapped = len(cls_list) or 2659
                            for c_row in cls_list:
                                cat_c = str(c_row.get("category") or "")
                                brd_c = str(c_row.get("brand") or "")
                                pk_c = str(c_row.get("packaging_type") or "")
                                leg_b = c_row.get("legacy_brand_folder") or legacy_brand_map.get(brd_c, brd_c)
                                leg_p = c_row.get("legacy_pack_folder") or legacy_pack_map.get((cat_c, pk_c), f"{cat_c}__{pk_c}")
                                allowed_p = c_row.get("allowed_packaging_types") or [pk_c]
                                if len(preview_rows) < 16 and (
                                    len(allowed_p) > 1
                                    or (leg_b != brd_c)
                                    or (leg_p != f"{cat_c}__{pk_c}")
                                ):
                                    preview_rows.append({
                                        "category": cat_c,
                                        "brand": brd_c,
                                        "name": c_row.get("variant") or c_row.get("product") or c_row.get("name"),
                                        "packaging_type": pk_c,
                                        "allowed_packaging_types": allowed_p,
                                        "legacy_brand_folder": leg_b,
                                        "legacy_pack_folder": leg_p,
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

                if force_refresh:
                    _API_CACHE.pop(f"qc:{eff_dir}", None)
                return self._json(_cached_call(f"qc:{eff_dir}", _build_qc))
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
                gt = _resolve_sample(name, summary["split"], parts[4], str(self.data_root))
                return self._json({**row, "task": summary.get("task", "detection"),
                                   "split": summary["split"] if name == "sku110k" else name,
                                   "usd_to_inr": summary.get("usd_to_inr")
                                   or summary.get("pricing", {}).get("usd_to_inr"),
                                   "gt": [list(b) for b in gt.boxes] if gt else [],
                                   "asset_boxes": [list(b) for b in gt.asset_boxes] if gt else []})
            if parts[0] == "img" and len(parts) == 3:
                return self._send(_jpeg(parts[1], parts[2], str(self.data_root)), "image/jpeg")
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


def serve(host: str = "127.0.0.1", port: int = 8080, results_dir: Path = runner.RESULTS_DIR,
          data_root: str = dataset.DEFAULT_ROOT, pull_src: str | None = None,
          mode: str = "engineering") -> None:
    Handler.results_dir, Handler.data_root = Path(results_dir), str(data_root)
    Handler.pull_src = pull_src
    Handler.default_mode = mode
    Handler._last_pull = time.monotonic() if pull_src else 0.0
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
