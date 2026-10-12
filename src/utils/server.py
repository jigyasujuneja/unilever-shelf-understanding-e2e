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

import io
import json
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

import runner
from utils import dataset

STATIC = Path(__file__).parent / "static"
LIGHT_KEYS = ("image_id", "gt_count", "pred_count", "accuracy", "recall", "f2",
              "latency_s", "cost_inr", "error", "labels")
GALLERY_SETS = dataset.RPC_FORMAT_SETS  # RPC layout: photos with identified boxes + reference photos
PULL_TTL_S = 30.0
_PULL_LOCK = threading.Lock()


@lru_cache(maxsize=64)
def _jpeg(split: str, image_id: str, root: str) -> bytes:
    name = Path(image_id).name  # no path traversal
    path: str | None = None
    if split in ("products", *GALLERY_SETS):
        sample = _samples(split, None, root).get(name)
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

    def log_message(self, *args):  # quiet
        pass

    def _maybe_pull(self) -> None:
        if not self.pull_src or (time.monotonic() - Handler._last_pull) < self.pull_ttl_s:
            return
        with _PULL_LOCK:
            if (time.monotonic() - Handler._last_pull) < self.pull_ttl_s:
                return
            Handler._last_pull = time.monotonic()
            try:
                runner.pull(self.pull_src, self.results_dir)
            except Exception:
                pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self) -> None:  # noqa: N802
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
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
                elif parts[1].endswith(".png"):
                    ctype = "image/png"
                else:
                    ctype = "application/octet-stream"
                return self._send((STATIC / parts[1]).read_bytes(), ctype)
            if parts == ["api", "leaderboard"]:
                self._maybe_pull()
                return self._json(runner.leaderboard(self.results_dir))
            if parts == ["api", "experiments"]:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                self._maybe_pull()
                cat = build_experiments_and_models_catalog(self.results_dir)
                qc_path = Path(self.results_dir) / "qc_audit_report.json"
                if qc_path.exists():
                    try:
                        cat["qc_report"] = json.loads(qc_path.read_text())
                    except Exception:
                        pass
                return self._json(cat)
            if parts in (["api", "exec-overview"], ["api", "cio-overview"]):
                from utils.experiment_matrix import (
                    build_experiments_and_models_catalog,
                    build_task_proof_examples,
                )

                self._maybe_pull()
                cat = build_experiments_and_models_catalog(self.results_dir)
                lb = runner.leaderboard(self.results_dir)
                task_proof = build_task_proof_examples(self.results_dir)
                qc_path = Path(self.results_dir) / "qc_audit_report.json"
                qc_rep = None
                if qc_path.exists():
                    try:
                        qc_rep = json.loads(qc_path.read_text())
                    except Exception:
                        qc_rep = None
                return self._json({
                    **cat,
                    "leaderboard": lb,
                    "task_proof_examples": task_proof,
                    "qc_report": qc_rep,
                    "portal_mode": self.default_mode,
                })
            if parts == ["api", "models"]:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                self._maybe_pull()
                return self._json(build_experiments_and_models_catalog(self.results_dir))
            if parts in (["api", "qc"], ["api", "qc-report"]):
                from utils.qc import run_12_gate_qc

                self._maybe_pull()
                qc_path = Path(self.results_dir) / "qc_audit_report.json"
                qs = parse_qs(urlparse(self.path).query)
                if qc_path.exists() and not qs.get("refresh"):
                    try:
                        return self._json(json.loads(qc_path.read_text()))
                    except Exception:
                        pass
                rep = run_12_gate_qc(results_dir=self.results_dir)
                try:
                    qc_path.write_text(json.dumps(rep, indent=2))
                except Exception:
                    pass
                return self._json(rep)
            if parts[:2] == ["api", "runs"] and len(parts) == 3:
                from utils.experiment_matrix import build_experiments_and_models_catalog

                summary, images = runner.load_run(parts[2], self.results_dir)
                cat = build_experiments_and_models_catalog(self.results_dir)
                return self._json({
                    "summary": summary,
                    "experiments": cat["run_to_experiments"].get(parts[2], []),
                    "images": [{k: r.get(k) for k in LIGHT_KEYS} for r in images],
                })
            if parts[:2] == ["api", "runs"] and len(parts) == 5 and parts[3] == "images":
                summary, images = runner.load_run(parts[2], self.results_dir)
                row = next((r for r in images if r["image_id"] == parts[4]), None)
                if row is None:
                    raise FileNotFoundError(parts[4])
                name = summary.get("dataset", "sku110k")
                gt = _samples(name, summary["split"], str(self.data_root)).get(parts[4])
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
