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
GALLERY_SETS = ("rpc", "shelves")  # RPC layout: photos with identified boxes + reference photos


@lru_cache(maxsize=64)
def _jpeg(split: str, image_id: str, root: str) -> bytes:
    name = Path(image_id).name  # no path traversal
    if split in ("products", *GALLERY_SETS):
        sample = _samples(split, None, root).get(name)
        if sample is None:
            raise FileNotFoundError(image_id)
        path = sample.path
    elif split.endswith("-ref") and split[:-4] in GALLERY_SETS:  # a product's first reference photo
        gallery = dataset.rpc_gallery(dataset.data_root(split[:-4]))
        paths = gallery.get(int(name) if name.isdigit() else -1)
        if not paths:
            raise FileNotFoundError(image_id)
        path = paths[0]
    elif name.startswith(f"{split}_"):
        path = dataset.join(root, "images", name)
    else:
        raise FileNotFoundError(image_id)
    try:
        data = dataset.read_bytes(path)
    except Exception as e:  # missing locally or in GCS
        raise FileNotFoundError(image_id) from e
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

    def log_message(self, *args):  # quiet
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
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[0] == "static" and len(parts) == 2 and parts[1] in ("app.js", "styles.css"):
                ctype = ("text/javascript" if parts[1].endswith(".js") else "text/css") + "; charset=utf-8"
                return self._send((STATIC / parts[1]).read_bytes(), ctype)
            if parts == ["api", "leaderboard"]:
                return self._json(runner.leaderboard(self.results_dir))
            if parts[:2] == ["api", "runs"] and len(parts) == 3:
                summary, images = runner.load_run(parts[2], self.results_dir)
                return self._json({"summary": summary,
                                   "images": [{k: r.get(k) for k in LIGHT_KEYS} for r in images]})
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
                                   "gt": [list(b) for b in gt.boxes] if gt else []})
            if parts[0] == "img" and len(parts) == 3:
                return self._send(_jpeg(parts[1], parts[2], str(self.data_root)), "image/jpeg")
            if parts[:2] == ["api", "runs"] and len(parts) == 6 and parts[3] == "images" and parts[5] == "audit":
                return self._audit(parts[2], parts[4])
            self._json({"error": "not found"}, 404)
        except FileNotFoundError as e:
            self._json({"error": f"not found: {e}"}, 404)

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
          data_root: str = dataset.DEFAULT_ROOT) -> None:
    Handler.results_dir, Handler.data_root = Path(results_dir), str(data_root)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Leaderboard: http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
