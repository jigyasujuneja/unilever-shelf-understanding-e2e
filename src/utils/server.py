"""Leaderboard UI server (stdlib only).

    GET /                                  leaderboard page
    GET /api/leaderboard                   ranked runs (optional ?task=&epic=&attribute=)
    GET /api/approaches                    registered approaches, tasks, epics, and stages
    GET /api/stages                        registered modular pipeline stages
    GET /api/runs/<run_id>                 run summary + per-image metrics
    GET /api/runs/<run_id>/images/<image>  one image: predictions, ground truth, step trace
    GET /img/<split>/<image>               downscaled JPEG from the dataset
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
LIGHT_KEYS = (
    "image_id",
    "gt_count",
    "pred_count",
    "accuracy",
    "recall",
    "f2",
    "latency_s",
    "cost_inr",
    "error",
)


@lru_cache(maxsize=64)
def _jpeg(split: str, image_id: str, root: str) -> bytes:
    name = Path(image_id).name  # no path traversal
    valid_prefixes = (f"{split}_", "sku110k_", "smart_retail_", "rpc_", "labeled_sku_")
    if not name.startswith(valid_prefixes):
        raise FileNotFoundError(image_id)
    data = None
    candidates = [
        dataset.join(root, "images", name),
        str(Path("data/SKU110K_fixed/images") / name),
        str(Path("data/SKU110K_fixed/images") / f"sku110k_{name}"),
    ]
    for cand in candidates:
        try:
            data = dataset.read_bytes(cand)
            if data:
                break
        except Exception:
            continue
    if not data:
        raise FileNotFoundError(image_id)
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((1400, 1400))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _build_registry_metadata(results_dir: Path) -> dict:
    """Return dynamic metadata for registered tasks, epics, approaches, stages, and runs."""
    import approaches
    import stages
    from approaches.base import EPICS, TASKS

    board = runner.leaderboard(results_dir)
    reg = approaches.all_approaches()
    discovered_epics: list[str] = [str(e) for e in EPICS]
    for ap in reg.values():
        if ap.epic not in discovered_epics:
            discovered_epics.append(str(ap.epic))
    for r in board:
        if r.get("epic") and str(r["epic"]) not in discovered_epics:
            discovered_epics.append(str(r["epic"]))
    approaches_meta = [
        {
            "name": ap.name,
            "task": ap.task,
            "epic": ap.epic,
            "target_field": getattr(ap, "target_field", "variant"),
            "architecture": ap.architecture,
            "steps": ap.steps,
        }
        for ap in reg.values()
    ]
    return {
        "tasks": list(TASKS),
        "epics": discovered_epics,
        "approaches": approaches_meta,
        "stages": stages.all_stages(),
        "leaderboard_runs": board,
    }


class Handler(BaseHTTPRequestHandler):
    results_dir = runner.RESULTS_DIR
    data_root = dataset.DEFAULT_ROOT

    def log_message(self, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
        qs = parse_qs(parsed.query)
        try:
            if not parts:
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[0] == "static" and len(parts) == 2 and parts[1] in ("app.js", "styles.css"):
                ctype = ("text/javascript" if parts[1].endswith(".js") else "text/css") + "; charset=utf-8"
                return self._send((STATIC / parts[1]).read_bytes(), ctype)
            if parts == ["api", "leaderboard"]:
                q_task = qs.get("task", [None])[0]
                q_epic = qs.get("epic", [None])[0]
                q_attr = qs.get("attribute", [None])[0]
                return self._json(
                    runner.leaderboard(self.results_dir, task=q_task, epic=q_epic, attribute=q_attr)
                )
            if parts in (["api", "approaches"], ["api", "v1", "approaches"], ["api", "v1", "eng-workbench"]):
                return self._json(_build_registry_metadata(self.results_dir))
            if parts in (["api", "stages"], ["api", "v1", "stages"]):
                import stages

                return self._json(stages.all_stages())
            if parts[:2] == ["api", "runs"] and len(parts) == 3:
                summary, images = runner.load_run(parts[2], self.results_dir)
                return self._json({
                    "summary": summary,
                    "images": [{k: r.get(k) for k in LIGHT_KEYS} for r in images],
                })
            if parts[:2] == ["api", "runs"] and len(parts) == 5 and parts[3] == "images":
                summary, images = runner.load_run(parts[2], self.results_dir)
                row = next((r for r in images if r["image_id"] == parts[4]), None)
                if row is None:
                    raise FileNotFoundError(parts[4])
                gt = dataset.load_split(summary["split"], str(self.data_root)).get(parts[4])
                return self._json({
                    **row,
                    "split": summary["split"],
                    "gt": [list(b) for b in gt.boxes] if gt else [],
                })
            if parts[0] == "img" and len(parts) == 3:
                return self._send(_jpeg(parts[1], parts[2], str(self.data_root)), "image/jpeg")
            self._json({"error": "not found"}, 404)
        except FileNotFoundError as e:
            self._json({"error": f"not found: {e}"}, 404)


def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    results_dir: Path = runner.RESULTS_DIR,
    data_root: str = dataset.DEFAULT_ROOT,
) -> None:
    Handler.results_dir, Handler.data_root = Path(results_dir), str(data_root)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Leaderboard: http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
