"""Leaderboard & EPIC Decision-First Control Plane UI server (stdlib only).

    GET /                                  Control Plane SPA (Leaderboard + Decision-First Reviewer)
    GET /api/leaderboard                   ranked runs (optional ?task=&epic=&attribute=)
    GET /api/approaches                    registered approaches, tasks, epics, and stages
    GET /api/stages                        registered modular pipeline stages
    GET /api/runs/<run_id>                 run summary + per-image metrics
    GET /api/runs/<run_id>/images/<image>  one image: predictions, ground truth, step trace
    GET /api/v1/audits                     EPIC Decision-First audit scenarios (MT + GT)
    POST /api/v1/audits/<id>/review        update review_status, notes, and shade overrides
    GET /img/<split>/<image>               downscaled JPEG from the dataset
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

import runner
from utils import dataset

WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"
STATIC = WEB_SRC if WEB_SRC.is_dir() else (Path(__file__).parent / "static")

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

# In-memory state store for the Decision-First EPIC Reviewer (supports MT and GT channels)
_AUDIT_STORE: list[dict[str, Any]] = [
    {
        "audit_id": "AUD-MT-2026-0929-01",
        "store_metadata": {
            "store_id": "HUL-MT-MUM-4021 (Reliance Smart Bazaar)",
            "channel_type": "MT",
            "endpoint_source": "Sales EDGE",
        },
        "media": {
            "raw_input_url": "/img/test/test_661.jpg",
            "processed_canvas_url": "/img/test/test_661.jpg",
        },
        "identification_detections": [
            {
                "box_id": "BOX-101",
                "sku_name": "Vaseline Intensive Care Deep Moisture",
                "confidence": 0.96,
                "status": "CONFIDENT",
                "coordinates": {"x": 80, "y": 140, "w": 150, "h": 310},
            },
            {
                "box_id": "BOX-102",
                "sku_name": "Dove Hair Therapy Daily Shine Shampoo",
                "confidence": 0.94,
                "status": "CONFIDENT",
                "coordinates": {"x": 255, "y": 145, "w": 145, "h": 305},
            },
            {
                "box_id": "BOX-103",
                "sku_name": "Lakme 9to5 CC Cream 02 Honey",
                "confidence": 0.78,
                "status": "AMBIGUOUS",
                "coordinates": {"x": 430, "y": 170, "w": 135, "h": 280},
            },
            {
                "box_id": "BOX-104",
                "sku_name": "Ponds Super Light Gel Oil Free Moisturizer",
                "confidence": 0.93,
                "status": "CONFIDENT",
                "coordinates": {"x": 590, "y": 160, "w": 155, "h": 290},
            },
            {
                "box_id": "BOX-105",
                "sku_name": "Unrecognized Competitor Body Lotion",
                "confidence": 0.44,
                "status": "UNRECOGNIZED",
                "coordinates": {"x": 775, "y": 150, "w": 140, "h": 300},
            },
        ],
        "compliance_scorecard": {
            "share_of_shelf_pct": {"target": 58.5, "detected": 54.2, "status": "FAILED"},
            "toker_compliance": {"expected_promos": 2, "detected_promos": 2, "status": "PASSED"},
            "red_line_alignment": {"status": "PASSED"},
        },
        "pipeline_trace": {
            "active_models": ["RT-DETR-v2", "gemini-embedding-2-preview (ScaNN)", "Gemini 3.8 Flash"],
            "latency_ms": 1460,
            "cost_saved_inr": 1.12,
        },
        "review_status": "PENDING",
        "review_notes": "",
    },
    {
        "audit_id": "AUD-GT-2026-0929-02",
        "store_metadata": {
            "store_id": "HUL-GT-DEL-1189 (Kirana Shikhar Outlet)",
            "channel_type": "GT",
            "endpoint_source": "Shikhar",
        },
        "media": {
            "raw_input_url": "/img/test/test_1956.jpg",
            "processed_canvas_url": "/img/test/test_1956.jpg",
        },
        "identification_detections": [
            {
                "box_id": "BOX-201",
                "sku_name": "Clinic Plus Strong & Long Health Shampoo",
                "confidence": 0.95,
                "status": "CONFIDENT",
                "coordinates": {"x": 110, "y": 180, "w": 140, "h": 290},
            },
            {
                "box_id": "BOX-202",
                "sku_name": "Sunsilk Stunning Black Shine Shampoo",
                "confidence": 0.92,
                "status": "CONFIDENT",
                "coordinates": {"x": 280, "y": 185, "w": 145, "h": 285},
            },
            {
                "box_id": "BOX-203",
                "sku_name": "Lakme 9to5 CC Cream 04 Almond",
                "confidence": 0.74,
                "status": "AMBIGUOUS",
                "coordinates": {"x": 460, "y": 210, "w": 130, "h": 260},
            },
            {
                "box_id": "BOX-204",
                "sku_name": "Surf Excel Matic Top Load Liquid",
                "confidence": 0.97,
                "status": "CONFIDENT",
                "coordinates": {"x": 620, "y": 175, "w": 165, "h": 310},
            },
        ],
        "compliance_scorecard": {
            "share_of_shelf_pct": {"target": 50.0, "detected": 61.4, "status": "PASSED"},
            "toker_compliance": {"expected_promos": 2, "detected_promos": 1, "status": "FAILED"},
            "red_line_alignment": {"status": "FAILED"},
        },
        "pipeline_trace": {
            "active_models": ["YoloN26", "gemini-embedding-2-preview (ScaNN)", "Gemini 3.5 Flash Lite"],
            "latency_ms": 920,
            "cost_saved_inr": 0.94,
        },
        "review_status": "PENDING",
        "review_notes": "",
    },
]


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
            if parts in (["api", "v1", "audits"], ["api", "audits"], ["api", "v1", "cx-storyboard"]):
                return self._json(_AUDIT_STORE)
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

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            payload = {}

        if len(parts) == 5 and parts[:3] == ["api", "v1", "audits"] and parts[4] == "review":
            audit_id = parts[3]
            target = next((a for a in _AUDIT_STORE if a["audit_id"] == audit_id), None)
            if not target:
                return self._json({"error": f"audit {audit_id!r} not found"}, 404)
            if payload.get("review_status") in ("PENDING", "APPROVED", "FLAGGED_FOR_AUDIT"):
                target["review_status"] = payload["review_status"]
            if "review_notes" in payload:
                target["review_notes"] = str(payload["review_notes"])
            if isinstance(payload.get("identification_detections"), list):
                target["identification_detections"] = payload["identification_detections"]
            return self._json(target)

        self._json({"error": "not found"}, 404)


def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    results_dir: Path = runner.RESULTS_DIR,
    data_root: str = dataset.DEFAULT_ROOT,
) -> None:
    Handler.results_dir, Handler.data_root = Path(results_dir), str(data_root)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Perfect Store Control Plane: http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
