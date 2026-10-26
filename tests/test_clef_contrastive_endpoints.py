"""Offline unit tests for CLEF Contrastive inference engine, HTTP REST endpoints, and CLI parsing.

All tests run 100% offline using in-memory synthetic PIL images and mocks.
Zero network access or GCS downloads are performed.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import time
import urllib.request
from typing import Any

import pytest
from PIL import Image

from utils import clef_contrastive_inference
from utils.clef_contrastive_inference import ClefContrastiveEngine
from utils.server import Handler, ThreadingHTTPServer


def _create_synthetic_jpeg_b64(width: int = 120, height: int = 240, color: tuple[int, int, int] = (200, 50, 50)) -> str:
    """Create a synthetic in-memory JPEG and return base64 string."""
    im = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _MockCropApproach:
    def __init__(self, sku_id: int = 42):
        self.sku_id = sku_id
        self.dataset = "hul_golden"
        self.catalog = {
            42: {
                "sku_id": 42,
                "is_hul": True,
                "brand": "Dove",
                "category": "Hair_Care",
                "packaging_type": "Bottle",
                "base_variant": "Dove__Daily_Shine",
                "product": "Dove Daily Shine Shampoo",
                "pack_size": "180 ml",
            },
            99: {
                "sku_id": 99,
                "is_hul": False,
                "brand": "Pantene",
                "category": "Hair_Care",
                "packaging_type": "Bottle",
                "base_variant": "Pantene__Pro_V",
                "product": "Pantene Pro V Shampoo",
                "pack_size": "200 ml",
            },
        }

    def setup(self, config: dict, ctx: Any) -> None:
        pass

    def identify(self, image: Image.Image, ctx: Any) -> int | None:
        ctx.trace.step(
            "Stage A CLM Fast Gate",
            "Accepted via CLM",
            info={
                "stage2a_bypassed": True,
                "clm_top1_prob": 0.942,
                "shortlist": [{"sku_id": self.sku_id, "cosine": 0.887}],
            },
        )
        return self.sku_id

    def identify_boxes(self, image: Image.Image, boxes: list[Any], ctx: Any) -> list[int | None]:
        out = []
        for idx in range(len(boxes)):
            s_id = 42 if idx % 2 == 0 else 99
            ctx.trace.step(
                f"Box {idx} decision",
                "Identified",
                info={
                    "box_index": idx,
                    "stage2a_bypassed": True,
                    "clm_top1_prob": 0.92,
                    "shortlist": [{"sku_id": s_id, "cosine": 0.85}],
                },
            )
            out.append(s_id)
        return out


class _MockShelfDetector:
    def __init__(self, boxes: list[tuple[float, float, float, float]] | None = None):
        self.boxes = boxes or [(10.0, 20.0, 110.0, 220.0), (120.0, 20.0, 220.0, 220.0)]
        self.dataset = "gtmerch_golden"

    def setup(self, config: dict, ctx: Any) -> None:
        pass

    def detect(self, image: Image.Image, ctx: Any) -> list[tuple[float, float, float, float]]:
        return self.boxes


@pytest.fixture
def mock_engine():
    crop_appr = _MockCropApproach()
    shelf_det = _MockShelfDetector()
    engine = ClefContrastiveEngine(
        crop_appr=crop_appr,
        shelf_det=shelf_det,
        llm=lambda img, p, **kw: None,
        catalog=crop_appr.catalog,
    )
    old_engine = clef_contrastive_inference._ENGINE_INSTANCE
    clef_contrastive_inference.set_engine(engine)
    yield engine
    clef_contrastive_inference.set_engine(old_engine)


def test_predict_crop_payload_single(mock_engine):
    b64 = _create_synthetic_jpeg_b64()
    payload = {
        "image_b64": b64,
        "model": "gemini-3.5-flash-lite",
        "tier": "priority",
    }
    res = mock_engine.predict_crop_payload(payload)
    assert res["pipeline"] == "clef_contrastive_contact_sheet"
    assert res["approach"] == "clef_contrastive_metadata"
    assert res["mode"] == "crop"
    assert res["sku_id"] == 42
    assert res["stage"] == "stage2a_fast_gate"
    assert res["confidence"] == 0.942
    assert res["prediction"]["brand"] == "Dove"
    assert res["prediction"]["is_hul"] is True
    assert len(res["predictions"]) == 1


def test_predict_crop_payload_batch(mock_engine):
    b64 = _create_synthetic_jpeg_b64()
    payload = {
        "images": [{"image_b64": b64, "id": "crop_1"}, {"image_b64": b64, "id": "crop_2"}],
        "model": "gemini-3.5-flash-lite",
    }
    res = mock_engine.predict_crop_payload(payload)
    assert res["pipeline"] == "clef_contrastive_contact_sheet"
    assert res["count"] == 2
    assert len(res["predictions"]) == 2
    assert res["predictions"][0]["sku_id"] == 42
    assert res["predictions"][1]["sku_id"] == 42


def test_predict_shelf_payload_single(mock_engine):
    b64 = _create_synthetic_jpeg_b64(width=800, height=600)
    payload = {
        "image_b64": b64,
        "model": "gemini-3.5-flash-lite",
    }
    res = mock_engine.predict_shelf_payload(payload)
    assert res["pipeline"] == "clef_contrastive_contact_sheet"
    assert res["approach"] == "clef_contrastive_metadata"
    assert res["mode"] == "shelf"
    assert res["total_facings"] == 2
    assert res["hul_facings"] == 1
    assert res["competitor_facings"] == 1
    assert res["hul_facing_sos"] == 0.5
    assert len(res["detections"]) == 2
    assert res["detections"][0]["prediction"]["brand"] == "Dove"
    assert res["detections"][1]["prediction"]["brand"] == "Pantene"


def test_http_rest_endpoints(mock_engine):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    base = f"http://127.0.0.1:{port}"
    try:
        # 1. Test GET /api/predict/health
        with urllib.request.urlopen(f"{base}/api/predict/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "healthy"
            assert data["pipeline"] == "clef_contrastive_contact_sheet"
            assert data["approach"] == "clef_contrastive_metadata"
            assert data["is_ready"] is True

        # 2. Test POST /api/predict/crop
        b64 = _create_synthetic_jpeg_b64()
        crop_req = urllib.request.Request(
            f"{base}/api/predict/crop",
            data=json.dumps({"image_b64": b64}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(crop_req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["sku_id"] == 42
            assert data["prediction"]["brand"] == "Dove"

        # 3. Test POST /api/predict/shelf
        shelf_req = urllib.request.Request(
            f"{base}/api/predict/shelf",
            data=json.dumps({"image_b64": b64}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(shelf_req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["total_facings"] == 2
            assert data["hul_facing_sos"] == 0.5
    finally:
        httpd.shutdown()


def test_cli_parsing():
    import cli

    # Test predict-crop argument parsing
    with pytest.raises(SystemExit) as exc:
        cli.main(["predict-crop", "--help"])
    assert exc.value.code == 0

    # Test predict-shelf argument parsing
    with pytest.raises(SystemExit) as exc:
        cli.main(["predict-shelf", "--help"])
    assert exc.value.code == 0

    # Test smoke-test-clef argument parsing
    with pytest.raises(SystemExit) as exc:
        cli.main(["smoke-test-clef", "--help"])
    assert exc.value.code == 0

    # Test deploy-clef-api argument parsing
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy-clef-api", "--help"])
    assert exc.value.code == 0
