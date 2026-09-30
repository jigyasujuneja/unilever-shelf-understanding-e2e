"""Modular tests for `yolo_n26_sku110k` (`src/approaches/yolo_n26_sku110k.py`) on real shelf images."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RealShelfFixture, make_real_context
from PIL import Image

import approaches


def test_yolo_n26_sku110k_real_workflow_and_technique(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    weights_path = Path("data/models/yolo26n_sku110k.pt")
    assert weights_path.is_file(), "Fine-tuned SKU-110K YOLO26n checkpoint must exist at data/models/yolo26n_sku110k.pt"

    det = approaches.get("yolo_n26_sku110k")
    det.setup({})

    # Standalone neural detector must make 0 VLM calls even with TrapLLM
    ctx_a = make_real_context(model="none", trap_vlm=True)
    ctx_b = make_real_context(model="none", trap_vlm=True)

    boxes_a = det.detect(shelf_fixture_a.image, ctx_a)
    boxes_b = det.detect(shelf_fixture_b.image, ctx_b)

    assert len(boxes_a) > 0 and len(boxes_b) > 0
    assert boxes_a != boxes_b, "Detections must be dynamically computed from each real shelf image"
    assert ctx_a.trace.usage.calls == 0 and ctx_b.trace.usage.calls == 0
    assert len(ctx_a.trace.steps) == 2
    assert "YOLO26n" in ctx_a.trace.steps[0]["detail"]
    assert "DIoU-NMS" in ctx_a.trace.steps[1]["name"]


def test_yolo_n26_sku110k_hard_fail_on_invalid_inputs() -> None:
    det = approaches.get("yolo_n26_sku110k")
    ctx = make_real_context(model="none", trap_vlm=True)
    with pytest.raises(ValueError):
        det.detect(None, ctx)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        det.detect(Image.new("RGB", (0, 0)), ctx)
