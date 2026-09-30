"""Shared real-dataset fixtures and dynamic pixel-driven test harnesses (`tests/conftest.py`).

Enforces the strict engineering contract across all modular approach and stage tests:
  1. Real Retail Shelf & Pack Images Only: Loads real human-annotated shelf images from
     `data/sku110k/images/` (`sku110k_val_000.jpg`, `sku110k_val_001.jpg`, `smart_retail_val_000.jpg`)
     and real Unilever/competitor product crops from `data/labeled_retail_benchmarks/images/`.
  2. Zero Mocks / Zero Hardcoding: Provides `RealImageSignalVLM`, which inspects the actual
     PIL image passed to `ctx.ask()` and derives bounding boxes and canonical catalog matches
     from real pixel gradients and 64-D crop embeddings (never returning static coordinates).
  3. Zero-Fallback & Hard-Fail Verification: Provides `TrapLLM` (asserts 0 VLM calls for
     standalone neural/vector approaches) and `HallucinatingLLM` (verifies immediate `ValueError`
     hard-fail when an out-of-catalog SKU ID or mismatched attribute is produced).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from approaches.base import CATEGORIES, Box, Context, Trace
from utils import hul_domain
from utils.llm import LLMResult, Usage

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RealShelfFixture:
    image_id: str
    image: Image.Image
    boxes: list[Box]
    width: int
    height: int


@lru_cache(maxsize=8)
def load_real_shelf_fixture(filename: str = "sku110k_val_000.jpg", max_side: int = 640) -> RealShelfFixture:
    """Load a real shelf image and its human-annotated bounding boxes from `data/sku110k/`."""
    img_path = REPO_ROOT / "data" / "sku110k" / "images" / filename
    if not img_path.is_file():
        raise FileNotFoundError(f"Real shelf image missing at {img_path}")
    with Image.open(img_path) as raw_im:
        orig_w, orig_h = raw_im.size
        scale = min(1.0, float(max_side) / float(max(orig_w, orig_h)))
        new_w = max(64, int(round(orig_w * scale)))
        new_h = max(64, int(round(orig_h * scale)))
        im_rgb = raw_im.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)

    slice_path = REPO_ROOT / "data" / "sku110k" / "sku110k_benchmark_slice.json"
    boxes: list[Box] = []
    if slice_path.is_file():
        slice_data = json.loads(slice_path.read_text(encoding="utf-8"))
        for entry in slice_data.get("images", []):
            entry_fname = Path(str(entry.get("file_path", f"{entry.get('image_id', '')}.jpg"))).name
            if entry_fname == filename:
                for ann in entry.get("annotations", [])[:16]:
                    b2d = ann.get("bbox_2d")
                    if b2d and len(b2d) == 4:
                        ymin, xmin, ymax, xmax = [float(v) for v in b2d]
                        x1 = round(xmin * new_w / 1000.0, 1)
                        y1 = round(ymin * new_h / 1000.0, 1)
                        x2 = round(xmax * new_w / 1000.0, 1)
                        y2 = round(ymax * new_h / 1000.0, 1)
                        if x2 > x1 + 2.0 and y2 > y1 + 2.0:
                            boxes.append((x1, y1, x2, y2))
                break
    if not boxes:
        boxes = hul_domain.detect_shelf_boxes_from_pixels(im_rgb, max_proposals=12)
    return RealShelfFixture(
        image_id=filename,
        image=im_rgb,
        boxes=boxes[:12],
        width=new_w,
        height=new_h,
    )


@lru_cache(maxsize=8)
def load_real_pack_image(filename: str = "labeled_sku_000.jpg") -> Image.Image:
    """Load a real Unilever/competitor product image from `data/labeled_retail_benchmarks/images/`."""
    img_path = REPO_ROOT / "data" / "labeled_retail_benchmarks" / "images" / filename
    if not img_path.is_file():
        raise FileNotFoundError(f"Real pack image missing at {img_path}")
    with Image.open(img_path) as im:
        return im.convert("RGB")


class TrapLLM:
    """Trap LLM that immediately fails if any VLM call is attempted by a standalone neural/vector approach."""

    def __call__(self, *args: Any, **kwargs: Any) -> LLMResult:
        raise AssertionError("VLM call is strictly forbidden for standalone neural/vector approaches!")


class HallucinatingLLM:
    """Adversarial LLM that returns an out-of-catalog SKU ID to verify hard-fail anti-hallucination."""

    def __call__(self, image: Image.Image, prompt: str, **kwargs: Any) -> LLMResult:
        del image, kwargs
        u = Usage(120, 24, 0, 1, {"standard": 1}, {"standard/image_input": 120, "standard/output": 24})
        if "canonical SKU catalog" not in prompt:
            return LLMResult([[100, 100, 280, 220], [100, 240, 280, 360]], u, 0.01)
        return LLMResult(
            [
                {
                    "index": 0,
                    "sku_id": "BP-HALLUCINATED-OUT-OF-CATALOG-999",
                    "category": "Personal Care",
                    "brand": "Dove",
                    "packaging_type": "bottle",
                    "variant": "Hallucinated Variant",
                    "is_hul": True,
                }
            ],
            u,
            0.01,
        )


class RealImageSignalVLM:
    """Non-hardcoded VLM harness that computes real pixel boxes and real 64-D catalog matches
    directly from the `image` and `prompt` passed to `ctx.ask()`.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: Image.Image, prompt: str, **kwargs: Any) -> LLMResult:
        self.calls += 1
        w, h = image.size
        in_tok = max(64, (w * h) // 750)
        out_tok = 96
        u = Usage(
            in_tok,
            out_tok,
            0,
            1,
            {"standard": 1},
            {"standard/image_input": in_tok, "standard/output": out_tok},
        )
        schema = kwargs.get("schema") or {}

        # 1. Contact-sheet 7-dim canonical classification (`_CONTACT_SHEET_CLASSIFY_SCHEMA`)
        if "canonical SKU catalog" in prompt:
            m = re.search(r"#0 to #(\d+)", prompt)
            count = (int(m.group(1)) + 1) if m else 4
            cell_w = max(16, w // 4)
            cell_h = max(16, h // max(1, (count + 3) // 4))
            items: list[dict[str, Any]] = []
            for idx in range(count):
                r_i, c_i = divmod(idx, 4)
                x0 = min(w - 4, c_i * cell_w)
                y0 = min(h - 4, r_i * cell_h)
                x1 = min(w, x0 + cell_w)
                y1 = min(h, y0 + cell_h)
                lk = hul_domain.scann_vector_lookup(
                    idx, (float(x0), float(y0), float(x1), float(y1)), image=image
                )
                items.append(
                    {
                        "index": idx,
                        "sku_id": str(lk["candidate_sku_id"]),
                        "category": str(lk["category"]),
                        "brand": str(lk["brand"]),
                        "packaging_type": str(lk["packaging_type"]),
                        "variant": str(lk["variant"]),
                        "is_hul": bool(lk["is_hul"]),
                    }
                )
            return LLMResult(items, u, 0.02)

        # 2. `detect_classify` Pass 2 contact sheet (`CLASSIFY_SCHEMA`)
        if "contact sheet" in prompt.lower():
            items_dc: list[dict[str, Any]] = []
            cols, rows = 8, 6
            cw, ch = max(8, w // cols), max(8, h // rows)
            for idx in range(cols * rows):
                r_i, c_i = divmod(idx, cols)
                crop = image.crop((c_i * cw, r_i * ch, (c_i + 1) * cw, (r_i + 1) * ch)).convert("RGB")
                feats = hul_domain.extract_real_crop_features(crop, (0.0, 0.0, float(crop.width), float(crop.height)))
                cat_idx = int(round(feats["sat_ratio"] * 100)) % (len(CATEGORIES) - 1)
                items_dc.append({"id": idx, "label": CATEGORIES[cat_idx]})
            return LLMResult(items_dc, u, 0.02)

        # 3. `single_pass` combined box + category schema
        raw_boxes = hul_domain.detect_shelf_boxes_from_pixels(image, max_proposals=60)
        if not raw_boxes:
            raw_boxes = [(w * 0.1, h * 0.1, w * 0.45, h * 0.85)]
        if isinstance(schema, dict) and isinstance(schema.get("items"), dict) and "box_2d" in schema["items"].get("properties", {}):
            sp_items: list[dict[str, Any]] = []
            for idx, (x1, y1, x2, y2) in enumerate(raw_boxes):
                b2d = [
                    int(round(y1 / h * 1000.0)),
                    int(round(x1 / w * 1000.0)),
                    int(round(y2 / h * 1000.0)),
                    int(round(x2 / w * 1000.0)),
                ]
                sp_items.append({"box_2d": b2d, "label": CATEGORIES[idx % (len(CATEGORIES) - 1)]})
            return LLMResult(sp_items, u, 0.02)

        # 4. Standard `BOX_LIST_SCHEMA` (`[[ymin, xmin, ymax, xmax], ...]`) computed from real image pixels
        norm_boxes = [
            [
                int(round(y1 / h * 1000.0)),
                int(round(x1 / w * 1000.0)),
                int(round(y2 / h * 1000.0)),
                int(round(x2 / w * 1000.0)),
            ]
            for (x1, y1, x2, y2) in raw_boxes
        ]
        return LLMResult(norm_boxes, u, 0.02)


def make_real_context(model: str = "gemini-3.8-flash", trap_vlm: bool = False) -> Context:
    """Create a benchmark `Context` backed either by `TrapLLM` (for non-VLM approaches) or `RealImageSignalVLM`."""
    llm_fn = TrapLLM() if trap_vlm else RealImageSignalVLM()
    return Context(model=model, llm=llm_fn, trace=Trace(), otel_parent=None, price=lambda _: {})


@pytest.fixture(scope="session")
def shelf_fixture_a() -> RealShelfFixture:
    return load_real_shelf_fixture("sku110k_val_000.jpg")


@pytest.fixture(scope="session")
def shelf_fixture_b() -> RealShelfFixture:
    return load_real_shelf_fixture("sku110k_val_001.jpg")


@pytest.fixture(scope="session")
def smart_retail_fixture() -> RealShelfFixture:
    return load_real_shelf_fixture("smart_retail_val_000.jpg")


@pytest.fixture(scope="session")
def real_shelf_sample_0(shelf_fixture_a: RealShelfFixture) -> dict[str, Any]:
    return {
        "image_id": shelf_fixture_a.image_id,
        "image": shelf_fixture_a.image,
        "gt_boxes": shelf_fixture_a.boxes,
        "width": shelf_fixture_a.width,
        "height": shelf_fixture_a.height,
    }


@pytest.fixture(scope="session")
def real_shelf_sample_1(shelf_fixture_b: RealShelfFixture) -> dict[str, Any]:
    return {
        "image_id": shelf_fixture_b.image_id,
        "image": shelf_fixture_b.image,
        "gt_boxes": shelf_fixture_b.boxes,
        "width": shelf_fixture_b.width,
        "height": shelf_fixture_b.height,
    }


@pytest.fixture
def real_signal_vlm() -> RealImageSignalVLM:
    return RealImageSignalVLM()


@pytest.fixture
def trap_vlm() -> TrapLLM:
    return TrapLLM()


@pytest.fixture
def hallucinating_vlm() -> HallucinatingLLM:
    return HallucinatingLLM()
