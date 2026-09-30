"""Shared offline fixtures for ``tests/``.

Every test runs completely offline (no network, no GCP credentials):
- ``fake_root``: tiny SKU-110K dataset on disk (3 images per split, 4 boxes per image) for ``detection``.
- ``products_root``: tiny 4-product FMCG catalog + 3 product photos for ``classification``.
- ``rpc_root`` + ``colour_embeddings``: tiny RPC checkout photo + 3 studio reference photos for
  ``retrieval`` and ``end_to_end``.
- ``OracleLLM``: fake Gemini callable returning ground-truth boxes.
- ``offline_prices`` (autouse): stubs the Cloud Billing Catalog API with a deterministic price sheet.
- ``no_cloud_telemetry`` (autouse): disables Cloud Trace / Cloud Logging export unless a test opts in.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image

import runner
from utils import dataset, pricing, telemetry
from utils.llm import LLMResult, Usage

W, H = 400, 300
GT = [(10, 10, 60, 90), (70, 10, 120, 90), (200, 150, 260, 280), (300, 20, 390, 100)]
PER_M = 1e-6  # USD per token for $1 / 1M tokens
REAL_PRICE_SHEET = pricing.price_sheet  # tests stub this out via offline_prices

CATALOG = [
    (1, "Knorr", "Knorr Cubes 10g", "cooking"),
    (2, "Knorr", "Knorr Cubes 60g", "cooking"),
    (3, "Dove", "Dove Bar 135g", "personal_care"),
    (4, "Surf", "Surf Powder 65g", "laundry"),
]

COLOURS = {1: "red", 2: "green", 3: "blue"}
RPC_GT = [
    (10, 10, 60, 90, 1, "red"),
    (70, 10, 120, 90, 2, "green"),
    (200, 150, 260, 280, 3, "green"),  # product 3 painted green: embeddings pick product 2
]


def fake_sheet(models, promotions=()):
    table = {}
    for tier, mult in (("standard", 1.0), ("priority", 1.8)):
        for kind, usd in (
            ("text_input", 0.3),
            ("image_input", 0.3),
            ("cached_text_input", 0.03),
            ("cached_image_input", 0.03),
            ("output", 2.5),
        ):
            table[f"{tier}/{kind}"] = {"sku": f"{tier}-{kind}", "usd": usd * mult * PER_M}
    return {
        "usd_to_inr": 95.5,
        "fetched_at": "2026-09-24T00:00:00+00:00",
        "gemini": {m: table for m in models},
        "cloud_run": {"vcpu_second": {"usd": 1.8e-5}, "gib_second": {"usd": 2e-6}},
        "storage": {"class_a_op": {"usd": 5e-6}, "class_b_op": {"usd": 4e-7}},
        "extra": {"embedding_image": {"usd": 1e-4}},
        "promotions": list(promotions),
    }


@pytest.fixture(autouse=True)
def offline_prices(monkeypatch):
    monkeypatch.setattr(
        pricing,
        "price_sheet",
        lambda models, config, session=None, extra_skus=None: fake_sheet(models),
    )


@pytest.fixture(autouse=True)
def no_cloud_telemetry(monkeypatch):
    monkeypatch.setenv("SHELF_BENCH_TELEMETRY", "0")  # tests opt in via telemetry.use_exporter
    yield
    telemetry.reset()


def usage(inp=1000, out=250):
    return Usage(
        inp,
        out,
        0,
        1,
        {"standard": 1},
        {"standard/image_input": inp, "standard/output": out},
    )


def board_inr(results_dir):
    return runner.leaderboard(results_dir)[0]["cost_per_image_inr"]


class OracleLLM:
    """Returns the ground-truth boxes that fit inside the image it is shown."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0

    def __call__(self, image, prompt, **kw):
        with self.lock:
            self.calls += 1
        w, h = image.size
        boxes = [
            [int(y1 / h * 1000), int(x1 / w * 1000), int(y2 / h * 1000), int(x2 / w * 1000)]
            for x1, y1, x2, y2 in GT
            if x2 <= w and y2 <= h
        ]
        return LLMResult(boxes, usage(), 0.01)


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "SKU110K_fixed"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    for split in dataset.SPLITS:
        lines = []
        for i in range(3):
            name = f"{split}_{i}.jpg"
            Image.new("RGB", (W, H), "white").save(root / "images" / name)
            lines += [f"{name},{x1},{y1},{x2},{y2},object,{W},{H}" for x1, y1, x2, y2 in GT]
        (root / "annotations" / f"annotations_{split}.csv").write_text("\n".join(lines) + "\n")
    dataset.load_split.cache_clear()
    return root


@pytest.fixture
def products_root(tmp_path, monkeypatch) -> Path:
    """Photos p1..p3 of catalog products 1..3 (product 4 has no photo). Photo i is 60+i px wide,
    so fakes can tell them apart."""
    root = tmp_path / "products"
    (root / "images").mkdir(parents=True)
    cat = [
        {"id": i, "brand": b, "product_name": n, "category": c, "extracted_size": ""}
        for i, b, n, c in CATALOG
    ]
    for c in cat[:3]:
        Image.new("RGB", (60 + c["id"], 40), "white").save(root / "images" / f"p{c['id']}.jpg")
    (root / dataset.PRODUCTS_JSON).write_text(
        json.dumps({
            "full_labeled_fmcg_catalog": cat,
            "downloaded_unilever_and_competitor_samples": [
                {**c, "local_image_path": f"data/x/images/p{c['id']}.jpg"} for c in cat[:3]
            ],
        })
    )
    monkeypatch.setenv("SHELF_BENCH_PRODUCTS", str(root))
    dataset.load_products.cache_clear()
    dataset._products_json.cache_clear()
    yield root
    dataset.load_products.cache_clear()
    dataset._products_json.cache_clear()


@pytest.fixture
def rpc_root(tmp_path, monkeypatch) -> Path:
    """One checkout photo with 3 products, and one single-colour reference photo per product."""
    root = tmp_path / "rpc"
    (root / "gallery").mkdir(parents=True)
    (root / "images").mkdir()
    for pid, colour in COLOURS.items():
        Image.new("RGB", (40, 60), colour).save(root / "gallery" / f"{pid}_0.jpg")
    im = Image.new("RGB", (W, H), "white")
    for x1, y1, x2, y2, _, colour in RPC_GT:
        im.paste(colour, (x1, y1, x2, y2))
    im.save(root / "images" / "test_0.jpg")
    cats = {1: "food", 2: "drink", 3: "drink"}
    (root / dataset.RPC_JSON).write_text(
        json.dumps({
            "classes": {
                str(p): {"sku_id": p, "product": f"RPC #{p} ({c})", "category": c}
                for p, c in cats.items()
            },
            "gallery": {str(p): [f"gallery/{p}_0.jpg"] for p in COLOURS},
            "splits": {
                "test": [
                    {
                        "image": "test_0.jpg",
                        "width": W,
                        "height": H,
                        "boxes": [list(g[:4]) for g in RPC_GT],
                        "products": [g[4] for g in RPC_GT],
                    }
                ],
                "val": [],
            },
        })
    )
    monkeypatch.setenv("SHELF_BENCH_RPC", str(root))
    dataset._rpc_json.cache_clear()
    dataset.load_rpc.cache_clear()
    yield root
    dataset._rpc_json.cache_clear()
    dataset.load_rpc.cache_clear()


@pytest.fixture
def colour_embeddings(monkeypatch):
    """Fake multimodal embedding: the image's mean colour. Each call bills one image."""
    from PIL import ImageStat

    from approaches.market_share.retrieval import embedding_retrieval

    class FakeEmbeddings:
        def __init__(self, config, model):
            assert model == "multimodalembedding@001"

        def image(self, image, ctx=None):
            ctx.bill("embedding_image", 1)
            return ImageStat.Stat(image).mean

    monkeypatch.setattr(embedding_retrieval, "VertexEmbeddings", FakeEmbeddings)


__all__ = [
    "CATALOG",
    "COLOURS",
    "GT",
    "H",
    "NS",
    "OracleLLM",
    "PER_M",
    "REAL_PRICE_SHEET",
    "RPC_GT",
    "W",
    "board_inr",
    "fake_sheet",
    "usage",
]
