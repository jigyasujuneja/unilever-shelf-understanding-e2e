"""Tests for ``src/utils/pricing.py``, ``src/utils/embeddings.py``, ``src/utils/alloydb.py``, and
custom approach SKU billing."""

from __future__ import annotations

import json
import threading
from datetime import date

import pytest
from conftest import NS, PER_M, REAL_PRICE_SHEET, OracleLLM, fake_sheet
from PIL import Image

import approaches
import runner
from utils import pricing
from utils.llm import usage_from_metadata


def _meta(traffic, image=1000, text=200, cached_image=0, out=300, think=100):
    return NS(
        traffic_type=NS(name=traffic),
        prompt_token_count=image + text,
        candidates_token_count=out,
        thoughts_token_count=think,
        prompt_tokens_details=[
            NS(modality="MediaModality.IMAGE", token_count=image),
            NS(modality="MediaModality.TEXT", token_count=text),
        ],
        cached_content_token_count=cached_image,
        cache_tokens_details=[NS(modality="MediaModality.IMAGE", token_count=cached_image)],
    )


def test_usage_buckets_follow_served_traffic_type_modality_and_cache():
    u = usage_from_metadata(_meta("ON_DEMAND_PRIORITY", cached_image=400))
    assert u.traffic == {"priority": 1}
    assert u.buckets == {
        "priority/image_input": 600,
        "priority/cached_image_input": 400,
        "priority/text_input": 200,
        "priority/output": 400,
    }
    down = usage_from_metadata(_meta("ON_DEMAND"))  # priority requested, downgraded
    assert down.traffic == {"standard": 1} and down.buckets["standard/image_input"] == 1000
    total = u + down
    assert total.calls == 2 and total.traffic == {"priority": 1, "standard": 1}


def test_gemini_cost_uses_the_sku_for_each_bucket_and_promo_credit():
    sheet = fake_sheet(["m"], [{"models": ["m"], "credit_share": 0.5, "until": "2026-12-31"}])
    buckets = {
        "standard/image_input": 1_000_000,
        "priority/output": 100_000,
        "standard/cached_text_input": 1_000_000,
    }
    c = pricing.gemini_cost("m", buckets, sheet, date(2026, 9, 24))
    assert c["list_usd"] == pytest.approx(0.3 + 0.1 * 2.5 * 1.8 + 0.03)
    assert c["credit_usd"] == pytest.approx(c["list_usd"] / 2)
    after = pricing.gemini_cost("m", buckets, sheet, date(2027, 1, 1))
    assert after["credit_usd"] == 0 and after["net_usd"] == after["list_usd"]
    assert pricing.gemini_cost("m", {"flex/output": 1_000_000}, sheet)["list_usd"] == pytest.approx(1.25)
    with pytest.raises(RuntimeError):
        pricing.gemini_cost("m", {"batch/output": 5}, sheet)


def test_price_sheet_parses_billing_catalog(monkeypatch):
    def sku(desc, units, nanos, unit="count", rate=1.0, tiers=None):
        return {
            "skuId": desc[:8],
            "description": desc,
            "pricingInfo": [
                {
                    "effectiveTime": "2026-09-24T07:00:00Z",
                    "currencyConversionRate": rate,
                    "pricingExpression": {
                        "usageUnit": unit,
                        "tieredRates": tiers
                        or [
                            {
                                "startUsageAmount": 0,
                                "unitPrice": {"units": str(units), "nanos": nanos},
                            }
                        ],
                    },
                }
            ],
        }

    gemini = [
        sku(pricing.sku_description("gemini-9-flash", "global", t, k), 0, n)
        for t, n in (("standard", 300), ("priority", 540))
        for k in pricing.KINDS
    ]
    catalog = {
        pricing.VERTEX_AI: gemini,
        pricing.CLOUD_RUN: [
            sku("Jobs CPU in us-central1", 0, 18_000, "s"),
            sku("Jobs Memory in us-central1", 0, 2_000, "GiBy.s"),
        ],
        pricing.CLOUD_STORAGE: [
            sku(
                "Regional Standard Class B Operations",
                0,
                0,
                tiers=[
                    {"startUsageAmount": 0, "unitPrice": {"nanos": 0}},
                    {"startUsageAmount": 50000, "unitPrice": {"nanos": 400}},
                ],
            )
        ],
    }
    inr = [sku("Jobs CPU in us-central1", 0, 1_719_810, "s", 95.545)]  # same skuId, INR
    monkeypatch.setattr(
        pricing,
        "fetch_skus",
        lambda svc, s, p, currency="USD": inr if currency == "INR" else catalog[svc],
    )
    cfg = {
        "gcp": {"project": "p", "location": "global", "region": "us-central1"},
        "promotions": [{"models": ["x"], "credit_share": 0.5, "until": date(2026, 12, 31)}],
    }
    # (YAML parses `until` as a date; the sheet must still be JSON-serialisable.)
    sheet = REAL_PRICE_SHEET(["gemini-9-flash"], cfg, session=object())
    assert sheet["gemini"]["gemini-9-flash"]["priority/output"]["usd"] == pytest.approx(5.4e-7)
    assert sheet["usd_to_inr"] == pytest.approx(95.545)
    assert sheet["cloud_run"]["vcpu_second"]["usd"] == 1.8e-5
    assert sheet["storage"]["class_b_op"]["usd"] == pytest.approx(4e-7)  # paid tier, not free
    json.dumps(sheet)  # stored in summary.json


def test_approach_setup_and_billed_units_are_priced_into_cost_per_image(
    fake_root, tmp_path, monkeypatch
):
    """An approach that creates its own clients in setup() and bills an embedding per box."""
    from approaches.base import Approach, to_pixels

    class FakeEmbeddings:
        def image(self, crop, ctx=None):
            if ctx:
                ctx.bill("embedding_image", 1)
            return [float(crop.width), 1.0]

    class EmbedEveryBox(Approach):
        name, architecture, steps = "embed_boxes", "test", ["detect", "embed"]
        skus = {"embedding_image": ("svc", "Embeddings for multimodal - Image (input)")}

        def setup(self, config, ctx):
            self.embed = FakeEmbeddings()

        def detect(self, image, ctx):
            boxes = to_pixels(ctx.ask(image, "detect").data, 0, 0, *image.size)
            widths = [int(self.embed.image(image.crop(b), ctx)[0]) for b in boxes]
            ctx.trace.step("Embed", ", ".join(map(str, widths)), boxes=boxes)
            return boxes

    monkeypatch.setitem(approaches.base.REGISTRY, "embed_boxes", EmbedEveryBox())
    s = runner.run(
        "embed_boxes",
        "gemini-t",
        "test",
        1,
        0,
        1,
        "t",
        results_dir=tmp_path,
        data_root=fake_root,
        llm=OracleLLM(),
        log=lambda *_: None,
    )
    # 4 boxes -> 4 image embeddings at $0.0001, on top of the Gemini tokens.
    assert s["cost"]["services_usd_per_image"] == pytest.approx(4e-4)
    assert s["cost_per_image_usd"] == pytest.approx((1000 * 0.3 + 250 * 2.5) * PER_M + 4e-4)
    _, images = runner.load_run(s["run_id"], tmp_path)
    assert images[0]["billed"] == {"embedding_image": 4}
    assert images[0]["steps"][1]["detail"] == "50, 50, 60, 90"


def test_unpriced_billed_unit_fails_loudly():
    with pytest.raises(RuntimeError, match="embedding_text_char"):
        pricing.extra_cost({"embedding_text_char": 10}, fake_sheet(["m"]))


def test_gemini_embedding_uses_embed_content_and_bills_reported_tokens(monkeypatch):
    from utils import embeddings

    posts = []

    class Session:
        def __init__(self, creds):
            pass

        def post(self, url, json, timeout):
            posts.append((url, json))
            modality = "IMAGE" if "inlineData" in json["content"]["parts"][0] else "TEXT"
            return NS(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {
                    "embedding": {"values": [0.6, 0.8]},
                    "usageMetadata": {
                        "promptTokensDetails": [{"modality": modality, "tokenCount": 258}]
                    },
                },
            )

    monkeypatch.setattr(embeddings.google.auth, "default", lambda scopes: (None, "p"))
    monkeypatch.setattr(embeddings, "AuthorizedSession", Session)
    bills = []
    ctx = NS(bill=lambda unit, n: bills.append((unit, n)))
    emb = embeddings.VertexEmbeddings({"gcp": {"project": "p"}}, model="gemini-embedding-2-preview")
    assert emb.image(Image.new("RGB", (8, 8)), ctx) == [0.6, 0.8] and emb.text("Knorr", ctx)
    assert posts[0][0].endswith("/models/gemini-embedding-2-preview:embedContent")
    assert posts[0][1]["outputDimensionality"] == 768
    assert bills == [("gemini_embedding_image_token", 258), ("gemini_embedding_text_token", 258)]
    assert set(u for u, _ in bills) <= set(embeddings.SKUS)
    with pytest.raises(ValueError, match="embedding model"):
        embeddings.VertexEmbeddings({"gcp": {"project": "p"}}, model="gemini-3.8-flash")


def test_alloydb_helper_runs_sql_on_a_per_thread_connection():
    """No AlloyDB here: a fake connector checks the connect args and the query round trip."""
    from utils.alloydb import AlloyDB, pgvector

    seen = {}

    class Cur:
        def execute(self, sql, params):
            seen["q"] = (sql, params)

        def fetchall(self):
            return [("sku-1", 0.93)]

    db = AlloyDB.__new__(AlloyDB)
    db.instance = "projects/p/.../instances/i"
    db.database, db.user, db.ip_type = "postgres", "u", "PRIVATE"
    db.connector = NS(
        connect=lambda *a, **k: seen.setdefault("connect", (a, k)) and NS(cursor=Cur)
    )
    db._local = threading.local()
    v = pgvector([0.1, 0.2])
    assert db.query(
        "SELECT id FROM products ORDER BY embedding <=> %s::vector LIMIT 1", (v,)
    ) == [("sku-1", 0.93)]
    assert seen["q"][1] == ("[0.1,0.2]",)
    assert seen["connect"][1]["enable_iam_auth"] is True


def test_sft_jsonl_builder_and_tuned_endpoint_resolution(rpc_root, tmp_path):
    from PIL import Image

    from utils import dataset, llm, tuning

    # Add a 2nd gallery view per product + 1 val image so build_sft_jsonl generates both splits
    for pid, colour in ((1, "red"), (2, "green"), (3, "blue")):
        Image.new("RGB", (40, 60), colour).save(rpc_root / "gallery" / f"{pid}_1.jpg")
    Image.new("RGB", (400, 300), "white").save(rpc_root / "images" / "val_0.jpg")
    meta = json.loads((rpc_root / dataset.RPC_JSON).read_text())
    for pid in ("1", "2", "3"):
        meta["gallery"][pid].append(f"gallery/{pid}_1.jpg")
    meta["splits"]["val"] = [
        {"image": "val_0.jpg", "width": 400, "height": 300, "boxes": [[10, 10, 60, 90]], "products": [1]}
    ]
    (rpc_root / dataset.RPC_JSON).write_text(json.dumps(meta))
    dataset._rpc_json.cache_clear()
    dataset.load_rpc.cache_clear()

    built = tuning.build_sft_jsonl("rpc", out_dir=tmp_path / "tuning", gcs_prefix="gs://b/tuning/rpc")
    assert built["train_count"] == 6 and built["val_count"] == 1
    first_train = json.loads(built["train_jsonl"].read_text().splitlines()[0])
    assert first_train["contents"][0]["parts"][0]["fileData"]["fileUri"].startswith("gs://b/tuning/rpc/sheets/")
    assert "choice" in json.loads(first_train["contents"][1]["parts"][0]["text"])

    cfg = {
        "gcp": {"project": "p", "location": "global", "region": "us-central1"},
        "tuned_models": {
            "gemini-2.5-flash-lite-sft": {
                "base_model": "gemini-2.5-flash-lite",
                "location": "us-central1",
                "endpoint": "projects/p/locations/us-central1/endpoints/12345",
            }
        },
    }
    target, base, loc = llm.resolve_model("gemini-2.5-flash-lite-sft", cfg)
    assert target == "projects/p/locations/us-central1/endpoints/12345"
    assert base == "gemini-2.5-flash-lite" and loc == "us-central1"

