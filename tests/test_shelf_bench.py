"""Offline tests: tiny fake SKU-110K on disk + a fake LLM + a fake price sheet. No network."""

from __future__ import annotations

import json
import threading
import urllib.request
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image

import approaches
import runner
from utils import dataset, metrics, pricing, telemetry
from utils.llm import LLMResult, Usage, parse_json, usage_from_metadata

W, H = 400, 300
GT = [(10, 10, 60, 90), (70, 10, 120, 90), (200, 150, 260, 280), (300, 20, 390, 100)]
PER_M = 1e-6  # USD per token for $1 / 1M tokens
REAL_PRICE_SHEET = pricing.price_sheet  # tests below stub it out


def fake_sheet(models, promotions=()):
    table = {}
    for tier, mult in (("standard", 1.0), ("priority", 1.8)):
        for kind, usd in (("text_input", 0.3), ("image_input", 0.3), ("cached_text_input", 0.03),
                          ("cached_image_input", 0.03), ("output", 2.5)):
            table[f"{tier}/{kind}"] = {"sku": f"{tier}-{kind}", "usd": usd * mult * PER_M}
    return {"usd_to_inr": 95.5, "fetched_at": "2026-09-24T00:00:00+00:00",
            "gemini": {m: table for m in models},
            "cloud_run": {"vcpu_second": {"usd": 1.8e-5}, "gib_second": {"usd": 2e-6}},
            "storage": {"class_a_op": {"usd": 5e-6}, "class_b_op": {"usd": 4e-7}},
            "extra": {"embedding_image": {"usd": 1e-4}},
            "promotions": list(promotions)}


@pytest.fixture(autouse=True)
def offline_prices(monkeypatch):
    monkeypatch.setattr(pricing, "price_sheet",
                        lambda models, config, session=None, extra_skus=None: fake_sheet(models))


@pytest.fixture(autouse=True)
def no_cloud_telemetry(monkeypatch):
    monkeypatch.setenv("SHELF_BENCH_TELEMETRY", "0")  # tests opt in via telemetry.use_exporter
    yield
    telemetry.reset()


def usage(inp=1000, out=250):
    return Usage(inp, out, 0, 1, {"standard": 1}, {"standard/image_input": inp, "standard/output": out})


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
            for x1, y1, x2, y2 in GT if x2 <= w and y2 <= h
        ]
        return LLMResult(boxes, usage(), 0.01)


def test_load_split_and_sampling(fake_root):
    s = dataset.load_split("test", fake_root)
    assert len(s) == 3 and len(s["test_0.jpg"].boxes) == 4
    a = dataset.sample_images("test", 2, 0, fake_root)
    b = dataset.sample_images("test", 2, 0, fake_root)
    assert [x.image_id for x in a] == [x.image_id for x in b]
    assert len(dataset.sample_images("test", 0, 0, fake_root)) == 3


def test_metrics():
    m = metrics.match(GT[:3] + [(0, 200, 5, 205)], GT)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 1)
    s = metrics.scores(3, 1, 1)
    assert s["precision"] == 0.75 and s["recall"] == 0.75 and s["accuracy"] == 0.6
    assert s["f2"] == pytest.approx(0.75)
    assert metrics.percentile([1, 2, 3, 4], 95) == 4
    assert metrics.percentile(list(range(1, 101)), 95) == 95


def test_each_gt_matched_once():
    m = metrics.match([GT[0], GT[0]], [GT[0]])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_parse_json_recovers_truncated_list():
    assert parse_json('[[1,2,3,4],[5,6,7,8],[9,10') == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert parse_json('```json\n[[1,2,3,4]]\n```') == [[1, 2, 3, 4]]


def _meta(traffic, image=1000, text=200, cached_image=0, out=300, think=100):
    return NS(traffic_type=NS(name=traffic), prompt_token_count=image + text,
              candidates_token_count=out, thoughts_token_count=think,
              prompt_tokens_details=[NS(modality="MediaModality.IMAGE", token_count=image),
                                     NS(modality="MediaModality.TEXT", token_count=text)],
              cached_content_token_count=cached_image,
              cache_tokens_details=[NS(modality="MediaModality.IMAGE", token_count=cached_image)])


def test_usage_buckets_follow_served_traffic_type_modality_and_cache():
    u = usage_from_metadata(_meta("ON_DEMAND_PRIORITY", cached_image=400))
    assert u.traffic == {"priority": 1}
    assert u.buckets == {"priority/image_input": 600, "priority/cached_image_input": 400,
                         "priority/text_input": 200, "priority/output": 400}
    down = usage_from_metadata(_meta("ON_DEMAND"))  # priority requested, downgraded
    assert down.traffic == {"standard": 1} and down.buckets["standard/image_input"] == 1000
    total = u + down
    assert total.calls == 2 and total.traffic == {"priority": 1, "standard": 1}


def test_gemini_cost_uses_the_sku_for_each_bucket_and_promo_credit():
    sheet = fake_sheet(["m"], [{"models": ["m"], "credit_share": 0.5, "until": "2026-12-31"}])
    buckets = {"standard/image_input": 1_000_000, "priority/output": 100_000,
               "standard/cached_text_input": 1_000_000}
    c = pricing.gemini_cost("m", buckets, sheet, date(2026, 9, 24))
    assert c["list_usd"] == pytest.approx(0.3 + 0.1 * 2.5 * 1.8 + 0.03)
    assert c["credit_usd"] == pytest.approx(c["list_usd"] / 2)
    after = pricing.gemini_cost("m", buckets, sheet, date(2027, 1, 1))
    assert after["credit_usd"] == 0 and after["net_usd"] == after["list_usd"]
    with pytest.raises(RuntimeError):
        pricing.gemini_cost("m", {"flex/output": 5}, sheet)


def test_price_sheet_parses_billing_catalog(monkeypatch):
    def sku(desc, units, nanos, unit="count", rate=1.0, tiers=None):
        return {"skuId": desc[:8], "description": desc, "pricingInfo": [{
            "effectiveTime": "2026-09-24T07:00:00Z", "currencyConversionRate": rate,
            "pricingExpression": {"usageUnit": unit, "tieredRates": tiers or [
                {"startUsageAmount": 0, "unitPrice": {"units": str(units), "nanos": nanos}}]}}]}

    gemini = [sku(pricing.sku_description("gemini-9-flash", "global", t, k), 0, n)
              for t, n in (("standard", 300), ("priority", 540)) for k in pricing.KINDS]
    catalog = {
        pricing.VERTEX_AI: gemini,
        pricing.CLOUD_RUN: [sku("Jobs CPU in us-central1", 0, 18_000, "s"),
                            sku("Jobs Memory in us-central1", 0, 2_000, "GiBy.s")],
        pricing.CLOUD_STORAGE: [sku("Regional Standard Class B Operations", 0, 0, tiers=[
            {"startUsageAmount": 0, "unitPrice": {"nanos": 0}},
            {"startUsageAmount": 50000, "unitPrice": {"nanos": 400}}])],
    }
    inr = [sku("Jobs CPU in us-central1", 0, 1_719_810, "s", 95.545)]  # same skuId, INR
    monkeypatch.setattr(pricing, "fetch_skus", lambda svc, s, p, currency="USD":
                        inr if currency == "INR" else catalog[svc])
    cfg = {"gcp": {"project": "p", "location": "global", "region": "us-central1"},
           "promotions": [{"models": ["x"], "credit_share": 0.5, "until": date(2026, 12, 31)}]}
    # (YAML parses `until` as a date; the sheet must still be JSON-serialisable.)
    sheet = REAL_PRICE_SHEET(["gemini-9-flash"], cfg, session=object())
    assert sheet["gemini"]["gemini-9-flash"]["priority/output"]["usd"] == pytest.approx(5.4e-7)
    assert sheet["usd_to_inr"] == pytest.approx(95.545)
    assert sheet["cloud_run"]["vcpu_second"]["usd"] == 1.8e-5
    assert sheet["storage"]["class_b_op"]["usd"] == pytest.approx(4e-7)  # paid tier, not free
    json.dumps(sheet)  # stored in summary.json


def test_single_pass_parses_labelled_boxes():
    from approaches.single_pass import labelled_boxes

    raw = [{"box_2d": [0, 0, 500, 500], "label": "food"},
           {"box_2d": [500, 500, 1000, 1000], "label": "not_a_product"},
           [0, 500, 500, 1000]]  # truncated output falls back to bare boxes
    boxes, labels = labelled_boxes(raw, 100, 100)
    assert boxes == [(0, 0, 50, 50), (50, 50, 100, 100), (50, 0, 100, 50)]
    assert labels == ["food", "not_a_product", "other_product"]


def test_registry_has_builtin_approaches():
    names = set(approaches.all_approaches())
    assert {"single_pass", "detect_classify"} <= names
    for ap in approaches.all_approaches().values():
        assert ap.architecture and ap.steps


def test_run_single_pass_end_to_end(fake_root, tmp_path):
    out = tmp_path / "results"
    s = runner.run("single_pass", "fake-model", "test", 0, 0, 2, "tester",
                   results_dir=out, data_root=fake_root, llm=OracleLLM(), log=lambda *_: None)
    assert s["images"] == 3 and s["errors"] == 0
    assert s["recall"] == pytest.approx(1.0, abs=0.01) and s["f2"] > 0.99
    assert s["owner"] == "tester" and s["cost_per_image_usd"] > 0
    assert board_inr(out) == pytest.approx(s["cost_per_image_usd"] * 95.5, abs=1e-3)
    # 1000 image-input + 250 output tokens per image at the fake sheet's prices; local runs
    # read the local dataset, so no compute or storage cost.
    assert s["cost_per_image_usd"] == pytest.approx((1000 * 0.3 + 250 * 2.5) * PER_M)
    assert s["cost"]["compute_source"].startswith("local") and s["traffic"] == {"standard": 3}
    cfg = {"defaults": {"split": "test", "limit": 0, "seed": 0}}
    board = runner.leaderboard(out, cfg)
    assert board[0]["rank"] == "dev" and board[0]["run_id"] == s["run_id"]  # local: unranked
    # A Cloud Run run on the configured image set is ranked, above dev runs even with lower F2;
    # one on a different image count is not.
    official = json.loads((out / s["run_id"] / "summary.json").read_text())
    for run_id, limit in (("cloud", 0), ("cloud-25", 25)):
        (out / run_id).mkdir()
        (out / run_id / "summary.json").write_text(json.dumps(
            {**official, "run_id": run_id, "f2": 0.1, "limit": limit,
             "environment": {"platform": "cloud-run"}}))
    assert [(r["run_id"], r["rank"]) for r in runner.leaderboard(out, cfg)] == [
        ("cloud", 1), (s["run_id"], "dev"), ("cloud-25", "dev")]
    summary, images = runner.load_run(s["run_id"], out)
    assert [st["name"] for st in images[0]["steps"]][-1] == "Score vs ground truth"


def test_detect_classify_drops_non_products(fake_root, tmp_path):
    junk = [int(260 / H * 1000), int(0 / W * 1000), int(299 / H * 1000), int(40 / W * 1000)]

    class TwoPass(OracleLLM):
        def __call__(self, image, prompt, **kw):
            with self.lock:
                self.calls += 1
            if "contact sheet" in prompt:  # pass 2: box 4 (the junk box) is not a product
                return LLMResult([{"id": i, "label": "not_a_product" if i == 4 else "food"}
                                  for i in range(5)], usage(500, 50), 0.01)
            boxes = [[int(y1 / H * 1000), int(x1 / W * 1000), int(y2 / H * 1000),
                      int(x2 / W * 1000)] for x1, y1, x2, y2 in GT]
            return LLMResult(boxes + [junk], usage(), 0.01)

    llm = TwoPass()
    s = runner.run("detect_classify", "fake-model", "val", 1, 0, 1, "t",
                   results_dir=tmp_path, data_root=fake_root, llm=llm, log=lambda *_: None)
    assert llm.calls == 2 and s["fp"] == 0 and s["recall"] == 1.0
    _, images = runner.load_run(s["run_id"], tmp_path)
    names = [st["name"] for st in images[0]["steps"]]
    assert names == ["Load image", "Pass 1: detection", "Pass 2: classification",
                     "Score vs ground truth"]
    assert len(images[0]["steps"][1]["boxes"]) == 5 and len(images[0]["steps"][2]["boxes"]) == 4


def test_failed_image_counts_as_zero_detections(fake_root, tmp_path):
    def broken(*a, **k):
        raise RuntimeError("boom")

    s = runner.run("single_pass", "m", "test", 0, 0, 1, "t", results_dir=tmp_path,
                   data_root=fake_root, llm=broken, log=lambda *_: None)
    assert s["errors"] == 3 and s["recall"] == 0.0


def test_server_endpoints(fake_root, tmp_path):
    from http.server import ThreadingHTTPServer

    from utils.server import Handler

    s = runner.run("single_pass", "m", "test", 1, 0, 1, "t", results_dir=tmp_path,
                   data_root=fake_root, llm=OracleLLM(), log=lambda *_: None)
    Handler.results_dir, Handler.data_root = tmp_path, fake_root
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        get = lambda p: urllib.request.urlopen(base + p).read()  # noqa: E731
        assert b"Leaderboard" in get("/")
        board = json.loads(get("/api/leaderboard"))
        assert board[0]["run_id"] == s["run_id"]
        run = json.loads(get(f"/api/runs/{s['run_id']}"))
        img_id = run["images"][0]["image_id"]
        detail = json.loads(get(f"/api/runs/{s['run_id']}/images/{img_id}"))
        assert len(detail["gt"]) == 4 and detail["steps"]
        assert get(f"/img/test/{img_id}")[:2] == b"\xff\xd8"
    finally:
        httpd.shutdown()


def board_inr(results_dir):
    return runner.leaderboard(results_dir)[0]["cost_per_image_inr"]


def test_cloud_run_task_runs_only_its_combination(fake_root, tmp_path, monkeypatch):
    import cli

    calls = []
    monkeypatch.setattr(runner, "run", lambda name, model, *a, **k: calls.append((name, model)))
    monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "4")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "3")
    cli.main(["run", "-a", "single_pass", "detect_classify", "-m", "m1", "m2"])
    assert calls == [("detect_classify", "m2")]


def test_cloud_run_compute_uses_real_task_duration(fake_root, tmp_path, monkeypatch):
    from utils import cloud

    cfg = {"cloud_run": {"cpu": 2, "memory_gib": 4}, "gcp": {"region": "us-central1"}}
    assert runner.environment(cfg) == {"platform": "local"}
    for k, v in {"CLOUD_RUN_JOB": "shelf-bench", "CLOUD_RUN_EXECUTION": "shelf-bench-abc",
                 "CLOUD_RUN_TASK_INDEX": "1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    s = runner.run("single_pass", "m", "test", 0, 0, 1, "t", results_dir=tmp_path,
                   data_root=fake_root, llm=OracleLLM(), log=lambda *_: None)
    assert s["environment"]["execution"] == "shelf-bench-abc" and s["environment"]["task_index"] == 1
    assert "provisional" in s["cost"]["compute_source"]

    class Tasks:  # Cloud Run Admin API: executions/{e}/tasks
        def get(self, url, params=None):
            assert url.endswith("/jobs/shelf-bench/executions/shelf-bench-abc/tasks")
            return NS(status_code=200, content=b"x", json=lambda: {"tasks": [
                {"index": 0, "startTime": "2026-09-24T20:00:00Z", "completionTime": "2026-09-24T20:01:00Z"},
                {"index": 1, "startTime": "2026-09-24T20:00:00.000000001Z",
                 "completionTime": "2026-09-24T20:02:00.040Z"}]})

    monkeypatch.setattr(cloud, "load_config", lambda: {"gcp": {"project": "p"}})
    secs = cloud.task_seconds(s["environment"], Tasks())
    assert secs == 120.1  # rounded up to 100 ms, like Cloud Run billing
    runner.set_compute(s, secs, "Cloud Run task start->completion (Admin API)")
    assert s["cost"]["compute_usd_per_image"] == pytest.approx(120.1 * (2 * 1.8e-5 + 4 * 2e-6) / 3)
    assert s["cost_per_image_usd"] == pytest.approx(
        s["cost"]["gemini_net_usd_per_image"] + s["cost"]["compute_usd_per_image"])


# ---- approaches that use other GCP APIs (embeddings, AlloyDB) ---------------------------------

def test_approach_setup_and_billed_units_are_priced_into_cost_per_image(fake_root, tmp_path, monkeypatch):
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

        def setup(self, config):
            self.embed = FakeEmbeddings()

        def detect(self, image, ctx):
            boxes = to_pixels(ctx.ask(image, "detect").data, 0, 0, *image.size)
            widths = [int(self.embed.image(image.crop(b), ctx)[0]) for b in boxes]
            ctx.trace.step("Embed", ", ".join(map(str, widths)), boxes=boxes)
            return boxes

    monkeypatch.setitem(approaches.base.REGISTRY, "embed_boxes", EmbedEveryBox())
    s = runner.run("embed_boxes", "m", "test", 1, 0, 1, "t", results_dir=tmp_path,
                   data_root=fake_root, llm=OracleLLM(), log=lambda *_: None)
    # 4 boxes -> 4 image embeddings at $0.0001, on top of the Gemini tokens.
    assert s["cost"]["services_usd_per_image"] == pytest.approx(4e-4)
    assert s["cost_per_image_usd"] == pytest.approx((1000 * 0.3 + 250 * 2.5) * PER_M + 4e-4)
    _, images = runner.load_run(s["run_id"], tmp_path)
    assert images[0]["billed"] == {"embedding_image": 4}
    assert images[0]["steps"][1]["detail"] == "50, 50, 60, 90"


def test_unpriced_billed_unit_fails_loudly():
    with pytest.raises(RuntimeError, match="embedding_text_char"):
        pricing.extra_cost({"embedding_text_char": 10}, fake_sheet(["m"]))


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
    db.instance, db.database, db.user, db.ip_type = "projects/p/.../instances/i", "postgres", "u", "PRIVATE"
    db.connector = NS(connect=lambda *a, **k: seen.setdefault("connect", (a, k)) and NS(cursor=Cur))
    db._local = threading.local()
    v = pgvector([0.1, 0.2])
    assert db.query("SELECT id FROM products ORDER BY embedding <=> %s::vector LIMIT 1", (v,)) == [("sku-1", 0.93)]
    assert seen["q"][1] == ("[0.1,0.2]",)
    assert seen["connect"][1]["enable_iam_auth"] is True


def test_retrieval_template_is_not_registered():
    assert "detect_retrieve" not in approaches.all_approaches()


# ---- OpenTelemetry --------------------------------------------------------------------------

def test_run_is_one_trace_with_image_and_gemini_spans_and_linked_logs(fake_root, tmp_path, caplog):
    """run -> image -> gemini spans carry every token count + cost; summary links to them."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    telemetry.use_exporter(exporter, project="proj")

    class Thinking(OracleLLM):  # detect_classify calls this from a thread pool (pass 2)
        def __call__(self, image, prompt, **kw):
            res = super().__call__(image, prompt, **kw)
            res.usage = Usage(1200, 250, 40, 1, {"priority": 1},
                              {"priority/image_input": 1000, "priority/text_input": 200,
                               "priority/output": 290})
            res.meta = {"attempts": 2, "finish_reason": "STOP", "model_version": "fake-001",
                        "response_id": "r-1", "tier_requested": "priority"}
            if "contact sheet" in prompt:
                res.data = [{"id": i, "label": "food"} for i in range(4)]
            return res

    with caplog.at_level("INFO", logger="shelf_bench.telemetry"):
        s = runner.run("detect_classify", "fake-model", "val", 2, 0, 2, "t", results_dir=tmp_path,
                       data_root=fake_root, llm=Thinking(), log=lambda *_: None)

    spans = exporter.get_finished_spans()
    run_span = next(x for x in spans if x.name.startswith("run "))
    images = [x for x in spans if x.name.startswith("image ")]
    calls = [x for x in spans if x.name == "gemini fake-model"]
    assert len(images) == 2 and len(calls) == 4  # 2 images x (detect + 1 contact sheet)
    assert {x.context.trace_id for x in spans} == {run_span.context.trace_id}
    assert all(i.parent.span_id == run_span.context.span_id for i in images)
    image_ids = {i.context.span_id for i in images}
    assert all(c.parent.span_id in image_ids for c in calls)  # also from pass-2 threads

    a = calls[0].attributes
    assert a["gen_ai.usage.input_tokens"] == 1200 and a["gen_ai.usage.output_tokens"] == 250
    assert a["shelf_bench.usage.thinking_tokens"] == 40
    assert a["shelf_bench.usage.image_input_tokens"] == 1000
    assert a["shelf_bench.usage.text_input_tokens"] == 200
    assert a["shelf_bench.traffic_type"] == "priority" and a["shelf_bench.attempts"] == 2
    assert a["gen_ai.response.model"] == "fake-001"
    assert a["shelf_bench.cost.net_usd"] > 0
    assert images[0].attributes["shelf_bench.usage.calls"] == 2
    assert [e.name for e in images[0].events][:2] == ["step: Load image", "step: Pass 1: detection"]
    assert run_span.attributes["shelf_bench.f2"] == s["f2"]
    assert run_span.attributes["shelf_bench.usage.thinking_tokens"] == 4 * 40

    tid = format(run_span.context.trace_id, "032x")
    t = s["telemetry"]
    assert t["trace_id"] == tid and tid in t["trace_url"] and "project=proj" in t["trace_url"]
    assert "logs/query" in t["logs_url"] and tid in t["logs_url"]
    _, rows = runner.load_run(s["run_id"], tmp_path)
    assert {r["telemetry"]["span_id"] for r in rows} == {format(i, "016x") for i in image_ids}

    events = [r.json_fields["event"] for r in caplog.records]
    assert events.count("gemini_call") == 4 and events.count("image_scored") == 2
    assert events[0] == "run_started" and events[-1] == "run_finished"
    call_log = next(r for r in caplog.records if r.json_fields["event"] == "gemini_call")
    assert call_log.trace == f"projects/proj/traces/{tid}" and call_log.json_fields["prompt"]


def test_telemetry_off_leaves_no_links(fake_root, tmp_path):
    s = runner.run("single_pass", "m", "test", 1, 0, 1, "t", results_dir=tmp_path,
                   data_root=fake_root, llm=OracleLLM(), log=lambda *_: None)
    assert s["telemetry"] is None
