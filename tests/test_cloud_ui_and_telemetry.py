"""Tests for ``src/cli.py``, ``src/utils/cloud.py``, ``src/utils/server.py``, and
``src/utils/telemetry.py``."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import NS, OracleLLM, usage

import cli
import runner
from utils import cloud, telemetry
from utils.llm import LLMResult, Usage
from utils.server import Handler


def test_server_endpoints(fake_root, tmp_path):
    s = runner.run(
        "single_pass",
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


def test_products_run_page_and_per_task_ranking(products_root, fake_root, tmp_path):
    det = runner.run(
        "single_pass",
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
    cls = runner.run(
        "gemini_classify",
        "gemini-t",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        llm=lambda *a, **k: LLMResult({"sku_id": 1}, usage(), 0.01),
        log=lambda *_: None,
    )
    for s in (det, cls):  # make both official Cloud Run runs on the board's image set
        p = tmp_path / s["run_id"] / "summary.json"
        p.write_text(
            json.dumps({
                **json.loads(p.read_text()),
                "limit": 50,
                "environment": {"platform": "cloud-run"},
            })
        )
    board = runner.leaderboard(tmp_path, {"defaults": {"split": "test", "limit": 50, "seed": 0}})
    assert [(r["task"], r["rank"]) for r in board] == [("classification", 1), ("detection", 1)]
    Handler.results_dir, Handler.data_root = tmp_path, fake_root
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    get = lambda p: urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}{p}").read()  # noqa: E731
    try:
        run = json.loads(get(f"/api/runs/{cls['run_id']}"))
        assert run["images"][0]["labels"][0]["gt"]["product"] == "Knorr Cubes 10g"
        detail = json.loads(get(f"/api/runs/{cls['run_id']}/images/p2.jpg"))
        assert detail["split"] == "products" and detail["gt"] == [[0.0, 0.0, 62.0, 40.0]]
        assert detail["labels"][0]["correct"]["product"] is False
        assert get("/img/products/p2.jpg")[:2] == b"\xff\xd8"
    finally:
        httpd.shutdown()


def test_cloud_run_task_runs_only_its_combination(fake_root, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda name, model, *a, **k: calls.append((name, model)))
    monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "4")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "3")
    cli.main(["run", "-a", "single_pass", "detect_classify", "-m", "gemini-1", "gemini-2"])
    assert calls == [("detect_classify", "gemini-2")]


def test_approaches_only_run_the_models_they_call(monkeypatch):
    combos = runner.combos(
        ["gemini_classify", "embedding_retrieval"],
        ["standard", "priority"],
        ["gemini-1", "multimodalembedding@001"],
        log=lambda *_: None,
    )
    assert combos == [
        ("gemini_classify", "standard", "gemini-1"),
        ("gemini_classify", "priority", "gemini-1"),
        ("embedding_retrieval", "standard", "multimodalembedding@001"),
    ]
    with pytest.raises(ValueError, match="doesn't run"):
        runner.run("gemini_classify", "multimodalembedding@001", log=lambda *_: None)
    calls = []
    monkeypatch.setattr(runner, "run", lambda name, model, *a, **k: calls.append((name, model)))
    monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "3")
    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "2")
    cli.main([
        "run",
        "-a",
        "gemini_classify",
        "embedding_retrieval",
        "-t",
        "standard",
        "priority",
        "-m",
        "gemini-1",
        "multimodalembedding@001",
    ])
    assert calls == [("embedding_retrieval", "multimodalembedding@001")]


def test_cloud_run_compute_uses_real_task_duration(fake_root, tmp_path, monkeypatch):
    cfg = {"cloud_run": {"cpu": 2, "memory_gib": 4}, "gcp": {"region": "us-central1"}}
    assert runner.environment(cfg) == {"platform": "local"}
    for k, v in {
        "CLOUD_RUN_JOB": "shelf-bench",
        "CLOUD_RUN_EXECUTION": "shelf-bench-abc",
        "CLOUD_RUN_TASK_INDEX": "1",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    s = runner.run(
        "single_pass",
        "gemini-t",
        "test",
        0,
        0,
        1,
        "t",
        results_dir=tmp_path,
        data_root=fake_root,
        llm=OracleLLM(),
        log=lambda *_: None,
    )
    assert s["environment"]["execution"] == "shelf-bench-abc"
    assert s["environment"]["task_index"] == 1
    assert "provisional" in s["cost"]["compute_source"]

    class Tasks:  # Cloud Run Admin API: executions/{e}/tasks
        def get(self, url, params=None):
            assert url.endswith("/jobs/shelf-bench/executions/shelf-bench-abc/tasks")
            return NS(
                status_code=200,
                content=b"x",
                json=lambda: {
                    "tasks": [
                        {
                            "index": 0,
                            "startTime": "2026-09-24T20:00:00Z",
                            "completionTime": "2026-09-24T20:01:00Z",
                        },
                        {
                            "index": 1,
                            "startTime": "2026-09-24T20:00:00.000000001Z",
                            "completionTime": "2026-09-24T20:02:00.040Z",
                        },
                    ]
                },
            )

    monkeypatch.setattr(cloud, "load_config", lambda: {"gcp": {"project": "p"}})
    secs = cloud.task_seconds(s["environment"], Tasks())
    assert secs == 120.1  # rounded up to 100 ms, like Cloud Run billing
    runner.set_compute(s, secs, "Cloud Run task start->completion (Admin API)")
    assert s["cost"]["compute_usd_per_image"] == pytest.approx(
        120.1 * (2 * 1.8e-5 + 4 * 2e-6) / 3
    )
    assert s["cost_per_image_usd"] == pytest.approx(
        s["cost"]["gemini_net_usd_per_image"] + s["cost"]["compute_usd_per_image"]
    )


def test_ui_proxy_forwards_with_identity_token(monkeypatch):
    class Upstream(BaseHTTPRequestHandler):  # stands in for the private Cloud Run service
        def do_GET(self):
            body = json.dumps({"path": self.path, "auth": self.headers["Authorization"]}).encode()
            self.send_response(200 if self.path != "/missing" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    uri = f"http://127.0.0.1:{upstream.server_port}/"

    class Services:  # Cloud Run Admin API: services/{name}
        def get(self, url):
            assert url.endswith("/projects/p/locations/r/services/ui")
            return NS(status_code=200, content=b"x", json=lambda: {"uri": uri})

    monkeypatch.setattr(
        cloud,
        "load_config",
        lambda: {"gcp": {"project": "p", "region": "r"}, "cloud_run": {"service": "ui"}},
    )
    monkeypatch.setattr(cloud, "_session", Services)
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, text: "tok\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    threading.Thread(
        target=cloud.proxy_ui, kwargs={"port": port, "log": lambda *_: None}, daemon=True
    ).start()
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs?x=1", timeout=2) as r:
                assert json.load(r) == {"path": "/api/runs?x=1", "auth": "Bearer tok"}
            break
        except urllib.error.URLError:  # proxy not listening yet
            threading.Event().wait(0.05)
    else:
        pytest.fail("proxy never started")
    with pytest.raises(urllib.error.HTTPError) as e:  # upstream errors are passed through
        urllib.request.urlopen(f"http://127.0.0.1:{port}/missing", timeout=2)
    assert e.value.code == 404
    upstream.shutdown()


def test_run_is_one_trace_with_image_and_gemini_spans_and_linked_logs(
    fake_root, tmp_path, caplog
):
    """run -> image -> gemini spans carry every token count + cost; summary links to them."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    telemetry.use_exporter(exporter, project="proj")

    class Thinking(OracleLLM):  # detect_classify calls this from a thread pool (pass 2)
        def __call__(self, image, prompt, **kw):
            res = super().__call__(image, prompt, **kw)
            res.usage = Usage(
                1200,
                250,
                40,
                1,
                {"priority": 1},
                {
                    "priority/image_input": 1000,
                    "priority/text_input": 200,
                    "priority/output": 290,
                },
            )
            res.meta = {
                "attempts": 2,
                "finish_reason": "STOP",
                "model_version": "fake-001",
                "response_id": "r-1",
                "tier_requested": "priority",
            }
            if "contact sheet" in prompt:
                res.data = [{"id": i, "label": "food"} for i in range(4)]
            return res

    with caplog.at_level("INFO", logger="shelf_bench.telemetry"):
        s = runner.run(
            "detect_classify",
            "gemini-fake",
            "val",
            2,
            0,
            2,
            "t",
            results_dir=tmp_path,
            data_root=fake_root,
            llm=Thinking(),
            log=lambda *_: None,
        )

    spans = exporter.get_finished_spans()
    run_span = next(x for x in spans if x.name.startswith("run "))
    images = [x for x in spans if x.name.startswith("image ")]
    calls = [x for x in spans if x.name == "gemini gemini-fake"]
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
    assert [e.name for e in images[0].events][:2] == [
        "step: Load image",
        "step: Pass 1: detection",
    ]
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
    s = runner.run(
        "single_pass",
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
    assert s["telemetry"] is None
