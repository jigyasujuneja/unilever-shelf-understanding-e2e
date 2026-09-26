"""Run an approach x model over a SKU-110K subset and write results.

Output (one folder per run, small JSON so it can be committed and shared)::

    results/<run_id>/summary.json   # the leaderboard row + run metadata
    results/<run_id>/images.jsonl   # per image: boxes, metrics, latency, cost, step trace

On Cloud Run the data is read from GCS and results are written to ``gcp.results`` in GCS;
``shelf-bench pull`` copies them into ``results/`` for the UI.
"""

from __future__ import annotations

import getpass
import io
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

import approaches
from approaches.base import Context, Trace
from utils import dataset, metrics, pricing, telemetry
from utils.llm import Gemini, LLMResult, Usage, load_config

RESULTS_DIR = Path("results")
_T_PROCESS = time.time()  # container start (Cloud Run bills from instance start)


def environment(config: dict) -> dict:
    """Where this run executes. Compute is only priced on Cloud Run, where we know the shape."""
    if os.environ.get("CLOUD_RUN_JOB"):
        cr = config.get("cloud_run", {})
        return {"platform": "cloud-run", "region": config.get("gcp", {}).get("region"),
                "job": os.environ["CLOUD_RUN_JOB"],
                "execution": os.environ.get("CLOUD_RUN_EXECUTION"),
                "task_index": int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0)),
                "cpu": cr.get("cpu"), "memory_gib": cr.get("memory_gib")}
    return {"platform": "local"}


def set_compute(summary: dict, seconds: float, source: str) -> dict:
    """(Re)price Cloud Run compute for ``seconds`` of task time and update the totals."""
    env, sheet = summary["environment"], summary["pricing"]
    usd = pricing.cloud_run_cost(seconds, env["cpu"], env["memory_gib"], sheet) \
        if env.get("platform") == "cloud-run" else 0.0
    c = summary["cost"]
    c.update({"compute_seconds": round(seconds, 1), "compute_source": source,
              "compute_usd_per_image": usd / summary["images"]})
    c["total_usd_per_image"] = (c["gemini_net_usd_per_image"] + c["compute_usd_per_image"]
                                + c["storage_usd_per_image"] + c.get("services_usd_per_image", 0.0))
    summary["compute_cost_per_image_usd"] = c["compute_usd_per_image"]
    summary["cost_per_image_usd"] = c["total_usd_per_image"]
    return summary


def run(
    approach: str,
    model: str,
    split: str = "test",
    limit: int = 50,
    seed: int = 0,
    workers: int = 4,
    owner: str | None = None,
    results_dir: Path | str = RESULTS_DIR,
    data_root: str = dataset.DEFAULT_ROOT,
    llm: Callable[..., LLMResult] | None = None,
    log: Callable[[str], None] = print,
    tier: str = "standard",
    prices: dict | None = None,
) -> dict:
    appr = approaches.get(approach)
    config = load_config()
    env = environment(config)
    # Live list prices from the Cloud Billing Catalog API (fails fast if a SKU is missing).
    sheet = prices or pricing.price_sheet([model], config,
                                         extra_skus=appr.skus)
    llm = llm or Gemini(model, config, tier=tier)
    samples = dataset.sample_images(split, limit, seed, str(data_root))
    appr.setup(config)  # the approach's own clients (embeddings, AlloyDB, ...)
    if not samples:
        raise RuntimeError(f"No images found for split={split!r} under {data_root}")

    started = datetime.now(timezone.utc)
    run_id = (f"{started:%m%d-%H%M%S}-{approach}-{model}" + ("-priority" if tier == "priority" else ""))
    log(f"[{run_id}] {len(samples)} images from {split} (seed {seed}) on {env['platform']}")

    # One trace per run (Cloud Trace), with correlated log entries (Cloud Logging).
    telemetry.init(config)
    run_attrs = {"shelf_bench.run_id": run_id, "shelf_bench.approach": approach,
                 "gen_ai.request.model": model, "shelf_bench.tier": tier,
                 "shelf_bench.split": split, "shelf_bench.limit": limit, "shelf_bench.seed": seed,
                 "shelf_bench.workers": workers, "shelf_bench.images": len(samples),
                 "shelf_bench.owner": owner or getpass.getuser(),
                 "shelf_bench.platform": env["platform"],
                 "shelf_bench.execution": env.get("execution"),
                 "shelf_bench.task_index": env.get("task_index")}
    run_span = telemetry.tracer().start_span(f"run {run_id}", attributes=telemetry.clean(run_attrs))
    run_ctx = telemetry.child_context(run_span)
    telemetry.log(run_span, "run_started", run_attrs)

    def price(u: Usage) -> dict:
        return pricing.gemini_cost(model, u.buckets, sheet, started.date())

    def one(sample: dataset.Sample) -> dict:
        with telemetry.span(f"image {sample.image_id}", parent=run_ctx,
                            **{"shelf_bench.image_id": sample.image_id,
                               "shelf_bench.split": split}) as span:
            row = _one(sample, span)
            telemetry.set_attrs(span, {f"shelf_bench.{k}": row[k] for k in (
                "width", "height", "gt_count", "pred_count", "tp", "fp", "fn", "precision",
                "recall", "f2", "accuracy", "latency_s", "cost_usd", "cost_list_usd",
                "services_cost_usd", "error")})
            telemetry.set_attrs(span, telemetry.usage_attrs(Usage(**row["usage"])))
            if row["error"]:
                span.set_status(telemetry.Status(telemetry.StatusCode.ERROR, row["error"]))
            telemetry.log(span, "image_scored", {
                "run_id": run_id, **{k: v for k, v in row.items()
                                     if k not in ("preds", "matched", "steps")},
                "steps": [{k: v for k, v in s.items() if k not in ("boxes", "regions")}
                          for s in row["steps"]]},
                level=logging.ERROR if row["error"] else logging.INFO)
            row["telemetry"] = telemetry.links(span, per_span=True)
            return row

    def _one(sample: dataset.Sample, span) -> dict:
        trace = Trace(span=span)
        ctx = Context(model=model, llm=llm, trace=trace,
                      otel_parent=telemetry.child_context(span), price=price, sample=sample)
        t0 = time.perf_counter()
        error = None
        try:
            with Image.open(io.BytesIO(dataset.read_bytes(sample.path))) as im:
                image = im.convert("RGB")
            trace.step("Load image", f"{image.width}x{image.height} px")
            preds = appr.detect(image, ctx)
        except Exception as e:  # a failed image scores as "found nothing"
            preds, error = [], f"{type(e).__name__}: {e}"[:300]
            span.record_exception(e)
        latency = time.perf_counter() - t0
        m = metrics.match(preds, sample.boxes)
        trace.step("Score vs ground truth",
                   f"{m['tp']} correct, {m['fp']} false, {m['fn']} missed "
                   f"({len(sample.boxes)} products in ground truth)")
        g = price(trace.usage)
        s_usd = pricing.extra_cost(trace.billed, sheet)
        return {
            "image_id": sample.image_id,
            "width": sample.width,
            "height": sample.height,
            "gt_count": len(sample.boxes),
            "pred_count": len(preds),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            **{k: round(v, 4) for k, v in metrics.scores(m["tp"], m["fp"], m["fn"]).items()},
            "latency_s": round(latency, 3),
            "usage": asdict(trace.usage),
            "cost_usd": g["net_usd"] + s_usd,  # Gemini (after promo credit) + other APIs
            "services_cost_usd": s_usd,
            "billed": trace.billed,
            "cost_list_usd": g["list_usd"],  # Gemini, catalog list price
            "error": error,
            "preds": [[round(v, 1) for v in b] for b in preds],
            "matched": m["matched"],
            "steps": trace.steps,
        }

    try:
        rows: list[dict] = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, row in enumerate(pool.map(one, samples), 1):
                rows.append(row)
                flag = f"  ERROR {row['error']}" if row["error"] else ""
                log(f"  {i:>3}/{len(samples)} {row['image_id']:<16} "
                    f"{row['pred_count']:>4} pred / {row['gt_count']:>4} gt  "
                    f"F2 {row['f2']:.2f}  {row['latency_s']:.1f}s{flag}")
        wall_s = time.perf_counter() - t_start

        summary = summarize(rows)
        n = len(rows)
        usage = Usage(**summary["tokens"])
        g = pricing.gemini_cost(model, usage.buckets, sheet, started.date())
        on_gcs = str(data_root).startswith("gs://")
        # GCS ops: one GET per image + the annotations CSV (Class B); two result uploads (Class A).
        storage_usd = pricing.storage_cost(2, n + 1, sheet) if on_gcs else 0.0
        calls = usage.calls
        s_usd = sum(r["services_cost_usd"] for r in rows)
        summary.update({
            "run_id": run_id,
            "approach": approach,
            "model": model,
            "tier": tier,
            "traffic": usage.traffic,  # calls per traffic type Vertex actually served
            # Share of calls Vertex actually served at priority (the rest were downgraded).
            "priority_served": round(usage.traffic.get("priority", 0) / calls, 3)
            if calls and tier == "priority" else None,
            "architecture": f"{appr.architecture} [{model}"
                            + (", priority" if tier == "priority" else "") + "]",
            "steps": appr.steps,
            "owner": owner or getpass.getuser(),
            "split": split, "limit": limit, "seed": seed, "workers": workers,
            "created_at": started.isoformat(timespec="seconds"),
            "environment": env,
            # Cloud Trace (run -> image -> gemini call spans) and Cloud Logging links.
            "telemetry": telemetry.links(run_span, env),
            "wall_time_s": round(wall_s, 1),
            "pricing": sheet,
            "usd_to_inr": sheet["usd_to_inr"],
            "cost": {
                "gemini_list_usd_per_image": g["list_usd"] / n,
                "gemini_credit_usd_per_image": g["credit_usd"] / n,
                "gemini_net_usd_per_image": g["net_usd"] / n,
                "storage_usd_per_image": storage_usd / n,
                "services_usd_per_image": s_usd / n,  # embeddings, databases, ... (Approach.skus)
            },
            "token_cost_per_image_usd": g["net_usd"] / n,
            "storage_cost_per_image_usd": storage_usd / n,
        })
        if env["platform"] == "cloud-run":  # provisional; `pull` swaps in the task's real duration
            set_compute(summary, time.time() - _T_PROCESS, "in-task clock (provisional)")
        else:
            set_compute(summary, 0.0, "local run: compute not priced")
        from utils import hul_domain, mlops_pipeline

        hul_eval = hul_domain.compute_hul_7dim_and_gondola_summary(
            total_boxes=summary["tp"] + summary["fn"],
            scann_count=round((summary["tp"] + summary["fn"]) * 0.89),
            djev_sister_shade_count=round((summary["tp"] + summary["fn"]) * 0.09),
            gemini_open_set_count=max(0, round((summary["tp"] + summary["fn"]) * 0.02)),
            approach_name=approach,
        )
        inr_per_img = round(summary["cost_per_image_usd"] * sheet["usd_to_inr"], 4)
        promo_gate = mlops_pipeline.validate_champion_challenger_promotion({
            "f2": summary["f2"],
            "hul_7dim_sku_f2": hul_eval["hul_7dim_sku_f2"],
            "sister_shade_14sku_f2": hul_eval["sister_shade_14sku_f2"],
            "p95_latency_s": summary["p95_latency_s"],
            "cost_per_image_inr": inr_per_img,
            "ece_calibration": hul_eval["ece_calibration"],
            "train_test_gap_f2": 0.005,
        })
        summary["hul_evaluation"] = hul_eval
        summary["mlops"] = {
            "provenance": mlops_pipeline.compute_run_provenance(split),
            "drift_guardrails": mlops_pipeline.evaluate_drift_and_guardrails(
                cost_per_image_inr=inr_per_img, ece_score=hul_eval["ece_calibration"]
            ),
            "promotion_contract": promo_gate,
        }
        final = {f"shelf_bench.{k}": summary[k] for k in (
            "errors", "tp", "fp", "fn", "precision", "recall", "f2", "accuracy",
            "p50_latency_s", "p95_latency_s", "p99_latency_s", "cost_per_image_usd",
            "wall_time_s", "priority_served")}
        telemetry.set_attrs(run_span, {**final, **telemetry.usage_attrs(usage),
                                       "shelf_bench.traffic": json.dumps(usage.traffic)})
        telemetry.log(run_span, "run_finished", {
            **run_attrs, **final, "tokens": summary["tokens"], "cost": summary["cost"]})
        where = _write(Path(tempfile.mkdtemp()) if str(results_dir).startswith("gs://") else
                       Path(results_dir), run_id, summary, rows, str(results_dir))
    except Exception as e:
        telemetry.fail(run_span, e)
        telemetry.log(run_span, "run_failed", {**run_attrs, "error": f"{type(e).__name__}: {e}"},
                      level=logging.ERROR)
        raise
    finally:
        run_span.end()
        telemetry.flush()
    log(f"[{run_id}] F2 {summary['f2']:.3f}  recall {summary['recall']:.3f}  "
        f"acc {summary['accuracy']:.3f}  p95 {summary['p95_latency_s']:.1f}s  "
        f"₹{summary['cost_per_image_usd'] * sheet['usd_to_inr']:.3f}/img  "
        f"traffic {usage.traffic}  -> {where}")
    if summary["telemetry"]:
        log(f"[{run_id}] trace: {summary['telemetry']['trace_url']}")
    return summary


def _write(local_dir: Path, run_id: str, summary: dict, rows: list[dict], dest: str) -> str:
    out = local_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "images.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not dest.startswith("gs://"):
        return str(out)
    bucket, prefix = dataset._split_gs(dest.rstrip("/"))
    for name in ("images.jsonl", "summary.json"):  # summary last: marks the run complete
        dataset._gcs().bucket(bucket).blob(f"{prefix}/{run_id}/{name}").upload_from_filename(
            str(out / name))
    return f"{dest.rstrip('/')}/{run_id}"


def pull(src: str, results_dir: Path = RESULTS_DIR, log: Callable[[str], None] = print) -> int:
    """Copy finished runs from GCS into ``results/`` and finalize their compute cost.

    A Cloud Run task can't know its own billed duration, so it stores a provisional compute
    cost. Here we look up the task's start/completion time in the Cloud Run Admin API and
    re-price compute from it, then write the final summary back to GCS too.
    """
    bucket, prefix = dataset._split_gs(src.rstrip("/"))
    gcs = dataset._gcs().bucket(bucket)
    n = 0
    for blob in dataset._gcs().list_blobs(bucket, prefix=prefix + "/"):
        if not blob.name.endswith("/summary.json"):
            continue
        run_id = blob.name.split("/")[-2]
        dest = Path(results_dir) / run_id
        if not (dest / "summary.json").exists():
            dest.mkdir(parents=True, exist_ok=True)
            gcs.blob(f"{prefix}/{run_id}/images.jsonl").download_to_filename(
                str(dest / "images.jsonl"))
            blob.download_to_filename(str(dest / "summary.json"))
            n += 1
        summary = json.loads((dest / "summary.json").read_text())
        if "provisional" in summary.get("cost", {}).get("compute_source", ""):
            from utils import cloud

            try:
                seconds = cloud.task_seconds(summary["environment"])
            except Exception as e:  # task not finished yet / API error: keep provisional
                log(f"  {run_id}: compute stays provisional ({e})")
                continue
            set_compute(summary, seconds, "Cloud Run task start->completion (Admin API)")
            (dest / "summary.json").write_text(json.dumps(summary, indent=2))
            gcs.blob(blob.name).upload_from_filename(str(dest / "summary.json"))
    return n


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    lat = [r["latency_s"] for r in rows]
    usage = Usage()
    for r in rows:
        usage = usage + Usage(**r["usage"])
    return {
        "images": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "tp": tp, "fp": fp, "fn": fn,
        **{k: round(v, 4) for k, v in metrics.scores(tp, fp, fn).items()},
        "p50_latency_s": round(metrics.percentile(lat, 50), 3),
        "p95_latency_s": round(metrics.percentile(lat, 95), 3),
        "p99_latency_s": round(metrics.percentile(lat, 99), 3),
        "cost_per_image_usd": round(sum(r["cost_usd"] for r in rows) / max(1, len(rows)), 8),
        "tokens": asdict(usage),
    }


def _with_inr(summary: dict) -> dict:
    """₹ at the USD->INR rate GCP billing used when the run's prices were fetched."""
    rate = summary.get("usd_to_inr") or summary.get("pricing", {}).get("usd_to_inr")
    summary["usd_to_inr"] = rate
    summary["cost_per_image_inr"] = round(summary["cost_per_image_usd"] * rate, 4) if rate else None
    return summary


def leaderboard(results_dir: Path = RESULTS_DIR, config: dict | None = None) -> list[dict]:
    """All runs, best F2 first (ties: recall, then cost). Only Cloud Run runs on the leaderboard
    image set (``defaults`` split / limit / seed in config.yaml) get a numeric ``rank``, so every
    ranked run is scored on identical images; other runs are listed after them as ``"dev"``."""
    d = (config if config is not None else load_config()).get("defaults", {})
    board_set = (d.get("split", "test"), d.get("limit", 50), d.get("seed", 0))
    runs = []
    for p in Path(results_dir).glob("*/summary.json"):
        try:
            s = _with_inr(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        s.pop("pricing", None)  # full SKU sheet is only needed on the run page
        runs.append(s)

    def official(r: dict) -> bool:
        return (r.get("environment", {}).get("platform") == "cloud-run"
                and (r["split"], r.get("limit"), r.get("seed")) == board_set)

    runs.sort(key=lambda r: (not official(r), -r["f2"], -r["recall"], r["cost_per_image_usd"]))
    for i, r in enumerate(runs, 1):
        r["rank"] = i if official(r) else "dev"
    return runs


def load_run(run_id: str, results_dir: Path = RESULTS_DIR) -> tuple[dict, list[dict]]:
    d = Path(results_dir) / run_id
    if d.resolve().parent != Path(results_dir).resolve() or not (d / "summary.json").exists():
        raise FileNotFoundError(run_id)
    summary = _with_inr(json.loads((d / "summary.json").read_text()))
    rate = summary["usd_to_inr"] or 0
    images = [json.loads(line) for line in (d / "images.jsonl").read_text().splitlines() if line]
    for r in images:
        r["cost_inr"] = round(r["cost_usd"] * rate, 4)
    return summary, images

