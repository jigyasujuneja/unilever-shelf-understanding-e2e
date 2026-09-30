"""Run an approach x model over a dataset subset (SKU-110K, labelled products or RPC) and write results.

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

from PIL import Image

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


def combos(names: list[str], tiers: list[str], models: list[str],
           log: Callable[[str], None] = print) -> list[tuple[str, str, str]]:
    """(approach, tier, model) runs to do, skipping models an approach doesn't call (and the
    priority tier for non-Gemini approaches). Raises on unknown approaches, before any spend."""
    out = []
    for n in names:
        appr = approaches.get(n)
        for t in tiers:
            for m in models:
                if appr.accepts(m) and (appr.uses_gemini or t == "standard"):
                    out.append((n, t, m))
                else:
                    runs = ", ".join(appr.models) if appr.models else "Gemini models"
                    log(f"skip {n} x {m} x {t}: {n} runs {runs}"
                        + ("" if appr.uses_gemini else " (standard tier)"))
    return out


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
    if not appr.accepts(model):
        raise ValueError(f"{approach} doesn't run {model}; it runs "
                         f"{', '.join(appr.models) if appr.models else 'Vertex Gemini models'}")
    config = load_config()
    env = environment(config)
    # Live list prices from the Cloud Billing Catalog API (fails fast if a SKU is missing).
    sheet = prices or pricing.price_sheet([model] if appr.uses_gemini else [], config,
                                         extra_skus=appr.skus)
    if appr.uses_gemini:
        llm = llm or Gemini(model, config, tier=tier)
    task = appr.task
    if appr.dataset != "sku110k":  # --root is SKU-110K's; the other sets have their own location
        data_root = dataset.data_root(appr.dataset)
    fields = {"rpc": dataset.RPC_FIELDS, "shelves": dataset.SHELF_FIELDS}.get(
        appr.dataset, dataset.PRODUCT_FIELDS)
    catalog = dataset.catalog(str(data_root), appr.dataset) if task != "detection" else {}
    samples = dataset.sample_images(split, limit, seed, str(data_root), appr.dataset)
    if not samples:
        raise RuntimeError(f"No images found for split={split!r} under {data_root}")

    started = datetime.now(timezone.utc)
    run_id = (f"{started:%m%d-%H%M%S}-{approach}-{model}" + ("-priority" if tier == "priority" else ""))
    log(f"[{run_id}] {len(samples)} images from {appr.dataset} {split} (seed {seed}) on {env['platform']}")

    def price(u: Usage) -> dict:
        return pricing.gemini_cost(model, u.buckets, sheet, started.date())

    # The approach's own clients and indexes (embeddings, AlloyDB, ...). What it bills here is
    # a one-off setup cost (e.g. embedding a reference gallery), reported apart from cost/img.
    setup_ctx = Context(model=model, llm=llm, price=price)
    t_setup = time.perf_counter()
    appr.setup(config, setup_ctx)
    setup_s = time.perf_counter() - t_setup
    setup_usd = price(setup_ctx.trace.usage)["net_usd"] + pricing.extra_cost(setup_ctx.trace.billed, sheet)

    # One trace per run (Cloud Trace), with correlated log entries (Cloud Logging).
    telemetry.init(config)
    run_attrs = {"shelf_bench.run_id": run_id, "shelf_bench.approach": approach,
                 "shelf_bench.task": task, "shelf_bench.dataset": appr.dataset,
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

    def identify_all(image: Image.Image, boxes: list, ctx: Context) -> list[int | None]:
        """Name the product in every box (``Approach.identify_boxes``)."""
        ids = appr.identify_boxes(image, boxes, ctx)
        ctx.trace.step("Identify each box", f"{sum(i is not None for i in ids)} of {len(boxes)} "
                       "boxes matched to a catalog product", boxes=boxes)
        return ids

    def _one(sample: dataset.Sample, span) -> dict:
        trace = Trace(span=span)
        ctx = Context(model=model, llm=llm, trace=trace,
                      otel_parent=telemetry.child_context(span), price=price)
        t0 = time.perf_counter()
        error, preds, ids = None, [], []
        try:
            with Image.open(io.BytesIO(dataset.read_bytes(sample.path))) as im:
                image = im.convert("RGB")
            trace.step("Load image", f"{image.width}x{image.height} px")
            if task == "classification":  # the photo is the product
                ids = [appr.identify(image, ctx)]
                preds = list(sample.boxes)
            elif task == "retrieval":  # the ground-truth boxes are given
                preds = list(sample.boxes)
                ids = identify_all(image, preds, ctx)
            elif task == "end_to_end":
                preds, ids = appr.detect_and_identify(image, ctx)
            else:
                preds = appr.detect(image, ctx)
        except Exception as e:  # a failed image scores as "found nothing"
            preds, ids, error = [], [], f"{type(e).__name__}: {e}"[:300]
            span.record_exception(e)
        latency = time.perf_counter() - t0
        extra = {}
        if task == "detection":
            m = metrics.match(preds, sample.boxes)
            trace.step("Score vs ground truth",
                       f"{m['tp']} correct, {m['fp']} false, {m['fn']} missed "
                       f"({len(sample.boxes)} products in ground truth)")
        elif task == "end_to_end":  # right box (IoU >= 0.5) and right product
            gt_ids = [g["sku_id"] for g in sample.labels]
            m = metrics.match_identified(preds, ids, sample.boxes, gt_ids)
            extra["found"] = m["found"]
            extra["labels"] = [
                {"box": [round(v, 1) for v in b], "pred": catalog.get(i),
                 "gt": sample.labels[m["pairs"][k]] if k in m["pairs"] else None,
                 "correct": {f: k in m["pairs"] and (catalog.get(i) or {}).get(f)
                             == sample.labels[m["pairs"][k]].get(f) for f in fields}}
                for k, (b, i) in enumerate(zip(preds, ids, strict=True))]
            trace.step("Score vs ground truth",
                       f"{m['found']} of {len(sample.boxes)} products boxed, {m['tp']} of those "
                       f"identified correctly; {m['fp']} boxes wrong or extra, {m['fn']} products missed")
        else:  # classification / retrieval: one answer per ground-truth box
            if not ids:  # the image failed: every product counts as unanswered
                ids = [None] * len(sample.boxes)
            m = {"tp": 0, "fp": 0, "fn": 0, "matched": []}
            extra["labels"] = []
            for k, (b, gt, i) in enumerate(zip(sample.boxes, sample.labels, ids, strict=True)):
                pred = catalog.get(i) if i is not None else None
                mk = metrics.match_product(pred, gt, fields)
                for key in ("tp", "fp", "fn"):
                    m[key] += mk[key]
                if mk["tp"]:
                    m["matched"].append(k)
                extra["labels"].append({"box": [round(v, 1) for v in b], "pred": pred, "gt": gt,
                                        "correct": mk["correct"]})
            trace.step("Score vs ground truth",
                       f"{m['tp']} of {len(sample.boxes)} products identified correctly"
                       if len(sample.boxes) > 1 else
                       f"predicted {(extra['labels'][0]['pred'] or {}).get('product', 'nothing')}; "
                       f"truth {sample.labels[0]['product']} "
                       f"({', '.join(f + (' ✓' if ok else ' ✗') for f, ok in extra['labels'][0]['correct'].items())})")
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
            **extra,
        }

    try:
        rows: list[dict] = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, row in enumerate(pool.map(one, samples), 1):
                rows.append(row)
                flag = f"  ERROR {row['error']}" if row["error"] else ""
                what = (f"{'✓' if row['tp'] else '✗'} {(row['labels'][0]['pred'] or {}).get('product', '-')}"
                        if task == "classification" else
                        f"{row['tp']:>3} / {row['gt_count']:>3} identified" if task == "retrieval" else
                        f"{row['pred_count']:>4} pred / {row['gt_count']:>4} gt  F2 {row['f2']:.2f}")
                log(f"  {i:>3}/{len(samples)} {row['image_id']:<16} {what}  "
                    f"{row['latency_s']:.1f}s{flag}")
        wall_s = time.perf_counter() - t_start

        summary = summarize(rows)
        n = len(rows)
        if task in ("classification", "retrieval"):  # accuracy = exact product, over all products
            boxes = [lab for r in rows for lab in r["labels"]]
            summary["field_accuracy"] = {
                f: round(sum(bool(lab["correct"][f]) for lab in boxes) / max(1, len(boxes)), 4)
                for f in fields}
            summary["accuracy"] = summary["field_accuracy"]["product"]
        if task == "end_to_end":  # share of products boxed at all, whatever the identity
            summary["found_recall"] = round(sum(r["found"] for r in rows)
                                            / max(1, sum(r["gt_count"] for r in rows)), 4)
        usage = Usage(**summary["tokens"])
        g = pricing.gemini_cost(model, usage.buckets, sheet, started.date())
        on_gcs = str(data_root).startswith("gs://")
        # GCS ops: one GET per image + the annotations file (Class B), plus one more GET per
        # products image (its size is read when the set loads); two result uploads (Class A).
        products = appr.dataset == "products"
        storage_usd = pricing.storage_cost(2, n + 1 + (n if products else 0), sheet) if on_gcs else 0.0
        calls = usage.calls
        s_usd = sum(r["services_cost_usd"] for r in rows)
        summary.update({
            "run_id": run_id,
            "approach": approach,
            "use_case": appr.use_case,
            "task": appr.task,
            "dataset": appr.dataset,
            "model": model,
            "tier": tier,
            "traffic": usage.traffic,  # calls per traffic type Vertex actually served
            # Share of calls Vertex actually served at priority (the rest were downgraded).
            "priority_served": round(usage.traffic.get("priority", 0) / calls, 3)
            if calls and tier == "priority" else None,
            "architecture": f"{appr.architecture} [{' + '.join([model, *appr.also_calls])}"
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
                # One-off setup (e.g. embedding the reference gallery), not in cost per image.
                "setup_usd": setup_usd, "setup_seconds": round(setup_s, 1),
            },
            "token_cost_per_image_usd": g["net_usd"] / n,
            "storage_cost_per_image_usd": storage_usd / n,
        })
        if env["platform"] == "cloud-run":  # provisional; `pull` swaps in the task's real duration
            set_compute(summary, time.time() - _T_PROCESS, "in-task clock (provisional)")
        else:
            set_compute(summary, 0.0, "local run: compute not priced")
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
            img_blob = gcs.blob(f"{prefix}/{run_id}/images.jsonl")
            if not img_blob.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            img_blob.download_to_filename(str(dest / "images.jsonl"))
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
    """All runs, grouped by ``use_case`` (Market share / Merchandising section) and ``task`` (one
    leaderboard tab each) and ranked within it: detection and end-to-end by F2, classification /
    retrieval by accuracy (exact product), ties by recall then cost.
    Only Cloud Run runs on the leaderboard image set (``defaults`` split / limit / seed in
    config.yaml) get a numeric ``rank``, so every ranked run in a task is scored on identical
    images; other runs are listed after them as ``"dev"``."""
    d = (config if config is not None else load_config()).get("defaults", {})
    board_set = (d.get("split", "test"), d.get("limit", 50), d.get("seed", 0))
    runs = []
    for p in Path(results_dir).glob("*/summary.json"):
        try:
            s = _with_inr(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        s.pop("pricing", None)  # full SKU sheet is only needed on the run page
        s.setdefault("use_case", "market_share")  # runs from before use cases existed
        s.setdefault("task", "detection")  # runs from before tasks existed
        s.setdefault("dataset", "sku110k")
        runs.append(s)

    def official(r: dict) -> bool:
        return (r.get("environment", {}).get("platform") == "cloud-run"
                and (r["split"], r.get("limit"), r.get("seed")) == board_set)

    def score(r: dict) -> float:
        return r["f2"] if r["task"] in ("detection", "end_to_end") else r["accuracy"]

    def board(r: dict) -> tuple[int, str, str]:  # one tab per task and dataset
        return approaches.base.USE_CASES.index(r["use_case"]), r["task"], r["dataset"]

    runs.sort(key=lambda r: (board(r), not official(r), -score(r), -r["recall"],
                             r["cost_per_image_usd"]))
    place: dict[tuple, int] = {}
    for r in runs:
        if official(r):
            place[board(r)] = place.get(board(r), 0) + 1
        r["rank"] = place[board(r)] if official(r) else "dev"
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

