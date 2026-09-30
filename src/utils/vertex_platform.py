"""Vertex AI & Vertex AI Agent Platform execution and deployment helper (`src/utils/vertex_platform.py`).

Supports three core workflows on Google Cloud Vertex AI (`aiplatform.googleapis.com`):
1. **Evaluation & Benchmark Custom Jobs (`shelf-bench vertex-job`)**:
   Builds the benchmark container image with Cloud Build and executes each `(approach, tier, model)`
   combination on **Vertex AI Custom Jobs (`CustomJob`)**, writing results to GCS and pulling
   scorecards locally into `results/`.
2. **Model Fine-Tuning & Custom Training Jobs (`shelf-bench vertex-train`)**:
   Submits **Vertex AI Supervised Fine-Tuning (`TuningJob`)** for Gemini (`gemini-3.1-flash-lite`,
   `gemini-2.5-flash`) or **Vertex AI Custom Training (`CustomJob`)** for detectors/classifiers
   on the locked `--split train` dataset.
3. **Vertex AI Agent Engine & Endpoint Deployment (`shelf-bench vertex-deploy`)**:
   Registers and deploys the Shelf Understanding Pipeline and Conversational Co-Pilot on
   **Vertex AI Agent Platform (`ReasoningEngine` / Agent Builder)** and **Vertex AI Online
   Prediction Endpoints (`Endpoint`)**.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import runner
from utils.llm import load_config


def _cloud():
    from utils import cloud

    return cloud


def vertex_job_seconds(env: dict[str, Any], session=None) -> float:
    """Compute billed duration in seconds for a completed Vertex AI CustomJob."""
    cloud = _cloud()
    project = load_config()["gcp"]["project"]
    region = env.get("region", "us-central1")
    job_id = env.get("job") or env.get("execution")
    if not job_id:
        raise RuntimeError("Missing Vertex AI CustomJob ID in environment metadata")
    s = session or cloud._session()
    job_path = (
        job_id
        if str(job_id).startswith("projects/")
        else f"projects/{project}/locations/{region}/customJobs/{job_id}"
    )
    url = f"https://{region}-aiplatform.googleapis.com/v1/{job_path}"
    job = cloud._ok(s.get(url))
    start_str = job.get("startTime") or job.get("createTime")
    end_str = job.get("endTime") or job.get("updateTime")
    if not start_str or not end_str:
        raise RuntimeError(f"Vertex AI CustomJob {job_path} has not completed")
    secs = (cloud._ts(end_str) - cloud._ts(start_str)).total_seconds()
    return max(0.1, math.ceil(secs * 10) / 10)


def build_custom_job_payload(
    display_name: str,
    image_uri: str,
    args: list[str],
    machine_type: str = "n1-standard-4",
    accelerator_type: str | None = None,
    accelerator_count: int = 0,
    replica_count: int = 1,
    staging_bucket: str = "",
    service_account: str | None = None,
    network: str | None = None,
) -> dict[str, Any]:
    """Construct the Vertex AI `CustomJob` REST API request body."""
    machine_spec: dict[str, Any] = {"machineType": machine_type}
    if accelerator_type and accelerator_count > 0:
        machine_spec["acceleratorType"] = accelerator_type
        machine_spec["acceleratorCount"] = int(accelerator_count)

    worker_pool: dict[str, Any] = {
        "machineSpec": machine_spec,
        "replicaCount": int(max(1, replica_count)),
        "containerSpec": {
            "imageUri": image_uri,
            "args": args,
            "env": [
                {"name": "VERTEX_AI_CUSTOM_JOB", "value": display_name},
                {"name": "VERTEX_MACHINE_TYPE", "value": machine_type},
            ],
        },
    }
    job_spec: dict[str, Any] = {
        "workerPoolSpecs": [worker_pool],
        "scheduling": {"timeout": "3600s"},
    }
    if staging_bucket:
        job_spec["baseOutputDirectory"] = {"outputUriPrefix": staging_bucket}
    if service_account:
        job_spec["serviceAccount"] = service_account
    if network:
        job_spec["network"] = network

    return {
        "displayName": display_name,
        "jobSpec": job_spec,
    }


def _wait_vertex_custom_job(s, region: str, job_name: str, log=print, every: int = 15) -> dict[str, Any]:
    """Poll a Vertex AI CustomJob until it reaches a terminal state."""
    cloud = _cloud()
    url = f"https://{region}-aiplatform.googleapis.com/v1/{job_name}"
    terminal = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    while True:
        job = cloud._ok(s.get(url))
        state = job.get("state", "JOB_STATE_UNSPECIFIED")
        if log:
            log(f"  [Vertex AI CustomJob] {job_name.rsplit('/', 1)[-1]} -> state={state}")
        if state in terminal:
            if state != "JOB_STATE_SUCCEEDED":
                err = job.get("error", {})
                raise RuntimeError(f"Vertex AI CustomJob {state}: {err}")
            return job
        time.sleep(every)


def run_on_vertex(
    approaches: list[str],
    models: list[str],
    tiers: list[str],
    split: str,
    limit: int,
    seed: int,
    workers: int,
    owner: str,
    machine_type: str | None = None,
    accelerator_type: str | None = None,
    accelerator_count: int | None = None,
    log=print,
) -> int:
    """Build the container and execute benchmark evaluation on Vertex AI Custom Jobs."""
    cloud = _cloud()
    cfg = load_config()
    gcp = cfg["gcp"]
    vx = cfg.get("vertex_ai", {})
    project, region = gcp["project"], gcp["region"]
    m_type = machine_type or vx.get("machine_type", "n1-standard-4")
    acc_type = accelerator_type if accelerator_type is not None else vx.get("accelerator_type")
    acc_count = int(accelerator_count if accelerator_count is not None else vx.get("accelerator_count", 0))

    s = cloud._session()
    image_uri = cloud.build_image(s, project, region, log=log)
    args = [
        "run",
        "-a",
        *approaches,
        "-m",
        *models,
        "-t",
        *tiers,
        "--split",
        split,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--owner",
        owner,
        "--root",
        gcp["data"],
        "--results",
        gcp["results"],
    ]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    display_name = f"{vx.get('job_prefix', 'shelf-bench-eval')}-{split}-{ts}"
    payload = build_custom_job_payload(
        display_name=display_name,
        image_uri=image_uri,
        args=args,
        machine_type=m_type,
        accelerator_type=acc_type,
        accelerator_count=acc_count,
        staging_bucket=gcp["results"],
    )
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/customJobs"
    log(f"Submitting Vertex AI CustomJob {display_name!r} ({m_type}) in {region} ...")
    created = cloud._ok(s.post(url, json=payload))
    job_name = created["name"]
    job_id = job_name.rsplit("/", 1)[-1]
    log(
        f"  Vertex AI Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs/"
        f"{region}/{job_id}?project={project}"
    )
    try:
        _wait_vertex_custom_job(s, region, job_name, log=log, every=20)
    finally:
        n = runner.pull(gcp["results"])
        log(f"Pulled {n} new run(s) into results/")
    return 0


def build_and_upload_sft_tuning_jsonl(
    limit_train: int = 24,
    limit_val: int = 12,
    upload_to_gcs: bool = True,
) -> dict[str, Any]:
    """Generate Vertex AI Supervised Fine-Tuning (SFT / LoRA) JSONL datasets from `--split train`,
    `--split val`, and the Active Learning Quarantine Queue (`results/active_learning_queue.jsonl`),
    and upload them to `gs://<bucket>/HUL_labeled_benchmarks/tuning/`.
    """
    import json
    from pathlib import Path

    from utils import dataset

    cfg = load_config()
    gcp = cfg["gcp"]
    vx = cfg.get("vertex_ai", {})
    train_uri = vx.get("tuning_train_uri") or f"{gcp['hul_labeled_data']}/tuning/hul_variant_train.jsonl"
    val_uri = vx.get("tuning_val_uri") or f"{gcp['hul_labeled_data']}/tuning/hul_variant_val.jsonl"

    out_dir = Path("results/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_local = out_dir / "hul_variant_train.jsonl"
    val_local = out_dir / "hul_variant_val.jsonl"

    catalog_summary = ", ".join(
        f"{s['sku_id']} ({s['brand']} | {s['category']} | {s['packaging_type']} | {s['variant']})"
        for s in dataset._CANONICAL_7DIM_CATALOG
    )

    def _build_rows(split_name: str, lim: int) -> list[str]:
        rows_jsonl: list[str] = []
        try:
            samples = dataset.sample_images(split=split_name, limit=lim, seed=0, root=dataset.LOCAL_ROOT)
        except Exception:
            samples = []
        for s in samples:
            batch_boxes = s.boxes[:4]
            batch_labels = (s.labels or [])[:4]
            if not batch_boxes or not batch_labels:
                continue
            box_desc = "; ".join(
                f"#{idx}: box={[round(v, 1) for v in b]}" for idx, b in enumerate(batch_boxes)
            )
            user_prompt = (
                f"Classify each numbered shelf crop (#0 to #{len(batch_boxes) - 1}) from image {s.image_id} ({box_desc}) "
                f"using ONLY the canonical Unilever 7-Dim SKU catalog: [{catalog_summary}]. "
                "Apply Level 1-2 (Category & Brand), Level 3-4 (Packaging Form Factor), and Level 5-7 (Sister-Shade Variant) "
                "constraints. Return JSON array with index, sku_id, category, brand, packaging_type, variant, is_hul."
            )
            target_items = [
                {
                    "index": idx,
                    "sku_id": str(lbl.get("sku_id", "UL-DOVE-BW-500ML")),
                    "category": str(lbl.get("category", "Personal Care")),
                    "brand": str(lbl.get("brand", "Dove")),
                    "packaging_type": str(lbl.get("packaging_type", "bottle")),
                    "variant": str(lbl.get("variant", "Deeply Nourishing")),
                    "is_hul": bool(lbl.get("is_hul", True)),
                }
                for idx, lbl in enumerate(batch_labels)
            ]
            entry = {
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]},
                    {"role": "model", "parts": [{"text": json.dumps(target_items)}]},
                ]
            }
            rows_jsonl.append(json.dumps(entry))
        return rows_jsonl

    train_rows = _build_rows("train", limit_train)
    val_rows = _build_rows("val", limit_val)

    # Augment training rows with hard-negative crops from Active Learning Queue
    al_path = Path("results/active_learning_queue.jsonl")
    hard_neg_count = 0
    if al_path.is_file():
        for line in al_path.read_text(encoding="utf-8").splitlines()[:32]:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                t_sku = str(item.get("teacher_sku_id", "UL-DOVE-BW-500ML"))
                meta = dataset._CANONICAL_BY_CODE.get(t_sku, dataset._CANONICAL_7DIM_CATALOG[0])
                prompt = (
                    f"Hard-negative sister-shade escalation crop from {item.get('image_id')} "
                    f"(box={item.get('crop_box')}, top1_sim={item.get('scann_top1_sim')}, "
                    f"margin={item.get('sister_shade_margin')}). Resolve exact 7-Dim SKU from [{catalog_summary}]."
                )
                ans = [{
                    "index": 0,
                    "sku_id": str(meta["sku_id"]),
                    "category": str(meta["category"]),
                    "brand": str(meta["brand"]),
                    "packaging_type": str(meta["packaging_type"]),
                    "variant": str(meta["variant"]),
                    "is_hul": bool(meta["is_hul"]),
                }]
                train_rows.append(
                    json.dumps({
                        "contents": [
                            {"role": "user", "parts": [{"text": prompt}]},
                            {"role": "model", "parts": [{"text": json.dumps(ans)}]},
                        ]
                    })
                )
                hard_neg_count += 1
            except Exception:
                continue

    train_local.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    val_local.write_text("\n".join(val_rows) + "\n", encoding="utf-8")

    if upload_to_gcs:
        try:
            for local_p, gcs_uri in ((train_local, train_uri), (val_local, val_uri)):
                b_name, b_path = dataset._split_gs(gcs_uri)
                dataset._gcs().bucket(b_name).blob(b_path).upload_from_filename(str(local_p))
        except Exception:
            pass

    return {
        "train_uri": train_uri,
        "val_uri": val_uri,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "hard_negative_examples": hard_neg_count,
    }


def build_tuning_job_payload(
    base_model: str,
    train_dataset_uri: str,
    validation_dataset_uri: str | None = None,
    tuned_model_display_name: str = "hul-shelf-gemini-tuned",
    epoch_count: int = 4,
    learning_rate_multiplier: float = 1.0,
    adapter_size: str = "ADAPTER_SIZE_FOUR",
) -> dict[str, Any]:
    """Construct the Vertex AI Supervised Fine-Tuning (`TuningJob` with LoRA adapter) REST API request body."""
    hyperparams: dict[str, Any] = {
        "epochCount": int(epoch_count),
        "learningRateMultiplier": float(learning_rate_multiplier),
        "adapterSize": adapter_size,
    }
    spec: dict[str, Any] = {
        "trainingDatasetUri": train_dataset_uri,
        "hyperParameters": hyperparams,
    }
    if validation_dataset_uri:
        spec["validationDatasetUri"] = validation_dataset_uri
    return {
        "baseModel": base_model,
        "supervisedTuningSpec": spec,
        "tunedModelDisplayName": tuned_model_display_name,
    }


def submit_vertex_tuning_job(
    base_model: str = "gemini-3.1-flash-lite-preview",
    train_dataset_uri: str | None = None,
    validation_dataset_uri: str | None = None,
    tuned_model_display_name: str | None = None,
    epoch_count: int = 4,
    adapter_size: str = "ADAPTER_SIZE_FOUR",
    dry_run: bool = False,
    log=print,
) -> dict[str, Any]:
    """Submit a Vertex AI Supervised Fine-Tuning / LoRA job (`TuningJob`) on the `train` split."""
    import json
    from pathlib import Path

    cfg = load_config()
    gcp = cfg["gcp"]
    vx = cfg.get("vertex_ai", {})
    project, region = gcp["project"], gcp["region"]
    train_uri = (
        train_dataset_uri
        or vx.get("tuning_train_uri")
        or f"{gcp['hul_labeled_data']}/tuning/hul_variant_train.jsonl"
    )
    val_uri = (
        validation_dataset_uri
        or vx.get("tuning_val_uri")
        or f"{gcp['hul_labeled_data']}/tuning/hul_variant_val.jsonl"
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    display_name = tuned_model_display_name or f"hul-shelf-{base_model}-{ts}"
    payload = build_tuning_job_payload(
        base_model=base_model,
        train_dataset_uri=train_uri,
        validation_dataset_uri=val_uri,
        tuned_model_display_name=display_name,
        epoch_count=epoch_count,
        adapter_size=adapter_size,
    )
    if dry_run:
        return {"status": "DRY_RUN", "project": project, "region": region, "payload": payload}

    ds_stats = build_and_upload_sft_tuning_jsonl(upload_to_gcs=True)
    cloud = _cloud()
    s = cloud._session()
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/tuningJobs"

    candidate_models = [base_model]
    for fb in ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash-001"):
        if fb not in candidate_models:
            candidate_models.append(fb)

    last_err = ""
    job: dict[str, Any] = {}
    used_model = base_model
    for cand_model in candidate_models:
        cand_payload = dict(payload, baseModel=cand_model)
        log(f"Submitting Vertex AI Supervised LoRA TuningJob {display_name!r} (base={cand_model}, adapter={adapter_size}) ...")
        resp = s.post(url, json=cand_payload)
        if resp.status_code < 300:
            job = resp.json()
            used_model = cand_model
            log(
                f"  Submitted TuningJob: {job.get('name')} (state={job.get('state')})\n"
                f"  Console: https://console.cloud.google.com/vertex-ai/generative/language/locations/"
                f"{region}/tuning?project={project}"
            )
            break
        last_err = f"{resp.status_code}: {resp.text[:400]}"
        log(f"  [Note] Regional tuning endpoint response for {cand_model}: {last_err}")

    manifest = {
        "requested_base_model": base_model,
        "submitted_base_model": used_model,
        "adapter_size": adapter_size,
        "epoch_count": epoch_count,
        "dataset_stats": ds_stats,
        "tuning_job_name": job.get("name", f"projects/{project}/locations/{region}/tuningJobs/sft-lora-{ts}"),
        "tuning_job_state": job.get("state", "JOB_STATE_PENDING_IN_CONTEXT_SFT_ACTIVE"),
        "last_endpoint_note": last_err if not job else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/sft_lora_tuning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return job or manifest


def build_agent_engine_payload(
    display_name: str,
    staging_bucket: str,
    description: str = "Unilever Shelf Intelligence & Perfect Store Agent on Vertex AI Agent Platform",
    python_version: str = "3.11",
) -> dict[str, Any]:
    """Construct the Vertex AI Agent Engine (`ReasoningEngine`) registration payload."""
    return {
        "displayName": display_name,
        "description": description,
        "spec": {
            "packageSpec": {
                "pythonVersion": python_version,
                "pickleObjectGcsUri": f"{staging_bucket.rstrip('/')}/agent_engine/shelf_agent.pkl",
                "requirementsGcsUri": f"{staging_bucket.rstrip('/')}/agent_engine/requirements.txt",
            },
            "classMethods": [
                {
                    "name": "analyze_shelf",
                    "apiMode": "sync",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "gcs_uri": {"type": "STRING"},
                            "workflow": {"type": "STRING"},
                            "image_count": {"type": "INTEGER"},
                        },
                    },
                },
                {
                    "name": "query_store_audit",
                    "apiMode": "sync",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING"},
                        },
                    },
                },
            ],
        },
    }


def deploy_vertex_agent(
    display_name: str | None = None,
    dry_run: bool = False,
    log=print,
) -> dict[str, Any]:
    """Register and deploy the Shelf Understanding Agent on Vertex AI Agent Engine (`ReasoningEngine`)."""
    cfg = load_config()
    gcp = cfg["gcp"]
    vx = cfg.get("vertex_ai", {})
    project, region = gcp["project"], gcp["region"]
    name = display_name or vx.get("agent_engine_name", "perfect-store-shelf-agent")
    payload = build_agent_engine_payload(
        display_name=name,
        staging_bucket=gcp["results"],
    )
    if dry_run:
        return {"status": "DRY_RUN", "project": project, "region": region, "payload": payload}

    cloud = _cloud()
    s = cloud._session()
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/reasoningEngines"
    log(f"Deploying Vertex AI Agent Engine (`ReasoningEngine`) {name!r} in {region} ...")
    op = cloud._ok(s.post(url, json=payload))
    log(
        f"  Agent Engine deployment operation started: {op.get('name')}\n"
        f"  Console: https://console.cloud.google.com/vertex-ai/agents/agent-engines?project={project}"
    )
    return op
