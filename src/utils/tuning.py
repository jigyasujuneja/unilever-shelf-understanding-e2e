"""Real Vertex AI Supervised Fine-Tuning (LoRA SFT) dataset builder and job manager.

Replaces static tuning manifests with live Vertex AI ``tuningJobs`` on ``gemini-2.5-flash-lite``
or ``gemini-2.5-flash``:

1. ``build_sft_jsonl(dataset_name, out_dir, gcs_prefix)``:
   - Builds zero-leakage ``train.jsonl`` and ``val.jsonl`` files strictly from non-test splits
     (``gallery`` and ``val`` splits of ``shelves`` / ``rpc``, and catalog attributes of
     ``products`` — **zero ``test`` split images** are ever included).
   - Formats examples in Vertex AI Gemini Supervised Fine-Tuning ``GenerateContent`` JSONL schema
     pairing GCS ``fileData`` image URIs + shortlist prompts with ground-truth JSON outputs.
2. ``upload_sft_jsonl(local_dir, gcs_prefix)``:
   - Uploads ``train.jsonl``, ``val.jsonl`` and any composite training sheets to GCS.
3. ``launch_sft_job(...)`` / ``list_sft_jobs(...)``:
   - Submits and inspects parameter-efficient LoRA Supervised Fine-Tuning jobs via the Vertex AI
     ``google.genai`` tunings API (``client.tunings.tune`` / ``client.tunings.list``) and returns
     the deployed Vertex AI endpoint resource name (``projects/.../locations/.../endpoints/...``).
"""

from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image

from approaches.base import crop
from approaches.market_share.retrieval.tiered_hybrid import PROMPT, K, sheet
from utils import dataset
from utils.llm import load_config

SUPPORTED_SFT_BASE_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
)


def _sft_example(gcs_image_uri: str, prompt: str, response_dict: dict) -> dict:
    """One Vertex AI Gemini SFT JSONL line with a GCS image URI and structured JSON target."""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"fileData": {"mimeType": "image/jpeg", "fileUri": gcs_image_uri}},
                    {"text": prompt},
                ],
            },
            {
                "role": "model",
                "parts": [{"text": json.dumps(response_dict, separators=(",", ":"))}],
            },
        ]
    }


def build_sft_jsonl(
    dataset_name: str = "shelves",
    out_dir: str | Path = "data/tuning",
    gcs_prefix: str | None = None,
    seed: int = 0,
    max_val_images: int = 20,
) -> dict[str, Any]:
    """Build zero-leakage ``train.jsonl`` and ``val.jsonl`` from non-test data only.

    For ``shelves`` or ``rpc``:
      - ``train`` examples are built from ``gallery`` crops paired against hard-negative reference
        panels (different angles of the same product vs same-category/brand distractors).
      - ``val`` examples are built from the dataset's ``val`` split boxes (never ``test``).
    """
    if dataset_name not in ("shelves", "rpc"):
        raise ValueError(f"dataset_name must be 'shelves' or 'rpc', got {dataset_name!r}")

    cfg = load_config()
    bucket_results = cfg.get("gcp", {}).get("results", "gs://unilever-shelf-understanding-shelf-images/results")
    bucket_root = bucket_results.rsplit("/", 1)[0]
    gcs_prefix = (gcs_prefix or f"{bucket_root}/tuning/{dataset_name}").rstrip("/")

    root = dataset.data_root(dataset_name)
    cat = dataset.catalog(root, dataset_name)
    gallery = dataset.rpc_gallery(root)
    val_samples = dataset.sample_images("val", max_val_images, seed, root, dataset_name)

    out_p = Path(out_dir) / dataset_name
    sheets_dir = out_p / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    all_pids = sorted(gallery.keys())

    # Cache gallery images in memory for fast contact-sheet construction
    ref_imgs: dict[str, Image.Image] = {}
    for paths in gallery.values():
        for p in paths:
            with Image.open(io.BytesIO(dataset.read_bytes(p))) as im:
                ref_imgs[p] = im.convert("RGB")

    # 1. Build TRAIN examples strictly from multi-view gallery crops (never scored in test)
    train_lines: list[str] = []
    for pid in all_pids:
        views = gallery[pid]
        if len(views) < 2:
            continue
        for v_idx, q_path in enumerate(views):
            pos_path = views[(v_idx + 1) % len(views)]
            distractor_pids = [p for p in all_pids if p != pid]
            # Prefer same-category or similar-name distractors when available
            p_cat = (cat.get(pid) or {}).get("category")
            same_cat = [p for p in distractor_pids if (cat.get(p) or {}).get("category") == p_cat]
            pool = same_cat if len(same_cat) >= K - 1 else distractor_pids
            chosen_neg = rng.sample(pool, min(K - 1, len(pool)))
            cand_pids = [pid, *chosen_neg]
            rng.shuffle(cand_pids)
            correct_choice = cand_pids.index(pid) + 1
            refs = [
                ref_imgs[pos_path if c_pid == pid else gallery[c_pid][0]]
                for c_pid in cand_pids
            ]
            sheet_name = f"train_{pid}_{v_idx}.jpg"
            sheet_path = sheets_dir / sheet_name
            sheet(ref_imgs[q_path], refs).save(sheet_path, format="JPEG", quality=88)
            ex = _sft_example(
                f"{gcs_prefix}/sheets/{sheet_name}",
                PROMPT,
                {"choice": correct_choice},
            )
            train_lines.append(json.dumps(ex))

    # 2. Build VAL examples strictly from the `val` split (zero `test` split leakage)
    val_lines: list[str] = []
    for sample in val_samples:
        with Image.open(io.BytesIO(dataset.read_bytes(sample.path))) as im:
            full_img = im.convert("RGB")
        for b_idx, (box, lab) in enumerate(zip(sample.boxes, sample.labels, strict=True)):
            gt_pid = lab["sku_id"]
            q_crop = crop(full_img, box)
            distractor_pids = [p for p in all_pids if p != gt_pid]
            if not distractor_pids:
                continue
            if gt_pid in gallery:
                chosen_neg = rng.sample(distractor_pids, min(K - 1, len(distractor_pids)))
                cand_pids = [gt_pid, *chosen_neg]
                rng.shuffle(cand_pids)
                correct_choice = cand_pids.index(gt_pid) + 1
            else:
                # Product not in gallery -> teach the model to answer 0 ("none of them")
                cand_pids = rng.sample(distractor_pids, min(K, len(distractor_pids)))
                correct_choice = 0
            refs = [ref_imgs[gallery[c_pid][0]] for c_pid in cand_pids]
            sheet_name = f"val_{Path(sample.image_id).stem}_{b_idx}.jpg"
            sheet_path = sheets_dir / sheet_name
            sheet(q_crop, refs).save(sheet_path, format="JPEG", quality=88)
            ex = _sft_example(
                f"{gcs_prefix}/sheets/{sheet_name}",
                PROMPT,
                {"choice": correct_choice},
            )
            val_lines.append(json.dumps(ex))

    train_file = out_p / "train.jsonl"
    val_file = out_p / "val.jsonl"
    train_file.write_text("\n".join(train_lines) + ("\n" if train_lines else ""))
    val_file.write_text("\n".join(val_lines) + ("\n" if val_lines else ""))
    return {
        "out_dir": out_p,
        "train_jsonl": train_file,
        "val_jsonl": val_file,
        "train_count": len(train_lines),
        "val_count": len(val_lines),
        "gcs_prefix": gcs_prefix,
    }


def upload_sft_dataset(local_dir: str | Path, gcs_prefix: str) -> dict[str, str]:
    """Upload the generated SFT JSONL files and contact sheets to GCS."""
    dataset.upload(str(local_dir), gcs_prefix.rstrip("/"))
    return {
        "train_uri": f"{gcs_prefix.rstrip('/')}/train.jsonl",
        "val_uri": f"{gcs_prefix.rstrip('/')}/val.jsonl",
    }


def launch_sft_job(
    train_uri: str,
    val_uri: str | None = None,
    base_model: str = "gemini-2.5-flash-lite",
    display_name: str = "shelf-bench-jev-laya-sft",
    epochs: int = 4,
    adapter_size: int = 4,
    learning_rate_multiplier: float = 1.0,
    config: dict | None = None,
) -> dict:
    """Submit a real parameter-efficient LoRA Supervised Fine-Tuning job on Vertex AI."""
    os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
    from google import genai
    from google.genai import types

    if base_model not in SUPPORTED_SFT_BASE_MODELS:
        raise ValueError(
            f"base_model must be one of {SUPPORTED_SFT_BASE_MODELS}, got {base_model!r}"
        )
    cfg = (config or load_config()).get("gcp", {})
    project = cfg.get("project", "unilever-shelf-understanding")
    region = cfg.get("region", "us-central1")
    client = genai.Client(vertexai=True, project=project, location=region)

    adapter_map = {
        1: types.AdapterSize.ADAPTER_SIZE_ONE,
        4: types.AdapterSize.ADAPTER_SIZE_FOUR,
        8: types.AdapterSize.ADAPTER_SIZE_EIGHT,
        16: types.AdapterSize.ADAPTER_SIZE_SIXTEEN,
    }
    if adapter_size not in adapter_map:
        raise ValueError(f"adapter_size must be one of {tuple(adapter_map)}, got {adapter_size}")

    tune_cfg = types.CreateTuningJobConfig(
        tuned_model_display_name=display_name,
        epoch_count=epochs,
        adapter_size=adapter_map[adapter_size],
        learning_rate_multiplier=learning_rate_multiplier,
    )
    if val_uri:
        tune_cfg.validation_dataset = types.TuningValidationDataset(gcs_uri=val_uri)

    job = client.tunings.tune(
        base_model=base_model,
        training_dataset=types.TuningDataset(gcs_uri=train_uri),
        config=tune_cfg,
    )
    endpoint = getattr(getattr(job, "tuned_model", None), "endpoint", None)
    return {
        "name": job.name,
        "state": getattr(job.state, "name", str(job.state)),
        "base_model": base_model,
        "display_name": display_name,
        "region": region,
        "train_uri": train_uri,
        "val_uri": val_uri,
        "endpoint": endpoint,
    }


def list_sft_jobs(config: dict | None = None) -> list[dict]:
    """List Vertex AI Supervised Fine-Tuning jobs in the configured GCP project and region."""
    os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
    from google import genai

    cfg = (config or load_config()).get("gcp", {})
    client = genai.Client(
        vertexai=True,
        project=cfg.get("project", "unilever-shelf-understanding"),
        location=cfg.get("region", "us-central1"),
    )
    out: list[dict] = []
    for job in client.tunings.list():
        tm = getattr(job, "tuned_model", None)
        out.append({
            "name": job.name,
            "display_name": getattr(job, "tuned_model_display_name", None),
            "base_model": getattr(job, "base_model", None),
            "state": getattr(job.state, "name", str(job.state)),
            "endpoint": getattr(tm, "endpoint", None) if tm else None,
            "model": getattr(tm, "model", None) if tm else None,
        })
    return out
