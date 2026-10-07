"""Real Vertex AI Supervised Fine-Tuning (LoRA SFT) dataset builder and job manager.

Replaces static tuning manifests with live Vertex AI ``tuningJobs`` on ``gemini-3.5-flash-lite``
or ``gemini-3.5-flash``:

1. ``build_sft_jsonl(dataset_name, out_dir, gcs_prefix, upload=True)``:
   - Builds zero-leakage ``train.jsonl`` and ``val.jsonl`` files strictly from non-test splits
     (``gallery`` and ``val`` splits of ``shelves`` / ``rpc``, and catalog attributes of
     ``products`` — **zero ``test`` split images** are ever included).
   - Formats examples in Vertex AI Gemini Supervised Fine-Tuning ``GenerateContent`` JSONL schema
     pairing GCS ``fileData`` image URIs + shortlist prompts with ground-truth JSON outputs, and
     uploads the generated JSONL files and contact sheets directly to GCS when ``upload=True``.
2. ``upload_sft_dataset(local_dir, gcs_prefix)``:
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
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from approaches.base import crop
from approaches.market_share.retrieval.tiered_hybrid import PROMPT, K, sheet
from utils import dataset
from utils.llm import load_config

SUPPORTED_SFT_BASE_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
)

TUNABLE_DATASETS = (
    "shelves",
    "rpc",
    "pack",
    "brand",
    "variant",
    "hul_golden",
    "hul_variants",
    "hul_brands",
    "hul_packs",
    "hul_promos",
)

MARKETSHARE_BUCKET = "gs://marketshare-datasets"

PACK_CATEGORIES = (
    "Beverages", "DMT", "Foods", "HFD", "HHC",
    "Hair", "Laundry", "Oral_Care", "PW", "Skin",
)
PACK_TYPES = (
    "Bar", "Bottle", "Box", "Can", "Carton", "Jar", "Pouch",
    "Pump", "Sachet", "Tetrapack", "Tin", "Tub", "Tube", "Wrapper",
)
PACK_PROMPT = (
    "Classify the retail product shown in the image by its category and pack_type. "
    f"Valid categories: {', '.join(PACK_CATEGORIES)}. "
    f"Valid pack_types: {', '.join(PACK_TYPES)}. "
    'Return ONLY JSON like {"category":"Beverages","pack_type":"Carton"}.'
)

BRAND_PROMPT = (
    'Classify the retail product shown in the image by its brand. '
    'Return ONLY JSON like {"brand":"Dove"}.'
)

VARIANT_CATEGORIES = ("gen", "oral", "p5", "skin", "wash")
VARIANT_BRANDS = (
    "3_Roses", "Annapurna", "Ayush", "Boost", "Brown_And_Polson", "Bru", "CIF", "Clear",
    "Clinic_Plus", "Closeup", "Comfort", "Domex", "Dove", "Elle18", "Freshwrapp",
    "Glow_And_Lovely", "Hamam", "Hellmans", "Horlicks", "Indulekha", "Kissan", "Knorr",
    "Lakme", "Lbp", "Lifebuoy", "Lipton", "Liril", "Lux", "Maltova", "Moti",
    "Nature_Protect", "Novology", "Oralb", "Pears", "Pepsodent", "Ponds", "Pukka",
    "Red_Label", "Rex", "Rexona", "Rin", "Ruby", "Simple", "St_Ives", "Sun",
    "Sunlight", "Sunsilk", "Surf_Excel", "Taaza", "Taj_Mahal", "Tresemme", "V_Wash",
    "Vaseline", "Vim", "Viva", "Wellbeing_Nutrition", "Wheel",
)
VARIANT_PROMPT_ALL3 = (
    "Classify the retail product shown in the image by its category, brand, and variant. "
    f"Valid categories: {', '.join(VARIANT_CATEGORIES)}. "
    'Return ONLY JSON like {"category":"gen","brand":"Clear","variant":"Active_Care_Shampoo"}.'
)
VARIANT_PROMPT_CATEGORY = (
    "Classify the retail product shown in the image by its category. "
    f"Valid categories: {', '.join(VARIANT_CATEGORIES)}. "
    'Return ONLY JSON like {"category":"gen"}.'
)
VARIANT_PROMPT_BRAND = (
    "Classify the retail product shown in the image by its brand. "
    f"Valid brands: {', '.join(VARIANT_BRANDS)}. "
    'Return ONLY JSON like {"brand":"Clear"}.'
)
VARIANT_PROMPT_ONLY = (
    "Classify the retail product shown in the image by its brand and variant. "
    f"Valid brands: {', '.join(VARIANT_BRANDS)}. "
    'Return ONLY JSON like {"brand":"Clear","variant":"Active_Care_Shampoo"}.'
)

MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

STRIP_PROMO_ART_SUFFIX_RE = re.compile(
    r"(?:_(?:New|Old|\d*Old_Pack|Square|PO[2-9]|B[1-9]G[1-9]|Pack_Of_\d+|Free_[A-Za-z0-9_]+|Rs_?\d+)(?:_\d+)?)+$",
    re.I,
)
COPY_FILE_RE = re.compile(
    r"(?:^copy[\s_-]*\d*\.(?:jpe?g|png)$|\s-\s*Copy(?:\s*\(\d+\))?|\(copy\)\.(?:jpe?g|png)$|^DA_(?:TBC_)?|(?:^|_)\d+\s+\(\d+\)\.(?:jpe?g|png)$)",
    re.I,
)



def _sft_example(gcs_image_uri: str, prompt: str, response_dict: dict) -> dict:
    """One Vertex AI Gemini SFT JSONL line with a GCS image URI and structured JSON target."""
    ext = Path(gcs_image_uri).suffix.lower()
    mime_type = MIME_BY_EXT.get(ext, "image/jpeg")
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"fileData": {"mimeType": mime_type, "fileUri": gcs_image_uri}},
                    {"text": prompt},
                ],
            },
            {
                "role": "model",
                "parts": [{"text": json.dumps(response_dict, separators=(",", ":"))}],
            },
        ]
    }


def _sft_line(gcs_image_uri: str, prompt: str, response_dict: dict) -> str:
    return json.dumps(_sft_example(gcs_image_uri, prompt, response_dict), separators=(",", ":"))


def _extract_uri(line: str) -> str:
    return json.loads(line)["contents"][0]["parts"][0]["fileData"]["fileUri"]


def clean_brand_name(folder: str) -> str:
    """Strip artificial leading 'A' / 'A_' sort prefixes (e.g. ADove -> Dove, A_3_roses -> 3_roses)."""
    if folder == "A_3_roses":
        return "3_roses"
    if len(folder) > 1 and folder[0] == "A" and folder[1].isupper():
        return folder[1:]
    return folder


def clean_variant_name(raw_variant: str) -> str:
    """Normalize stray spaces/underscores and strip promo/price/artwork suffixes (`_Rs_10`, `_B3G1`, `_PO3`, `_New`, `_Old`)."""
    cleaned = re.sub(r"_+", "_", raw_variant.replace(" ", "_")).strip("_")
    return STRIP_PROMO_ART_SUFFIX_RE.sub("", cleaned) or cleaned


def is_copy_or_synthetic(uri_or_filename: str) -> bool:
    """True when the image filename is a duplicate copy (`copy*.jpg`, `- Copy`, `(copy)`) or synthetic `DA_TBC_*` crop."""
    name = uri_or_filename.rsplit("/", 1)[-1]
    return bool(COPY_FILE_RE.search(name))


def is_copy_uri(uri: str) -> bool:
    """True when the image filename is a duplicate copy (`copy*.jpg`, `- Copy`, `(copy)`) or synthetic `DA_TBC_*` crop."""
    return is_copy_or_synthetic(uri)


def list_gcs_image_blobs(gcs_prefix: str) -> dict[str, str]:
    """Return `{gcs_uri: md5_hash}` under `gcs_prefix` using GCS blob metadata only (never reads bytes)."""
    bucket_name, prefix = dataset._split_gs(gcs_prefix.rstrip("/"))
    blobs = dataset._gcs().list_blobs(bucket_name, prefix=prefix + "/")
    out: dict[str, str] = {}
    for b in blobs:
        if Path(b.name).suffix.lower() in MIME_BY_EXT and "/jsonl/" not in f"/{b.name}":
            uri = f"gs://{bucket_name}/{b.name}"
            out[uri] = b.md5_hash or uri
    return out


def list_gcs_image_uris(gcs_prefix: str) -> list[str]:
    """List image object URIs under `gcs_prefix` using GCS blob metadata only (never reads bytes)."""
    return sorted(list_gcs_image_blobs(gcs_prefix))


def _dedup_uris(
    uris: list[str],
    md5_by_uri: dict[str, str] | None = None,
) -> list[str]:
    """Deduplicate URIs on `b.md5_hash` (falling back to URI when offline), preferring original filenames over copies."""
    ordered = [u for u in uris if not is_copy_or_synthetic(u)] + [
        u for u in uris if is_copy_or_synthetic(u)
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for u in ordered:
        key = (md5_by_uri.get(u) if md5_by_uri else None) or u
        if key not in seen:
            seen.add(key)
            unique.append(u)
    return unique


def _find_conflicted_md5s(
    by_class: dict[Any, list[str]],
    md5_by_uri: dict[str, str] | None = None,
) -> set[str]:
    """Identify `md5_hash` values that appear in more than one distinct class folder."""
    if not md5_by_uri:
        return set()
    md5_to_classes: dict[str, set[Any]] = defaultdict(set)
    for cls_key, uris in by_class.items():
        for u in uris:
            h = md5_by_uri.get(u)
            if h:
                md5_to_classes[h].add(cls_key)
    return {h for h, cls_set in md5_to_classes.items() if len(cls_set) > 1}


def _prioritize_originals(
    uris: list[str],
    rng: random.Random,
    md5_by_uri: dict[str, str] | None = None,
    exclude_md5s: set[str] | None = None,
) -> list[str]:
    """Deduplicate on `b.md5_hash`, drop cross-folder MD5 conflicts, and shuffle originals ahead of copy/synthetic files."""
    unique = _dedup_uris(uris, md5_by_uri)
    if exclude_md5s and md5_by_uri:
        unique = [u for u in unique if md5_by_uri.get(u, u) not in exclude_md5s]
    orig = [u for u in unique if not is_copy_or_synthetic(u)]
    copies = [u for u in unique if is_copy_or_synthetic(u)]
    rng.shuffle(orig)
    rng.shuffle(copies)
    return orig + copies


def _filter_out_val_records(
    all_records: list[str],
    val_records: list[str],
    md5_by_uri: dict[str, str] | None = None,
) -> list[str]:
    """Exclude any validation `fileUri` (and any matching `b.md5_hash`) from `train.jsonl`."""
    val_uris = {_extract_uri(r) for r in val_records}
    val_md5s = {
        md5_by_uri[u] for u in val_uris if md5_by_uri and u in md5_by_uri
    }
    out: list[str] = []
    for r in all_records:
        u = _extract_uri(r)
        if u in val_uris:
            continue
        if md5_by_uri and md5_by_uri.get(u) in val_md5s:
            continue
        out.append(r)
    return out


def _assert_disjoint_train_val(
    split_name: str,
    train_records: list[str],
    val_records: list[str],
    md5_by_uri: dict[str, str] | None = None,
) -> None:
    tr_uris = {_extract_uri(x) for x in train_records}
    vl_uris = {_extract_uri(x) for x in val_records}
    if tr_uris & vl_uris:
        raise RuntimeError(f"URI overlap detected in {split_name}")
    if md5_by_uri:
        tr_md5 = {md5_by_uri.get(u, u) for u in tr_uris}
        vl_md5 = {md5_by_uri.get(u, u) for u in vl_uris}
        if tr_md5 & vl_md5:
            raise RuntimeError(f"MD5 overlap detected in {split_name}")


def _upload_sft_dir(local_dir: str | Path, gcs_dest: str) -> None:
    try:
        dataset.upload(str(local_dir), gcs_dest, overwrite=True)
    except TypeError:
        dataset.upload(str(local_dir), gcs_dest)


def build_marketshare_pack_jsonl(
    gcs_bucket: str = MARKETSHARE_BUCKET,
    out_dir: str | Path = "pack_data/pack/jsonl",
    seed: int = 42,
    val_fraction: float = 0.10,
    min_val_per_class: int = 5,
    max_val_per_class: int = 10,
    max_train_per_class: int = 120,
    upload: bool = True,
    uris: list[str] | None = None,
    md5_by_uri: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build per-folder, full, and stratified SFT JSONL files for `pack_data/pack/train`."""
    train_prefix = f"{gcs_bucket.rstrip('/')}/pack_data/pack/train/"
    gcs_dest = f"{gcs_bucket.rstrip('/')}/pack_data/pack/jsonl"
    if uris is None:
        blobs_map = list_gcs_image_blobs(train_prefix)
        raw_uris = sorted(blobs_map)
        md5_map = md5_by_uri if md5_by_uri is not None else blobs_map
    else:
        raw_uris = sorted(uris)
        md5_map = md5_by_uri

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    by_folder: dict[str, list[str]] = defaultdict(list)
    for uri in raw_uris:
        if not uri.startswith(train_prefix):
            continue
        parts = uri[len(train_prefix) :].split("/")
        if len(parts) == 2 and parts[1] and parts[0] != "jsonl" and "__" in parts[0]:
            by_folder[parts[0]].append(uri)

    conflicted_md5s = _find_conflicted_md5s(by_folder, md5_map)
    rng = random.Random(seed)
    all_records: list[str] = []
    train_stratified: list[str] = []
    val_stratified: list[str] = []

    for folder, f_uris in sorted(by_folder.items()):
        category, pack_type = folder.split("__", 1)
        target = {"category": category, "pack_type": pack_type}
        unique_uris = _dedup_uris(f_uris, md5_map)
        records = [_sft_line(u, PACK_PROMPT, target) for u in unique_uris]
        (out_p / f"{folder}.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
        all_records.extend(records)

        prioritized_uris = _prioritize_originals(unique_uris, rng, md5_map, conflicted_md5s)
        orig_only = [u for u in prioritized_uris if not is_copy_or_synthetic(u)]
        val_pool = orig_only if len(orig_only) >= min_val_per_class else prioritized_uris
        n_val = min(max_val_per_class, max(min_val_per_class, int(len(val_pool) * val_fraction)))
        v_uris = val_pool[:n_val]
        v_set = set(v_uris)
        t_pool = [u for u in prioritized_uris if u not in v_set]
        t_uris = t_pool[:max_train_per_class]
        val_stratified.extend(_sft_line(u, PACK_PROMPT, target) for u in v_uris)
        train_stratified.extend(_sft_line(u, PACK_PROMPT, target) for u in t_uris)

    rng.shuffle(train_stratified)
    rng.shuffle(val_stratified)
    _assert_disjoint_train_val("pack stratified", train_stratified, val_stratified, md5_map)

    full_train_records = _filter_out_val_records(all_records, val_stratified, md5_map)
    _assert_disjoint_train_val("pack full train", full_train_records, val_stratified, md5_map)

    (out_p / "train.jsonl").write_text(
        "\n".join(full_train_records) + ("\n" if full_train_records else ""), encoding="utf-8"
    )
    train_f = out_p / "train_stratified.jsonl"
    val_f = out_p / "val_stratified.jsonl"
    train_f.write_text("\n".join(train_stratified) + ("\n" if train_stratified else ""), encoding="utf-8")
    val_f.write_text("\n".join(val_stratified) + ("\n" if val_stratified else ""), encoding="utf-8")

    if upload:
        _upload_sft_dir(out_p, gcs_dest)
    return {
        "out_dir": out_p,
        "train_jsonl": train_f,
        "val_jsonl": val_f,
        "full_train_count": len(full_train_records),
        "train_count": len(train_stratified),
        "val_count": len(val_stratified),
        "class_count": len(by_folder),
        "gcs_prefix": gcs_dest,
        "train_uri": f"{gcs_dest}/train_stratified.jsonl",
        "val_uri": f"{gcs_dest}/val_stratified.jsonl",
    }


def build_marketshare_brand_jsonl(
    gcs_bucket: str = MARKETSHARE_BUCKET,
    out_dir: str | Path = "brand_data/brand/jsonl",
    seed: int = 42,
    val_per_brand: int = 2,
    train_per_brand: int = 10,
    upload: bool = True,
    uris: list[str] | None = None,
    md5_by_uri: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build per-brand, full, and stratified SFT JSONL files for `brand_data/brand/brand_data`."""
    train_prefix = f"{gcs_bucket.rstrip('/')}/brand_data/brand/brand_data/"
    gcs_dest = f"{gcs_bucket.rstrip('/')}/brand_data/brand/jsonl"
    if uris is None:
        blobs_map = list_gcs_image_blobs(train_prefix)
        raw_uris = sorted(blobs_map)
        md5_map = md5_by_uri if md5_by_uri is not None else blobs_map
    else:
        raw_uris = sorted(uris)
        md5_map = md5_by_uri

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    by_brand: dict[str, list[str]] = defaultdict(list)
    for uri in raw_uris:
        if not uri.startswith(train_prefix):
            continue
        parts = uri[len(train_prefix) :].split("/")
        if len(parts) == 2 and parts[1] and parts[0] != "jsonl":
            by_brand[clean_brand_name(parts[0])].append(uri)

    conflicted_md5s = _find_conflicted_md5s(by_brand, md5_map)
    rng = random.Random(seed)
    all_records: list[str] = []
    train_stratified: list[str] = []
    val_stratified: list[str] = []

    for brand, b_uris in sorted(by_brand.items()):
        unique_uris = _dedup_uris(b_uris, md5_map)
        target = {"brand": brand}
        records = [_sft_line(u, BRAND_PROMPT, target) for u in unique_uris]
        (out_p / f"{brand}.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
        all_records.extend(records)

        prioritized_uris = _prioritize_originals(unique_uris, rng, md5_map, conflicted_md5s)
        orig_only = [u for u in prioritized_uris if not is_copy_or_synthetic(u)]
        val_pool = orig_only if len(orig_only) >= val_per_brand else prioritized_uris
        v_uris = val_pool[:val_per_brand]
        v_set = set(v_uris)
        t_pool = [u for u in prioritized_uris if u not in v_set]
        t_uris = t_pool[:train_per_brand]
        val_stratified.extend(_sft_line(u, BRAND_PROMPT, target) for u in v_uris)
        train_stratified.extend(_sft_line(u, BRAND_PROMPT, target) for u in t_uris)

    rng.shuffle(train_stratified)
    rng.shuffle(val_stratified)
    _assert_disjoint_train_val("brand stratified", train_stratified, val_stratified, md5_map)

    full_train_records = _filter_out_val_records(all_records, val_stratified, md5_map)
    _assert_disjoint_train_val("brand full train", full_train_records, val_stratified, md5_map)

    (out_p / "train.jsonl").write_text(
        "\n".join(full_train_records) + ("\n" if full_train_records else ""), encoding="utf-8"
    )
    train_f = out_p / "train_stratified.jsonl"
    val_f = out_p / "val_stratified.jsonl"
    train_f.write_text("\n".join(train_stratified) + ("\n" if train_stratified else ""), encoding="utf-8")
    val_f.write_text("\n".join(val_stratified) + ("\n" if val_stratified else ""), encoding="utf-8")

    if upload:
        _upload_sft_dir(out_p, gcs_dest)
    return {
        "out_dir": out_p,
        "train_jsonl": train_f,
        "val_jsonl": val_f,
        "full_train_count": len(full_train_records),
        "train_count": len(train_stratified),
        "val_count": len(val_stratified),
        "class_count": len(by_brand),
        "gcs_prefix": gcs_dest,
        "train_uri": f"{gcs_dest}/train_stratified.jsonl",
        "val_uri": f"{gcs_dest}/val_stratified.jsonl",
    }


def build_marketshare_variant_jsonl(
    gcs_bucket: str = MARKETSHARE_BUCKET,
    out_dir: str | Path = "input_data/jsonl",
    seed: int = 42,
    val_per_folder: int = 2,
    train_per_folder: int = 8,
    task: str = "all3",
    upload: bool = True,
    uris: list[str] | None = None,
    md5_by_uri: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build per-folder, full, and multi-task stratified SFT JSONL files for `input_data/variant_*`."""
    input_prefix = f"{gcs_bucket.rstrip('/')}/input_data/"
    gcs_dest = f"{gcs_bucket.rstrip('/')}/input_data/jsonl"
    if uris is None:
        blobs_map: dict[str, str] = {}
        for cat in VARIANT_CATEGORIES:
            blobs_map.update(list_gcs_image_blobs(f"{input_prefix}variant_{cat}/train"))
        raw_uris = sorted(blobs_map)
        md5_map = md5_by_uri if md5_by_uri is not None else blobs_map
    else:
        raw_uris = sorted(uris)
        md5_map = md5_by_uri

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for uri in raw_uris:
        if not uri.startswith(input_prefix):
            continue
        parts = uri[len(input_prefix) :].split("/")
        if len(parts) == 4 and parts[0].startswith("variant_") and parts[1] == "train" and parts[3]:
            cat = parts[0][len("variant_") :]
            folder = parts[2]
            if "__" in folder:
                brand, raw_var = folder.split("__", 1)
                by_key[(cat, brand, clean_variant_name(raw_var))].append(uri)

    conflicted_md5s = _find_conflicted_md5s(by_key, md5_map)
    rng = random.Random(seed)
    all_records: list[str] = []
    train_all3, val_all3 = [], []
    train_cat, val_cat = [], []
    train_var, val_var = [], []
    brand_train_pool: dict[str, list[str]] = defaultdict(list)
    brand_val_pool: dict[str, list[str]] = defaultdict(list)

    for (cat, brand, variant), f_uris in sorted(by_key.items()):
        unique_uris = _dedup_uris(f_uris, md5_map)
        target_all3 = {"category": cat, "brand": brand, "variant": variant}
        target_var = {"brand": brand, "variant": variant}
        records = [_sft_line(u, VARIANT_PROMPT_ALL3, target_all3) for u in unique_uris]
        (out_p / f"{cat}__{brand}__{variant}.jsonl").write_text(
            "\n".join(records) + "\n", encoding="utf-8"
        )
        all_records.extend(records)

        prioritized = _prioritize_originals(unique_uris, rng, md5_map, conflicted_md5s)
        orig_only = [u for u in prioritized if not is_copy_or_synthetic(u)]
        val_pool = orig_only if len(orig_only) >= val_per_folder else prioritized
        v_uris = val_pool[:val_per_folder]
        v_set = set(v_uris)
        t_pool = [u for u in prioritized if u not in v_set]
        t_uris = t_pool[:train_per_folder]

        for u in v_uris:
            val_all3.append(_sft_line(u, VARIANT_PROMPT_ALL3, target_all3))
            val_var.append(_sft_line(u, VARIANT_PROMPT_ONLY, target_var))
        for u in t_uris:
            train_all3.append(_sft_line(u, VARIANT_PROMPT_ALL3, target_all3))
            train_var.append(_sft_line(u, VARIANT_PROMPT_ONLY, target_var))

        for u in v_uris[:1]:
            val_cat.append(_sft_line(u, VARIANT_PROMPT_CATEGORY, {"category": cat}))
        for u in t_uris[:5]:
            train_cat.append(_sft_line(u, VARIANT_PROMPT_CATEGORY, {"category": cat}))

        brand_val_pool[brand].extend(v_uris)
        brand_train_pool[brand].extend(t_pool)

    train_brand, val_brand = [], []
    for brand in sorted(brand_train_pool):
        v_pool = _dedup_uris(brand_val_pool[brand], md5_map)
        v_keys = {(md5_map.get(u) if md5_map else None) or u for u in v_pool}
        t_pool = [
            u for u in _dedup_uris(brand_train_pool[brand], md5_map)
            if ((md5_map.get(u) if md5_map else None) or u) not in v_keys
        ]
        rng.shuffle(v_pool)
        rng.shuffle(t_pool)
        val_brand.extend(_sft_line(u, VARIANT_PROMPT_BRAND, {"brand": brand}) for u in v_pool[:10])
        train_brand.extend(_sft_line(u, VARIANT_PROMPT_BRAND, {"brand": brand}) for u in t_pool[:100])

    for lst in (train_all3, val_all3, train_cat, val_cat, train_brand, val_brand, train_var, val_var):
        rng.shuffle(lst)

    for name, tr, vl in (
        ("all3", train_all3, val_all3),
        ("category", train_cat, val_cat),
        ("brand", train_brand, val_brand),
        ("variant", train_var, val_var),
    ):
        _assert_disjoint_train_val(f"variant {name}", tr, vl, md5_map)

    all_val_records = val_all3 + val_cat + val_brand + val_var
    full_train_records = _filter_out_val_records(all_records, all_val_records, md5_map)
    _assert_disjoint_train_val("variant full train", full_train_records, all_val_records, md5_map)

    (out_p / "train.jsonl").write_text(
        "\n".join(full_train_records) + ("\n" if full_train_records else ""), encoding="utf-8"
    )
    files_by_task = {
        "all3": ("train_stratified.jsonl", "val_stratified.jsonl", train_all3, val_all3),
        "category": ("train_category_stratified.jsonl", "val_category_stratified.jsonl", train_cat, val_cat),
        "brand": ("train_brand_stratified.jsonl", "val_brand_stratified.jsonl", train_brand, val_brand),
        "variant": ("train_variant_stratified.jsonl", "val_variant_stratified.jsonl", train_var, val_var),
    }
    for t_name, v_name, tr, vl in files_by_task.values():
        (out_p / t_name).write_text("\n".join(tr) + ("\n" if tr else ""), encoding="utf-8")
        (out_p / v_name).write_text("\n".join(vl) + ("\n" if vl else ""), encoding="utf-8")

    if upload:
        _upload_sft_dir(out_p, gcs_dest)

    sel_train_name, sel_val_name, sel_tr, sel_vl = files_by_task.get(task, files_by_task["all3"])
    return {
        "out_dir": out_p,
        "train_jsonl": out_p / sel_train_name,
        "val_jsonl": out_p / sel_val_name,
        "full_train_count": len(full_train_records),
        "train_count": len(sel_tr),
        "val_count": len(sel_vl),
        "class_count": len(by_key),
        "gcs_prefix": gcs_dest,
        "train_uri": f"{gcs_dest}/{sel_train_name}",
        "val_uri": f"{gcs_dest}/{sel_val_name}",
    }


def build_sft_jsonl(
    dataset_name: str = "shelves",
    out_dir: str | Path = "data/tuning",
    gcs_prefix: str | None = None,
    seed: int = 0,
    max_val_images: int = 20,
    upload: bool = True,
    task: str = "all3",
) -> dict[str, Any]:
    """Build zero-leakage ``train.jsonl`` and ``val.jsonl`` from non-test data only.

    Supports benchmark datasets (``shelves``, ``rpc``) and metadata-only marketshare datasets
    (``pack``, ``brand``, ``variant``).
    """
    if dataset_name == "pack":
        return build_marketshare_pack_jsonl(
            out_dir="pack_data/pack/jsonl" if out_dir == "data/tuning" else out_dir,
            upload=upload,
        )
    if dataset_name == "brand":
        return build_marketshare_brand_jsonl(
            out_dir="brand_data/brand/jsonl" if out_dir == "data/tuning" else out_dir,
            upload=upload,
        )
    if dataset_name == "variant":
        return build_marketshare_variant_jsonl(
            out_dir="input_data/jsonl" if out_dir == "data/tuning" else out_dir,
            task=task,
            upload=upload,
        )
    if dataset_name not in TUNABLE_DATASETS:
        raise ValueError(
            f"dataset_name must be one of {TUNABLE_DATASETS}, got {dataset_name!r}"
        )

    cfg = load_config()
    tuning_root = cfg.get("gcp", {}).get("tuning", "gs://ul-fs-s-sandbx-16-prj-shelf-images/tuning")
    gcs_prefix = (gcs_prefix or f"{tuning_root.rstrip('/')}/{dataset_name}").rstrip("/")

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
    result: dict[str, Any] = {
        "out_dir": out_p,
        "train_jsonl": train_file,
        "val_jsonl": val_file,
        "train_count": len(train_lines),
        "val_count": len(val_lines),
        "gcs_prefix": gcs_prefix,
    }
    if upload:
        result.update(upload_sft_dataset(out_p, gcs_prefix))
    return result


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
    base_model: str = "gemini-3.5-flash-lite",
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
    project = cfg.get("project", "ul-fs-s-sandbx-16-prj")
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
        project=cfg.get("project", "ul-fs-s-sandbx-16-prj"),
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
