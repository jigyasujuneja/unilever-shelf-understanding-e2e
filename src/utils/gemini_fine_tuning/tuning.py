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
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from approaches.base import crop
from approaches.market_share.retrieval.tiered_hybrid import PROMPT, K, sheet
from utils import dataset
from utils.llm import load_config
from utils.taxonomy.brands import NOISE_BRAND_FOLDERS, canonicalize_brand_name

SUPPORTED_SFT_BASE_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
)

TUNABLE_DATASETS = (
    "shelves",
    "rpc",
    "pack",
    "brand",
    "unified_brand",
    "variant",
    "hul_golden",
    "hul_variants",
    "hul_brands",
    "hul_packs",
    "hul_promos",
)

MARKETSHARE_BUCKET = "gs://marketshare-datasets"
# GCS folders and per-task (train, val) file names of the build_marketshare_*_jsonl outputs.
SFT_JSONL_DIRS = {
    "pack": "pack_data/pack/jsonl",
    "brand": "brand_data/brand/jsonl",
    "unified_brand": "derived_manifests/unified_brand/sft",
    "variant": "input_data/jsonl",
}
SFT_SPLIT_FILES = {
    "all3": ("train_stratified.jsonl", "val_stratified.jsonl"),
    "category": ("train_category_stratified.jsonl", "val_category_stratified.jsonl"),
    "brand": ("train_brand_stratified.jsonl", "val_brand_stratified.jsonl"),
    "variant": ("train_variant_stratified.jsonl", "val_variant_stratified.jsonl"),
}

PACK_CATEGORIES = (
    "Beverages", "DMT", "Foods", "HFD", "HHC",
    "Hair", "Hair_Care", "Laundry", "Oral_Care", "PW", "Skin",
)
PACK_TYPES = (
    "Bar", "Bottle", "Box", "Can", "Carton", "Jar", "Pouch",
    "Pump", "Roll_On", "Sachet", "Spray", "Tetrapack", "Tin", "Tub", "Tube", "Wrapper",
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

UNIFIED_BRAND_PROMPT = (
    "Classify the retail product shown in the image by its canonical brand and Unilever (HUL) ownership. "
    "Output canonical Title_Case_With_Underscores brand names (e.g. Dove, Surf_Excel, Red_Label, Clinic_Plus, "
    "Nescafe, Maggi, Others). "
    'Return ONLY JSON like {"brand":"Dove","is_hul":true}.'
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

_SKIP_NOISE_BRANDS = {b.lower() for b in NOISE_BRAND_FOLDERS}


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
    """Strip artificial leading 'A' / 'A_' sort prefixes and apply canonical brand aliases."""
    return canonicalize_brand_name(folder)


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


def _delete_stale_gcs_blobs(gcs_prefix: str, stale_names: set[str] | None = None) -> int:
    """Remove obsolete or typo-named JSONL blobs under `gcs_prefix`."""
    bucket_name, prefix = dataset._split_gs(gcs_prefix.rstrip("/"))
    bkt = dataset._gcs().bucket(bucket_name)
    deleted = 0
    for b in dataset._gcs().list_blobs(bucket_name, prefix=prefix + "/"):
        fname = b.name.rsplit("/", 1)[-1]
        if stale_names and fname in stale_names:
            try:
                bkt.blob(b.name).delete()
                deleted += 1
            except Exception:
                pass
    return deleted


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
    clean_local: bool = False,
) -> dict[str, Any]:
    """Build per-folder, full, and stratified SFT JSONL files for `pack_data/pack/train`."""

    train_prefix = f"{gcs_bucket.rstrip('/')}/pack_data/pack/train/"
    gcs_dest = f"{gcs_bucket.rstrip('/')}/{SFT_JSONL_DIRS['pack']}"
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
    train_f, val_f = (out_p / f for f in SFT_SPLIT_FILES["all3"])
    train_f.write_text("\n".join(train_stratified) + ("\n" if train_stratified else ""), encoding="utf-8")
    val_f.write_text("\n".join(val_stratified) + ("\n" if val_stratified else ""), encoding="utf-8")

    if upload:
        _upload_sft_dir(out_p, gcs_dest)
        if clean_local:
            shutil.rmtree(out_p, ignore_errors=True)
    return {
        "out_dir": out_p,
        "train_jsonl": train_f,
        "val_jsonl": val_f,
        "full_train_count": len(full_train_records),
        "train_count": len(train_stratified),
        "val_count": len(val_stratified),
        "class_count": len(by_folder),
        "gcs_prefix": gcs_dest,
        "train_uri": f"{gcs_dest}/{train_f.name}",
        "val_uri": f"{gcs_dest}/{val_f.name}",
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
    clean_local: bool = False,
) -> dict[str, Any]:
    """Build per-brand, full, and stratified SFT JSONL files for `brand_data/brand/brand_data`."""

    train_prefix = f"{gcs_bucket.rstrip('/')}/brand_data/brand/brand_data/"
    gcs_dest = f"{gcs_bucket.rstrip('/')}/{SFT_JSONL_DIRS['brand']}"
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
            raw_folder = parts[0].strip()
            if raw_folder.lower() in _SKIP_NOISE_BRANDS:
                continue
            canon_b = clean_brand_name(raw_folder)
            if canon_b.lower() in _SKIP_NOISE_BRANDS:
                continue
            by_brand[canon_b].append(uri)

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
    train_f, val_f = (out_p / f for f in SFT_SPLIT_FILES["all3"])
    train_f.write_text("\n".join(train_stratified) + ("\n" if train_stratified else ""), encoding="utf-8")
    val_f.write_text("\n".join(val_stratified) + ("\n" if val_stratified else ""), encoding="utf-8")

    if upload:
        _delete_stale_gcs_blobs(
            gcs_dest,
            stale_names={
                "Others3.jsonl",
                "Hair_Care.jsonl",
                "Twinnings.jsonl",
                "3_roses.jsonl",
                "temp_train_stratified_100.jsonl",
                "temp_val_stratified_10.jsonl",
            },
        )
        _upload_sft_dir(out_p, gcs_dest)
        if clean_local:
            shutil.rmtree(out_p, ignore_errors=True)
    return {
        "out_dir": out_p,
        "train_jsonl": train_f,
        "val_jsonl": val_f,
        "full_train_count": len(full_train_records),
        "train_count": len(train_stratified),
        "val_count": len(val_stratified),
        "class_count": len(by_brand),
        "gcs_prefix": gcs_dest,
        "train_uri": f"{gcs_dest}/{train_f.name}",
        "val_uri": f"{gcs_dest}/{val_f.name}",
    }


def build_unified_brand_sft_jsonl(
    gcs_bucket: str = MARKETSHARE_BUCKET,
    out_dir: str | Path = "data/tuning/unified_brand",
    seed: int = 42,
    val_per_brand: int = 3,
    train_per_brand: int = 10,
    upload: bool = True,
    manifests: list[dict[str, Any]] | None = None,
    md5_by_uri: dict[str, str] | None = None,
    clean_local: bool = False,
) -> dict[str, Any]:
    """Build zero-leakage ~950-brand SFT JSONL files across all canonical HUL + Competitor + Others brands.

    Combines ``hul_golden`` (gallery + train + val) and ``hul_brands`` (train + val) while strictly
    excluding every ``test`` and ``test_5k`` URI and every held-out store-visit ``session_id``.
    """
    from utils.taxonomy.brands import verify_brand_ownership
    from utils.taxonomy.filenames import parse_crop_filename

    if manifests is None:
        manifests = [
            dataset._rpc_json(dataset.data_root("hul_brands")),
            dataset._rpc_json(dataset.data_root("hul_golden")),
        ]

    gcs_dest = f"{gcs_bucket.rstrip('/')}/{SFT_JSONL_DIRS['unified_brand']}"
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    def _session_of_item(item: dict[str, Any]) -> str:
        sid = str(item.get("session_id") or "").strip()
        if sid:
            return sid
        uri = str(item.get("uri") or "")
        if not uri:
            return ""
        try:
            return str(parse_crop_filename(uri.rsplit("/", 1)[-1]).get("session_id") or "")
        except Exception:
            return ""

    def _resolve_brand_and_hul(lbl: dict[str, Any] | None) -> tuple[str, bool] | None:
        if not lbl:
            return None
        raw_b = str(lbl.get("brand") or "").strip()
        if not raw_b or raw_b.lower() in _SKIP_NOISE_BRANDS or raw_b.lower() in ("unknown", "none", "null", "na"):
            return None
        if raw_b.lower() in ("others", "other"):
            return "Others", False
        canon_b = clean_brand_name(raw_b)
        if not canon_b or canon_b.lower() in _SKIP_NOISE_BRANDS or canon_b == "Unknown":
            return None
        is_hul = bool(lbl["is_hul"]) if "is_hul" in lbl else bool(verify_brand_ownership(canon_b)["is_hul"])
        return canon_b, is_hul

    held_out_uris: set[str] = set()
    held_out_sessions: set[str] = set()
    for mf in manifests:
        splits = mf.get("splits") or {}
        for sp in ("test", "test_5k"):
            for it in splits.get(sp) or []:
                u = str(it.get("uri") or "")
                if u:
                    held_out_uris.add(u)
                sid = _session_of_item(it)
                if sid:
                    held_out_sessions.add(sid)
        for it in splits.get("val") or []:
            sid = _session_of_item(it)
            if sid:
                held_out_sessions.add(sid)

    val_by_brand: dict[str, list[str]] = defaultdict(list)
    train_by_brand: dict[str, list[str]] = defaultdict(list)
    hul_by_brand: dict[str, bool] = {}
    seen_val_uris: set[str] = set()
    seen_train_uris: set[str] = set()

    # 1. Collect validation crops strictly from val splits (disjoint from test / test_5k)
    for mf in manifests:
        classes = mf.get("classes") or {}
        for it in (mf.get("splits") or {}).get("val") or []:
            u = str(it.get("uri") or "")
            if not u or not u.startswith("gs://") or u in held_out_uris or u in seen_val_uris:
                continue
            lbls = it.get("labels") or [classes.get(str(p)) for p in (it.get("products") or [])]
            resolved = _resolve_brand_and_hul(lbls[0] if lbls else None)
            if not resolved:
                continue
            canon_b, is_hul = resolved
            hul_by_brand.setdefault(canon_b, is_hul)
            val_by_brand[canon_b].append(u)
            seen_val_uris.add(u)

    # 2. Collect training crops from gallery + train splits (disjoint from val, test, test_5k, and held-out sessions)
    for mf in manifests:
        classes = mf.get("classes") or {}
        for pid_s, uris in (mf.get("gallery") or {}).items():
            resolved = _resolve_brand_and_hul(classes.get(str(pid_s)))
            if not resolved:
                continue
            canon_b, is_hul = resolved
            hul_by_brand.setdefault(canon_b, is_hul)
            for u in uris:
                u_str = str(u or "")
                if not u_str.startswith("gs://") or u_str in held_out_uris or u_str in seen_val_uris or u_str in seen_train_uris:
                    continue
                train_by_brand[canon_b].append(u_str)
                seen_train_uris.add(u_str)
        for it in (mf.get("splits") or {}).get("train") or []:
            u = str(it.get("uri") or "")
            if not u or not u.startswith("gs://") or u in held_out_uris or u in seen_val_uris or u in seen_train_uris:
                continue
            sid = _session_of_item(it)
            if sid and sid in held_out_sessions:
                continue
            lbls = it.get("labels") or [classes.get(str(p)) for p in (it.get("products") or [])]
            resolved = _resolve_brand_and_hul(lbls[0] if lbls else None)
            if not resolved:
                continue
            canon_b, is_hul = resolved
            hul_by_brand.setdefault(canon_b, is_hul)
            train_by_brand[canon_b].append(u)
            seen_train_uris.add(u)

    all_brands = sorted(set(train_by_brand) | set(val_by_brand))
    conflicted_md5s = _find_conflicted_md5s(train_by_brand, md5_by_uri) | _find_conflicted_md5s(val_by_brand, md5_by_uri)
    rng = random.Random(seed)
    all_train_records: list[str] = []
    train_stratified: list[str] = []
    val_stratified: list[str] = []

    for brand in all_brands:
        target = {"brand": brand, "is_hul": bool(hul_by_brand.get(brand, False))}
        v_raw = _prioritize_originals(val_by_brand.get(brand, []), rng, md5_by_uri, conflicted_md5s)
        t_raw = _prioritize_originals(train_by_brand.get(brand, []), rng, md5_by_uri, conflicted_md5s)

        v_uris = v_raw[:val_per_brand]
        v_set = set(v_uris)
        t_clean = [u for u in t_raw if u not in v_set]
        t_uris = t_clean[:train_per_brand]

        val_stratified.extend(_sft_line(u, UNIFIED_BRAND_PROMPT, target) for u in v_uris)
        train_stratified.extend(_sft_line(u, UNIFIED_BRAND_PROMPT, target) for u in t_uris)
        all_train_records.extend(_sft_line(u, UNIFIED_BRAND_PROMPT, target) for u in t_clean)

    rng.shuffle(train_stratified)
    rng.shuffle(val_stratified)
    _assert_disjoint_train_val("unified_brand stratified", train_stratified, val_stratified, md5_by_uri)

    full_train_records = _filter_out_val_records(all_train_records, val_stratified, md5_by_uri)
    _assert_disjoint_train_val("unified_brand full train", full_train_records, val_stratified, md5_by_uri)

    out_uris = {_extract_uri(r) for r in full_train_records} | {_extract_uri(r) for r in val_stratified}
    if out_uris & held_out_uris:
        raise RuntimeError(f"Held-out test leakage detected in unified_brand SFT: {len(out_uris & held_out_uris)} URIs")

    (out_p / "train.jsonl").write_text(
        "\n".join(full_train_records) + ("\n" if full_train_records else ""), encoding="utf-8"
    )
    train_f, val_f = (out_p / f for f in SFT_SPLIT_FILES["all3"])
    train_f.write_text("\n".join(train_stratified) + ("\n" if train_stratified else ""), encoding="utf-8")
    val_f.write_text("\n".join(val_stratified) + ("\n" if val_stratified else ""), encoding="utf-8")

    if upload:
        _upload_sft_dir(out_p, gcs_dest)
        if clean_local:
            shutil.rmtree(out_p, ignore_errors=True)

    return {
        "out_dir": out_p,
        "train_jsonl": train_f,
        "val_jsonl": val_f,
        "full_train_count": len(full_train_records),
        "train_count": len(train_stratified),
        "val_count": len(val_stratified),
        "class_count": len(all_brands),
        "hul_brand_count": sum(1 for b in all_brands if hul_by_brand.get(b)),
        "non_hul_brand_count": sum(1 for b in all_brands if not hul_by_brand.get(b)),
        "gcs_prefix": gcs_dest,
        "train_uri": f"{gcs_dest}/{train_f.name}",
        "val_uri": f"{gcs_dest}/{val_f.name}",
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
    clean_local: bool = False,
) -> dict[str, Any]:
    """Build per-folder, full, and multi-task stratified SFT JSONL files for `input_data/variant_*`."""

    input_prefix = f"{gcs_bucket.rstrip('/')}/input_data/"
    gcs_dest = f"{gcs_bucket.rstrip('/')}/{SFT_JSONL_DIRS['variant']}"
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
                raw_brand, raw_var = folder.split("__", 1)
                brand = clean_brand_name(raw_brand)
                if brand.lower() in _SKIP_NOISE_BRANDS:
                    continue
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
        is_generic_others_variant = variant.lower() in ("others", "other")
        target_all3 = {"category": cat, "brand": brand, "variant": variant}
        target_var = {"brand": brand, "variant": variant}
        if not is_generic_others_variant:
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

        if not is_generic_others_variant:
            for u in v_uris:
                val_all3.append(_sft_line(u, VARIANT_PROMPT_ALL3, target_all3))
                val_var.append(_sft_line(u, VARIANT_PROMPT_ONLY, target_var))
            for u in t_uris:
                train_all3.append(_sft_line(u, VARIANT_PROMPT_ALL3, target_all3))
                train_var.append(_sft_line(u, VARIANT_PROMPT_ONLY, target_var))

        # Keep <Brand>__Others in category and brand SFT pools where category and brand are 100% verified
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
        "all3": (*SFT_SPLIT_FILES["all3"], train_all3, val_all3),
        "category": (*SFT_SPLIT_FILES["category"], train_cat, val_cat),
        "brand": (*SFT_SPLIT_FILES["brand"], train_brand, val_brand),
        "variant": (*SFT_SPLIT_FILES["variant"], train_var, val_var),
    }
    for t_name, v_name, tr, vl in files_by_task.values():
        (out_p / t_name).write_text("\n".join(tr) + ("\n" if tr else ""), encoding="utf-8")
        (out_p / v_name).write_text("\n".join(vl) + ("\n" if vl else ""), encoding="utf-8")

    if upload:
        _upload_sft_dir(out_p, gcs_dest)
        if clean_local:
            shutil.rmtree(out_p, ignore_errors=True)

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


def resolve_sft_dataset_uris(
    dataset_name: str = "pack",
    task: str = "all3",
    gcs_bucket: str = MARKETSHARE_BUCKET,
) -> dict[str, Any]:
    """GCS ``train_uri`` / ``val_uri`` of already-built SFT JSONL (``tune --launch-only``)."""
    bkt = gcs_bucket.rstrip("/")
    ds = dataset_name.strip().lower()
    if ds == "hul_golden":  # hul_golden_sft.build_clean_splits_in_gcp
        from utils.gemini_fine_tuning.hul_golden_sft import HulGoldenSftConfig

        dest = f"{bkt}/{HulGoldenSftConfig.output_prefix}"
        return {"train_uri": f"{dest}/train_clean.jsonl", "val_uri": f"{dest}/val_clean.jsonl"}
    if ds not in SFT_JSONL_DIRS:
        raise ValueError(f"Unsupported marketshare SFT dataset {dataset_name!r}")
    train_name, val_name = SFT_SPLIT_FILES[task if ds == "variant" and task in SFT_SPLIT_FILES else "all3"]
    dest = f"{bkt}/{SFT_JSONL_DIRS[ds]}"
    return {"train_uri": f"{dest}/{train_name}", "val_uri": f"{dest}/{val_name}"}


def _subset_single_gcs_jsonl(src_uri: str, max_lines: int, dest_uri: str | None = None) -> tuple[str, int]:
    """Slice the first `max_lines` JSONL records from `src_uri` and upload to `dest_uri` in GCS without local disk writes."""
    bucket_name, obj_path = dataset._split_gs(src_uri)
    if not dest_uri:
        parent = obj_path.rsplit("/", 1)[0] if "/" in obj_path else ""
        stem = Path(obj_path).stem
        dest_obj = f"{parent}/temp_{stem}_{max_lines}.jsonl" if parent else f"temp_{stem}_{max_lines}.jsonl"
        dest_uri = f"gs://{bucket_name}/{dest_obj}"
    dest_bucket_name, dest_obj_path = dataset._split_gs(dest_uri)
    client = dataset._gcs()
    src_blob = client.bucket(bucket_name).blob(obj_path)
    raw_text = src_blob.download_as_text(encoding="utf-8")
    lines = [ln for ln in raw_text.splitlines() if ln.strip()][:max_lines]
    payload = "\n".join(lines) + ("\n" if lines else "")
    dest_blob = client.bucket(dest_bucket_name).blob(dest_obj_path)
    dest_blob.upload_from_string(payload, content_type="application/jsonl")
    return dest_uri, len(lines)


def subset_gcs_sft_jsonl(
    train_uri: str,
    val_uri: str | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
) -> dict[str, Any]:
    """Create temporary subset SFT JSONL files directly in GCS from existing `train_uri` and `val_uri`."""
    out_train_uri = train_uri
    train_count = None
    if max_train is not None and max_train > 0:
        out_train_uri, train_count = _subset_single_gcs_jsonl(train_uri, max_train)
    out_val_uri = val_uri
    val_count = None
    if val_uri and max_val is not None and max_val > 0:
        out_val_uri, val_count = _subset_single_gcs_jsonl(val_uri, max_val)
    return {
        "train_uri": out_train_uri,
        "val_uri": out_val_uri,
        "train_count": train_count,
        "val_count": val_count,
    }


def build_sft_jsonl(
    dataset_name: str = "shelves",
    out_dir: str | Path = "data/tuning",
    gcs_prefix: str | None = None,
    seed: int = 0,
    max_val_images: int = 20,
    upload: bool = True,
    task: str = "all3",
    clean_local: bool = False,
) -> dict[str, Any]:
    """Build zero-leakage ``train.jsonl`` and ``val.jsonl`` from non-test data only.

    Supports benchmark datasets (``shelves``, ``rpc``) and metadata-only marketshare datasets
    (``pack``, ``brand``, ``variant``).
    """

    if dataset_name == "pack":
        return build_marketshare_pack_jsonl(
            out_dir="pack_data/pack/jsonl" if out_dir == "data/tuning" else out_dir,
            upload=upload,
            clean_local=clean_local,
        )
    if dataset_name == "brand":
        return build_marketshare_brand_jsonl(
            out_dir="brand_data/brand/jsonl" if out_dir == "data/tuning" else out_dir,
            upload=upload,
            clean_local=clean_local,
        )
    if dataset_name == "unified_brand":
        return build_unified_brand_sft_jsonl(
            out_dir="data/tuning/unified_brand" if out_dir == "data/tuning" else out_dir,
            upload=upload,
            clean_local=clean_local,
        )
    if dataset_name == "variant":
        return build_marketshare_variant_jsonl(
            out_dir="input_data/jsonl" if out_dir == "data/tuning" else out_dir,
            task=task,
            upload=upload,
            clean_local=clean_local,
        )
    if dataset_name not in TUNABLE_DATASETS:
        raise ValueError(
            f"dataset_name must be one of {TUNABLE_DATASETS}, got {dataset_name!r}"
        )

    cfg = load_config()
    tuning_root = cfg.get("gcp", {}).get("tuning", "gs://ul-fs-s-sandbx-16-prj-shelf-images/tuning")
    use_clef_contrastive = task == "clef_contrastive"
    ds_folder = f"{dataset_name}_clef_contrastive" if use_clef_contrastive else dataset_name
    gcs_prefix = (gcs_prefix or f"{tuning_root.rstrip('/')}/{ds_folder}").rstrip("/")

    root = dataset.data_root(dataset_name)
    cat = dataset.catalog(root, dataset_name)
    gallery = dataset.rpc_gallery(root)
    eff_val_images = max(max_val_images, 50) if use_clef_contrastive else max_val_images
    val_samples = dataset.sample_images("val", eff_val_images, seed, root, dataset_name)

    out_p = Path(out_dir) / ds_folder
    sheets_dir = out_p / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    all_pids = sorted(gallery.keys())

    # 1. Build TRAIN examples strictly from multi-view gallery crops (never scored in test)
    from concurrent.futures import ThreadPoolExecutor

    from approaches.market_share.retrieval.clef_standalone import (
        CLEF_ZOOM_SINGLE_PROMPT,
        build_contrastive_candidate_rubric,
    )
    from utils.clm import zoom_sheet

    sheet_fn = zoom_sheet if use_clef_contrastive else sheet

    def _make_prompt_and_target(cand_pids: list[int], correct_choice: int, gt_pid: int | None) -> tuple[str, dict[str, Any]]:
        if not use_clef_contrastive:
            return PROMPT, {"choice": correct_choice}
        base_p = CLEF_ZOOM_SINGLE_PROMPT.format(k=len(cand_pids))
        top_stub = [(1.0, c_pid, gallery[c_pid][0]) for c_pid in cand_pids]
        rubric = build_contrastive_candidate_rubric(top_stub, cat)
        full_p = f"{base_p}\n\n{rubric}" if rubric else base_p
        ps = str((cat.get(gt_pid) or {}).get("pack_size") or "") if gt_pid is not None and correct_choice > 0 else ""
        return full_p, {"pack_size": ps, "choice": correct_choice}

    # Select stratified multi-view SKUs across brands/categories so large catalogs (2,659 SKUs)
    # produce a balanced ~600-example LoRA dataset without downloading 10,000 unused images.
    multi_pids = [p for p in all_pids if len(gallery[p]) >= 2]
    max_train_pids = 600 if use_clef_contrastive else 800
    if len(multi_pids) > max_train_pids:
        by_brand: dict[str, list[int]] = defaultdict(list)
        for p in multi_pids:
            b = str((cat.get(p) or {}).get("brand") or "Unknown")
            by_brand[b].append(p)
        for b_list in by_brand.values():
            rng.shuffle(b_list)
        selected_pids: list[int] = []
        brand_keys = sorted(by_brand.keys())
        round_idx = 0
        while len(selected_pids) < max_train_pids:
            added_any = False
            for b in brand_keys:
                if round_idx < len(by_brand[b]):
                    selected_pids.append(by_brand[b][round_idx])
                    added_any = True
                    if len(selected_pids) >= max_train_pids:
                        break
            if not added_any:
                break
            round_idx += 1
        train_pids = sorted(selected_pids)
    else:
        train_pids = multi_pids

    train_specs: list[tuple[int, int, str, str, list[int], int]] = []
    needed_paths: set[str] = set()
    for pid in train_pids:
        views = gallery[pid]
        max_views_per_pid = 1 if len(train_pids) >= 400 else len(views)
        for v_idx, q_path in enumerate(views[:max_views_per_pid]):
            pos_path = views[(v_idx + 1) % len(views)]
            distractor_pids = [p for p in all_pids if p != pid]
            p_meta = cat.get(pid) or {}
            p_brand = p_meta.get("brand")
            p_cat = p_meta.get("category")
            same_brand = [p for p in distractor_pids if p_brand and (cat.get(p) or {}).get("brand") == p_brand]
            same_cat = [p for p in distractor_pids if p_cat and (cat.get(p) or {}).get("category") == p_cat and p not in same_brand]
            chosen_neg: list[int] = []
            if same_brand:
                chosen_neg.extend(rng.sample(same_brand, min(K - 1, len(same_brand))))
            if len(chosen_neg) < K - 1 and same_cat:
                rem = (K - 1) - len(chosen_neg)
                chosen_neg.extend(rng.sample(same_cat, min(rem, len(same_cat))))
            if len(chosen_neg) < K - 1:
                rem = (K - 1) - len(chosen_neg)
                rest = [p for p in distractor_pids if p not in chosen_neg]
                chosen_neg.extend(rng.sample(rest, min(rem, len(rest))))
            cand_pids = [pid, *chosen_neg]
            rng.shuffle(cand_pids)
            correct_choice = cand_pids.index(pid) + 1
            train_specs.append((pid, v_idx, q_path, pos_path, cand_pids, correct_choice))
            needed_paths.add(q_path)
            needed_paths.add(pos_path)
            for c_pid in cand_pids:
                needed_paths.add(gallery[c_pid][0])

    # Pre-plan val specs so we know all needed gallery image paths up front
    val_specs: list[tuple[Any, int, Any, dict[str, Any], list[int], int]] = []
    for sample in val_samples:
        for b_idx, (box, lab) in enumerate(zip(sample.boxes, sample.labels, strict=True)):
            gt_pid = lab["sku_id"]
            distractor_pids = [p for p in all_pids if p != gt_pid]
            if not distractor_pids:
                continue
            if gt_pid in gallery:
                g_meta = cat.get(gt_pid) or {}
                g_brand = g_meta.get("brand")
                g_cat = g_meta.get("category")
                same_brand = [p for p in distractor_pids if g_brand and (cat.get(p) or {}).get("brand") == g_brand]
                same_cat = [p for p in distractor_pids if g_cat and (cat.get(p) or {}).get("category") == g_cat and p not in same_brand]
                chosen_neg = []
                if same_brand:
                    chosen_neg.extend(rng.sample(same_brand, min(K - 1, len(same_brand))))
                if len(chosen_neg) < K - 1 and same_cat:
                    rem = (K - 1) - len(chosen_neg)
                    chosen_neg.extend(rng.sample(same_cat, min(rem, len(same_cat))))
                if len(chosen_neg) < K - 1:
                    rem = (K - 1) - len(chosen_neg)
                    rest = [p for p in distractor_pids if p not in chosen_neg]
                    chosen_neg.extend(rng.sample(rest, min(rem, len(rest))))
                cand_pids = [gt_pid, *chosen_neg]
                rng.shuffle(cand_pids)
                correct_choice = cand_pids.index(gt_pid) + 1
            else:
                cand_pids = rng.sample(distractor_pids, min(K, len(distractor_pids)))
                correct_choice = 0
            val_specs.append((sample, b_idx, box, lab, cand_pids, correct_choice))
            for c_pid in cand_pids:
                needed_paths.add(gallery[c_pid][0])

    def _load_rgb(path_str: str) -> tuple[str, Image.Image]:
        with Image.open(io.BytesIO(dataset.read_bytes(path_str))) as im:
            return path_str, im.convert("RGB")

    ref_imgs: dict[str, Image.Image] = {}
    with ThreadPoolExecutor(max_workers=24) as pool:
        for p_str, img_obj in pool.map(_load_rgb, sorted(needed_paths)):
            ref_imgs[p_str] = img_obj

    train_lines: list[str] = []
    for pid, v_idx, q_path, pos_path, cand_pids, correct_choice in train_specs:
        refs = [
            ref_imgs[pos_path if c_pid == pid else gallery[c_pid][0]]
            for c_pid in cand_pids
        ]
        sheet_name = f"train_{pid}_{v_idx}.jpg"
        sheet_path = sheets_dir / sheet_name
        sheet_fn(ref_imgs[q_path], refs).save(sheet_path, format="JPEG", quality=88)
        prompt_str, target_dict = _make_prompt_and_target(cand_pids, correct_choice, pid)
        ex = _sft_example(
            f"{gcs_prefix}/sheets/{sheet_name}",
            prompt_str,
            target_dict,
        )
        train_lines.append(json.dumps(ex))

    # 2. Build VAL examples strictly from the `val` split (zero `test` split leakage)
    val_lines: list[str] = []
    val_full_imgs: dict[str, Image.Image] = {}
    unique_val_sample_paths = sorted({s.path for s, *_ in val_specs})
    with ThreadPoolExecutor(max_workers=16) as pool:
        for p_str, img_obj in pool.map(_load_rgb, unique_val_sample_paths):
            val_full_imgs[p_str] = img_obj

    for sample, b_idx, box, lab, cand_pids, correct_choice in val_specs:
        gt_pid = lab["sku_id"]
        q_crop = crop(val_full_imgs[sample.path], box)
        refs = [ref_imgs[gallery[c_pid][0]] for c_pid in cand_pids]
        sheet_name = f"val_{Path(sample.image_id).stem}_{b_idx}.jpg"
        sheet_path = sheets_dir / sheet_name
        sheet_fn(q_crop, refs).save(sheet_path, format="JPEG", quality=88)
        prompt_str, target_dict = _make_prompt_and_target(cand_pids, correct_choice, gt_pid)
        ex = _sft_example(
            f"{gcs_prefix}/sheets/{sheet_name}",
            prompt_str,
            target_dict,
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
    if clean_local and out_p.exists():
        shutil.rmtree(out_p, ignore_errors=True)
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
    region: str | None = None,
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
    target_region = region or ("us" if base_model.startswith("gemini-3.5") else cfg.get("region", "us-central1"))
    client = genai.Client(vertexai=True, project=project, location=target_region)

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

    try:
        job = client.tunings.tune(
            base_model=base_model,
            training_dataset=types.TuningDataset(gcs_uri=train_uri),
            config=tune_cfg,
        )
    except Exception as exc:
        if "not supported" in str(exc).lower() and target_region not in ("us", "eu"):
            target_region = "eu" if "europe" in target_region else "us"
            client = genai.Client(vertexai=True, project=project, location=target_region)
            job = client.tunings.tune(
                base_model=base_model,
                training_dataset=types.TuningDataset(gcs_uri=train_uri),
                config=tune_cfg,
            )
        else:
            raise
    endpoint = getattr(getattr(job, "tuned_model", None), "endpoint", None)
    return {
        "name": job.name,
        "state": getattr(job.state, "name", str(job.state)),
        "base_model": base_model,
        "display_name": display_name,
        "region": target_region,
        "train_uri": train_uri,
        "val_uri": val_uri,
        "endpoint": endpoint,
    }


def list_sft_jobs(config: dict | None = None, region: str | None = None) -> list[dict]:
    """List Vertex AI Supervised Fine-Tuning jobs in the configured GCP project and region(s)."""
    os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
    from google import genai

    cfg = (config or load_config()).get("gcp", {})
    project = cfg.get("project", "ul-fs-s-sandbx-16-prj")
    regions = [region] if region else ["us", "eu", cfg.get("region", "us-central1"), "us-east4", "europe-west4"]
    out: list[dict] = []
    seen_regions: set[str] = set()
    for reg in regions:
        if reg in seen_regions:
            continue
        seen_regions.add(reg)
        client = genai.Client(vertexai=True, project=project, location=reg)
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
