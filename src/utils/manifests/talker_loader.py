"""Loader & Golden Split Builder for ``gs://ms-json-dataset/toker_dataset/mar_talker_final/`` (``hul_talkers_golden``).

Rebuilds the 4,767 retail shelf-talker and price-tag images across 532 outlets into a
leakage-free 80/10/10 outlet-partitioned Golden benchmark (``hul_talkers_golden``).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

OUTLET_DATE_RE = re.compile(r"_(\d{6,12})_(\d{4}-\d{2}-\d{2})_")

TALKER_CLASSES: dict[str, dict[str, Any]] = {
    "1": {
        "sku_id": 1,
        "product": "Promo_Talker__Shelf_Talker_Card",
        "brand": "Unilever_Promo",
        "variant": "Shelf_Talker_Card",
        "base_variant": "Shelf_Talker_Card",
        "basepack_id": "Unilever_Promo__Shelf_Talker_Card__Carton",
        "category": "Promo_Talker",
        "packaging_type": "Carton",
        "allowed_packaging_types": ["Carton"],
        "is_hul": True,
        "has_promo": True,
        "promo_type": "Shelf_Talker",
    },
    "2": {
        "sku_id": 2,
        "product": "Store_Rail__Printed_Price_Tag",
        "brand": "Store_Rail",
        "variant": "Printed_Price_Tag",
        "base_variant": "Printed_Price_Tag",
        "basepack_id": "Store_Rail__Printed_Price_Tag__Wrapper",
        "category": "Promo_Talker",
        "packaging_type": "Wrapper",
        "allowed_packaging_types": ["Wrapper"],
        "is_hul": False,
        "has_promo": True,
        "promo_type": "Price_Tag",
    },
    "3": {
        "sku_id": 3,
        "product": "Store_Rail__Handwritten_Price_Tag",
        "brand": "Store_Rail",
        "variant": "Handwritten_Price_Tag",
        "base_variant": "Handwritten_Price_Tag",
        "basepack_id": "Store_Rail__Handwritten_Price_Tag__Wrapper",
        "category": "Promo_Talker",
        "packaging_type": "Wrapper",
        "allowed_packaging_types": ["Wrapper"],
        "is_hul": False,
        "has_promo": True,
        "promo_type": "Handwritten_Tag",
    },
}


def extract_talker_outlet_code(filename_or_stem: str) -> str:
    """Extract the 6-12 digit supermarket ``outlet_code`` from a talker filename."""
    stem = PurePosixPath(filename_or_stem).stem
    m = OUTLET_DATE_RE.search(stem)
    if m:
        return m.group(1)
    parts = stem.split("_")
    for p in parts:
        if p.isdigit() and 6 <= len(p) <= 12:
            return p
    return f"OUTLET_{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:8]}"


def _exif_transposed_dimensions_from_blob(jpg_blob: Any) -> tuple[int, int]:
    """Read EXIF-transposed (width, height) from JPEG header bytes, falling back to full download if needed."""
    try:
        head_bytes = jpg_blob.download_as_bytes(start=0, end=65535)
        with Image.open(io.BytesIO(head_bytes)) as im:
            w, h = im.size
            exif = im.getexif()
            orientation = int(exif.get(274, 1)) if exif else 1
            if orientation in (5, 6, 7, 8):
                return int(h), int(w)
            return int(w), int(h)
    except Exception:
        raw_bytes = jpg_blob.download_as_bytes()
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im_t = ImageOps.exif_transpose(im)
            w, h = im_t.size
            return int(w), int(h)


def _yolo_lines_to_boxes_and_classes(
    txt_content: str,
    width: int,
    height: int,
) -> tuple[list[list[float]], list[int], list[dict[str, Any]]]:
    """Convert YOLO normalized ``[cls, xc, yc, w, h]`` lines into absolute pixel ``[x1, y1, x2, y2]``."""
    boxes: list[list[float]] = []
    products: list[int] = []
    labels: list[dict[str, Any]] = []
    for raw_line in txt_content.splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            yolo_cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw_n = float(parts[3])
            bh_n = float(parts[4])
        except ValueError:
            continue
        cid = yolo_cls + 1
        if str(cid) not in TALKER_CLASSES:
            continue
        x1 = round(max(0.0, min(float(width), (xc - bw_n / 2.0) * float(width))), 1)
        y1 = round(max(0.0, min(float(height), (yc - bh_n / 2.0) * float(height))), 1)
        x2 = round(max(0.0, min(float(width), (xc + bw_n / 2.0) * float(width))), 1)
        y2 = round(max(0.0, min(float(height), (yc + bh_n / 2.0) * float(height))), 1)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        products.append(cid)
        labels.append(dict(TALKER_CLASSES[str(cid)]))
    return boxes, products, labels


def build_talker_golden_manifest(
    client: Any = None,
    ms_json_bucket: str = "ms-json-dataset",
    talker_prefix: str = "toker_dataset/mar_talker_final/",
    marketshare_bucket: str = "marketshare-datasets",
    local_dir: str | Path = "data/hul_talkers_golden",
    seed: int = 42,
    upload: bool = True,
) -> dict[str, Any]:
    """Build and publish the 80/10/10 outlet-grouped ``hul_talkers_golden`` manifest."""
    if client is None:
        from google.cloud import storage

        from utils.llm import ensure_gcloud_auth_fallback

        ensure_gcloud_auth_fallback()
        client = storage.Client()

    from utils.dataset import export_rpc_splits_jsonl, format_split_jsonl

    b_json = client.bucket(ms_json_bucket)
    all_blobs = list(client.list_blobs(b_json, prefix=talker_prefix))

    jpg_by_key: dict[str, Any] = {}
    txt_by_key: dict[str, Any] = {}
    for bl in all_blobs:
        if bl.name.endswith(".jpg"):
            jpg_by_key[bl.name[:-4]] = bl
        elif bl.name.endswith(".txt"):
            txt_by_key[bl.name[:-4]] = bl

    paired_keys = sorted(set(jpg_by_key.keys()) & set(txt_by_key.keys()))
    raw_pair_count = len(paired_keys)

    # Deduplicate by GCS MD5 hash and stem so zero byte-identical image or stem can cross splits
    seen_md5s: set[str] = set()
    seen_stems: set[str] = set()
    unique_pairs: list[tuple[str, Any, Any, str]] = []
    quarantined_md5_dups = 0
    for key in paired_keys:
        jpg_bl = jpg_by_key[key]
        txt_bl = txt_by_key[key]
        stem = PurePosixPath(jpg_bl.name).stem
        md5_hex = (
            base64.b64decode(jpg_bl.md5_hash).hex()
            if getattr(jpg_bl, "md5_hash", None)
            else hashlib.md5(jpg_bl.name.encode("utf-8")).hexdigest()
        )
        if md5_hex in seen_md5s or stem in seen_stems:
            quarantined_md5_dups += 1
            continue
        seen_md5s.add(md5_hex)
        seen_stems.add(stem)
        unique_pairs.append((stem, jpg_bl, txt_bl, md5_hex))

    def _load_one_pair(item: tuple[str, Any, Any, str]) -> dict[str, Any] | None:
        stem, jpg_bl, txt_bl, md5_hex = item
        txt_content = txt_bl.download_as_bytes().decode("utf-8", errors="replace")
        w, h = _exif_transposed_dimensions_from_blob(jpg_bl)
        boxes, products, labels = _yolo_lines_to_boxes_and_classes(txt_content, w, h)
        if not boxes:
            return None
        outlet_code = extract_talker_outlet_code(stem)
        return {
            "stem": stem,
            "uri": f"gs://{ms_json_bucket}/{jpg_bl.name}",
            "outlet_code": outlet_code,
            "md5": md5_hex,
            "width": w,
            "height": h,
            "boxes": boxes,
            "products": products,
            "labels": labels,
        }

    with ThreadPoolExecutor(max_workers=32) as pool:
        records = [r for r in pool.map(_load_one_pair, unique_pairs) if r is not None]

    # Group by outlet_code and stratify 80/10/10 across train/val/test
    by_outlet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_outlet[rec["outlet_code"]].append(rec)

    all_outlets = sorted(
        by_outlet.keys(),
        key=lambda oc: (
            any(3 in r["products"] for r in by_outlet[oc]),
            any(1 in r["products"] for r in by_outlet[oc]),
            any(2 in r["products"] for r in by_outlet[oc]),
            hashlib.sha1(f"{seed}:{oc}".encode()).hexdigest(),
        ),
        reverse=True,
    )
    n_outlets = len(all_outlets)
    target_val_outlets = round(n_outlets * 0.10)
    target_test_outlets = round(n_outlets * 0.10)

    outlet_to_split: dict[str, str] = {}
    val_count = 0
    test_count = 0
    for idx, oc in enumerate(all_outlets):
        if idx % 10 == 0 and val_count < target_val_outlets:
            outlet_to_split[oc] = "val"
            val_count += 1
        elif idx % 10 == 5 and test_count < target_test_outlets:
            outlet_to_split[oc] = "test"
            test_count += 1
        else:
            outlet_to_split[oc] = "train"

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for oc in sorted(all_outlets, key=lambda x: hashlib.sha1(f"{seed}:order:{x}".encode()).hexdigest()):
        sp_name = outlet_to_split[oc]
        for rec in sorted(by_outlet[oc], key=lambda r: r["stem"]):
            sp_idx = len(splits[sp_name])
            splits[sp_name].append({
                "image": f"{sp_name}_{sp_idx:05d}.jpg",
                "uri": rec["uri"],
                "session_id": f"talker_{rec['outlet_code']}",
                "outlet_code": rec["outlet_code"],
                "md5": rec["md5"],
                "width": rec["width"],
                "height": rec["height"],
                "boxes": rec["boxes"],
                "asset_boxes": [],
                "products": rec["products"],
                "labels": rec["labels"],
            })

    split_hashes: dict[str, str] = {}
    for sp_name in ("train", "val", "test"):
        uris = sorted(r["uri"] for r in splits[sp_name])
        dg = hashlib.sha256("\n".join(uris).encode("utf-8")).hexdigest()
        split_hashes[sp_name] = dg[:16]
        split_hashes[f"{sp_name}_sha256"] = dg

    manifest: dict[str, Any] = {
        "source": f"gs://{ms_json_bucket}/{talker_prefix} (v2.2 80/10/10 Outlet-Grouped Shelf Talker & Price Tag Benchmark)",
        "classes": TALKER_CLASSES,
        "gallery": {k: [] for k in TALKER_CLASSES},
        "splits": splits,
        "stats": {
            "mode": "hul_talkers_golden",
            "schema_version": "2.2.0",
            "total_classes": len(TALKER_CLASSES),
            "raw_paired_samples": raw_pair_count,
            "quarantined_md5_duplicates": quarantined_md5_dups,
            "total_outlets": n_outlets,
            "train_outlets": sum(1 for v in outlet_to_split.values() if v == "train"),
            "val_outlets": sum(1 for v in outlet_to_split.values() if v == "val"),
            "test_outlets": sum(1 for v in outlet_to_split.values() if v == "test"),
            "train_samples": len(splits["train"]),
            "val_samples": len(splits["val"]),
            "test_samples": len(splits["test"]),
            "train_boxes": sum(len(r["boxes"]) for r in splits["train"]),
            "val_boxes": sum(len(r["boxes"]) for r in splits["val"]),
            "test_boxes": sum(len(r["boxes"]) for r in splits["test"]),
            "split_hashes": split_hashes,
        },
    }

    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_rpc_splits_jsonl(manifest, out_dir)
    rpc_text = json.dumps(manifest, indent=2)
    (out_dir / "rpc.json").write_text(rpc_text, encoding="utf-8")

    if upload:
        b_ms = client.bucket(marketshare_bucket)
        base_gcs = "derived_manifests/hul_talkers_golden"
        b_ms.blob(f"{base_gcs}/rpc.json").upload_from_string(rpc_text, content_type="application/json")
        for sp_name in ("train", "val", "test"):
            b_ms.blob(f"{base_gcs}/{sp_name}.jsonl").upload_from_string(
                format_split_jsonl(splits[sp_name]), content_type="application/jsonl"
            )

    return manifest
