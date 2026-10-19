"""AWP Campaign Reference Template Indexer & Loader for ``gs://sos-datasets/awp_vectors/awp_code/``.

Indexes all 63 Unilever Always-Winning-Presence (AWP) campaign planogram/bay reference images
into ``data/awp_code_templates/index.json`` and ``gs://marketshare-datasets/derived_manifests/awp_code_templates/index.json``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

CATEGORY_PREFIXES: tuple[str, ...] = ("DETS", "FNB", "HFD", "ORAL", "PP", "SUNSILK")
FIXTURE_TYPES: tuple[str, ...] = ("FULLWALL", "MB3", "MB2", "CTU", "HAN", "WIN")

_BRAND_HINT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("SUNSILK",), "SUNSILK"),
    (("BOOST",), "BOOST"),
    (("HORLICK", "DIAP", "MOTP", "WOMP"), "HORLICKS"),
    (("PEPSODENT", "ORALPEP"), "PEPSODENT"),
    (("CLUP", "CLOSEUP"), "CLOSEUP"),
    (("DOPEARS", "PEARS"), "PEARS"),
    (("DOVE",), "DOVE"),
    (("INDU",), "INDULEKHA"),
    (("LAK",), "LAKME"),
    (("PON",), "PONDS"),
    (("SIM",), "SIMPLE"),
    (("TRES",), "TRESEMME"),
    (("VAS",), "VASELINE"),
    (("LUX",), "LUX"),
    (("LBP",), "LBP"),
    (("VWASH",), "VWASH"),
    (("ELLE",), "ELLE_18"),
    (("FOODS",), "FOODS"),
    (("MINI",), "MULTI_BRAND"),
)


def parse_awp_template_filename(filename: str) -> dict[str, str]:
    """Parse an AWP template filename into ``awp_code``, ``category_prefix``, ``fixture_type``, and ``brand_hint``."""
    name = PurePosixPath(filename).name
    stem = PurePosixPath(name).stem
    up = stem.upper()

    category_prefix = "PP"
    for pfx in CATEGORY_PREFIXES:
        if up.startswith(pfx):
            category_prefix = pfx
            break

    fixture_type = "WIN"
    for ftype in FIXTURE_TYPES:
        if ftype in up:
            fixture_type = ftype
            break

    brand_hint = "UNILEVER"
    for tokens, brand in _BRAND_HINT_RULES:
        if any(tok in up for tok in tokens):
            brand_hint = brand
            break

    return {
        "awp_code": stem,
        "filename": name,
        "category_prefix": category_prefix,
        "fixture_type": fixture_type,
        "brand_hint": brand_hint,
    }


def build_awp_template_index(
    client: Any = None,
    sos_bucket: str = "sos-datasets",
    sos_prefix: str = "awp_vectors/awp_code/",
    marketshare_bucket: str = "marketshare-datasets",
    local_dir: str | Path = "data/awp_code_templates",
    upload: bool = True,
) -> dict[str, Any]:
    """Build ``index.json`` for all 63 AWP reference templates in ``gs://sos-datasets/awp_vectors/awp_code/``."""
    if client is None:
        from google.cloud import storage

        from utils.llm import ensure_gcloud_auth_fallback

        ensure_gcloud_auth_fallback()
        client = storage.Client()

    b_sos = client.bucket(sos_bucket)
    blobs = [
        b for b in client.list_blobs(b_sos, prefix=sos_prefix)
        if not b.name.endswith("/") and PurePosixPath(b.name).suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    def _read_template(blob: Any) -> dict[str, Any]:
        meta = parse_awp_template_filename(blob.name)
        raw_bytes = blob.download_as_bytes()
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im_t = ImageOps.exif_transpose(im)
            w, h = im_t.size
        md5_hex = (
            base64.b64decode(blob.md5_hash).hex()
            if getattr(blob, "md5_hash", None)
            else hashlib.md5(raw_bytes).hexdigest()
        )
        return {
            "awp_code": meta["awp_code"],
            "filename": meta["filename"],
            "uri": f"gs://{sos_bucket}/{blob.name}",
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(float(w) / max(1.0, float(h)), 4),
            "category_prefix": meta["category_prefix"],
            "fixture_type": meta["fixture_type"],
            "brand_hint": meta["brand_hint"],
            "md5": md5_hex,
        }

    with ThreadPoolExecutor(max_workers=32) as pool:
        templates = sorted(pool.map(_read_template, blobs), key=lambda r: r["awp_code"])

    payload: dict[str, Any] = {
        "source": f"gs://{sos_bucket}/{sos_prefix}",
        "total_templates": len(templates),
        "templates": templates,
        "by_code": {t["awp_code"]: t for t in templates},
    }

    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    serialized = json.dumps(payload, indent=2)
    index_path.write_text(serialized, encoding="utf-8")

    if upload:
        b_ms = client.bucket(marketshare_bucket)
        b_ms.blob("derived_manifests/awp_code_templates/index.json").upload_from_string(
            serialized, content_type="application/json"
        )

    return payload


_AWP_INDEX_CACHE: dict[str, Any] | None = None

_HINT_TO_CANONICAL_BRAND: dict[str, str] = {
    "SUNSILK": "Sunsilk",
    "BOOST": "Boost",
    "HORLICKS": "Horlicks",
    "PEPSODENT": "Pepsodent",
    "CLOSEUP": "Closeup",
    "PEARS": "Pears",
    "DOVE": "Dove",
    "INDULEKHA": "Indulekha",
    "LAKME": "Lakme",
    "PONDS": "Ponds",
    "SIMPLE": "Simple",
    "TRESEMME": "Tresemme",
    "VASELINE": "Vaseline",
    "LUX": "Lux",
    "LBP": "Love_Beauty_And_Planet",
    "VWASH": "V_Wash",
    "ELLE_18": "Elle_18",
}


def load_awp_template_index(path_or_uri: str | Path | None = None) -> dict[str, Any]:
    """Load the AWP template index from local ``data/awp_code_templates/index.json`` or GCS."""
    global _AWP_INDEX_CACHE
    if path_or_uri is None and _AWP_INDEX_CACHE is not None:
        return _AWP_INDEX_CACHE

    if path_or_uri is not None:
        target = str(path_or_uri)
        if not target.endswith(".json"):
            target = f"{target.rstrip('/')}/index.json"
        from utils.dataset import read_bytes

        return json.loads(read_bytes(target))

    local_path = Path("data/awp_code_templates/index.json")
    if local_path.exists():
        loaded: dict[str, Any] = json.loads(local_path.read_text(encoding="utf-8"))
        _AWP_INDEX_CACHE = loaded
        return loaded

    from utils.dataset import read_bytes
    from utils.llm import load_config

    gcp_root = load_config().get("gcp", {}).get("awp_code_templates")
    if gcp_root:
        loaded_gcs: dict[str, Any] = json.loads(read_bytes(f"{str(gcp_root).rstrip('/')}/index.json"))
        _AWP_INDEX_CACHE = loaded_gcs
        return loaded_gcs
    raise FileNotFoundError("data/awp_code_templates/index.json not found locally or in config.yaml.")


def resolve_awp_planogram_spec(hint: str | None) -> dict[str, Any] | None:
    """Match an image filename or AWP_Code against the 63 studio templates and return a planogram_spec."""
    if not hint:
        return None
    try:
        idx = load_awp_template_index()
    except Exception:
        return None
    by_code: dict[str, dict[str, Any]] = idx.get("by_code") or {}
    if not by_code:
        return None

    stem = PurePosixPath(str(hint).strip()).stem.upper()
    matched = by_code.get(stem)
    if matched is None:
        for code in sorted(by_code.keys(), key=len, reverse=True):
            if code.upper() in stem:
                matched = by_code[code]
                break
    if matched is None:
        return None

    brand_hint = str(matched.get("brand_hint") or "")
    canon_brand = _HINT_TO_CANONICAL_BRAND.get(brand_hint)
    ftype = str(matched.get("fixture_type") or "WIN")
    return {
        "awp_code": matched["awp_code"],
        "fixture_type": ftype,
        "category_prefix": matched.get("category_prefix"),
        "template_uri": matched.get("uri"),
        "required_brands": [canon_brand] if canon_brand else [],
        "min_hul_sos": 0.90 if ftype != "HAN" else 0.70,
        "min_brand_block_purity": 0.80,
        "forbid_competitors_in_bay": ftype != "HAN",
    }

