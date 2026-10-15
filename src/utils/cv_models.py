"""Legacy InceptionV3 Keras (``.h5``) brand, pack, and category-routed variant classifiers (``-a cv_classify``).

The ``.h5`` files (``cv_models`` in config.yaml) are registered in Vertex AI Model Registry
and deployed to Vertex AI Endpoints (``shelf-bench deploy-cv-endpoints``). On an endpoint, this
repo's image runs ``shelf-bench serve``, and its ``/api/cv/predict`` route runs the model with
``KerasClassifier``. The benchmark calls the endpoints through ``EndpointClassifier``.

All input conditioning vectors and output class mappings are sanitized against the authoritative
ground-truth configuration ``gt-mapping.yml``:
- brand model: a 941-way softmax over ``brand_data`` folder names (``<model>_classes.json``),
  sanitized against ``brand_mapping`` and ``*_brand_mapping`` in ``gt-mapping.yml``.
- pack model: a 46-way softmax over ``pack_type_mapping`` (``Category__PackType``) in
  ``gt-mapping.yml``, conditioned on a 10-d one-hot vector from ``category_mapping`` (``0..9``).
- 7 variant models (``wash_variant``, ``skin_variant``, ``p5_variant``, ``oral_variant``,
  ``gen_variant``, ``nonhul_p6_variant``, ``nonhul_gen_variant``): each takes a brand one-hot
  vector from its ``<role>_brand_mapping`` in ``gt-mapping.yml`` alongside the 299x299 image
  tensor, and predicts a 5-part ``<HUL|NON_HUL>__<Category>__<SubCategory>__<Brand>__<Variant>``
  entry from its ``<role>_var_mapping`` / ``<role>_variant_mapping`` in ``gt-mapping.yml``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from utils import dataset

VARIANT_ROLES = (
    "wash_variant",
    "skin_variant",
    "p5_variant",
    "oral_variant",
    "gen_variant",
    "nonhul_p6_variant",
    "nonhul_gen_variant",
)
ROLES = ("brand", "pack", *VARIANT_ROLES)
# Model overrides: set by `run --brand-model/--pack-model/--<variant>-model` and by the serving container's env.
MODEL_ENV = {r: f"SHELF_BENCH_CV_{r.upper()}_MODEL" for r in ROLES}
VARIANT_SILOS = {
    "wash_variant": "variant_wash",
    "skin_variant": "variant_skin",
    "p5_variant": "variant_p5",
    "oral_variant": "variant_oral",
    "gen_variant": "variant_gen",
    "nonhul_p6_variant": "variant_NH_P6",
    "nonhul_gen_variant": "variant_NH_gen",
}
_GT_VARIANT_KEYS: dict[str, tuple[str, str]] = {
    "gen_variant": ("gen_brand_mapping", "gen_var_mapping"),
    "oral_variant": ("oral_brand_mapping", "oral_var_mapping"),
    "skin_variant": ("skin_brand_mapping", "skin_var_mapping"),
    "wash_variant": ("was_brand_mapping", "was_var_mapping"),
    "p5_variant": ("p5_brand_mapping", "p5_variant_mapping"),
    "nonhul_gen_variant": ("nh_gen_brand_mapping", "nh_gen_variant_mapping"),
    "nonhul_p6_variant": ("nh_p6_brand_mapping", "nh_p6_variant_mapping"),
}
_DEFAULT_ROLE_MODEL_URIS: dict[str, str] = {
    "brand": "gs://sos-datasets/legacy-models/mt_brand/marketshare_brand.h5",
    "pack": "gs://sos-datasets/legacy-models/mt_pack/marketshare_packtype.h5",
    "wash_variant": "gs://sos-datasets/legacy-models/wash_variant/wash_variant.h5",
    "skin_variant": "gs://sos-datasets/legacy-models/skin_variant/skin_variant.h5",
    "p5_variant": "gs://sos-datasets/legacy-models/p5_variant/p5_variant.h5",
    "oral_variant": "gs://sos-datasets/legacy-models/oral_variant/oral_variant.h5",
    "gen_variant": "gs://sos-datasets/legacy-models/gen_variant/gen_variant.h5",
    "nonhul_p6_variant": "gs://sos-datasets/legacy-models/nonhul_p6_variant/nonhul_p6_variant.h5",
    "nonhul_gen_variant": "gs://sos-datasets/legacy-models/nonhul_gen_variant/nonhul_gen_variant.h5",
}
# The 4 single-SKU brands omitted when nonhul_p6_variant.h5 (61 brands, 367 classes) was exported from the 65-brand / 371-class table
_NH_P6_DROPPED_BRANDS = frozenset({"citricplus", "maxkleen", "nestle", "slurrpfarm"})

_ROLE_ALIASES = {
    "brand": "brand",
    "pack": "pack",
    "pack_type": "pack",
    "packaging_type": "pack",
    "wash_variant": "wash_variant",
    "variant_wash": "wash_variant",
    "wash": "wash_variant",
    "skin_variant": "skin_variant",
    "variant_skin": "skin_variant",
    "skin": "skin_variant",
    "p5_variant": "p5_variant",
    "variant_p5": "p5_variant",
    "p5": "p5_variant",
    "oral_variant": "oral_variant",
    "variant_oral": "oral_variant",
    "oral": "oral_variant",
    "gen_variant": "gen_variant",
    "variant_gen": "gen_variant",
    "gen": "gen_variant",
    "nonhul_p6_variant": "nonhul_p6_variant",
    "variant_nh_p6": "nonhul_p6_variant",
    "nonhul_p6": "nonhul_p6_variant",
    "nh_p6": "nonhul_p6_variant",
    "nonhul_gen_variant": "nonhul_gen_variant",
    "variant_nh_gen": "nonhul_gen_variant",
    "nonhul_gen": "nonhul_gen_variant",
    "nh_gen": "nonhul_gen_variant",
}
_P5_P6_CATEGORIES = {
    "Foods",
    "Beverages",
    "Health_Food_Drinks",
    "Health_Wellbeing",
    "Household_Care",
    "Home_Care",
    "HHC",
    "HFD",
    "Food",
    "Tea",
    "Coffee",
    "Dishwash",
}
_ORAL_CATEGORIES = {"Oral_Care", "Oral"}
_WASH_CATEGORIES = {"Laundry", "Personal_Wash", "PW", "Skin_Cleansing", "Fabric_Wash", "Fabric_Care", "Wash"}
_GEN_CATEGORIES = {"Hair_Care", "Hair", "DMT", "Deodorants", "Deo"}
_BRAND_KEY_ALIASES: dict[str, str] = {
    "lovebeautyandplanet": "lbp",
    "alovebeautyandplanet": "lbp",
    "fairandlovely": "glowandlovely",
    "afairandlovely": "glowandlovely",
    "glowandhandsome": "glowandlovely",
    "aclinic": "clinicplus",
    "clinic": "clinicplus",
    "a3roses": "3roses",
    "abrookbond": "redlabel",
    "brookbond": "redlabel",
    "brookebond": "redlabel",
    "yippee": "yipee",
    "tetley": "tetely",
    "twinings": "twinnings",
    "hellmanns": "hellmans",
}
_ENDPOINT_RE = re.compile(r"^projects/[^/]+/locations/([^/]+)/endpoints/[^/]+$")
_RETRY_STATUS = (429, 500, 502, 503, 504)


def normalize_role(role: str | None) -> str:
    """``brand``, ``pack``, or one of ``VARIANT_ROLES`` (aliases mapped), else ``""``."""
    r = str(role or "").strip().lower()
    if r in _ROLE_ALIASES:
        return _ROLE_ALIASES[r]
    return "pack" if r.startswith("pack") else ""


def _norm_brand_key(brand: str | None) -> str:
    """Normalize a brand token so ``gt-mapping.yml``, ``brand_data`` ``ABrand`` folders, and v2.2 canonical names align."""
    raw = str(brand or "").strip()
    if not raw:
        return ""
    if raw.startswith(("A_", "Z_")):
        raw = raw[2:]
    elif len(raw) >= 3 and raw[0] == "A" and raw[1].isupper():
        raw = raw[1:]
    s = re.sub(r"[^a-z0-9]+", "", raw.replace("&", "and").lower())
    return _BRAND_KEY_ALIASES.get(s, s)


def _class_brand_token(c: str) -> str:
    """Extract the brand token from a 5-part ``gt-mapping.yml`` class or a 2-part ``Brand__Variant`` class."""
    parts = str(c).strip().split("__")
    if len(parts) >= 5 and parts[0].upper() in ("HUL", "NON_HUL"):
        return parts[3]
    return parts[0] if len(parts) >= 2 else ""


@lru_cache(maxsize=1)
def gt_mapping() -> dict[str, dict[int, str]]:
    """Load the authoritative customer ground-truth mapping dictionary from ``gt-mapping.yml``."""
    import yaml

    candidates = (
        Path(__file__).resolve().parents[2] / "gt-mapping.yml",
        Path("gt-mapping.yml"),
        Path("/app/gt-mapping.yml"),
    )
    for path in candidates:
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {
                str(k): {int(idx): str(val) for idx, val in sorted((v or {}).items(), key=lambda kv: int(kv[0]))}
                for k, v in raw.items()
                if isinstance(v, Mapping)
            }
    return {}


@lru_cache(maxsize=1)
def pack_classes() -> tuple[str, ...]:
    """The pack model's 46 output classes from ``gt-mapping.yml`` ``pack_type_mapping``."""
    gtm = gt_mapping().get("pack_type_mapping")
    if gtm and len(gtm) == 46:
        return tuple(gtm[i] for i in sorted(gtm))
    from utils.legacy_benchmark import build_legacy_mapping_table

    return tuple(sorted({r["legacy_pack_folder"] for r in build_legacy_mapping_table()["category_packs"]}))


@lru_cache(maxsize=1)
def pack_categories() -> tuple[str, ...]:
    """The 10 legacy categories (indices ``0..9``) from ``gt-mapping.yml`` ``category_mapping``."""
    gtm = gt_mapping().get("category_mapping")
    if gtm:
        cats = tuple(gtm[i] for i in sorted(gtm) if int(i) < 10 and gtm[i] != "Others")
        if len(cats) == 10:
            return cats
    return tuple(sorted({c.split("__", 1)[0] for c in pack_classes()}))


@lru_cache(maxsize=32)
def gt_brand_classes(role: str, size: int | None = None) -> tuple[str, ...]:
    """The brand one-hot vocabulary for ``role`` from ``gt-mapping.yml`` ``<role>_brand_mapping``."""
    r = normalize_role(role)
    if r not in _GT_VARIANT_KEYS:
        return ()
    b_key, _ = _GT_VARIANT_KEYS[r]
    b_map = gt_mapping().get(b_key) or {}
    brands = tuple(b_map[i] for i in sorted(b_map))
    if r == "nonhul_p6_variant" and size == 61 and len(brands) == 65:
        brands = tuple(b for b in brands if _norm_brand_key(b) not in _NH_P6_DROPPED_BRANDS)
    return brands


@lru_cache(maxsize=32)
def gt_variant_classes(role: str, size: int | None = None) -> tuple[str, ...]:
    """The variant output classes for ``role`` from ``gt-mapping.yml`` ``<role>_var_mapping``."""
    r = normalize_role(role)
    if r not in _GT_VARIANT_KEYS:
        return ()
    _, v_key = _GT_VARIANT_KEYS[r]
    v_map = gt_mapping().get(v_key) or {}
    classes = tuple(v_map[i] for i in sorted(v_map))
    if r == "nonhul_p6_variant" and size == 367 and len(classes) == 371:
        classes = tuple(c for c in classes if _norm_brand_key(_class_brand_token(c)) not in _NH_P6_DROPPED_BRANDS)
    return classes


@lru_cache(maxsize=16)
def load_classes(model_uri: str | None) -> tuple[str, ...]:
    """Class names stored as ``<model>_classes.json`` next to a ``.h5`` (local or gs://), else ()."""
    from google.api_core.exceptions import NotFound

    if not model_uri or not str(model_uri).endswith(".h5"):
        return ()
    try:
        return tuple(str(c) for c in json.loads(dataset.read_bytes(str(model_uri)[:-3] + "_classes.json")))
    except (FileNotFoundError, NotFound, Exception):
        return ()


@lru_cache(maxsize=16)
def variant_brands(classes: tuple[str, ...]) -> tuple[str, ...]:
    """The sorted legacy brand prefixes ``<Brand>`` from ``<Brand>__<Variant>`` or 5-part ``gt-mapping.yml`` classes."""
    return tuple(sorted({_class_brand_token(c) for c in classes if _class_brand_token(c)}))


def brand_index(
    brand: str | None,
    classes: Sequence[str] = (),
    *,
    role: str | None = None,
    size: int | None = None,
) -> int | None:
    """Index of ``brand`` in ``gt-mapping.yml``'s ``<role>_brand_mapping`` or ``variant_brands(classes)``."""
    key = _norm_brand_key(brand)
    if not key:
        return None
    # If custom classes were provided (e.g. unit test fixtures) whose unique brand count matches `size` (or `role` is unset),
    # check `variant_brands(classes)` first when `classes` are 2-part strings or when `size` matches `len(cls_brands)`.
    cls_brands = variant_brands(tuple(classes)) if classes else ()
    gt_brands = gt_brand_classes(role, size=size) if role else ()

    if classes and (size is None or len(cls_brands) == size) and len(cls_brands) != len(gt_brands):
        for i, b in enumerate(cls_brands):
            if _norm_brand_key(b) == key:
                return i
    if gt_brands and (size is None or len(gt_brands) == size):
        for i, b in enumerate(gt_brands):
            if _norm_brand_key(b) == key:
                return i
    for i, b in enumerate(cls_brands):
        if _norm_brand_key(b) == key:
            return i
    return None


def route_variant_role(
    category: str | None = None,
    brand: str | None = None,
    is_hul: bool | None = None,
    runners: Mapping[str, Any] | None = None,
) -> str:
    """Route an image to the appropriate variant model role using ``gt-mapping.yml`` brand tables and category."""
    cat_raw = str(category or "").strip()
    if not brand and cat_raw.lower() in (set(VARIANT_ROLES) | set(VARIANT_SILOS.values())):
        return normalize_role(cat_raw)
    cat = cat_raw
    if is_hul is None and brand:
        from utils.legacy_benchmark import normalize_legacy_prediction

        norm_b = normalize_legacy_prediction(brand_pred=brand)
        if norm_b.get("brand") != "Unknown":
            is_hul = bool(norm_b.get("is_hul"))
            if not cat and norm_b.get("category") != "Unknown":
                cat = str(norm_b["category"])

    # Determine category-based fallback role
    if is_hul is False:
        cat_role = "nonhul_p6_variant" if cat in _P5_P6_CATEGORIES else "nonhul_gen_variant"
    elif cat in _ORAL_CATEGORIES:
        cat_role = "oral_variant"
    elif cat in _P5_P6_CATEGORIES:
        cat_role = "p5_variant"
    elif cat in _WASH_CATEGORIES:
        cat_role = "wash_variant"
    elif cat in _GEN_CATEGORIES and (not runners or "gen_variant" in runners):
        cat_role = "gen_variant"
    else:
        cat_role = "skin_variant"

    # Authoritative brand lookup across configured runners (or gt-mapping.yml brand tables)
    if brand:
        candidate_roles = [r for r in VARIANT_ROLES if not runners or r in runners]
        matches: list[str] = []
        for r in candidate_roles:
            r_classes = runners[r].classes if (runners and r in runners) else ()
            if brand_index(brand, r_classes, role=r) is not None:
                matches.append(r)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if cat_role in matches:
                return cat_role
            if is_hul is False:
                for nh_r in ("nonhul_gen_variant", "nonhul_p6_variant"):
                    if nh_r in matches:
                        return nh_r
            return matches[0]

    if runners:
        available = [r for r in VARIANT_ROLES if r in runners]
        if available:
            return cat_role if cat_role in available else available[0]
    return cat_role


def configured_variant_roles(config: dict) -> list[str]:
    """Variant roles that have an endpoint or CLI override configured in ``config`` / env."""
    cv = (config or {}).get("cv_models") or {}
    return [r for r in VARIANT_ROLES if os.environ.get(MODEL_ENV[r]) or cv.get(f"{r}_endpoint")]


def has_configured_variants(config: dict) -> bool:
    """Return True if at least one variant endpoint or CLI model override is configured."""
    return bool(configured_variant_roles(config))


def parse_gt_variant_label(role: str, label: str) -> dict[str, Any]:
    """Parse a 5-part ``gt-mapping.yml`` variant label (or 2-part ``Brand__Variant`` folder) with full metadata."""
    from utils.legacy_benchmark import LEGACY_PACK_CATEGORY_TO_CANONICAL
    from utils.taxonomy.brands import HUL_BRANDS, canonicalize_brand_name
    from utils.taxonomy.categories import normalize_category_name
    from utils.taxonomy.filenames import parse_variant_folder

    raw = str(label or "").strip()
    if not raw or raw.isdigit():
        return {}
    r = normalize_role(role)
    silo = VARIANT_SILOS.get(r, "variant_skin")
    parts = raw.split("__")

    # Check if `raw` is a 5-part `gt-mapping.yml` string: `<HUL|NON_HUL>__<Category>__<SubCategory>__<Brand>__<Variant>`
    if len(parts) >= 5 and parts[0].upper() in ("HUL", "NON_HUL"):
        owner_tok, cat_tok, subcat_tok, brand_tok = parts[0].upper(), parts[1], parts[2], parts[3]
        var_tok = "__".join(parts[4:])
        is_hul = owner_tok == "HUL"
        canon_brand = canonicalize_brand_name(brand_tok)
        if canon_brand == "Unknown" or not canon_brand:
            canon_brand = brand_tok
        canon_cat = LEGACY_PACK_CATEGORY_TO_CANONICAL.get(
            cat_tok.lower(), normalize_category_name(cat_tok)
        )

        # If `<role>_classes.json` (the GCS training folder list) is available at the same index,
        # extract its exact `base_variant` so both v2.2 Golden manifests and `gt-mapping.yml` align 100%.
        folder_entry = f"{brand_tok}__{var_tok}"
        for sz in (367, None):
            gt_list = gt_variant_classes(r, size=sz)
            if raw in gt_list:
                idx = gt_list.index(raw)
                uri = os.environ.get(MODEL_ENV.get(r, "")) or _DEFAULT_ROLE_MODEL_URIS.get(r)
                f_classes = load_classes(uri)
                if len(f_classes) == len(gt_list) and idx < len(f_classes):
                    f_cand = f_classes[idx]
                    if "__" in f_cand and len(f_cand.split("__")) == 2:
                        folder_entry = f_cand
                break

        p = parse_variant_folder(silo, folder_entry)
        base_variant = str(p.get("base_variant") or var_tok)
        variant_val = str(p.get("variant") or var_tok)
        pkg_type = str(p.get("packaging_type") or "")
        return {
            "brand": canon_brand,
            "base_variant": base_variant,
            "variant": variant_val,
            "category": canon_cat if canon_cat != "Unknown" else str(p.get("category") or ""),
            "packaging_type": pkg_type if pkg_type != "Unknown" else "",
            "product": f"{canon_brand}__{base_variant}" if canon_brand and base_variant else "",
            "is_hul": is_hul,
            "gt_brand": brand_tok,
            "gt_category": cat_tok,
            "gt_subcategory": subcat_tok,
            "gt_variant": var_tok,
            "gt_label": raw,
        }

    # 2-part `<Brand>__<Variant>` folder label (also cross-reference `gt-mapping.yml` at the same class index if present)
    gt_meta: dict[str, Any] = {}
    uri = os.environ.get(MODEL_ENV.get(r, "")) or _DEFAULT_ROLE_MODEL_URIS.get(r)
    f_classes = load_classes(uri)
    if f_classes and raw in f_classes:
        idx = f_classes.index(raw)
        gt_list = gt_variant_classes(r, size=len(f_classes))
        if idx < len(gt_list):
            gt_raw = gt_list[idx]
            gt_parts = gt_raw.split("__")
            if len(gt_parts) >= 5 and gt_parts[0].upper() in ("HUL", "NON_HUL"):
                gt_cat = gt_parts[1]
                gt_meta = {
                    "is_hul": gt_parts[0].upper() == "HUL",
                    "gt_brand": gt_parts[3],
                    "gt_category": gt_cat,
                    "gt_subcategory": gt_parts[2],
                    "gt_variant": "__".join(gt_parts[4:]),
                    "gt_label": gt_raw,
                    "category_override": LEGACY_PACK_CATEGORY_TO_CANONICAL.get(
                        gt_cat.lower(), normalize_category_name(gt_cat)
                    ),
                }

    p = parse_variant_folder(silo, raw)
    brand_val = str(p.get("brand") or "")
    bv_val = str(p.get("base_variant") or "")
    cat_val = str(gt_meta.get("category_override") or p.get("category") or "")
    pkg_val = str(p.get("packaging_type") or "")
    return {
        "brand": brand_val if brand_val != "Unknown" else "",
        "base_variant": bv_val if bv_val != "Unknown" else "",
        "variant": str(p.get("variant") or bv_val),
        "category": cat_val if cat_val != "Unknown" else "",
        "packaging_type": pkg_val if pkg_val != "Unknown" else "",
        "product": f"{brand_val}__{bv_val}" if brand_val and bv_val else "",
        "is_hul": bool(gt_meta["is_hul"]) if "is_hul" in gt_meta else (brand_val.lower() in HUL_BRANDS),
        **{k: v for k, v in gt_meta.items() if k != "category_override"},
    }


def parse_label(role: str, label: str) -> dict[str, str]:
    """Benchmark fields for one model class (brand_data, Category__PackType, or Brand__Variant / gt-mapping.yml)."""
    from utils.legacy_benchmark import normalize_legacy_prediction

    if not label or str(label).isdigit():
        return {}
    r = normalize_role(role)
    if r == "brand":
        p = normalize_legacy_prediction(brand_pred=label)
        keys = ("brand", "category")
        return {k: str(p[k]) for k in keys if p.get(k) and p[k] != "Unknown"}
    if r in VARIANT_ROLES:
        info = parse_gt_variant_label(r, str(label).strip())
        out: dict[str, str] = {}
        for k in ("brand", "base_variant", "variant", "category", "packaging_type", "product"):
            if info.get(k) and info[k] != "Unknown":
                out[k] = str(info[k])
        return out
    p = normalize_legacy_prediction(pack_pred=label)
    keys = ("category", "packaging_type")
    return {k: str(p[k]) for k in keys if p.get(k) and p[k] != "Unknown"}


def resolve_evaluated_fields(
    dataset_name: str,
    requested: str | Sequence[str] | None = None,
    with_variants: bool = False,
) -> tuple[str, ...]:
    """The attributes to score. ``requested`` is ``--fields``: '1', '2', '3', '4', '5' or names like
    'brand,packaging_type'. Without it: what the legacy models predict on ``dataset_name``."""
    packs_only = dataset_name == "hul_packs"
    presets = {
        "1": ("packaging_type",) if packs_only else ("brand",),
        "2": ("packaging_type", "category") if packs_only else ("brand", "packaging_type"),
        "3": ("brand", "packaging_type", "category"),
        "4": ("brand", "base_variant", "packaging_type", "category"),
        "5": ("product", "brand", "base_variant", "packaging_type", "category"),
    }
    if requested is None or requested == "" or requested == ():
        if dataset_name == "hul_brands":
            return ("brand",)
        if packs_only:
            return presets["2"]
        if dataset_name == "hul_variants" or with_variants:
            return presets["5"]
        return presets["3"]
    if isinstance(requested, str):
        if requested.strip() in presets:
            return presets[requested.strip()]
        requested = re.split(r"[,+/\s]+", requested)
    aliases = {
        "pack": "packaging_type",
        "pack_type": "packaging_type",
        "packtype": "packaging_type",
        "packaging": "packaging_type",
        "cat": "category",
        "brands": "brand",
    }
    out = [aliases.get(f.strip().lower(), f.strip().lower()) for f in requested if f and f.strip()]
    if not out:
        raise ValueError(f"--fields {requested!r} names no attribute")
    return tuple(dict.fromkeys(out))


def _find_models(model_dir: str) -> dict[str, str]:
    """``{role: uri}`` for the ``.h5`` files in a local or gs:// directory (role from the file name)."""
    root = str(model_dir).rstrip("/")
    if root.startswith("gs://"):
        bucket, prefix = dataset._split_gs(root)
        names = [
            f"gs://{bucket}/{b.name}"
            for b in dataset._gcs().list_blobs(bucket, prefix=f"{prefix}/" if prefix else "")
        ]
    else:
        names = [str(p) for p in Path(root).rglob("*.h5")]
    found: dict[str, str] = {}
    for uri in sorted(names):
        if not uri.endswith(".h5"):
            continue
        stem = Path(uri).stem.lower()
        role = normalize_role(stem) or ("pack" if "pack" in stem else "brand" if "brand" in stem else "")
        if role:
            found.setdefault(role, uri)
    return found


class KerasClassifier:
    """One legacy ``.h5`` classifier run in this process (inside the Vertex serving container).

    - ``brand`` (``marketshare_brand.h5``): PIL RGB + nearest-neighbour resize to 299x299 +
      ``inception_v3.preprocess_input`` (``x / 127.5 - 1.0`` in ``[-1, 1]``).
    - ``pack`` (``marketshare_packtype.h5``) and all 7 ``*_variant.h5`` models: OpenCV BGR channel
      order + bilinear resize to 299x299 + ``x / 255.0`` in ``[0, 1]``.
    Input conditioning vectors and output classes are sanitized via ``gt-mapping.yml``.
    """

    def __init__(self, role: str, model_uri: str, classes: Sequence[str] = ()):
        self.role = normalize_role(role)
        self.source_uri = str(model_uri)
        self.name = Path(self.source_uri).name
        self.classes = list(classes)
        self._model = None
        self._lock = threading.RLock()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import tensorflow as tf
            except ModuleNotFoundError as e:
                raise RuntimeError(f"{e}: install the CV extra, pip install '.[cv]'") from e
            path = self.source_uri
            if path.startswith("gs://"):
                local = Path(tempfile.mkdtemp(prefix="shelf_bench_cv_")) / self.name
                local.write_bytes(dataset.read_bytes(path))
                path = str(local)
            try:
                model = tf.keras.models.load_model(path, compile=False)
            except Exception:  # Keras 3 cannot read every legacy Keras 2 .h5; tf-keras can
                import tf_keras

                model = tf_keras.models.load_model(path, compile=False)
            n_out = int(model.output_shape[-1])
            if not self.classes and self.role in VARIANT_ROLES:
                gt_cls = gt_variant_classes(self.role, size=n_out)
                if len(gt_cls) == n_out:
                    self.classes = list(gt_cls)
            if not self.classes:
                self.classes = list(load_classes(self.source_uri))
            if not self.classes and self.role in MODEL_ENV:
                self.classes = list(load_classes(os.environ.get(MODEL_ENV[self.role])))
            if not self.classes and self.role == "pack" and n_out == len(pack_classes()):
                self.classes = list(pack_classes())
            self._model = model
            return model

    def _inputs(self, image: Image.Image, category: str | None = None, brand: str | None = None) -> list:
        import numpy as np

        feeds = []
        for inp in self._model.inputs:
            shape = tuple(inp.shape)
            if len(shape) == 4:  # the image; InceptionV3's 299 x 299 when the size is not fixed
                h, w = (shape[1] or 299), (shape[2] or 299)
                if self.role == "brand":
                    rgb = image.convert("RGB").resize((int(w), int(h)), Image.Resampling.NEAREST)
                    feeds.append(np.asarray(rgb, dtype=np.float32)[None] / 127.5 - 1.0)
                else:
                    rgb = image.convert("RGB").resize((int(w), int(h)), Image.Resampling.BILINEAR)
                    bgr = np.asarray(rgb, dtype=np.float32)[..., ::-1]
                    feeds.append(bgr[None] / 255.0)
            elif self.role in VARIANT_ROLES:  # variant model's one-hot brand conditioning vector
                feeds.append(self._brand_onehot(brand or category, int(shape[-1])))
            else:  # the pack model's one-hot legacy category
                feeds.append(self._category_onehot(category, int(shape[-1])))
        return feeds

    def _brand_onehot(self, brand: str | None, size: int):
        import numpy as np

        vec = np.zeros((1, size), dtype=np.float32)
        idx = brand_index(brand, self.classes, role=self.role, size=size)
        if idx is not None and 0 <= idx < size:
            vec[0, idx] = 1.0
        return vec

    @staticmethod
    def _category_onehot(category: str | None, size: int):
        import numpy as np

        from utils.legacy_benchmark import CANONICAL_TO_LEGACY_PACK_CATEGORY

        vec = np.zeros((1, size), dtype=np.float32)
        cats = pack_categories()
        legacy = CANONICAL_TO_LEGACY_PACK_CATEGORY.get(str(category or ""), str(category or ""))
        if len(cats) == size and legacy in cats:
            vec[0, cats.index(legacy)] = 1.0
        return vec

    def predict_class(
        self,
        image: Image.Image,
        category: str | None = None,
        brand: str | None = None,
    ) -> tuple[str, float]:
        """``(class name, softmax probability)``; masks variant outputs to the conditioned brand's classes."""
        import numpy as np

        model = self._load()
        feeds = self._inputs(image, category=category, brand=brand)
        with self._lock:
            probs = np.asarray(model.predict(feeds if len(feeds) > 1 else feeds[0], verbose=0))[0]
        # Brand-consistent output masking for variant models: restrict argmax to classes belonging to `brand`
        if self.role in VARIANT_ROLES and brand and self.classes:
            b_key = _norm_brand_key(brand)
            if b_key:
                limit = min(len(self.classes), len(probs))
                valid_idxs = [
                    i for i in range(limit)
                    if _norm_brand_key(_class_brand_token(self.classes[i])) == b_key
                ]
                if valid_idxs:
                    idx = max(valid_idxs, key=lambda i: float(probs[i]))
                    return self.classes[idx], float(probs[idx])
        idx = int(np.argmax(probs))
        return (self.classes[idx] if idx < len(self.classes) else str(idx)), float(probs[idx])


_SESSION = None
_SESSION_LOCK = threading.Lock()


def _session():
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            _SESSION = AuthorizedSession(creds)
        return _SESSION


def call_cv_endpoint(
    endpoint: str,
    image: Image.Image,
    role: str,
    category: str | None = None,
    brand: str | None = None,
) -> dict:
    """POST one image to a Vertex AI Endpoint serving ``/api/cv/predict``; returns its prediction."""
    m = _ENDPOINT_RE.match(endpoint)
    if not m:
        raise ValueError(f"not a Vertex AI Endpoint resource name: {endpoint!r}")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    inst = {"image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "role": role}
    if category:
        inst["category"] = category
    if brand:
        inst["brand"] = brand
    url = f"https://{m.group(1)}-aiplatform.googleapis.com/v1/{endpoint}:predict"
    for attempt in range(4):
        r = _session().post(url, json={"instances": [inst]}, timeout=120)
        if r.status_code not in _RETRY_STATUS or attempt == 3:
            break
        time.sleep(2 ** attempt)
    if r.status_code != 200:
        raise RuntimeError(f"{endpoint}: HTTP {r.status_code} {r.text[:300]}")
    pred = (r.json().get("predictions") or [{}])[0]
    if pred.get("error"):
        raise RuntimeError(f"{endpoint}: {pred['error']}")
    return pred


class EndpointClassifier:
    """A legacy classifier deployed on a Vertex AI Endpoint (``projects/../endpoints/<id>``)."""

    def __init__(self, role: str, endpoint: str, model_uri: str | None = None):
        if not _ENDPOINT_RE.match(endpoint):
            raise ValueError(f"not a Vertex AI Endpoint resource name: {endpoint!r}")
        self.role = normalize_role(role)
        self.endpoint, self.model_uri = endpoint, model_uri
        self.name = Path(model_uri).name if model_uri else endpoint.rsplit("/", 1)[-1]
        self.source_uri = f"{model_uri} @ {endpoint}" if model_uri else endpoint

    @property
    def classes(self) -> tuple[str, ...]:
        if self.role == "pack":
            return pack_classes()
        if self.role in VARIANT_ROLES:
            gt_cls = gt_variant_classes(self.role, size=367 if self.role == "nonhul_p6_variant" else None)
            if gt_cls:
                return gt_cls
        loaded = load_classes(self.model_uri)
        if loaded:
            return loaded
        return ()

    def predict_class(
        self,
        image: Image.Image,
        category: str | None = None,
        brand: str | None = None,
    ) -> tuple[str, float]:
        pred = call_cv_endpoint(self.endpoint, image, self.role, category=category, brand=brand)
        label = str(pred.get("raw_class") or "").strip()
        if label.isdigit():  # the server had no class list (no <model>_classes.json in its artifacts)
            classes = self.classes
            label = classes[int(label)] if int(label) < len(classes) else label
        return label, float(pred.get("confidence") or 0.0)


def _classifier(role: str, ref: str, model_uri: str | None = None) -> KerasClassifier | EndpointClassifier:
    if str(ref).startswith("projects/"):
        return EndpointClassifier(role, str(ref), model_uri)
    return KerasClassifier(role, str(ref))


class CVPipeline:
    """The brand, pack, and category-routed variant classifiers (in-process ``.h5`` or Vertex AI Endpoints)."""

    def __init__(self, refs: Mapping[str, str | None], model_uris: Mapping[str, str | None] | None = None):
        self.runners: dict[str, KerasClassifier | EndpointClassifier] = {
            normalize_role(role): _classifier(role, ref, (model_uris or {}).get(normalize_role(role)))
            for role, ref in refs.items() if ref
        }

    @classmethod
    def from_config(cls, config: dict, roles: Iterable[str] | None = None) -> CVPipeline:
        """Benchmark side: the Vertex AI Endpoints in ``cv_models``. ``--brand-model`` /
        ``--pack-model`` / ``--<variant>-model`` override them with another endpoint or ``.h5``."""
        cv = config.get("cv_models") or {}
        if roles is None:
            roles = ["brand", "pack", *configured_variant_roles(config)]
        norm_roles = [normalize_role(r) for r in roles if normalize_role(r)]
        refs = {r: os.environ.get(MODEL_ENV[r]) or cv.get(f"{r}_endpoint") for r in norm_roles}
        missing = [r for r in norm_roles if not refs[r]]
        if missing:
            raise ValueError(f"no {', '.join(missing)} model: set cv_models.<role>_endpoint in config.yaml")
        return cls(refs, {r: cv.get(f"{r}_model") for r in norm_roles})

    @classmethod
    def for_serving(cls, config: dict) -> CVPipeline:
        """Prediction-server side: the ``.h5`` in the Vertex artifact dir (``AIP_STORAGE_URI``), else
        ``SHELF_BENCH_CV_<ROLE>_MODEL`` / ``cv_models.<role>_model``, limited to
        ``SHELF_BENCH_CV_ROLE`` when it is set."""
        cv = config.get("cv_models") or {}
        aip = os.environ.get("AIP_STORAGE_URI")
        refs = _find_models(aip) if aip else {r: os.environ.get(MODEL_ENV[r]) or cv.get(f"{r}_model") for r in ROLES}
        role = normalize_role(os.environ.get("SHELF_BENCH_CV_ROLE"))
        return cls({role: refs.get(role)} if role else refs)

    def predict(self, image: Image.Image, fields: Sequence[str], category_hint: str | None = None) -> dict[str, str]:
        """The benchmark fields for one product photo (see ``predict_coupled``)."""
        fset = set(fields)
        with_pack = "pack" in self.runners and bool({"packaging_type", "category"} & fset)
        with_variant = any(r in self.runners for r in VARIANT_ROLES) and (
            bool({"product", "base_variant", "variant"} & fset) or "brand" not in self.runners
        )
        pred = self.predict_coupled(
            image,
            with_pack=with_pack,
            with_variant=with_variant,
            category_hint=category_hint,
        )
        return {f: pred.get(f, "") for f in fields}

    def predict_coupled(
        self,
        image: Image.Image,
        with_pack: bool = True,
        with_variant: bool | None = None,
        category_hint: str | None = None,
    ) -> dict[str, Any]:
        """Brand model -> pack model -> ``gt-mapping.yml``-routed variant model, coupled:
        1. Brand model predicts ``brand`` and implies a category + HUL ownership hint.
        2. Pack model takes that category as its 10-d one-hot input (``category_mapping``) and
           predicts ``Category__PackType`` (``pack_type_mapping``).
        3. The brand + category route the image to one of the 7 variant models (``wash_variant``,
           ``skin_variant``, ``p5_variant``, ``oral_variant``, ``gen_variant``,
           ``nonhul_p6_variant``, ``nonhul_gen_variant``) using ``gt-mapping.yml``'s
           ``*_brand_mapping`` tables. If the predicted variant's ``gt-mapping.yml`` category
           differs from the initial brand category hint (for multi-category umbrella brands like
           Dove, Ayush, Lux, LBP, Tata), the pack model is re-conditioned with the variant's true
           ``gt-mapping.yml`` category.
        """
        from utils.legacy_benchmark import CANONICAL_TO_LEGACY_PACK_CATEGORY, normalize_legacy_prediction

        out: dict[str, Any] = {}
        b: dict[str, str] = {}
        is_hul_flag: bool | None = None

        if "brand" in self.runners:
            brand_cls, brand_conf = self.runners["brand"].predict_class(image)
            b = parse_label("brand", brand_cls)
            norm_b = normalize_legacy_prediction(brand_pred=brand_cls) if brand_cls and not brand_cls.isdigit() else {}
            if norm_b.get("brand") and norm_b["brand"] != "Unknown":
                is_hul_flag = bool(norm_b.get("is_hul"))
            out.update({
                "brand": b.get("brand", ""),
                "brand_class": brand_cls,
                "brand_confidence": round(float(brand_conf), 4),
            })
            if is_hul_flag is not None:
                out["is_hul"] = is_hul_flag

        initial_pack_hint = b.get("category") or category_hint or None
        if with_pack and "pack" in self.runners:
            pack_cls, pack_conf = self.runners["pack"].predict_class(image, initial_pack_hint)
            p = parse_label("pack", pack_cls)
            out.update({
                "packaging_type": p.get("packaging_type", ""),
                "category": p.get("category", "") or b.get("category", ""),
                "pack_class": pack_cls,
                "pack_confidence": round(float(pack_conf), 4),
                "category_hint": initial_pack_hint or "",
            })
        elif initial_pack_hint:
            out.setdefault("category", initial_pack_hint)

        if with_variant is None:
            with_variant = any(r in self.runners for r in VARIANT_ROLES)

        if with_variant and any(r in self.runners for r in VARIANT_ROLES):
            routed_cat = category_hint or out.get("category") or b.get("category") or ""
            routed_brand = out.get("brand") or ""
            v_role = route_variant_role(
                category=routed_cat,
                brand=routed_brand,
                is_hul=is_hul_flag,
                runners=self.runners,
            )
            if v_role and v_role in self.runners:
                brand_hint = routed_brand or routed_cat or None
                var_cls, var_conf = self.runners[v_role].predict_class(
                    image,
                    category=routed_cat or None,
                    brand=brand_hint,
                )
                v = parse_gt_variant_label(v_role, var_cls)
                out.update({
                    "variant_role": v_role,
                    "variant_class": var_cls,
                    "variant_confidence": round(float(var_conf), 4),
                    "brand_hint": routed_brand,
                    "routed_category": routed_cat,
                    "base_variant": v.get("base_variant", ""),
                    "variant": v.get("variant", ""),
                })
                for meta_k in ("gt_brand", "gt_category", "gt_subcategory", "gt_variant", "gt_label"):
                    if v.get(meta_k):
                        out[meta_k] = v[meta_k]
                if not out.get("brand") and v.get("brand"):
                    out["brand"] = v["brand"]
                if out.get("brand") and out.get("base_variant"):
                    out["product"] = f"{out['brand']}__{out['base_variant']}"
                elif v.get("product"):
                    out["product"] = v["product"]

                # If gt-mapping.yml specifies a variant-level category that differs from the initial
                # umbrella-brand category hint (and the predicted variant belongs to the routed brand),
                # update category and re-condition the pack model.
                var_gt_cat = str(v.get("gt_category") or "")
                var_canon_cat = str(v.get("category") or "")
                var_brand_matches = not routed_brand or (
                    _norm_brand_key(v.get("gt_brand") or v.get("brand")) == _norm_brand_key(routed_brand)
                )
                init_leg_cat = CANONICAL_TO_LEGACY_PACK_CATEGORY.get(
                    str(initial_pack_hint or ""), str(initial_pack_hint or "")
                )
                if var_brand_matches and var_gt_cat and var_gt_cat in pack_categories() and var_gt_cat != init_leg_cat:
                    if with_pack and "pack" in self.runners:
                        pack_cls2, pack_conf2 = self.runners["pack"].predict_class(image, var_gt_cat)
                        p2 = parse_label("pack", pack_cls2)
                        out.update({
                            "packaging_type": p2.get("packaging_type", "") or out.get("packaging_type", ""),
                            "category": p2.get("category", "") or var_canon_cat or out.get("category", ""),
                            "pack_class": pack_cls2,
                            "pack_confidence": round(float(pack_conf2), 4),
                            "category_hint": var_gt_cat,
                        })
                    elif var_canon_cat:
                        out["category"] = var_canon_cat
                elif not out.get("category") and var_canon_cat:
                    out["category"] = var_canon_cat
                if not out.get("packaging_type") and v.get("packaging_type"):
                    out["packaging_type"] = v["packaging_type"]
        return out
