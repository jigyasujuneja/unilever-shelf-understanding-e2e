"""Legacy InceptionV3 Keras (``.h5``) brand, pack, and category-routed variant classifiers (``-a cv_classify``).

The ``.h5`` files (``cv_models`` in config.yaml) are registered in Vertex AI Model Registry
and deployed to Vertex AI Endpoints (``shelf-bench deploy-cv-endpoints``). On an endpoint, this
repo's image runs ``shelf-bench serve``, and its ``/api/cv/predict`` route runs the model with
``KerasClassifier``. The benchmark calls the endpoints through ``EndpointClassifier``.

- brand model: a 941-way softmax over the sorted ``brand_data`` folder names. The class list is
  stored next to the weights in ``<model>_classes.json``.
- pack model: a 46-way softmax over the legacy ``Category__PackType`` pack_data folders. It has a
  second input, a 10-d one-hot of the legacy category; all zeros means the category is unknown.
- 6 category-routed variant models (``wash_variant``, ``skin_variant``, ``p5_variant``,
  ``oral_variant``, ``nonhul_p6_variant``, ``nonhul_gen_variant``): each predicts ``Brand__Variant``
  folders for its category/ownership silo and takes a one-hot vector of the silo's sorted brands
  as its first input alongside the 299x299 InceptionV3 image tensor.

Class names become benchmark fields through ``legacy_benchmark.normalize_legacy_prediction`` and
``taxonomy.filenames.parse_variant_folder``.
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
    "nonhul_p6_variant": "variant_NH_P6",
    "nonhul_gen_variant": "variant_NH_gen",
}
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
_WASH_CATEGORIES = {"Laundry", "Personal_Wash", "Fabric_Wash", "Wash"}
_ENDPOINT_RE = re.compile(r"^projects/[^/]+/locations/([^/]+)/endpoints/[^/]+$")
_RETRY_STATUS = (429, 500, 502, 503, 504)


def normalize_role(role: str | None) -> str:
    """``brand``, ``pack``, or one of ``VARIANT_ROLES`` (aliases mapped), else ``""``."""
    r = str(role or "").strip().lower()
    if r in _ROLE_ALIASES:
        return _ROLE_ALIASES[r]
    return "pack" if r.startswith("pack") else ""


@lru_cache(maxsize=1)
def pack_classes() -> tuple[str, ...]:
    """The pack model's 46 output classes: the sorted legacy ``Category__PackType`` folders."""
    from utils.legacy_benchmark import build_legacy_mapping_table

    return tuple(sorted({r["legacy_pack_folder"] for r in build_legacy_mapping_table()["category_packs"]}))


@lru_cache(maxsize=1)
def pack_categories() -> tuple[str, ...]:
    """The 10 legacy categories, in the order of the pack model's one-hot category input."""
    return tuple(sorted({c.split("__", 1)[0] for c in pack_classes()}))


@lru_cache(maxsize=16)
def load_classes(model_uri: str | None) -> tuple[str, ...]:
    """Class names stored as ``<model>_classes.json`` next to a ``.h5`` (local or gs://), else ()."""
    from google.api_core.exceptions import NotFound

    if not model_uri or not str(model_uri).endswith(".h5"):
        return ()
    try:
        return tuple(str(c) for c in json.loads(dataset.read_bytes(str(model_uri)[:-3] + "_classes.json")))
    except (FileNotFoundError, NotFound):
        return ()


@lru_cache(maxsize=16)
def variant_brands(classes: tuple[str, ...]) -> tuple[str, ...]:
    """The sorted legacy brand prefixes ``<Brand>`` from ``<Brand>__<Variant>`` class folders."""
    return tuple(sorted({c.split("__", 1)[0] for c in classes if "__" in c}))


def brand_index(brand: str | None, classes: Sequence[str]) -> int | None:
    """Index of ``brand`` (canonical or raw legacy folder prefix) in ``variant_brands(classes)``."""
    from utils.taxonomy.brands import canonicalize_brand_name

    raw = str(brand or "").strip()
    if not raw or not classes:
        return None
    brands = variant_brands(tuple(classes))
    low = raw.lower()
    for i, b in enumerate(brands):
        if b.lower() == low:
            return i
    canon = canonicalize_brand_name(raw).lower()
    for i, b in enumerate(brands):
        if canonicalize_brand_name(b).lower() == canon:
            return i
    return None


def route_variant_role(
    category: str | None = None,
    brand: str | None = None,
    is_hul: bool | None = None,
    runners: Mapping[str, Any] | None = None,
) -> str:
    """Route an image to the appropriate variant model role based on category, brand, and HUL ownership."""
    direct = normalize_role(category)
    if direct in VARIANT_ROLES:
        return direct
    cat = str(category or "").strip()
    if is_hul is None and brand:
        from utils.legacy_benchmark import normalize_legacy_prediction

        norm_b = normalize_legacy_prediction(brand_pred=brand)
        if norm_b.get("brand") != "Unknown":
            is_hul = bool(norm_b.get("is_hul"))
            if not cat and norm_b.get("category") != "Unknown":
                cat = str(norm_b["category"])

    if is_hul is False:
        cat_role = "nonhul_p6_variant" if cat in _P5_P6_CATEGORIES else "nonhul_gen_variant"
    elif cat in _ORAL_CATEGORIES:
        cat_role = "oral_variant"
    elif cat in _P5_P6_CATEGORIES:
        cat_role = "p5_variant"
    elif cat in _WASH_CATEGORIES:
        cat_role = "wash_variant"
    else:
        cat_role = "skin_variant"

    if brand and runners:
        matches = [
            r for r in VARIANT_ROLES
            if r in runners and brand_index(brand, runners[r].classes) is not None
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return cat_role if cat_role in matches else matches[0]

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


def parse_label(role: str, label: str) -> dict[str, str]:
    """Benchmark fields for one model class (brand_data, Category__PackType, or Brand__Variant)."""
    from utils.legacy_benchmark import normalize_legacy_prediction
    from utils.taxonomy.filenames import parse_variant_folder

    if not label or label.isdigit():
        return {}
    r = normalize_role(role)
    if r == "brand":
        p = normalize_legacy_prediction(brand_pred=label)
        keys = ("brand", "category")
        return {k: str(p[k]) for k in keys if p.get(k) and p[k] != "Unknown"}
    if r in VARIANT_ROLES:
        silo = VARIANT_SILOS.get(r, "variant_skin")
        p = parse_variant_folder(silo, str(label).strip())
        out: dict[str, str] = {}
        for k in ("brand", "base_variant", "variant", "category", "packaging_type"):
            if p.get(k) and p[k] != "Unknown":
                out[k] = str(p[k])
        if out.get("brand") and out.get("base_variant"):
            out["product"] = f"{out['brand']}__{out['base_variant']}"
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

    Preprocessing matches ``keras.preprocessing.image.load_img(target_size=(299, 299))``
    (nearest-neighbour resize) followed by ``inception_v3.preprocess_input`` (x / 127.5 - 1).
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
            if not self.classes:
                self.classes = list(load_classes(self.source_uri))
            if not self.classes and self.role in MODEL_ENV:
                self.classes = list(load_classes(os.environ.get(MODEL_ENV[self.role])))
            if not self.classes and self.role == "pack" and int(model.output_shape[-1]) == len(pack_classes()):
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
                rgb = image.convert("RGB").resize((int(w), int(h)), Image.Resampling.NEAREST)
                feeds.append(np.asarray(rgb, dtype=np.float32)[None] / 127.5 - 1.0)
            elif self.role in VARIANT_ROLES:  # variant model's one-hot brand conditioning vector
                feeds.append(self._brand_onehot(brand or category, int(shape[-1])))
            else:  # the pack model's one-hot legacy category
                feeds.append(self._category_onehot(category, int(shape[-1])))
        return feeds

    def _brand_onehot(self, brand: str | None, size: int):
        import numpy as np

        vec = np.zeros((1, size), dtype=np.float32)
        idx = brand_index(brand, self.classes)
        if idx is not None and len(variant_brands(tuple(self.classes))) == size and idx < size:
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
        """``(class name, softmax probability)``; the class index as a string if names are unknown."""
        import numpy as np

        model = self._load()
        feeds = self._inputs(image, category=category, brand=brand)
        with self._lock:
            probs = np.asarray(model.predict(feeds if len(feeds) > 1 else feeds[0], verbose=0))[0]
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
        return pack_classes() if self.role == "pack" else load_classes(self.model_uri)

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
        """Brand model -> pack model -> category-routed variant model, coupled:
        1. Brand model predicts ``brand`` and implies a category + HUL ownership hint.
        2. Pack model takes that category as its 10-d one-hot input and predicts ``Category__PackType``.
        3. The category + brand route the image to one of the 6 variant models (``wash_variant``,
           ``skin_variant``, ``p5_variant``, ``oral_variant``, ``nonhul_p6_variant``,
           ``nonhul_gen_variant``), passing the predicted brand as its one-hot brand input to
           predict ``Brand__Variant`` (``base_variant`` and ``product``).
        """
        from utils.legacy_benchmark import normalize_legacy_prediction

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

        if with_pack and "pack" in self.runners:
            hint = b.get("category") or category_hint or None
            pack_cls, pack_conf = self.runners["pack"].predict_class(image, hint)
            p = parse_label("pack", pack_cls)
            out.update({
                "packaging_type": p.get("packaging_type", ""),
                "category": p.get("category", "") or b.get("category", ""),
                "pack_class": pack_cls,
                "pack_confidence": round(float(pack_conf), 4),
                "category_hint": hint or "",
            })
        elif b.get("category") or category_hint:
            out.setdefault("category", b.get("category") or category_hint or "")

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
                v = parse_label(v_role, var_cls)
                out.update({
                    "variant_role": v_role,
                    "variant_class": var_cls,
                    "variant_confidence": round(float(var_conf), 4),
                    "brand_hint": routed_brand,
                    "routed_category": routed_cat,
                    "base_variant": v.get("base_variant", ""),
                    "variant": v.get("variant", ""),
                })
                if not out.get("brand") and v.get("brand"):
                    out["brand"] = v["brand"]
                if out.get("brand") and out.get("base_variant"):
                    out["product"] = f"{out['brand']}__{out['base_variant']}"
                elif v.get("product"):
                    out["product"] = v["product"]
                if not out.get("category") and v.get("category"):
                    out["category"] = v["category"]
                if not out.get("packaging_type") and v.get("packaging_type"):
                    out["packaging_type"] = v["packaging_type"]
        return out
