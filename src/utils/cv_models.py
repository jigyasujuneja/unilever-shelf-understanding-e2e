"""Legacy InceptionV3 Keras (``.h5``) brand and pack classifiers, used by ``-a cv_classify``.

The two ``.h5`` files (``cv_models`` in config.yaml) are registered in Vertex AI Model Registry
and deployed to Vertex AI Endpoints (``shelf-bench deploy-cv-endpoints``). On an endpoint, this
repo's image runs ``shelf-bench serve``, and its ``/api/cv/predict`` route runs the model with
``KerasClassifier``. The benchmark calls the endpoints through ``EndpointClassifier``.

- brand model: a 941-way softmax over the sorted ``brand_data`` folder names. The class list is
  stored next to the weights in ``<model>_classes.json``.
- pack model: a 46-way softmax over the legacy ``Category__PackType`` pack_data folders. It has a
  second input, a 10-d one-hot of the legacy category; all zeros means the category is unknown.

Class names become benchmark fields through ``legacy_benchmark.normalize_legacy_prediction``.
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

ROLES = ("brand", "pack")
# Model overrides: set by `run --brand-model/--pack-model` and by the serving container's env.
MODEL_ENV = {"brand": "SHELF_BENCH_CV_BRAND_MODEL", "pack": "SHELF_BENCH_CV_PACK_MODEL"}
_ENDPOINT_RE = re.compile(r"^projects/[^/]+/locations/([^/]+)/endpoints/[^/]+$")
_RETRY_STATUS = (429, 500, 502, 503, 504)


def normalize_role(role: str | None) -> str:
    """``brand`` / ``pack`` (``pack_type`` and ``packaging_type`` mean ``pack``), else ``""``."""
    r = str(role or "").strip().lower()
    return "pack" if r.startswith("pack") else "brand" if r == "brand" else ""


@lru_cache(maxsize=1)
def pack_classes() -> tuple[str, ...]:
    """The pack model's 46 output classes: the sorted legacy ``Category__PackType`` folders."""
    from utils.legacy_benchmark import build_legacy_mapping_table

    return tuple(sorted({r["legacy_pack_folder"] for r in build_legacy_mapping_table()["category_packs"]}))


@lru_cache(maxsize=1)
def pack_categories() -> tuple[str, ...]:
    """The 10 legacy categories, in the order of the pack model's one-hot category input."""
    return tuple(sorted({c.split("__", 1)[0] for c in pack_classes()}))


@lru_cache(maxsize=8)
def load_classes(model_uri: str | None) -> tuple[str, ...]:
    """Class names stored as ``<model>_classes.json`` next to a ``.h5`` (local or gs://), else ()."""
    from google.api_core.exceptions import NotFound

    if not model_uri or not str(model_uri).endswith(".h5"):
        return ()
    try:
        return tuple(str(c) for c in json.loads(dataset.read_bytes(str(model_uri)[:-3] + "_classes.json")))
    except (FileNotFoundError, NotFound):
        return ()


def parse_label(role: str, label: str) -> dict[str, str]:
    """Benchmark fields for one model class (a brand_data folder or a ``Category__PackType``)."""
    from utils.legacy_benchmark import normalize_legacy_prediction

    if not label or label.isdigit():
        return {}
    if normalize_role(role) == "brand":
        p = normalize_legacy_prediction(brand_pred=label)
        keys = ("brand", "category")
    else:
        p = normalize_legacy_prediction(pack_pred=label)
        keys = ("category", "packaging_type")
    return {k: str(p[k]) for k in keys if p.get(k) and p[k] != "Unknown"}


def resolve_evaluated_fields(
    dataset_name: str,
    requested: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    """The attributes to score. ``requested`` is ``--fields``: '1', '2', '3' or names like
    'brand,packaging_type'. Without it: what the legacy models predict on ``dataset_name``."""
    packs_only = dataset_name == "hul_packs"
    presets = {
        "1": ("packaging_type",) if packs_only else ("brand",),
        "2": ("packaging_type", "category") if packs_only else ("brand", "packaging_type"),
        "3": ("brand", "packaging_type", "category"),
    }
    if requested is None or requested == "" or requested == ():
        if dataset_name == "hul_brands":
            return ("brand",)
        return presets["2"] if packs_only else presets["3"]
    if isinstance(requested, str):
        if requested.strip() in presets:
            return presets[requested.strip()]
        requested = re.split(r"[,+/\s]+", requested)
    aliases = {"pack": "packaging_type", "pack_type": "packaging_type", "packtype": "packaging_type",
               "packaging": "packaging_type", "cat": "category", "brands": "brand"}
    out = [aliases.get(f.strip().lower(), f.strip().lower()) for f in requested if f and f.strip()]
    if not out:
        raise ValueError(f"--fields {requested!r} names no attribute")
    return tuple(dict.fromkeys(out))


def _find_models(model_dir: str) -> dict[str, str]:
    """``{role: uri}`` for the ``.h5`` files in a local or gs:// directory (role from the file name)."""
    root = str(model_dir).rstrip("/")
    if root.startswith("gs://"):
        bucket, prefix = dataset._split_gs(root)
        names = [f"gs://{bucket}/{b.name}" for b in dataset._gcs().list_blobs(bucket, prefix=f"{prefix}/" if prefix else "")]
    else:
        names = [str(p) for p in Path(root).rglob("*.h5")]
    found: dict[str, str] = {}
    for uri in sorted(names):
        stem = Path(uri).name.lower()
        role = "pack" if "pack" in stem else "brand" if "brand" in stem else ""
        if uri.endswith(".h5") and role:
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
            if not self.classes and self.role == "pack" and int(model.output_shape[-1]) == len(pack_classes()):
                self.classes = list(pack_classes())
            self._model = model
            return model

    def _inputs(self, image: Image.Image, category: str | None) -> list:
        import numpy as np

        feeds = []
        for inp in self._model.inputs:
            shape = tuple(inp.shape)
            if len(shape) == 4:  # the image; InceptionV3's 299 x 299 when the size is not fixed
                h, w = (shape[1] or 299), (shape[2] or 299)
                rgb = image.convert("RGB").resize((int(w), int(h)), Image.Resampling.NEAREST)
                feeds.append(np.asarray(rgb, dtype=np.float32)[None] / 127.5 - 1.0)
            else:  # the pack model's one-hot legacy category
                feeds.append(self._category_onehot(category, int(shape[-1])))
        return feeds

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

    def predict_class(self, image: Image.Image, category: str | None = None) -> tuple[str, float]:
        """``(class name, softmax probability)``; the class index as a string if names are unknown."""
        import numpy as np

        model = self._load()
        feeds = self._inputs(image, category)
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


def call_cv_endpoint(endpoint: str, image: Image.Image, role: str, category: str | None = None) -> dict:
    """POST one image to a Vertex AI Endpoint serving ``/api/cv/predict``; returns its prediction."""
    m = _ENDPOINT_RE.match(endpoint)
    if not m:
        raise ValueError(f"not a Vertex AI Endpoint resource name: {endpoint!r}")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    inst = {"image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "role": role}
    if category:
        inst["category"] = category
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
        return load_classes(self.model_uri) if self.role == "brand" else pack_classes()

    def predict_class(self, image: Image.Image, category: str | None = None) -> tuple[str, float]:
        pred = call_cv_endpoint(self.endpoint, image, self.role, category)
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
    """The brand and/or pack classifier (in-process ``.h5`` or Vertex AI Endpoint) for one run."""

    def __init__(self, refs: Mapping[str, str | None], model_uris: Mapping[str, str | None] | None = None):
        self.runners: dict[str, KerasClassifier | EndpointClassifier] = {
            normalize_role(role): _classifier(role, ref, (model_uris or {}).get(role))
            for role, ref in refs.items() if ref
        }

    @classmethod
    def from_config(cls, config: dict, roles: Iterable[str] = ROLES) -> CVPipeline:
        """Benchmark side: the Vertex AI Endpoints in ``cv_models``. ``--brand-model`` /
        ``--pack-model`` override them with another endpoint or a local / gs:// ``.h5``."""
        cv = config.get("cv_models") or {}
        roles = [normalize_role(r) for r in roles]
        refs = {r: os.environ.get(MODEL_ENV[r]) or cv.get(f"{r}_endpoint") for r in roles}
        missing = [r for r in roles if not refs[r]]
        if missing:
            raise ValueError(f"no {', '.join(missing)} model: set cv_models.<role>_endpoint in config.yaml")
        return cls(refs, {r: cv.get(f"{r}_model") for r in roles})

    @classmethod
    def for_serving(cls, config: dict) -> CVPipeline:
        """Prediction-server side: the ``.h5`` in the Vertex artifact dir (``AIP_STORAGE_URI``), else
        ``SHELF_BENCH_CV_{BRAND,PACK}_MODEL`` / ``cv_models.{brand,pack}_model``, limited to
        ``SHELF_BENCH_CV_ROLE`` when it is set."""
        cv = config.get("cv_models") or {}
        aip = os.environ.get("AIP_STORAGE_URI")
        refs = _find_models(aip) if aip else {r: os.environ.get(MODEL_ENV[r]) or cv.get(f"{r}_model") for r in ROLES}
        role = normalize_role(os.environ.get("SHELF_BENCH_CV_ROLE"))
        return cls({role: refs.get(role)} if role else refs)

    def predict(self, image: Image.Image, fields: Sequence[str]) -> dict[str, str]:
        """The benchmark fields for one product photo (see ``predict_coupled``)."""
        pred = self.predict_coupled(image, with_pack=bool({"packaging_type", "category"} & set(fields)))
        return {f: pred.get(f, "") for f in fields}

    def predict_coupled(self, image: Image.Image, with_pack: bool = True) -> dict[str, Any]:
        """Brand model -> pack model, coupled: the category implied by the predicted brand is the
        pack model's one-hot category input. ``brand`` comes from the brand model; ``packaging_type``
        and ``category`` come from the pack model (its ``Category__PackType`` class). The pack
        model is skipped when only ``brand`` is scored. Also returns both raw classes,
        confidences and the category hint for the trace."""
        brand_cls, brand_conf = self.runners["brand"].predict_class(image)
        b = parse_label("brand", brand_cls)
        out: dict[str, Any] = {"brand": b.get("brand", ""), "brand_class": brand_cls,
                               "brand_confidence": round(float(brand_conf), 4)}
        if with_pack:
            hint = b.get("category") or None
            pack_cls, pack_conf = self.runners["pack"].predict_class(image, hint)
            p = parse_label("pack", pack_cls)
            out.update({"packaging_type": p.get("packaging_type", ""), "category": p.get("category", ""),
                        "pack_class": pack_cls, "pack_confidence": round(float(pack_conf), 4),
                        "category_hint": hint or ""})
        return out
