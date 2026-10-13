"""Legacy CV models (cv_classify): the .h5 classifier, the Vertex AI endpoint client, the
prediction server route and the Vertex AI deployment. No network, no TensorFlow."""

from __future__ import annotations

import base64
import io
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from approaches.base import Context
from approaches.market_share.classification.cv_models_classify import CVClassify
from utils import cloud, cv_models, metrics, server

BRAND_EP = "projects/p/locations/us-central1/endpoints/111"
PACK_EP = "projects/p/locations/us-central1/endpoints/222"
CFG = {"cv_models": {"brand_model": "gs://b/mt_brand/marketshare_brand.h5", "brand_endpoint": BRAND_EP,
                     "pack_model": "gs://b/mt_pack/marketshare_packtype.h5", "pack_endpoint": PACK_EP}}


class _Input:
    def __init__(self, shape):
        self.shape = shape


class FakeKeras:
    """The legacy pack model's signature: [10-d category one-hot, image], 46-way softmax."""

    def __init__(self, winner: int, n: int = 46, inputs=((None, 10), (None, None, None, 3))):
        self.inputs = [_Input(s) for s in inputs]
        self.output_shape = (None, n)
        self.winner, self.n, self.feeds = winner, n, None

    def predict(self, feeds, verbose=0):
        self.feeds = feeds
        out = np.full((1, self.n), 0.1 / (self.n - 1), dtype=np.float32)
        out[0, self.winner] = 0.9
        return out


def test_pack_classes_come_from_main_legacy_mapping():
    classes = cv_models.pack_classes()
    assert len(classes) == 46 and list(classes) == sorted(classes) and "Hair__Bottle" in classes
    assert len(cv_models.pack_categories()) == 10
    assert cv_models.parse_label("pack", "Hair__Bottle") == {"category": "Hair_Care", "packaging_type": "Bottle"}
    assert cv_models.parse_label("brand", "APepsodent")["brand"] == "Pepsodent"
    assert cv_models.parse_label("brand", "107") == {}  # an unmapped class index is no answer


def test_keras_classifier_inception_preprocessing_category_onehot_and_argmax():
    idx = cv_models.pack_classes().index("Hair__Bottle")
    clf = cv_models.KerasClassifier("pack_type", "gs://b/mt_pack/marketshare_packtype.h5", cv_models.pack_classes())
    clf._model = FakeKeras(idx)
    img = Image.new("RGB", (500, 400), (255, 0, 128))

    assert clf.predict_class(img) == ("Hair__Bottle", pytest.approx(0.9))
    onehot, x = clf._model.feeds
    assert x.shape == (1, 299, 299, 3)  # InceptionV3 size when the model's input size is free
    assert x[0, 0, 0].tolist() == pytest.approx([1.0, -1.0, 128 / 127.5 - 1.0], abs=1e-6)  # x / 127.5 - 1
    assert not onehot.any()  # no category -> zeros (what the deployed endpoint feeds)

    clf.predict_class(img, category="Hair_Care")  # canonical name -> legacy "Hair" slot
    assert clf._model.feeds[0][0].tolist() == [float(c == "Hair") for c in cv_models.pack_categories()]


class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        status, body = self.replies[url.split("/endpoints/")[1].split(":")[0]]

        class R:
            status_code, text = status, "x"

            def json(self):
                return body

        return R()


def test_pipeline_calls_vertex_endpoints_and_maps_brand_index(monkeypatch):
    sess = FakeSession({"111": (200, {"predictions": [{"raw_class": "1", "confidence": 0.98}]}),
                        "222": (200, {"predictions": [{"raw_class": "Oral_Care__Tube", "confidence": 0.7}]})})
    monkeypatch.setattr(cv_models, "_session", lambda: sess)
    monkeypatch.setattr(cv_models, "load_classes", lambda uri: ("AColgate", "APepsodent") if uri else ())
    monkeypatch.delenv("SHELF_BENCH_CV_BRAND_MODEL", raising=False)
    monkeypatch.delenv("SHELF_BENCH_CV_PACK_MODEL", raising=False)

    pipe = cv_models.CVPipeline.from_config(CFG)
    pred = pipe.predict(Image.new("RGB", (64, 64)), ("brand", "packaging_type", "category"))

    assert pred == {"brand": "Pepsodent", "packaging_type": "Tube", "category": "Oral_Care"}
    (brand_url, brand_body), (pack_url, pack_body) = sess.calls
    assert brand_url == f"https://us-central1-aiplatform.googleapis.com/v1/{BRAND_EP}:predict"
    assert brand_body["instances"][0]["role"] == "brand" and brand_body["instances"][0]["image_b64"]
    assert pack_body["instances"][0]["category"] == "Oral_Care"  # brand-derived hint for the pack model
    assert [r.name for r in pipe.runners.values()] == ["marketshare_brand.h5", "marketshare_packtype.h5"]

    # packaging_type alone still runs brand first (for the hint); category comes from the pack class.
    sess.calls.clear()
    sess.replies["222"] = (200, {"predictions": [{"raw_class": "Skin__Tube", "confidence": 0.6}]})
    assert pipe.predict(Image.new("RGB", (64, 64)), ("packaging_type", "category")) == {
        "packaging_type": "Tube", "category": "Skin"}
    assert len(sess.calls) == 2 and sess.calls[1][1]["instances"][0]["category"] == "Oral_Care"
    sess.calls.clear()
    assert pipe.predict(Image.new("RGB", (64, 64)), ("brand",)) == {"brand": "Pepsodent"}
    assert len(sess.calls) == 1  # brand-only scoring skips the pack model

    sess.replies["111"] = (403, {})
    with pytest.raises(RuntimeError, match="HTTP 403"):
        pipe.predict(Image.new("RGB", (64, 64)), ("brand",))
    with pytest.raises(ValueError, match="Vertex AI Endpoint"):
        cv_models.EndpointClassifier("brand", "https://x.run.app")


def test_cv_classify_fields_follow_dataset_and_model(monkeypatch):
    monkeypatch.delenv("SHELF_BENCH_CV_BRAND_MODEL", raising=False)
    monkeypatch.delenv("SHELF_BENCH_CV_PACK_MODEL", raising=False)

    def setup(dataset_name, model, fields=None):
        a = CVClassify()
        a.dataset, a.evaluated_fields = dataset_name, fields
        a.setup(CFG, Context(model=model))
        return a.evaluated_fields, sorted(a.pipeline.runners), a.resolved_model

    assert setup("hul_golden", "keras-h5") == (
        ("brand", "packaging_type", "category"), ["brand", "pack"], "marketshare_brand.h5+marketshare_packtype.h5")
    assert setup("hul_brands", "keras-h5") == (("brand",), ["brand"], "marketshare_brand.h5")
    # Pack fields always run the brand model first: its category is the pack model's input.
    assert setup("hul_packs", "keras-h5")[:2] == (("packaging_type", "category"), ["brand", "pack"])
    assert setup("hul_golden", "keras-h5", ("packaging_type",))[:2] == (("packaging_type",), ["brand", "pack"])
    assert setup("hul_golden", "keras-h5", ("brand",))[:2] == (("brand",), ["brand"])
    assert cv_models.resolve_evaluated_fields("hul_golden", "pack_type,cat") == ("packaging_type", "category")
    assert cv_models.resolve_evaluated_fields("hul_packs", "1") == ("packaging_type",)


def test_fields_subset_needs_every_field_right():
    gt = {"product": "Dove__Soap", "brand": "Dove", "packaging_type": "Bar", "category": "PW"}
    assert metrics.match_product({"brand": "Dove", "packaging_type": "Bar"}, gt, ("brand", "packaging_type"))["tp"] == 1
    assert metrics.match_product({"brand": "Dove", "packaging_type": "Tube"}, gt, ("brand", "packaging_type"))["tp"] == 0


def test_server_cv_health_and_predict_routes(monkeypatch):
    idx = cv_models.pack_classes().index("Skin__Bottle")
    clf = cv_models.KerasClassifier("pack", "gs://b/mt_pack/marketshare_packtype.h5", cv_models.pack_classes())
    clf._model = FakeKeras(idx)
    monkeypatch.setattr(server, "_CV_PIPELINE", cv_models.CVPipeline({}))
    server._CV_PIPELINE.runners = {"pack": clf}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api/cv"
    try:
        health = json.loads(urllib.request.urlopen(f"{base}/health").read())
        assert health == {"status": "healthy", "models": {"pack": "gs://b/mt_pack/marketshare_packtype.h5"}}
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="JPEG")
        body = json.dumps({"instances": [{"image_b64": base64.b64encode(buf.getvalue()).decode(),
                                          "role": "pack_type", "category": "Skin"}]}).encode()
        req = urllib.request.Request(f"{base}/predict", data=body, headers={"Content-Type": "application/json"})
        (pred,) = json.loads(urllib.request.urlopen(req).read())["predictions"]
    finally:
        httpd.shutdown()
    assert pred["raw_class"] == "Skin__Bottle" and pred["confidence"] == pytest.approx(0.9)
    assert (pred["category"], pred["packaging_type"]) == ("Skin", "Bottle")
    assert clf._model.feeds[0][0, cv_models.pack_categories().index("Skin")] == 1.0


def test_deploy_cv_endpoints_uses_vertex_operations_only(monkeypatch):
    calls = []
    region_api = "https://us-central1-aiplatform.googleapis.com/v1"

    class Resp:
        def __init__(self, body):
            self.body, self.status_code, self.content = body, 200, b"{}"

        def json(self):
            return self.body

    class Sess:
        def post(self, url, json=None, timeout=None):
            calls.append(("POST", url, json))
            return Resp({"name": f"{url.split('/v1/')[1].split(':')[0]}/operations/1"})

        def get(self, url, timeout=None, params=None):
            calls.append(("GET", url, None))
            if "/operations/" in url:
                return Resp({"done": True, "response": {"model": "projects/1/locations/us-central1/models/9"}})
            if url.endswith("/endpoints"):
                return Resp({"endpoints": [{"name": "projects/1/locations/us-central1/endpoints/7",
                                            "displayName": "marketshare-brand-vertex-endpoint"},
                                           {"name": "projects/1/locations/us-central1/endpoints/8",
                                            "displayName": "marketshare-packtype-vertex-endpoint"}]})
            return Resp({"deployedModels": [{"id": "old1"}]})

    monkeypatch.setattr(cloud, "_session", Sess)
    monkeypatch.setattr(cloud, "load_config", lambda: {"gcp": {"project": "p", "region": "us-central1"}, **CFG})
    out = cloud.deploy_cv_endpoints(image="img:1", log=lambda *a: None)

    assert out["brand"]["endpoint"].endswith("/endpoints/7") and out["pack"]["endpoint"].endswith("/endpoints/8")
    assert all(url.startswith(region_api) for _, url, _ in calls)  # no run.googleapis.com polling
    upload = next(body for verb, url, body in calls if url.endswith("/models:upload"))["model"]
    assert upload["artifactUri"] == "gs://b/mt_brand"
    assert upload["containerSpec"]["predictRoute"] == "/api/cv/predict"
    assert {"name": "SHELF_BENCH_CV_ROLE", "value": "brand"} in upload["containerSpec"]["env"]
    posts = [url.rsplit(":", 1)[-1] for verb, url, _ in calls if verb == "POST"]
    assert posts == ["upload", "deployModel", "undeployModel"] * 2
