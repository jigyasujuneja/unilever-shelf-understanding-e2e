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


def test_variant_routing_and_brand_onehot(monkeypatch):
    wash_classes = ("Comfort__Blue_Morning_Fresh", "Lifebuoy__Lifebuoy_Total", "St_Ives__St_Ives_Oatmeal")
    skin_classes = ("Dove__Dove_Lotion", "Lakme__Lakme_9to5", "Vaseline__Vaseline_Cocoa")
    p5_classes = ("Bru__Bru_Gold", "Knorr__Knorr_Soup", "Taj_Mahal__Taj_Mahal_Tea")
    oral_classes = ("Close_Up__Close_Up_Red", "Oral_B__Oral_B_Fresh", "Pepsodent__Pepsodent_2in1")
    gen_classes = ("Clear__Active_Care", "Clinic_Plus__Strong_And_Long", "Sunsilk__Black_Shine")
    nh_p6_classes = ("Dettol__Dettol_Liquid", "Maggi__Maggi_Noodles", "Savlon__Savlon_Spray")
    nh_gen_classes = ("Colgate__Colgate_Strong", "Dettol__Dettol_Soap", "Nivea__Nivea_Soft")
    class_map = {
        "gs://b/wash_variant.h5": wash_classes,
        "gs://b/skin_variant.h5": skin_classes,
        "gs://b/p5_variant.h5": p5_classes,
        "gs://b/oral_variant.h5": oral_classes,
        "gs://b/gen_variant.h5": gen_classes,
        "gs://b/nonhul_p6_variant.h5": nh_p6_classes,
        "gs://b/nonhul_gen_variant.h5": nh_gen_classes,
    }
    monkeypatch.setattr(cv_models, "load_classes", lambda uri: class_map.get(uri, ()))
    cv_models.variant_brands.cache_clear()
    runners = {
        r: cv_models.KerasClassifier(r, f"gs://b/{r}.h5", class_map[f"gs://b/{r}.h5"])
        for r in cv_models.VARIANT_ROLES
    }

    # Category + ownership routing and brand-silo disambiguation (including gen_variant and PW)
    assert cv_models.route_variant_role(category="Oral_Care", brand="Pepsodent", is_hul=True, runners=runners) == "oral_variant"
    assert cv_models.route_variant_role(category="Oral_Care", brand="Colgate", is_hul=False, runners=runners) == "nonhul_gen_variant"
    assert cv_models.route_variant_role(category="Hair_Care", brand="Clinic_Plus", is_hul=True, runners=runners) == "gen_variant"
    assert cv_models.route_variant_role(category="PW", brand="Lifebuoy", is_hul=True, runners=runners) == "wash_variant"
    assert cv_models.route_variant_role(category="PW", brand="Dove", is_hul=True, runners=runners) == "skin_variant"
    assert cv_models.route_variant_role(category="Foods", brand="Knorr", is_hul=True, runners=runners) == "p5_variant"
    assert cv_models.route_variant_role(category="Foods", brand="Maggi", is_hul=False, runners=runners) == "nonhul_p6_variant"
    assert cv_models.route_variant_role(category="Household_Care", brand="Dettol", is_hul=False, runners=runners) == "nonhul_p6_variant"
    assert cv_models.route_variant_role(category="Personal_Wash", brand="Dettol", is_hul=False, runners=runners) == "nonhul_gen_variant"

    # KerasClassifier 2-input brand one-hot conditioning for variant models
    clf = cv_models.KerasClassifier("oral_variant", "gs://b/oral_variant.h5", oral_classes)
    clf._model = FakeKeras(winner=2, n=3, inputs=((None, 3), (None, None, None, 3)))
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    label, conf = clf.predict_class(img, brand="Pepsodent")
    assert label == "Pepsodent__Pepsodent_2in1" and conf == pytest.approx(0.9)
    assert clf._model.feeds[0][0].tolist() == [0.0, 0.0, 1.0]
    assert cv_models.parse_label("oral_variant", label) == {
        "brand": "Pepsodent",
        "base_variant": "Pepsodent_2in1",
        "variant": "Pepsodent_2in1",
        "category": "Oral_Care",
        "packaging_type": "Tube",
        "product": "Pepsodent__Pepsodent_2in1",
    }


def test_gt_mapping_ground_truth_and_gen_variant(monkeypatch):
    monkeypatch.setattr(cv_models, "load_classes", lambda uri: ())
    gtm = cv_models.gt_mapping()
    assert len(gtm["brand_mapping"]) == 227
    assert len(cv_models.pack_categories()) == 10
    assert len(cv_models.pack_classes()) == 46
    assert len(cv_models.gt_brand_classes("gen_variant")) == 7
    assert len(cv_models.gt_variant_classes("gen_variant")) == 145
    assert len(cv_models.gt_brand_classes("oral_variant")) == 3
    assert len(cv_models.gt_variant_classes("oral_variant")) == 57
    assert len(cv_models.gt_brand_classes("skin_variant")) == 9
    assert len(cv_models.gt_variant_classes("skin_variant")) == 438
    assert len(cv_models.gt_brand_classes("wash_variant")) == 14
    assert len(cv_models.gt_variant_classes("wash_variant")) == 173
    assert len(cv_models.gt_brand_classes("p5_variant")) == 24
    assert len(cv_models.gt_variant_classes("p5_variant")) == 291
    assert len(cv_models.gt_brand_classes("nonhul_gen_variant")) == 106
    assert len(cv_models.gt_variant_classes("nonhul_gen_variant")) == 595
    assert len(cv_models.gt_brand_classes("nonhul_p6_variant", size=61)) == 61
    assert len(cv_models.gt_variant_classes("nonhul_p6_variant", size=367)) == 367

    # 5-part gt-mapping.yml label parsing and brand-consistent argmax masking
    gen_cls = cv_models.gt_variant_classes("gen_variant")
    clf = cv_models.KerasClassifier("gen_variant", "gs://b/gen_variant.h5", gen_cls)
    # Winner 0 is Clear__Active_Care, but conditioning on Clinic_Plus masks argmax to Clinic_Plus classes
    clf._model = FakeKeras(winner=0, n=145, inputs=((None, 7), (None, None, None, 3)))
    label, _ = clf.predict_class(Image.new("RGB", (64, 64)), brand="Clinic_Plus")
    assert "__Clinic_Plus__" in label
    assert clf._model.feeds[0][0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    parsed = cv_models.parse_gt_variant_label("gen_variant", "HUL__Hair__Shampoo__Clear__Active_Care")
    assert parsed["brand"] == "Clear" and parsed["category"] == "Hair_Care" and parsed["is_hul"] is True
    assert metrics.match_product(
        {"brand": "Clear", "base_variant": parsed["base_variant"], "product": parsed["product"]},
        {"brand": "Clear", "base_variant": "Active_Care_Shampoo", "product": "Clear__Active_Care_Shampoo"},
        ("product", "brand", "base_variant"),
    )["tp"] == 1



def test_coupled_3_stage_pipeline_with_variant_endpoints(monkeypatch):
    var_cfg = {
        "cv_models": {
            **CFG["cv_models"],
            "oral_variant_model": "gs://b/oral_variant/oral_variant.h5",
            "oral_variant_endpoint": "projects/p/locations/us-central1/endpoints/333",
            "wash_variant_model": "gs://b/wash_variant/wash_variant.h5",
            "wash_variant_endpoint": "projects/p/locations/us-central1/endpoints/444",
        }
    }
    sess = FakeSession({
        "111": (200, {"predictions": [{"raw_class": "APepsodent", "confidence": 0.96}]}),
        "222": (200, {"predictions": [{"raw_class": "Oral_Care__Tube", "confidence": 0.88}]}),
        "333": (200, {"predictions": [{"raw_class": "Pepsodent__Pepsodent_Germicheck", "confidence": 0.91}]}),
    })
    monkeypatch.setattr(cv_models, "_session", lambda: sess)
    monkeypatch.setattr(cv_models, "load_classes", lambda uri: (
        ("Close_Up__Close_Up_Red", "Oral_B__Oral_B_Fresh", "Pepsodent__Pepsodent_Germicheck")
        if "oral_variant" in str(uri) else ("Dove__Dove_Bar", "Lifebuoy__Lifebuoy_Total")
    ))
    cv_models.variant_brands.cache_clear()

    a = CVClassify()
    a.dataset, a.evaluated_fields = "hul_golden", None
    ctx = Context(model="keras-h5")
    a.setup(var_cfg, ctx)
    assert a.evaluated_fields == ("product", "brand", "base_variant", "packaging_type", "category")
    assert sorted(a.pipeline.runners) == ["brand", "oral_variant", "pack", "wash_variant"]

    pred = a.identify(Image.new("RGB", (64, 64)), ctx)
    assert pred == {
        "product": "Pepsodent__Pepsodent_Germicheck",
        "brand": "Pepsodent",
        "base_variant": "Pepsodent_Germicheck",
        "packaging_type": "Tube",
        "category": "Oral_Care",
    }
    assert sess.calls[2][1]["instances"][0]["role"] == "oral_variant"
    assert sess.calls[2][1]["instances"][0]["brand"] == "Pepsodent"

