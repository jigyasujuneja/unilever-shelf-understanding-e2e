"""Offline testing utilities: run and score the whole benchmark with no GCP access.

This module exists so that a new engineer can clone the repo and immediately run a full,
deterministic benchmark on their laptop. Without it, every test path reached for real credentials,
which meant the only way to find out whether a change broke scoring was to spend money.

What you get:

* `FakeGenAIClient` / `ReplayGenAIClient` -- drop-in stand-ins for `genai.Client` that return
  canned or recorded responses with realistic token usage.
* `offline_config()` -- a `BenchmarkConfig` with every network dependency disabled and telemetry
  pointed at a temp directory.
* `make_offline_sdk()` -- a fully wired `ShelfBenchmarkSDK` using the above.
* `sample_ground_truth()` / `sample_predictions()` -- small in-memory fixtures with known-correct
  answers, used by the golden tests that pin down the scoring maths.

Typical use in a test or a scratch script:

    from shelf_benchmark.testing import make_offline_sdk, FakeGenAIClient, sample_shelf_payload

    sdk = make_offline_sdk(tmp_path)
    sdk.register_model("my-model", custom_handler=lambda p, uri, schema: sample_shelf_payload())
    summary = sdk.run_suite(models=["my-model"], tasks=["classification"],
                            approaches=["single_pass_full_shelf"])

See `docs/EVALUATION_PROTOCOL.md` for what the resulting numbers mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.models import GroundTruthProductItem, ImageGroundTruth

__all__ = [
    "FakeGenAIClient",
    "ReplayGenAIClient",
    "offline_config",
    "make_offline_sdk",
    "benchmark_harness",
    "run_offline_approach",
    "offline_universal_payload_handler",
    "sample_shelf_payload",
    "sample_ground_truth",
    "sample_predictions",
    "perfect_prediction_payload",
    "shifted_prediction_payload",
    "wrong_brand_payload",
    "fixture_image_path",
    "write_sample_ground_truth_file",
    "OFFLINE_IMAGE_URI",
]

_FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures"


def fixture_image_path() -> Path:
    """Absolute path to the bundled synthetic shelf image.

    This is a 600x400 PNG with three coloured rectangles positioned to match
    `sample_ground_truth()` exactly. It ships inside the package (rather than living in `tests/`)
    so that `shelf_benchmark.testing` works for anyone who pip-installs the suite, not only for
    someone running from a source checkout.
    """
    path = _FIXTURE_DIR / "shelf_sample_01.png"
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled fixture image missing at '{path}'. Reinstall the package, or regenerate it "
            f"with the snippet in docs/EVALUATION_PROTOCOL.md."
        )
    return path


# Offline runs point at the bundled fixture so the real image-loading, cropping, and montage code
# paths are genuinely exercised rather than stubbed out.
OFFLINE_IMAGE_URI = str(_FIXTURE_DIR / "shelf_sample_01.png")


class _UsageMetadata:
    """Minimal stand-in for the `usage_metadata` returned by the real client."""

    def __init__(self, input_tokens: int, thinking_tokens: int, output_tokens: int) -> None:
        self.prompt_token_count = input_tokens
        self.thoughts_token_count = thinking_tokens
        self.candidates_token_count = output_tokens
        self.cached_content_token_count = 0
        self.total_token_count = input_tokens + thinking_tokens + output_tokens
        self.traffic_type = "ON_DEMAND"


class _Response:
    def __init__(self, text: str, input_tokens: int, thinking_tokens: int, output_tokens: int) -> None:
        self.text = text
        self.usage_metadata = _UsageMetadata(input_tokens, thinking_tokens, output_tokens)


class FakeGenAIClient:
    """Returns one canned payload for every call, with deterministic token counts.

    Use this when the test cares about the plumbing (cost maths, row construction, report
    generation) rather than about model behaviour.
    """

    def __init__(
        self,
        response_payload: Dict[str, Any] | Sequence[Dict[str, Any]],
        prompt_tokens: int = 1106,
        thought_tokens: int = 250,
        out_tokens: int = 300,
    ) -> None:
        self.payload = response_payload
        self.prompt_tokens = prompt_tokens
        self.thought_tokens = thought_tokens
        self.out_tokens = out_tokens
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs: Any) -> _Response:
        self.call_count += 1
        self.calls.append(kwargs)
        return _Response(
            text=json.dumps(self.payload),
            input_tokens=self.prompt_tokens,
            thinking_tokens=self.thought_tokens,
            output_tokens=self.out_tokens,
        )


class ReplayGenAIClient:
    """Replays a sequence of recorded responses, in order, then repeats the last one.

    Use this for multi-stage approaches (detect, then classify each crop) where each call should
    return something different. Recording real responses once and replaying them gives you a
    regression test for a real model's behaviour without paying for it on every run.

    `responses` may be dicts (serialized to JSON) or raw strings.
    """

    def __init__(
        self,
        responses: Sequence[Dict[str, Any] | str],
        prompt_tokens: int = 900,
        thought_tokens: int = 120,
        out_tokens: int = 240,
    ) -> None:
        if not responses:
            raise ValueError("ReplayGenAIClient requires at least one recorded response.")
        self.responses = list(responses)
        self.prompt_tokens = prompt_tokens
        self.thought_tokens = thought_tokens
        self.out_tokens = out_tokens
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "ReplayGenAIClient":
        """Load a JSON list of recorded responses from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Replay file '{path}' must contain a JSON list of responses.")
        return cls(data, **kwargs)

    def _generate_content(self, **kwargs: Any) -> _Response:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        self.calls.append(kwargs)
        item = self.responses[idx]
        text = item if isinstance(item, str) else json.dumps(item)
        return _Response(
            text=text,
            input_tokens=self.prompt_tokens,
            thinking_tokens=self.thought_tokens,
            output_tokens=self.out_tokens,
        )


def offline_config(
    tmp_dir: str | Path,
    *,
    ground_truth_uri: Optional[str] = None,
    gt_provider: str = "none",
    gt_version: str = "offline-fixture-v1",
    iou_threshold: float = 0.50,
) -> BenchmarkConfig:
    """Build a `BenchmarkConfig` that touches no network and writes everything under `tmp_dir`.

    Every switch that could reach out to GCP is turned off explicitly rather than relying on
    credentials happening to be absent, so an offline test stays offline even on a machine that is
    logged in.
    """
    out = Path(tmp_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    cfg = BenchmarkConfig()
    cfg.offline.enabled = True
    cfg.offline.fail_on_network_access = True

    cfg.telemetry.otel_log_path = str(out / "reports" / "otel_logs.jsonl")
    cfg.telemetry.export_to_gcp_cloud_logging = False
    cfg.telemetry.sync_otel_jsonl_to_gcs = False
    cfg.telemetry.sync_otel_logs_to_gcs = False
    cfg.telemetry.export_to_console = False

    cfg.reporting.output_dir = str(out / "reports")
    cfg.reporting.sync_reports_to_gcs = False
    cfg.reporting.isolate_runs = False

    cfg.billing.use_live_cloud_billing_catalog_api = False
    cfg.billing.include_infrastructure_costs = False
    cfg.billing.bigquery_billing_export_table = None

    cfg.fine_tuning.submit_live_tuning_job = False
    cfg.embeddings.reference_catalog.source_uri = None

    cfg.evaluation.iou_threshold = iou_threshold
    cfg.ground_truth.provider_type = gt_provider
    cfg.ground_truth.source_uri = ground_truth_uri
    cfg.ground_truth.gt_version = gt_version
    cfg.ground_truth.strict = True

    return cfg


def make_offline_sdk(tmp_dir: str | Path, **config_overrides: Any):
    """Return a `ShelfBenchmarkSDK` wired for offline execution under `tmp_dir`.

    Imported lazily so that `shelf_benchmark.testing` stays cheap to import.
    """
    from shelf_benchmark.sdk import ShelfBenchmarkSDK

    cfg = offline_config(tmp_dir, **config_overrides)
    return ShelfBenchmarkSDK.from_config(cfg)


# ---------------------------------------------------------------------------
# Fixtures with known-correct answers.
#
# These describe a tiny 3-facing shelf. The numbers are arbitrary but internally consistent, which
# is the point: a scoring change that breaks the maths will move a metric away from an obviously
# correct value (1.0 for a perfect prediction, 0.0 for a total miss).
# ---------------------------------------------------------------------------

def sample_ground_truth(image_key: str = OFFLINE_IMAGE_URI) -> ImageGroundTruth:
    """Three front facings with exact bounding boxes, in suite-native 0-1000 coordinates."""
    return ImageGroundTruth(
        image_id=image_key,
        source_key=image_key,
        gt_version="offline-fixture-v1",
        total_main_shelf_facings=3,
        expected_brands=["Brand_A", "Brand_B", "Brand_C"],
        items=[
            GroundTruthProductItem(
                item_id=1, brand="Brand_A", product_name="Brand_A Radiance Daily Cleanser",
                sku_id="SKU-001", bbox_2d=[100, 100, 300, 200], shelf_row="top",
            ),
            GroundTruthProductItem(
                item_id=2, brand="Brand_B", product_name="Brand_B Herbal Purifying Cleanser",
                sku_id="SKU-002", bbox_2d=[100, 220, 300, 320], shelf_row="top",
            ),
            GroundTruthProductItem(
                item_id=3, brand="Brand_C", product_name="Brand_C Vitamin Gel Cleanser",
                sku_id="SKU-003", bbox_2d=[400, 100, 600, 200], shelf_row="middle",
            ),
        ],
    )


def perfect_prediction_payload() -> Dict[str, Any]:
    """A model output that exactly reproduces `sample_ground_truth`.

    Scored against `sample_ground_truth()` this must yield 1.0 for detection precision/recall/F1,
    brand accuracy, product accuracy, and count accuracy. Any change that moves these off 1.0 has
    broken the scorer, which is exactly what the golden tests assert.

    Shape matches `ProductClassificationOutput`.
    """
    return {
        "total_classified_products": 3,
        "distinct_brands_found": ["Brand_A", "Brand_B", "Brand_C"],
        "classified_products": [
            {
                "product_index": 1, "brand": "Brand_A",
                "product_name": "Brand_A Radiance Daily Cleanser",
                "category": "Skin Cleansing", "subcategory": "Face Wash",
                "variant": "Bright Beauty", "packaging_type": "tube", "pack_type": "Single",
                "is_hul_brand": True,
                "bbox_2d": [100, 100, 300, 200],
                "shelf_row": "top", "position_on_shelf": 1, "confidence": 0.95,
            },
            {
                "product_index": 2, "brand": "Brand_B",
                "product_name": "Brand_B Herbal Purifying Cleanser",
                "category": "Skin Cleansing", "subcategory": "Face Wash",
                "variant": "Purifying Neem", "packaging_type": "tube", "pack_type": "Single",
                "is_hul_brand": False,
                "bbox_2d": [100, 220, 300, 320],
                "shelf_row": "top", "position_on_shelf": 2, "confidence": 0.93,
            },
            {
                "product_index": 3, "brand": "Brand_C",
                "product_name": "Brand_C Vitamin Gel Cleanser",
                "category": "Skin Cleansing", "subcategory": "Face Wash",
                "variant": "Blush and Glow", "packaging_type": "tube", "pack_type": "Single",
                "is_hul_brand": True,
                "bbox_2d": [400, 100, 600, 200],
                "shelf_row": "middle", "position_on_shelf": 1, "confidence": 0.91,
            },
        ],
        "_token_usage": {"input_tokens": 350, "thinking_tokens": 25, "output_tokens": 160},
    }


def sample_shelf_payload() -> Dict[str, Any]:
    """Alias for `perfect_prediction_payload`, for use as a generic canned model response."""
    return perfect_prediction_payload()


def sample_predictions() -> List[Dict[str, Any]]:
    """Raw prediction dicts, as written to `predictions.json`, for re-scoring tests."""
    return perfect_prediction_payload()["classified_products"]


def offline_universal_payload_handler(
    prompt: str, image_uri: str, response_schema: Any = None
) -> Dict[str, Any]:
    """Universal multi-stage offline handler that serves detection, class-agnostic detection, matching, or classification payloads."""
    schema_name = getattr(response_schema, "__name__", "")
    if schema_name == "ProductDetectionOutput":
        return {
            "total_detected_products": 3,
            "detected_products": [
                {
                    "product_index": 1,
                    "bbox_2d": [100, 100, 300, 200],
                    "shelf_row": "top",
                    "position_on_shelf": 1,
                    "is_front_facing": True,
                    "preliminary_brand_hint": "Brand_A",
                    "visual_description": "Brand_A Radiance Daily Cleanser",
                    "confidence": 0.95,
                },
                {
                    "product_index": 2,
                    "bbox_2d": [100, 220, 300, 320],
                    "shelf_row": "top",
                    "position_on_shelf": 2,
                    "is_front_facing": True,
                    "preliminary_brand_hint": "Brand_B",
                    "visual_description": "Brand_B Herbal Purifying Cleanser",
                    "confidence": 0.93,
                },
                {
                    "product_index": 3,
                    "bbox_2d": [400, 100, 600, 200],
                    "shelf_row": "middle",
                    "position_on_shelf": 1,
                    "is_front_facing": True,
                    "preliminary_brand_hint": "Brand_C",
                    "visual_description": "Brand_C Vitamin Gel Cleanser",
                    "confidence": 0.91,
                },
            ],
            "_token_usage": {"input_tokens": 320, "thinking_tokens": 20, "output_tokens": 140},
        }
    if schema_name == "ClassAgnosticDetectionResponse":
        return {
            "products": [
                {"product_index": 1, "class_label": "product", "bbox_2d": [100, 100, 300, 200], "shelf_row": "top", "position_on_shelf": 1, "is_front_facing": True, "confidence": 0.95},
                {"product_index": 2, "class_label": "product", "bbox_2d": [100, 220, 300, 320], "shelf_row": "top", "position_on_shelf": 2, "is_front_facing": True, "confidence": 0.93},
                {"product_index": 3, "class_label": "product", "bbox_2d": [400, 100, 600, 200], "shelf_row": "middle", "position_on_shelf": 1, "is_front_facing": True, "confidence": 0.91},
            ],
            "_token_usage": {"input_tokens": 280, "thinking_tokens": 15, "output_tokens": 110},
        }
    if schema_name == "ProductMatchingOutput":
        return {
            "total_matched_products": 3,
            "matched_products": [
                {
                    "product_index": 1,
                    "bbox_2d": [100, 100, 300, 200],
                    "shelf_row": "top",
                    "position_on_shelf": 1,
                    "category": "Skin Cleansing",
                    "subcategory": "Face Wash",
                    "brand": "Brand_A",
                    "is_hul_brand": True,
                    "product_name": "Brand_A Radiance Daily Cleanser",
                    "variant": "Bright Beauty",
                    "packaging_type": "tube",
                    "pack_type": "Single",
                    "size": "100g",
                    "lexical_search_keywords": ["Brand_A", "Bright Beauty", "Face Wash"],
                    "dense_embedding_text": "Brand: Brand_A | Product: Brand_A Radiance Daily Cleanser | Size: 100g",
                    "matched_sku_id": "SKU-001",
                    "match_confidence": 0.95,
                    "planogram_compliant": True,
                },
                {
                    "product_index": 2,
                    "bbox_2d": [100, 220, 300, 320],
                    "shelf_row": "top",
                    "position_on_shelf": 2,
                    "category": "Skin Cleansing",
                    "subcategory": "Face Wash",
                    "brand": "Brand_B",
                    "is_hul_brand": False,
                    "product_name": "Brand_B Herbal Purifying Cleanser",
                    "variant": "Purifying Neem",
                    "packaging_type": "tube",
                    "pack_type": "Single",
                    "size": "100g",
                    "lexical_search_keywords": ["Brand_B", "Neem", "Face Wash"],
                    "dense_embedding_text": "Brand: Brand_B | Product: Brand_B Herbal Purifying Cleanser | Size: 100g",
                    "matched_sku_id": "SKU-002",
                    "match_confidence": 0.93,
                    "planogram_compliant": True,
                },
                {
                    "product_index": 3,
                    "bbox_2d": [400, 100, 600, 200],
                    "shelf_row": "middle",
                    "position_on_shelf": 1,
                    "category": "Skin Cleansing",
                    "subcategory": "Face Wash",
                    "brand": "Brand_C",
                    "is_hul_brand": True,
                    "product_name": "Brand_C Vitamin Gel Cleanser",
                    "variant": "Blush and Glow",
                    "packaging_type": "tube",
                    "pack_type": "Single",
                    "size": "100g",
                    "lexical_search_keywords": ["Brand_C", "Blush and Glow", "Face Wash"],
                    "dense_embedding_text": "Brand: Brand_C | Product: Brand_C Vitamin Gel Cleanser | Size: 100g",
                    "matched_sku_id": "SKU-003",
                    "match_confidence": 0.91,
                    "planogram_compliant": True,
                },
            ],
            "_token_usage": {"input_tokens": 380, "thinking_tokens": 30, "output_tokens": 190},
        }
    return perfect_prediction_payload()


def benchmark_harness(
    tmp_dir: str | Path,
    *,
    model_id: str = "offline-demo-model",
    payload: Optional[Dict[str, Any]] = None,
    handler: Optional[Any] = None,
    with_ground_truth: bool = False,
    **config_overrides: Any,
):
    """One-call offline test harness that returns a pre-configured `ShelfBenchmarkSDK` with a stub model registered.

    Example:
        sdk = benchmark_harness(tmp_path, with_ground_truth=True)
        summary = sdk.run_suite(approaches=["single_pass_full_shelf"], tasks=["classification"])
    """
    from shelf_benchmark.config import ModelPricing

    sdk = make_offline_sdk(tmp_dir, **config_overrides)
    effective_handler = handler or (
        (lambda prompt, uri, schema: payload)
        if payload is not None
        else offline_universal_payload_handler
    )
    sdk.register_model(
        model_id,
        custom_handler=effective_handler,
        pricing=ModelPricing(input=0.30, thinking=0.30, output=2.50),
    )
    sdk.swap_models([model_id])
    if with_ground_truth:
        gt_file = write_sample_ground_truth_file(Path(tmp_dir) / "sample_gt.json")
        sdk.connect_ground_truth(
            provider_type="json",
            source_uri=str(gt_file),
            gt_version="offline-fixture-v1",
        )
    return sdk


def run_offline_approach(
    tmp_dir: str | Path,
    approach_id: str = "single_pass_full_shelf",
    *,
    model_id: str = "offline-demo-model",
    payload: Optional[Dict[str, Any]] = None,
    handler: Optional[Any] = None,
    ground_truth: Optional[ImageGroundTruth] = None,
    with_ground_truth: bool = False,
):
    """Run a single approach offline in one function call and return its `TaskExecutionResult`."""
    sdk = benchmark_harness(
        tmp_dir,
        model_id=model_id,
        payload=payload,
        handler=handler,
        with_ground_truth=with_ground_truth,
    )
    summary = sdk.run_suite(
        models=[model_id],
        tasks=["classification"],
        approaches=[approach_id],
        shelf_image_uri=OFFLINE_IMAGE_URI,
        ground_truth=ground_truth,
    )
    return summary["results"][0]


def write_sample_ground_truth_file(path: str | Path, image_key: Optional[str] = None) -> Path:
    """Write the fixture ground truth to `path` in the suite-native JSON schema.

    Doubles as the canonical worked example of the file format: point
    `connect_ground_truth(provider_type="json", source_uri=...)` at the result and everything
    scores. Use it to sanity-check your pipeline before the real annotations land.
    """
    gt = sample_ground_truth(image_key or OFFLINE_IMAGE_URI)
    image_entry = {
        "image_id": gt.image_id,
        "gt_version": gt.gt_version,
        "total_main_shelf_facings": gt.total_main_shelf_facings,
        "expected_brands": list(gt.expected_brands),
        "items": [
            {
                "item_id": it.item_id,
                "brand": it.brand,
                "product_name": it.product_name,
                "sku_id": it.sku_id,
                "bbox_2d": list(it.bbox_2d),
                "shelf_row": it.shelf_row,
                "back_row": it.back_row,
                "occluded": it.occluded,
            }
            for it in gt.items
        ],
    }
    payload = {
        "gt_version": gt.gt_version,
        "bbox_format": "ymin_xmin_ymax_xmax_1000",
        "images": {gt.image_id: image_entry},
        gt.image_id: image_entry,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def shifted_prediction_payload(offset: int = 400) -> Dict[str, Any]:
    """The perfect prediction with every box translated far enough to drop IoU to zero.

    Used to prove that detection recall actually responds to localization quality. An earlier
    version of the scorer thresholded at IoU 0.25 while reporting 'IoU 0.50', so boxes this wrong
    still counted as hits. Coordinates stay clamped inside the 0-1000 space, so these remain
    well-formed boxes that simply do not overlap the truth.
    """
    payload = perfect_prediction_payload()
    for prod in payload["classified_products"]:
        y0, x0, y1, x1 = prod["bbox_2d"]
        prod["bbox_2d"] = [
            min(1000, y0 + offset), min(1000, x0 + offset),
            min(1000, y1 + offset), min(1000, x1 + offset),
        ]
    return payload


def wrong_brand_payload() -> Dict[str, Any]:
    """Correct boxes, wrong brands. Detection should be perfect, classification should be zero.

    This separation is the point of pairing on geometry alone: localization quality and naming
    quality are different capabilities and must be reportable independently.
    """
    payload = perfect_prediction_payload()
    payload["distinct_brands_found"] = ["CompletelyDifferentBrand"]
    for prod in payload["classified_products"]:
        prod["brand"] = "CompletelyDifferentBrand"
        prod["product_name"] = "Completely Different Product"
        prod["is_hul_brand"] = False
    return payload
