"""Stage 1: Class-Agnostic Single-Class ('product') Object Detection.

Supports three Google Cloud detection backends:
1. `google_open_vocab_class_agnostic_box2d` (Default): Uses Google's spatial `box_2d`
   grounding head in pure class-agnostic single-class mode (`label="product"`, zero brand/text
   generation) + shared `deduplicate_depth_stacked_facings()` from `facing_utils.py`.
2. `cloud_vision_object_localization`: Calls Google Cloud Vision API (`vision.googleapis.com/v1/images:annotate`
   with `OBJECT_LOCALIZATION`), converting normalized polygon vertices to `[ymin, xmin, ymax, xmax]`.
3. `vertex_ai_custom_detector_endpoint`: Calls a custom Vertex AI Online Prediction Endpoint
   hosting `EfficientDet-D2`, `YOLOv8/v11`, `Faster-RCNN`, or `OWL-ViT v2` trained on single-class
   retail shelf datasets (such as `SKU-110K`).
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

import requests
from google.genai import types
from pydantic import BaseModel, Field

from shelf_benchmark.approaches.base import CommonLayerContext
from shelf_benchmark.auth import create_genai_client, get_gcp_credentials
from shelf_benchmark.models import TokenUsageMetrics


class ClassAgnosticBoxItem(BaseModel):
    """Single-class ('product') bounding box output."""

    bbox_2d: List[int] = Field(
        description="Normalized 2D bounding box [ymin, xmin, ymax, xmax] from 0 to 1000 for a front-most physical product facing"
    )
    shelf_row: str = Field(
        default="middle",
        description="Shelf row ('top', 'middle', or 'bottom')",
    )
    is_front_facing: bool = Field(
        default=True,
        description="True if this is the front-most visible unit in its horizontal shelf slot",
    )
    confidence: float = Field(default=0.95)


class ClassAgnosticDetectionResponse(BaseModel):
    """Pure class-agnostic ('product') object detection schema (zero brand/text attributes)."""

    products: List[ClassAgnosticBoxItem]


CLASS_AGNOSTIC_DETECTION_PROMPT = """You are a pure Class-Agnostic Object Detector (single class: 'product') operating like a YOLO / EfficientDet / OWL-ViT model trained on SKU-110K.

RULES:
1. Detect every physical front-most product facing on the shelf as the single class 'product'.
2. Do NOT read brands, variants, or text. Output ONLY bounding box coordinates `bbox_2d` [ymin, xmin, ymax, xmax] (0..1000) and `shelf_row`.
3. Do NOT box products stacked behind the front unit in the same horizontal column (depth duplicates).
"""


def run_stage1_class_agnostic_detection(
    ctx: CommonLayerContext,
    image_bytes: bytes,
    model_name: str,
    detector_backend: str = "google_open_vocab_class_agnostic_box2d",
    vertex_endpoint_url: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int, TokenUsageMetrics, float]:
    """Executes Stage 1 Class-Agnostic ('product') Object Detection and applies
    shared `ctx.deduplicate_depth_stacked_facings()` from `facing_utils.py`.

    Returns:
        (kept_front_facings, depth_duplicates_filtered, token_usage, extra_api_cost_usd)
    """
    raw_candidates: List[Dict[str, Any]] = []
    tokens = TokenUsageMetrics()
    extra_api_cost_usd = 0.0

    if detector_backend == "cloud_vision_object_localization":
        # Google Cloud Vision API: OBJECT_LOCALIZATION ($0.0015 per image)
        creds = get_gcp_credentials(project_id=ctx.config.gcp.project_id)
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "x-goog-user-project": ctx.config.gcp.project_id,
            "Content-Type": "application/json",
        }
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = requests.post(
            "https://vision.googleapis.com/v1/images:annotate",
            headers=headers,
            json={
                "requests": [
                    {
                        "image": {"content": img_b64},
                        "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 60}],
                    }
                ]
            },
            timeout=30,
        )
        extra_api_cost_usd = 0.0015
        if resp.status_code == 200:
            annotations = (
                resp.json()
                .get("responses", [{}])[0]
                .get("localizedObjectAnnotations", [])
            )
            for idx, obj in enumerate(annotations, start=1):
                verts = obj.get("boundingPoly", {}).get("normalizedVertices", [])
                if len(verts) >= 3:
                    xs = [int(round(float(v.get("x", 0.0)) * 1000)) for v in verts]
                    ys = [int(round(float(v.get("y", 0.0)) * 1000)) for v in verts]
                    ymin, ymax = max(0, min(ys)), min(1000, max(ys))
                    xmin, xmax = max(0, min(xs)), min(1000, max(xs))
                    shelf_row = "bottom" if ymax > 840 else "middle"
                    raw_candidates.append(
                        {
                            "product_index": idx,
                            "class_label": "product",
                            "bbox_2d": [ymin, xmin, ymax, xmax],
                            "shelf_row": shelf_row,
                            "position_on_shelf": idx,
                            "is_front_facing": True,
                            "confidence": round(float(obj.get("score", 0.75)), 3),
                        }
                    )

    elif detector_backend == "vertex_ai_custom_detector_endpoint" and vertex_endpoint_url:
        # Custom Vertex AI Online Prediction Endpoint (EfficientDet / YOLOv8 / Faster-RCNN)
        creds = get_gcp_credentials(project_id=ctx.config.gcp.project_id)
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "x-goog-user-project": ctx.config.gcp.project_id,
            "Content-Type": "application/json",
        }
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = requests.post(
            vertex_endpoint_url,
            headers=headers,
            json={"instances": [{"image_bytes": {"b64": img_b64}}]},
            timeout=30,
        )
        extra_api_cost_usd = 0.00025
        if resp.status_code == 200:
            preds = resp.json().get("predictions", [{}])[0]
            boxes = preds.get("detection_boxes", [])
            scores = preds.get("detection_scores", [])
            for idx, (b, sc) in enumerate(zip(boxes, scores), start=1):
                if float(sc) < 0.35:
                    continue
                ymin, xmin, ymax, xmax = [int(round(float(v) * 1000)) for v in b]
                raw_candidates.append(
                    {
                        "product_index": idx,
                        "class_label": "product",
                        "bbox_2d": [ymin, xmin, ymax, xmax],
                        "shelf_row": "bottom" if ymax > 840 else "middle",
                        "position_on_shelf": idx,
                        "is_front_facing": True,
                        "confidence": round(float(sc), 3),
                    }
                )

    else:
        # Default: Google Open-Vocabulary Single-Class ("product") Spatial Box2D Detector
        client = create_genai_client(
            project_id=ctx.config.gcp.project_id,
            location=ctx.config.gcp.location,
        )
        part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        gen_cfg = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ClassAgnosticDetectionResponse,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[part, CLASS_AGNOSTIC_DETECTION_PROMPT],
            config=gen_cfg,
        )
        tokens = ctx.extract_tokens(response)
        parsed = ClassAgnosticDetectionResponse.model_validate_json(response.text)
        for idx, item in enumerate(parsed.products, start=1):
            raw_candidates.append(
                {
                    "product_index": idx,
                    "class_label": "product",
                    "bbox_2d": list(item.bbox_2d),
                    "shelf_row": item.shelf_row,
                    "position_on_shelf": idx,
                    "is_front_facing": item.is_front_facing,
                    "confidence": item.confidence,
                }
            )

    # Apply shared Front-Facing Column Depth NMS (`ctx.deduplicate_depth_stacked_facings`)
    kept_facings, depth_filtered = ctx.deduplicate_depth_stacked_facings(
        raw_candidates, x_overlap_threshold=0.55
    )

    # Re-index left-to-right per shelf row
    for new_idx, f in enumerate(kept_facings, start=1):
        f["product_index"] = new_idx
        f["position_on_shelf"] = new_idx
        f["class_label"] = "product"

    return kept_facings, depth_filtered, tokens, extra_api_cost_usd
