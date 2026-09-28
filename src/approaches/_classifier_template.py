"""TEMPLATE (not registered: files starting with ``_`` are skipped).
Copy to ``src/approaches/my_classifier.py`` to benchmark a Fine-Tuned Gemini model,
custom compound classifier, or vector retriever on ground-truth shelf crops.

Supports both:
  * ``epic = "MT Market Share - Variant Classification"`` (``target_field = "variant"``)
  * ``epic = "MT Market Share - Other (Category, Brand and Package Type) Classifiers"`` (``target_field = "compound"``)

Run locally or on Cloud Run (pass a base Gemini model id OR a fine-tuned Vertex Endpoint id):
    shelf-bench run -a my_classifier -m gemini-3.1-flash-lite --split val --limit 25 --owner riley
    shelf-bench cloud-run -a my_classifier -m projects/PROJ/locations/us-central1/endpoints/ENDPOINT_ID --owner riley
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register


@register
class MyClassifier(Approach):
    name = "my_classifier"
    task = "classification"
    epic = "MT Market Share - Variant Classification"
    target_field = "variant"  # "variant" | "compound" | "category" | "brand" | "packaging_type"
    architecture = "Fine-Tuned Gemini 3.1 Flash Lite compound output classifier on shelf crops"
    steps = [
        "Crop each product box from the shelf image",
        "Predict compound JSON {category, brand, packaging_type, variant, sku_id} per crop",
    ]

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        preds: list[dict[str, Any]] = []
        for idx, b in enumerate(boxes):
            hint = prior[idx] if prior and idx < len(prior) else {}
            preds.append({
                "category": hint.get("category", "Hair Care - DMT"),
                "brand": hint.get("brand", "Dove"),
                "packaging_type": hint.get("packaging_type", "bottle"),
                "variant": "Hair Fall Rescue Shampoo 340ml",
                "sku_id": "BP-HUL-DOVE-HAIR-FALL-340ML",
                "is_hul": True,
            })
        ctx.trace.step("Classify crops", label_counts(preds), boxes=boxes, labels=preds)
        return preds
