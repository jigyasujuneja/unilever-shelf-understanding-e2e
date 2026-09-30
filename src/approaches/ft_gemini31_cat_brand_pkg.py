"""Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #2: Category, Brand & Package Type.

Epic: ``MT Market Share - Other (Category, Brand and Package Type) Classifiers`` (``task = "classification"``)
Implements Task #2 from the team's roadmap: Fine-Tuned Gemini 3.1 Flash Lite (``gemini-3.1-flash-lite``)
predicting structured compound JSON outputs ``{"category", "brand", "packaging_type", "is_hul"}`` for each
shelf crop. Can be benchmarked standalone in Epic 2 or chained into ``ft_gemini31_variant_compound`` in
``compound_pipeline_1_plus_2`` (Epic 4).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from core.retrieval import CONFIG_FT_GEMINI31_CAT_BRAND_PKG, classify_shelf_boxes_7dim

COMPOUND_CAT_BRAND_PKG_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "category": {"type": "string"},
            "brand": {"type": "string"},
            "packaging_type": {"type": "string"},
            "is_hul": {"type": "boolean"},
        },
        "required": ["id", "category", "brand", "packaging_type"],
    },
}


@register
class FTGemini31CatBrandPkgClassifier(Approach):
    name = "ft_gemini31_cat_brand_pkg"
    task = "classification"
    epic = "MT Market Share - Other (Category, Brand and Package Type) Classifiers"
    target_field = "compound"
    architecture = (
        "Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #2: "
        "Constrained JSON Schema for (Category, Brand, Package Type, Is-HUL)"
    )
    steps = [
        "Step 1: Contact-Sheet Crop Batching & High-Purity Clustering",
        "Step 2: Fine-Tuned Gemini 3.1 Flash Lite Compound (Category + Brand + Package Type) Decode",
    ]
    sft_lora_endpoint: str = ""

    def setup(self, config: dict) -> None:
        del config
        import json
        from pathlib import Path

        manifest_path = Path(__file__).resolve().parents[2] / "results" / "sft_lora_tuning_manifest.json"
        if not manifest_path.is_file():
            manifest_path = Path("results/sft_lora_tuning_manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError("Missing results/sft_lora_tuning_manifest.json for FTGemini31CatBrandPkgClassifier")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        endpoint = str(manifest.get("tuned_model", {}).get("endpoint") or "").strip()
        if not endpoint:
            raise RuntimeError("Empty tuned_model.endpoint in results/sft_lora_tuning_manifest.json")
        self.sft_lora_endpoint = endpoint

    def classify(
        self,
        image: Image.Image,
        boxes: list[Box],
        ctx: Context,
        prior: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.sft_lora_endpoint:
            self.setup({})
        ctx.trace.meta["sft_lora_endpoint"] = self.sft_lora_endpoint
        ctx.trace.step(
            "Step 1: Contact-Sheet Crop Batching",
            f"Packing {len(boxes)} crops into clustered contact sheets for Fine-Tuned Gemini 3.1 Flash Lite ({self.sft_lora_endpoint})",
            boxes=boxes,
        )
        preds = classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, config=CONFIG_FT_GEMINI31_CAT_BRAND_PKG, prior=prior
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Step 2: FT Gemini 3.1 Flash Lite Compound Decode (Cat + Brand + Pkg)",
            f"Decoded {len(preds)} compound (Category, Brand, Package Type) predictions ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        return preds
