"""Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #1: Fine-Grained Variant Classification.

Epic: ``MT Market Share - Variant Classification`` (``task = "classification"``)
Implements Task #1 from the team's roadmap: Fine-Tuned Gemini 3.1 Flash Lite (``gemini-3.1-flash-lite``)
for Compound Outputs predicting fine-grained SKU ``variant`` alongside ``category``, ``brand``, and
``packaging_type``. Supports optional ``prior`` conditioning from Task #2 (``ft_gemini31_cat_brand_pkg``)
when chained inside ``compound_pipeline_1_plus_2``.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from approaches.base import Approach, Box, Context, label_counts, register
from core.retrieval import CONFIG_FT_GEMINI31_VARIANT, classify_shelf_boxes_7dim


@register
class FTGemini31VariantCompoundClassifier(Approach):
    name = "ft_gemini31_variant_compound"
    task = "classification"
    epic = "MT Market Share - Variant Classification"
    target_field = "variant"
    architecture = (
        "Fine-Tuned Gemini 3.1 Flash Lite for Compound Outputs #1: "
        "Fine-Grained Variant & Shade Disambiguation (Supports Hierarchical Prior Conditioning)"
    )
    steps = [
        "Step 1: Sub-ROI Shade & Claim Zone Crop Preparation",
        "Step 2: Fine-Tuned Gemini 3.1 Flash Lite Compound Variant Decode",
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
            raise RuntimeError("Missing results/sft_lora_tuning_manifest.json for FTGemini31VariantCompoundClassifier")
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
        prior_msg = "conditioned on Stage-1 (Category, Brand, Package Type) prior" if prior else "unconditioned standalone mode"
        ctx.trace.step(
            "Step 1: Sub-ROI Shade & Claim Crop Preparation",
            f"Preparing {len(boxes)} crops ({prior_msg}, endpoint={self.sft_lora_endpoint})",
            boxes=boxes,
        )
        preds = classify_shelf_boxes_7dim(
            image, boxes, ctx=ctx, config=CONFIG_FT_GEMINI31_VARIANT, prior=prior
        )
        ctx.trace.labels = preds
        ctx.trace.step(
            "Step 2: FT Gemini 3.1 Flash Lite Variant Compound Decode",
            f"Decoded {len(preds)} fine-grained variants ({label_counts(preds)})",
            boxes=boxes,
            labels=preds,
        )
        return preds
