"""Google MaxViT (maxvit_tiny_tf_224) 1,922-Class Variant Classifier (``maxvit_classify``).

Fine-tuned on hul_variants dataset (1,922 Brand__Variant classes).
Supports:
  - TorchScript traced model deployment (traced_maxvit_brand_variant.pt)
  - Standard PyTorch weights checkpoint (best_maxvit_brand_variant.pth)
  - Contiguous 0..1921 index -> Canonical Unilever SKU ID mapping via variant_classes.json
  - Candidate masking when `allowed_ids` is passed (hierarchical brand/category narrowing)
  - Zero Gemini API calls ($0 token spend, pure edge/GPU/CPU PyTorch execution)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torchvision import transforms

from approaches.base import Approach, Box, Context, crop, register

# Model architecture identifier
MODEL_ARCH = "maxvit_tiny_tf_224"


@register
class MaxVitClassify(Approach):
    """Fine-tuned Google MaxViT Tiny (maxvit_tiny_tf_224) 1,922-Class Variant Classifier."""

    name = "maxvit_classify"
    task = "classification"
    dataset = "hul_variants"
    models = [MODEL_ARCH]
    architecture = (
        "Fine-tuned Google MaxViT Tiny (maxvit_tiny_tf_224) 1,922 Brand__Variant Classifier "
        "(224x224 RGB, 0 VLM Calls, Pure PyTorch/TorchScript)"
    )
    steps = [
        "Preprocess crop (Resize 224x224, ImageNet RGB normalization)",
        "Forward pass through fine-tuned MaxViT backbone (TorchScript / PyTorch)",
        "Compute softmax logits over 1,922 canonical Brand__Variant classes",
        "Map top-1 class index to canonical SKU ID using variant_classes.json",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any = None
        self.idx_to_meta: dict[int, dict[str, Any]] = {}
        self.idx_to_sku: dict[int, int] = {}
        self.sku_to_idx: dict[int, int] = {}
        self.transform: transforms.Compose | None = None

    def setup(self, config: dict, ctx: Context) -> None:
        """Load fine-tuned model artifacts and class metadata."""
        # 1. Search for model directory
        candidates = [
            Path("models/maxvit_variants"),
            Path(__file__).resolve().parents[4] / "models" / "maxvit_variants",
            Path.cwd() / "models" / "maxvit_variants",
        ]
        model_dir: Path | None = None
        for cand in candidates:
            if (cand / "variant_classes.json").exists():
                model_dir = cand
                break

        if model_dir is None:
            raise FileNotFoundError(
                "Could not locate 'models/maxvit_variants/variant_classes.json'. "
                "Please ensure model artifacts are placed in 'models/maxvit_variants/'."
            )

        # 2. Load deployment class mapping (0..1921 -> sku_id, brand, base_variant, etc.)
        classes_path = model_dir / "variant_classes.json"
        with open(classes_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

        self.idx_to_meta = {int(k): v for k, v in raw_meta.items()}
        self.idx_to_sku = {int(k): int(v["sku_id"]) for k, v in self.idx_to_meta.items()}
        self.sku_to_idx = {sku: idx for idx, sku in self.idx_to_sku.items()}

        # 3. Load model (prefer TorchScript traced graph, fallback to .pth state dict)
        weight_candidates_pt = [
            model_dir / "traced_maxvit_brand_variant.pt",
            Path("models/maxvit_variants/traced_maxvit_brand_variant.pt"),
            Path("traced_maxvit_brand_variant.pt"),
        ]
        weight_candidates_pth = [
            model_dir / "best_maxvit_brand_variant.pth",
            Path("models/maxvit_variants/best_maxvit_brand_variant.pth"),
            Path("best_maxvit_brand_variant.pth"),
        ]
        traced_path = next((p for p in weight_candidates_pt if p.exists()), weight_candidates_pt[0])
        pth_path = next((p for p in weight_candidates_pth if p.exists()), weight_candidates_pth[0])

        if traced_path.exists():
            try:
                self.model = torch.jit.load(str(traced_path), map_location=self.device)
                self.model.eval()
            except Exception:
                self.model = None

        if self.model is None and pth_path.exists():
            import timm

            num_classes = len(self.idx_to_meta)
            self.model = timm.create_model(
                MODEL_ARCH, pretrained=False, num_classes=num_classes
            )
            state_dict = torch.load(str(pth_path), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

        if self.model is None:
            raise RuntimeError(
                f"Failed to load MaxViT model from either {traced_path} or {pth_path}."
            )

        # 4. Standard validation/inference transform matching training Cell 2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        """Classify a single product crop into one of 1,922 Brand__Variant classes."""
        if self.model is None or self.transform is None:
            return None

        # Convert to 3-channel RGB and apply validation transform
        rgb_img = image.convert("RGB")
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        # Mask unallowed classes if allowed_ids is provided (e.g. two-stage brand filter)
        if allowed_ids is not None:
            mask = torch.full_like(logits, -float("inf"))
            allowed_indices = [
                idx for idx, sku in self.idx_to_sku.items() if sku in allowed_ids
            ]
            if allowed_indices:
                mask[:, allowed_indices] = logits[:, allowed_indices]
                logits = mask

        pred_idx = int(logits.argmax(dim=-1).item())
        pred_sku = self.idx_to_sku.get(pred_idx)

        # Record step in UI trace
        if ctx and hasattr(ctx, "trace") and ctx.trace:
            meta = self.idx_to_meta.get(pred_idx, {})
            probs = torch.softmax(logits, dim=-1)
            conf = (
                float(probs[0, pred_idx].item())
                if logits.shape[-1] > pred_idx
                else 0.0
            )
            ctx.trace.step(
                "MaxViT Classification",
                f"Predicted SKU {pred_sku} ({meta.get('brand')} - {meta.get('base_variant')}) [conf={conf:.1%}]",
                info={
                    "pred_idx": pred_idx,
                    "sku_id": pred_sku,
                    "confidence": round(conf, 4),
                    "brand": meta.get("brand"),
                    "base_variant": meta.get("base_variant"),
                    "category": meta.get("category"),
                },
            )

        return pred_sku

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        """Crop and identify each bounding box on the shelf."""
        if not boxes:
            return []
        crops = [crop(image, b) for b in boxes]
        return [self.identify(c, ctx) for c in crops]


@register
class MaxVitRetrieval(MaxVitClassify):
    """Retrieval-task alias for MaxViT on GT-box crop benchmarks (hul_variants, hul_golden)."""

    name = "maxvit_retrieval"
    task = "retrieval"
    dataset = "hul_variants"
    models = [MODEL_ARCH]
    architecture = (
        "Fine-tuned Google MaxViT Tiny (maxvit_tiny_tf_224) 1,922 Brand__Variant Crop Matcher "
        "(Task: Retrieval, 0 VLM Calls)"
    )

