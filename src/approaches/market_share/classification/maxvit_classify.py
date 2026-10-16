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
from utils import dataset

# Model architecture identifier and GCS model storage path
MODEL_ARCH = "maxvit_tiny_tf_224"
GCS_MODEL_PATH = "gs://marketshare-datasets/models/maxvit_variants"


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
        """Load fine-tuned model artifacts directly from gs://marketshare-datasets/models/maxvit_variants/."""
        cache_dir = Path("models/maxvit_variants")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load variant_classes.json mapping (0..1921 -> sku_id, brand, base_variant)
        classes_path = cache_dir / "variant_classes.json"
        if not classes_path.exists():
            print(f"[MaxViT] Downloading variant_classes.json from {GCS_MODEL_PATH}/variant_classes.json ...")
            raw_meta = dataset.read_bytes(f"{GCS_MODEL_PATH}/variant_classes.json")
            classes_path.write_bytes(raw_meta)

        with open(classes_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

        self.idx_to_meta = {int(k): v for k, v in raw_meta.items()}
        self.idx_to_sku = {int(k): int(v["sku_id"]) for k, v in self.idx_to_meta.items()}
        self.sku_to_idx = {sku: idx for idx, sku in self.idx_to_sku.items()}
        num_classes = len(self.idx_to_meta)

        # Helper to validate a candidate model with a dummy forward pass
        def _test_forward(m: Any) -> bool:
            try:
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                with torch.no_grad():
                    out = m(dummy)
                return out is not None and out.shape[-1] == num_classes
            except Exception as exc:
                print(f"[MaxViT] Forward pass check failed: {exc}")
                return False

        # 2. Primary: Native PyTorch weights (best_maxvit_brand_variant.pth via timm)
        pth_path = cache_dir / "best_maxvit_brand_variant.pth"
        if not pth_path.exists():
            try:
                uri = f"{GCS_MODEL_PATH}/best_maxvit_brand_variant.pth"
                print(f"[MaxViT] Downloading PyTorch weights from {uri} ...")
                raw = dataset.read_bytes(uri)
                pth_path.write_bytes(raw)
            except Exception as e:
                print(f"[MaxViT] Note: Could not fetch .pth checkpoint ({e})")

        if pth_path.exists():
            try:
                print(f"[MaxViT] Loading PyTorch weights from {pth_path}...")
                import timm

                m = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=num_classes)
                state_dict = torch.load(str(pth_path), map_location=self.device)
                if isinstance(state_dict, dict):
                    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                        state_dict = state_dict["state_dict"]
                    elif "model" in state_dict and isinstance(state_dict["model"], dict):
                        state_dict = state_dict["model"]
                state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
                m.load_state_dict(state_dict)
                m.to(self.device)
                m.eval()
                if _test_forward(m):
                    self.model = m
                    print(f"✔ MaxViT PyTorch state_dict loaded successfully ({num_classes} classes)")
                else:
                    print("[MaxViT] PyTorch model forward pass validation failed.")
            except Exception as e:
                print(f"[MaxViT] PyTorch state_dict load error: {e}")
                self.model = None

        # 3. Fallback: TorchScript traced model (traced_maxvit_brand_variant.pt)
        if self.model is None:
            traced_path = cache_dir / "traced_maxvit_brand_variant.pt"
            if not traced_path.exists():
                try:
                    uri = f"{GCS_MODEL_PATH}/traced_maxvit_brand_variant.pt"
                    print(f"[MaxViT] Downloading TorchScript model from {uri} ...")
                    raw = dataset.read_bytes(uri)
                    traced_path.write_bytes(raw)
                except Exception as e:
                    print(f"[MaxViT] Note: Could not fetch TorchScript model ({e})")

            if traced_path.exists():
                try:
                    print(f"[MaxViT] Loading TorchScript model from {traced_path}...")
                    m = torch.jit.load(str(traced_path), map_location=self.device)
                    m.eval()
                    if _test_forward(m):
                        self.model = m
                        print(f"✔ MaxViT TorchScript model loaded successfully ({num_classes} classes)")
                    else:
                        print("[MaxViT] TorchScript model failed forward pass validation on this runtime.")
                except Exception as e:
                    print(f"[MaxViT] TorchScript load error: {e}")
                    self.model = None

        if self.model is None:
            raise RuntimeError(
                f"Could not initialize a functional MaxViT model from {GCS_MODEL_PATH}. "
                "Ensure 'best_maxvit_brand_variant.pth' is uploaded to the bucket and 'timm' is installed."
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
        print(f"✔ MaxViT pipeline ready on {self.device} ({len(self.idx_to_sku)} classes)")

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

