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
import os
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torchvision import transforms

from approaches.base import Approach, Box, Context, crop, register
from utils import dataset

# Model architecture identifier
MODEL_ARCH = "maxvit_tiny_tf_224"

DEFAULT_GCS_ROOTS = [
    "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/maxvit_variants",
    "gs://marketshare-datasets/models/maxvit_variants",
]


def _resolve_or_download_artifact(
    filename: str,
    local_candidates: list[Path],
    gcs_candidates: list[str],
    cache_dir: Path,
) -> Path | None:
    """Find file locally or download on-demand from GCP bucket into cache_dir."""
    # 1. Fast path: check local candidates
    for cand in local_candidates:
        if cand.exists():
            return cand

    # 2. Check if already cached in cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_cache = cache_dir / filename
    if target_cache.exists():
        return target_cache

    # 3. Download from GCS candidates
    for gcs_root in gcs_candidates:
        if not gcs_root or not str(gcs_root).startswith("gs://"):
            continue
        gcs_uri = f"{str(gcs_root).rstrip('/')}/{filename}"
        try:
            print(f"[MaxViT] Downloading {filename} from {gcs_uri} ...")
            raw = dataset.read_bytes(gcs_uri)
            target_cache.write_bytes(raw)
            print(f"[MaxViT] Successfully cached to {target_cache} ({len(raw) / (1024 * 1024):.2f} MB)")
            return target_cache
        except Exception as e:
            # Try next candidate
            continue

    return None


@register
class MaxVitClassify(Approach):
    """Fine-tuned Google MaxViT Tiny (maxvit_tiny_tf_224) 1,922-Class Variant Classifier."""

    name = "maxvit_classify"
    task = "classification"
    dataset = "hul_variants"
    models = [MODEL_ARCH]
    architecture = (
        "Fine-tuned Google MaxViT Tiny (maxvit_tiny_tf_224) 1,922 Brand__Variant Classifier "
        "(224x224 RGB, 0 VLM Calls, GCS Auto-Download, Pure PyTorch/TorchScript)"
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
        """Load fine-tuned model artifacts and class metadata from local disk or GCS bucket."""
        cache_dir = Path("models/maxvit_variants")

        # 1. Assemble GCS search roots (supports custom MAXVIT_GCS_URI environment variable)
        gcs_candidates: list[str] = []
        custom_gcs = os.environ.get("MAXVIT_GCS_URI") or config.get("gcp", {}).get("maxvit_model_gcs")
        if custom_gcs:
            gcs_candidates.append(str(custom_gcs))
        gcs_candidates.extend(DEFAULT_GCS_ROOTS)

        # 2. Resolve deployment class mapping (variant_classes.json)
        local_meta = [
            cache_dir / "variant_classes.json",
            Path("variant_classes.json"),
            Path(__file__).resolve().parents[4] / "models" / "maxvit_variants" / "variant_classes.json",
        ]
        classes_path = _resolve_or_download_artifact(
            "variant_classes.json", local_meta, gcs_candidates, cache_dir
        )
        if classes_path is None or not classes_path.exists():
            raise FileNotFoundError(
                "Could not locate 'variant_classes.json' locally or in GCS "
                f"(checked GCS: {', '.join(gcs_candidates)}). "
                "Set MAXVIT_GCS_URI='gs://<your-bucket>/<path>' or place file locally."
            )

        with open(classes_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

        self.idx_to_meta = {int(k): v for k, v in raw_meta.items()}
        self.idx_to_sku = {int(k): int(v["sku_id"]) for k, v in self.idx_to_meta.items()}
        self.sku_to_idx = {sku: idx for idx, sku in self.idx_to_sku.items()}

        # 3. Load model weights (prefer TorchScript traced graph, fallback to .pth state dict)
        local_pt = [
            cache_dir / "traced_maxvit_brand_variant.pt",
            Path("traced_maxvit_brand_variant.pt"),
            Path(__file__).resolve().parents[4] / "models" / "maxvit_variants" / "traced_maxvit_brand_variant.pt",
        ]
        traced_path = _resolve_or_download_artifact(
            "traced_maxvit_brand_variant.pt", local_pt, gcs_candidates, cache_dir
        )

        if traced_path and traced_path.exists():
            try:
                print(f"[MaxViT] Loading TorchScript graph from {traced_path}...")
                self.model = torch.jit.load(str(traced_path), map_location=self.device)
                self.model.eval()
            except Exception as e:
                print(f"[MaxViT] TorchScript load warning: {e}, attempting .pth fallback...")
                self.model = None

        if self.model is None:
            local_pth = [
                cache_dir / "best_maxvit_brand_variant.pth",
                Path("best_maxvit_brand_variant.pth"),
                Path(__file__).resolve().parents[4] / "models" / "maxvit_variants" / "best_maxvit_brand_variant.pth",
            ]
            pth_path = _resolve_or_download_artifact(
                "best_maxvit_brand_variant.pth", local_pth, gcs_candidates, cache_dir
            )
            if pth_path and pth_path.exists():
                print(f"[MaxViT] Loading PyTorch weights from {pth_path} via timm...")
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
                "Failed to load MaxViT model weights from local disk or GCP bucket. "
                f"Checked GCS locations: {', '.join(gcs_candidates)}. "
                "Please upload 'traced_maxvit_brand_variant.pt' or 'best_maxvit_brand_variant.pth' to GCS, "
                "or set MAXVIT_GCS_URI='gs://<your-bucket>/<path>'."
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
        print(f"✔ MaxViT model successfully initialized on {self.device} ({len(self.idx_to_sku)} classes)")


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

