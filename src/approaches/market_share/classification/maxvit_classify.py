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
try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None  # type: ignore
    transforms = None  # type: ignore

from approaches.base import Approach, Box, Context, crop, register
from utils import dataset

# Model architecture identifier and GCS model storage path
MODEL_ARCH = "maxvit_tiny_tf_224"
GCS_MODEL_PATH = "gs://marketshare-datasets/models/maxvit_variants"


def _preload_nvidia_libraries() -> None:
    """Pre-load modular NVIDIA cuDNN v9+ and cuBLAS libraries if installed via pip wheels."""
    import ctypes
    import glob
    import os
    import site
    import sys

    search_paths = []
    try:
        search_paths.extend(site.getsitepackages())
    except Exception:
        pass
    search_paths.extend(sys.path)

    for p in search_paths:
        for pkg in ("cudnn", "cublas", "cuda_runtime"):
            lib_dir = os.path.join(p, "nvidia", pkg, "lib")
            if os.path.isdir(lib_dir):
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                if lib_dir not in ld:
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld}"
                for pat in ("libcudnn*.so*", "libcublas*.so*", "libcudart*.so*"):
                    for match in sorted(glob.glob(os.path.join(lib_dir, pat))):
                        try:
                            ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass


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
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch is not None
            else "cpu"
        )
        self.model: Any = None
        self.idx_to_meta: dict[int, dict[str, Any]] = {}
        self.idx_to_sku: dict[int, int] = {}
        self.sku_to_idx: dict[int, int] = {}
        self.transform: Any = None

    def setup(self, config: dict, ctx: Context) -> None:
        """Load fine-tuned model artifacts directly from gs://marketshare-datasets/models/maxvit_variants/."""
        global torch, transforms
        if torch is None or transforms is None:
            import torch
            from torchvision import transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _preload_nvidia_libraries()

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
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            try:
                with torch.no_grad():
                    out = m(dummy)
                return out is not None and out.shape[-1] == num_classes
            except Exception as exc:
                exc_str = str(exc)
                if "CUDNN" in exc_str or "cudnn" in exc_str.lower():
                    print(
                        f"[MaxViT] cuDNN backend issue detected ({exc_str[:120]}...). "
                        "Switching to native PyTorch CUDA kernels (torch.backends.cudnn.enabled = False)..."
                    )
                    torch.backends.cudnn.enabled = False
                    try:
                        with torch.no_grad():
                            out = m(dummy)
                        return out is not None and out.shape[-1] == num_classes
                    except Exception as e2:
                        print(f"[MaxViT] Native CUDA fallback check failed: {e2}")
                        return False
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
            allowed_indices = [
                idx for idx, sku in self.idx_to_sku.items() if sku in allowed_ids
            ]
            if not allowed_indices:
                return None
            mask = torch.full_like(logits, -float("inf"))
            mask[:, allowed_indices] = logits[:, allowed_indices]
            logits = mask

        pred_idx = int(logits.argmax(dim=-1).item())
        pred_sku = self.idx_to_sku.get(pred_idx)

        # Record step in UI trace
        if ctx and hasattr(ctx, "trace") and ctx.trace:
            meta = self.idx_to_meta.get(pred_idx, {})
            probs = torch.softmax(logits, dim=-1)
            k_cand = min(5, logits.shape[-1])
            top_vals, top_inds = torch.topk(probs, k=k_cand, dim=-1)
            top5 = []
            for val, ind in zip(top_vals[0].tolist(), top_inds[0].tolist()):
                c_sku = self.idx_to_sku.get(ind)
                c_meta = self.idx_to_meta.get(ind, {})
                top5.append({
                    "sku_id": c_sku,
                    "brand": c_meta.get("brand"),
                    "base_variant": c_meta.get("base_variant"),
                    "category": c_meta.get("category"),
                    "probability": round(float(val), 4),
                })
            conf = float(top_vals[0][0].item())
            margin = float(top_vals[0][0].item() - top_vals[0][1].item()) if k_cand > 1 else 1.0

            ctx.trace.step(
                "MaxViT Classification",
                f"Predicted SKU {pred_sku} ({meta.get('brand')} - {meta.get('base_variant')}) [conf={conf:.1%}, margin={margin:.3f}]",
                info={
                    "model": MODEL_ARCH,
                    "pred_idx": pred_idx,
                    "sku_id": pred_sku,
                    "confidence": round(conf, 4),
                    "margin": round(margin, 4),
                    "brand": meta.get("brand"),
                    "base_variant": meta.get("base_variant"),
                    "category": meta.get("category"),
                    "packaging_type": meta.get("packaging_type"),
                    "top5": top5,
                },
            )

        return pred_sku

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        """Crop and identify each bounding box on the shelf in high-throughput batched mode."""
        if not boxes:
            return []
        if self.model is None or self.transform is None:
            return [None] * len(boxes)

        crops = [crop(image, b).convert("RGB") for b in boxes]
        batch_tensors = [self.transform(c) for c in crops]
        batch = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            batch_logits = self.model(batch)

        preds: list[int | None] = []
        batch_probs = torch.softmax(batch_logits, dim=-1)
        k_cand = min(5, batch_logits.shape[-1])
        top_vals, top_inds = torch.topk(batch_probs, k=k_cand, dim=-1)

        for i, box in enumerate(boxes):
            p_idx = int(batch_logits[i].argmax(dim=-1).item())
            p_sku = self.idx_to_sku.get(p_idx)
            preds.append(p_sku)

            if ctx and hasattr(ctx, "trace") and ctx.trace:
                meta = self.idx_to_meta.get(p_idx, {})
                conf = float(top_vals[i, 0].item())
                margin = float(top_vals[i, 0].item() - top_vals[i, 1].item()) if k_cand > 1 else 1.0
                top5 = []
                for val, ind in zip(top_vals[i].tolist(), top_inds[i].tolist()):
                    c_sku = self.idx_to_sku.get(ind)
                    c_meta = self.idx_to_meta.get(ind, {})
                    top5.append({
                        "sku_id": c_sku,
                        "brand": c_meta.get("brand"),
                        "base_variant": c_meta.get("base_variant"),
                        "category": c_meta.get("category"),
                        "probability": round(float(val), 4),
                    })
                ctx.trace.step(
                    "MaxViT Shelf Box Identification",
                    f"Box #{i+1}: SKU {p_sku} ({meta.get('brand')} - {meta.get('base_variant')}) [conf={conf:.1%}, margin={margin:.3f}]",
                    boxes=[box],
                    info={
                        "box_index": i,
                        "model": MODEL_ARCH,
                        "sku_id": p_sku,
                        "confidence": round(conf, 4),
                        "margin": round(margin, 4),
                        "brand": meta.get("brand"),
                        "base_variant": meta.get("base_variant"),
                        "category": meta.get("category"),
                        "top5": top5,
                    },
                )

        return preds


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

