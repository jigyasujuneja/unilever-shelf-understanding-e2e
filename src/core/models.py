"""Singleton neural model registry (`src/core/models.py`).

Provides thread-safe lazy singleton loaders for all production vision and multimodal backbones:
  - `load_yolo26n_detector()`: Ultralytics `YOLO` (`yolo26n_sku110k.pt` or `yolo26n.pt`)
  - `load_rtdetr_detector()`: Ultralytics `RTDETR` (`rtdetr-l.pt`)
  - `load_mobilesam_segmenter()`: Ultralytics `SAM` (`mobile_sam.pt`)
  - `load_maxvit_t_backbone()`: Torchvision `maxvit_t` (`MaxVit_T_Weights.DEFAULT`)
  - `load_efficientnet_b4_backbone()`: Torchvision `efficientnet_b4` (`EfficientNet_B4_Weights.DEFAULT`)
  - `load_siglip_classifier()`: HuggingFace `google/siglip-base-patch16-224`
  - `load_owlv2_detector()`: HuggingFace `google/owlv2-base-patch16-ensemble`
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_maxvit_t_backbone() -> Any:
    """Load real pre-trained ``torchvision.models.maxvit_t(weights=MaxVit_T_Weights.DEFAULT)``."""
    import torch
    import torchvision.models as tvm

    model = tvm.maxvit_t(weights=tvm.MaxVit_T_Weights.DEFAULT).eval()
    model.classifier = torch.nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        model.classifier[2],
        model.classifier[3],
    )
    return model


@lru_cache(maxsize=1)
def load_efficientnet_b4_backbone() -> Any:
    """Load real pre-trained ``torchvision.models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)``."""
    import torch
    import torchvision.models as tvm

    model = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.DEFAULT).eval()
    model.classifier = torch.nn.Identity()
    return model


@lru_cache(maxsize=1)
def load_yolo26n_detector() -> Any:
    """Load real SKU-110K fine-tuned Ultralytics ``YOLO`` detector (or ``yolo26n.pt`` base weights)."""
    from ultralytics import YOLO

    ft_path = Path(__file__).resolve().parents[2] / "data" / "models" / "yolo26n_sku110k.pt"
    if ft_path.exists():
        return YOLO(str(ft_path))
    return YOLO("yolo26n.pt")


@lru_cache(maxsize=1)
def load_rtdetr_detector() -> Any:
    """Load real pre-trained Ultralytics ``RTDETR('rtdetr-l.pt')`` neural transformer detector."""
    from ultralytics import RTDETR

    return RTDETR("rtdetr-l.pt")


@lru_cache(maxsize=1)
def load_mobilesam_segmenter() -> Any:
    """Load real pre-trained Ultralytics ``SAM('mobile_sam.pt')`` instance segmentation model."""
    from ultralytics import SAM

    return SAM("mobile_sam.pt")


@lru_cache(maxsize=1)
def load_siglip_classifier() -> tuple[Any, Any]:
    """Load real HuggingFace ``google/siglip-base-patch16-224`` zero-shot vision-language model."""
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    mod = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
    return proc, mod


@lru_cache(maxsize=1)
def load_owlv2_detector() -> tuple[Any, Any]:
    """Load real HuggingFace ``google/owlv2-base-patch16-ensemble`` open-vocabulary detector."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    mod = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
    return proc, mod


__all__ = [
    "load_efficientnet_b4_backbone",
    "load_maxvit_t_backbone",
    "load_mobilesam_segmenter",
    "load_owlv2_detector",
    "load_rtdetr_detector",
    "load_siglip_classifier",
    "load_yolo26n_detector",
]
