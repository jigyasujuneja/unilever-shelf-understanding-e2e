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
import time
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
from utils import dataset, pricing, telemetry

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
    skus = {
        "maxvit_compute_second": (pricing.CLOUD_RUN, "Jobs CPU in us-central1"),
    }

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

        otel_parent = getattr(ctx, "otel_parent", None) if ctx else None

        # 1. Image preprocessing
        t_prep_start = time.perf_counter()
        rgb_img = image.convert("RGB")
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        t_prep = time.perf_counter() - t_prep_start

        # 2. Forward inference through fine-tuned MaxViT
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
        if torch is not None and torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        t_infer = time.perf_counter() - t_infer_start

        # 3. Post-processing & Candidate Masking
        t_post_start = time.perf_counter()
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
        t_post = time.perf_counter() - t_post_start
        t_total = t_prep + t_infer + t_post

        # 4. Transparent 100% all-in cost calculation
        cost_sheet = getattr(ctx, "sheet", None) if ctx else None
        cost_breakdown = pricing.maxvit_all_in_cost(
            infer_seconds=t_infer,
            batch_size=1,
            sheet=cost_sheet,
        )

        # 5. Bill compute unit to runner trace for catalog pricing audit
        if ctx and hasattr(ctx, "bill") and callable(ctx.bill):
            try:
                ctx.bill("maxvit_compute_second", round(t_infer, 6))
            except Exception:
                pass

        # 6. OpenTelemetry Span & Correlated Cloud Logging
        span_attrs = {
            "gen_ai.operation.name": "classify_crop",
            "gen_ai.provider.name": "google.maxvit",
            "gen_ai.request.model": MODEL_ARCH,
            "shelf_bench.task": getattr(self, "task", "classification"),
            "shelf_bench.dataset": getattr(self, "dataset", "hul_variants"),
            "shelf_bench.device": str(self.device),
            "shelf_bench.batch_size": 1,
            "shelf_bench.input_resolution": "224x224",
            "shelf_bench.num_classes": len(self.idx_to_sku),
            "shelf_bench.weights_source": GCS_MODEL_PATH,
            "shelf_bench.latency.preprocess_ms": round(t_prep * 1000, 2),
            "shelf_bench.latency.inference_ms": round(t_infer * 1000, 2),
            "shelf_bench.latency.postprocess_ms": round(t_post * 1000, 2),
            "shelf_bench.latency.total_ms": round(t_total * 1000, 2),
            "shelf_bench.throughput.fps": round(1.0 / max(t_total, 1e-6), 1),
            "shelf_bench.pred_sku_id": pred_sku,
            "shelf_bench.confidence": round(conf, 4),
            "shelf_bench.margin": round(margin, 4),
            "shelf_bench.brand": meta.get("brand"),
            "shelf_bench.base_variant": meta.get("base_variant"),
            "shelf_bench.category": meta.get("category"),
            "shelf_bench.packaging_type": meta.get("packaging_type"),
            "shelf_bench.cost.token_cost_usd": 0.0,
            "shelf_bench.cost.compute_cost_usd": round(cost_breakdown["compute_cost_usd"], 8),
            "shelf_bench.cost.compute_cost_inr": round(cost_breakdown["compute_cost_inr"], 6),
            "shelf_bench.cost.storage_cost_usd": round(cost_breakdown["storage_cost_usd"], 8),
            "shelf_bench.cost.all_in_cost_usd": round(cost_breakdown["all_in_cost_usd"], 8),
            "shelf_bench.cost.all_in_cost_inr": round(cost_breakdown["all_in_cost_inr"], 6),
        }

        with telemetry.span(f"maxvit {MODEL_ARCH}", parent=otel_parent, **span_attrs) as span:
            telemetry.log(
                span,
                "maxvit_prediction",
                {
                    "event": "maxvit_prediction",
                    "model": MODEL_ARCH,
                    "device": str(self.device),
                    "sku_id": pred_sku,
                    "confidence": round(conf, 4),
                    "margin": round(margin, 4),
                    "brand": meta.get("brand"),
                    "base_variant": meta.get("base_variant"),
                    "category": meta.get("category"),
                    "packaging_type": meta.get("packaging_type"),
                    "top5": top5,
                    "timing": {
                        "preprocess_ms": round(t_prep * 1000, 2),
                        "inference_ms": round(t_infer * 1000, 2),
                        "postprocess_ms": round(t_post * 1000, 2),
                        "total_ms": round(t_total * 1000, 2),
                        "throughput_fps": round(1.0 / max(t_total, 1e-6), 1),
                    },
                    "all_in_cost": cost_breakdown,
                },
            )
            if span:
                span.add_event(
                    "shortlist_evaluated",
                    telemetry.clean({
                        "pred_sku": pred_sku,
                        "confidence": round(conf, 4),
                        "margin": round(margin, 4),
                        "top1_brand": meta.get("brand"),
                    }),
                )

        # 7. Record step in UI trace
        if ctx and hasattr(ctx, "trace") and ctx.trace:
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
                    "timing": {
                        "preprocess_ms": round(t_prep * 1000, 2),
                        "inference_ms": round(t_infer * 1000, 2),
                        "postprocess_ms": round(t_post * 1000, 2),
                        "total_ms": round(t_total * 1000, 2),
                    },
                    "all_in_cost": cost_breakdown,
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

        otel_parent = getattr(ctx, "otel_parent", None) if ctx else None

        # 1. Batch cropping and tensor preprocessing
        t_prep_start = time.perf_counter()
        crops = [crop(image, b).convert("RGB") for b in boxes]
        batch_tensors = [self.transform(c) for c in crops]
        batch = torch.stack(batch_tensors).to(self.device)
        t_prep = time.perf_counter() - t_prep_start

        # 2. Batch forward inference
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            batch_logits = self.model(batch)
        if torch is not None and torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        t_infer = time.perf_counter() - t_infer_start

        # 3. Batch post-processing
        t_post_start = time.perf_counter()
        preds: list[int | None] = []
        batch_probs = torch.softmax(batch_logits, dim=-1)
        k_cand = min(5, batch_logits.shape[-1])
        top_vals, top_inds = torch.topk(batch_probs, k=k_cand, dim=-1)
        per_box_info = []

        for i, box in enumerate(boxes):
            p_idx = int(batch_logits[i].argmax(dim=-1).item())
            p_sku = self.idx_to_sku.get(p_idx)
            preds.append(p_sku)
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
            per_box_info.append({
                "box_index": i,
                "box": [round(v, 1) for v in box],
                "sku_id": p_sku,
                "confidence": round(conf, 4),
                "margin": round(margin, 4),
                "brand": meta.get("brand"),
                "base_variant": meta.get("base_variant"),
                "category": meta.get("category"),
                "top5": top5,
            })
        t_post = time.perf_counter() - t_post_start
        t_total = t_prep + t_infer + t_post

        # 4. Transparent 100% all-in cost calculation for the shelf batch
        cost_sheet = getattr(ctx, "sheet", None) if ctx else None
        cost_breakdown = pricing.maxvit_all_in_cost(
            infer_seconds=t_infer,
            batch_size=len(boxes),
            sheet=cost_sheet,
        )

        # 5. Bill compute unit to runner trace
        if ctx and hasattr(ctx, "bill") and callable(ctx.bill):
            try:
                ctx.bill("maxvit_compute_second", round(t_infer, 6))
            except Exception:
                pass

        # 6. OpenTelemetry Batch Span & Cloud Logging
        batch_span_attrs = {
            "gen_ai.operation.name": "classify_shelf_boxes",
            "gen_ai.provider.name": "google.maxvit",
            "gen_ai.request.model": MODEL_ARCH,
            "shelf_bench.task": getattr(self, "task", "retrieval"),
            "shelf_bench.dataset": getattr(self, "dataset", "hul_variants"),
            "shelf_bench.device": str(self.device),
            "shelf_bench.batch_size": len(boxes),
            "shelf_bench.input_resolution": "224x224",
            "shelf_bench.num_classes": len(self.idx_to_sku),
            "shelf_bench.latency.preprocess_ms": round(t_prep * 1000, 2),
            "shelf_bench.latency.inference_ms": round(t_infer * 1000, 2),
            "shelf_bench.latency.postprocess_ms": round(t_post * 1000, 2),
            "shelf_bench.latency.total_ms": round(t_total * 1000, 2),
            "shelf_bench.throughput.fps": round(len(boxes) / max(t_total, 1e-6), 1),
            "shelf_bench.cost.token_cost_usd": 0.0,
            "shelf_bench.cost.compute_cost_usd": round(cost_breakdown["compute_cost_usd"], 8),
            "shelf_bench.cost.compute_cost_inr": round(cost_breakdown["compute_cost_inr"], 6),
            "shelf_bench.cost.storage_cost_usd": round(cost_breakdown["storage_cost_usd"], 8),
            "shelf_bench.cost.all_in_cost_usd": round(cost_breakdown["all_in_cost_usd"], 8),
            "shelf_bench.cost.all_in_cost_inr": round(cost_breakdown["all_in_cost_inr"], 6),
            "shelf_bench.cost.per_crop_inr": round(cost_breakdown["per_crop_inr"], 6),
        }

        with telemetry.span(f"maxvit_batch {MODEL_ARCH}", parent=otel_parent, **batch_span_attrs) as span:
            telemetry.log(
                span,
                "maxvit_batch_prediction",
                {
                    "event": "maxvit_batch_prediction",
                    "model": MODEL_ARCH,
                    "device": str(self.device),
                    "boxes_count": len(boxes),
                    "timing": {
                        "preprocess_ms": round(t_prep * 1000, 2),
                        "inference_ms": round(t_infer * 1000, 2),
                        "postprocess_ms": round(t_post * 1000, 2),
                        "total_ms": round(t_total * 1000, 2),
                        "throughput_fps": round(len(boxes) / max(t_total, 1e-6), 1),
                    },
                    "all_in_cost": cost_breakdown,
                    "predictions_summary": [
                        {
                            "box_index": b["box_index"],
                            "sku_id": b["sku_id"],
                            "brand": b["brand"],
                            "confidence": b["confidence"],
                        }
                        for b in per_box_info[:10]
                    ],
                },
            )
            if span:
                for b_info in per_box_info:
                    span.add_event(
                        f"box_{b_info['box_index']+1}_matched",
                        telemetry.clean({
                            "sku_id": b_info["sku_id"],
                            "brand": b_info["brand"],
                            "confidence": b_info["confidence"],
                            "margin": b_info["margin"],
                        }),
                    )

        # 7. Record individual box steps in UI trace
        if ctx and hasattr(ctx, "trace") and ctx.trace:
            for b_info in per_box_info:
                ctx.trace.step(
                    "MaxViT Shelf Box Identification",
                    f"Box #{b_info['box_index']+1}: SKU {b_info['sku_id']} ({b_info['brand']} - {b_info['base_variant']}) [conf={b_info['confidence']:.1%}, margin={b_info['margin']:.3f}]",
                    boxes=[boxes[b_info["box_index"]]],
                    info={
                        "box_index": b_info["box_index"],
                        "model": MODEL_ARCH,
                        "sku_id": b_info["sku_id"],
                        "confidence": b_info["confidence"],
                        "margin": b_info["margin"],
                        "brand": b_info["brand"],
                        "base_variant": b_info["base_variant"],
                        "category": b_info["category"],
                        "top5": b_info["top5"],
                        "timing": {
                            "shelf_total_ms": round(t_total * 1000, 2),
                            "crop_amortized_ms": round(t_infer * 1000 / len(boxes), 2),
                        },
                        "all_in_cost_per_crop_inr": round(cost_breakdown["per_crop_inr"], 6),
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

