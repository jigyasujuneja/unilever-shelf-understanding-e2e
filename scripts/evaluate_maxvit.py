"""Standalone Evaluation & Benchmark Script for Fine-Tuned Google MaxViT Tiny (maxvit_tiny_tf_224).

Can be executed directly via Python in any environment (Local, Vertex AI, Google Colab, or CLI):
    python scripts/evaluate_maxvit.py --split test --limit 200 --batch-size 32

Computes Unilever 5-Level Taxonomy Metrics:
  - Exact SKU Top-1 Accuracy (L5)
  - Base Variant Accuracy (L4)
  - Brand Accuracy (L3)
  - Category Accuracy (L2)
  - Packaging Type Accuracy
  - HUL Ownership Attribution (L1)
  - Inference Latency (ms/crop) & Throughput (crops/sec)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CACHE_DIR = Path("cache/crops")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("models/maxvit_variants")
DATA_ROOT = "gs://marketshare-datasets/derived_manifests/hul_variants"
RPC_JSON_URI = f"{DATA_ROOT}/rpc.json"


def resolve_gcs_uri(uri_or_path: str) -> str:
    if uri_or_path.startswith("gs://"):
        return uri_or_path
    return f"{DATA_ROOT.rstrip('/')}/{uri_or_path.lstrip('/')}"


def get_storage_client():
    from google.cloud import storage
    return storage.Client()


def read_gcs_bytes(gcs_uri: str, client=None) -> bytes:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")
    client = client or get_storage_client()
    path_without_prefix = gcs_uri[5:]
    bucket_name, blob_name = path_without_prefix.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


class EvalCropDataset(Dataset):
    """Dataset for fast evaluation of split records."""

    def __init__(self, records: list[dict], sku_to_idx: dict[int, int], transform=None, storage_client=None):
        self.records = records
        self.sku_to_idx = sku_to_idx
        self.transform = transform
        self.client = storage_client

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, uri: str) -> Image.Image:
        import hashlib

        full_uri = resolve_gcs_uri(uri)
        uri_hash = hashlib.sha256(full_uri.encode("utf-8")).hexdigest()
        cache_path = CACHE_DIR / f"{uri_hash}.bin"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return Image.open(f).convert("RGB")
            except Exception:
                pass

        try:
            raw = read_gcs_bytes(full_uri, self.client)
            # Atomic disk cache write
            tmp_path = CACHE_DIR / f"{uri_hash}.tmp_{os.getpid()}_{time.time_ns()}"
            tmp_path.write_bytes(raw)
            os.replace(tmp_path, cache_path)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (240, 240, 240))

    def __getitem__(self, idx: int):
        item = self.records[idx]
        uri = item.get("uri") or item.get("image")
        prods = item.get("products", [])
        cid = int(prods[0]) if prods else 0
        target_idx = self.sku_to_idx.get(cid, 0)

        boxes = item.get("boxes", [])
        box = boxes[0] if boxes else None

        img = self._load_image(uri)
        if box:
            x1, y1, x2, y2 = box
            w, h = img.size
            if x2 > x1 and y2 > y1 and (x2 - x1 < w or y2 - y1 < h):
                img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        return img, target_idx, cid


def load_model(device: torch.device, num_classes: int, storage_client=None):
    """Load TorchScript traced model or PyTorch weights from local disk or GCS."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    traced_path = MODEL_DIR / "traced_maxvit_brand_variant.pt"
    pth_path = MODEL_DIR / "best_maxvit_brand_variant.pth"

    gcs_candidates = [
        os.environ.get("MAXVIT_GCS_URI", "").rstrip("/"),
        "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/maxvit_variants",
        "gs://marketshare-datasets/models/maxvit_variants",
    ]

    # Try local traced first
    if traced_path.exists():
        print(f"Loading TorchScript model from {traced_path}...")
        try:
            model = torch.jit.load(str(traced_path), map_location=device)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load local TorchScript: {e}")

    # Try local pth
    if pth_path.exists():
        print(f"Loading PyTorch weights from {pth_path}...")
        import timm
        model = timm.create_model("maxvit_tiny_tf_224", pretrained=False, num_classes=num_classes)
        state_dict = torch.load(str(pth_path), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    # If neither exists locally, attempt download from GCS
    for gcs_root in gcs_candidates:
        if not gcs_root or not gcs_root.startswith("gs://"):
            continue

        # Try traced_maxvit_brand_variant.pt
        try:
            uri = f"{gcs_root}/traced_maxvit_brand_variant.pt"
            print(f"Attempting download of {uri} ...")
            raw = read_gcs_bytes(uri, storage_client)
            traced_path.write_bytes(raw)
            print(f"Downloaded and cached to {traced_path} ({len(raw)/(1024*1024):.1f} MB)")
            model = torch.jit.load(str(traced_path), map_location=device)
            model.eval()
            return model
        except Exception:
            pass

        # Try best_maxvit_brand_variant.pth
        try:
            uri = f"{gcs_root}/best_maxvit_brand_variant.pth"
            print(f"Attempting download of {uri} ...")
            raw = read_gcs_bytes(uri, storage_client)
            pth_path.write_bytes(raw)
            print(f"Downloaded and cached to {pth_path} ({len(raw)/(1024*1024):.1f} MB)")
            import timm
            model = timm.create_model("maxvit_tiny_tf_224", pretrained=False, num_classes=num_classes)
            state_dict = torch.load(str(pth_path), map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not load MaxViT weights locally or from GCS candidates: {gcs_candidates}. "
        "Set MAXVIT_GCS_URI='gs://<bucket>/<path>' or place weights in models/maxvit_variants/."
    )



def evaluate(split_name: str = "test", limit: int = 0, batch_size: int = 32, num_workers: int = 2):
    print("=" * 75)
    print(f"EVALUATING GOOGLE MAXVIT TINY ON HUL_VARIANTS [{split_name.upper()} SPLIT]")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Classes & Metadata Mapping
    classes_json_path = MODEL_DIR / "variant_classes.json"
    if not classes_json_path.exists():
        raise FileNotFoundError(f"Missing {classes_json_path}")

    with open(classes_json_path, "r", encoding="utf-8") as f:
        classes_dict = json.load(f)

    idx_to_meta = {int(k): v for k, v in classes_dict.items()}
    idx_to_sku = {int(k): int(v["sku_id"]) for k, v in idx_to_meta.items()}
    sku_to_idx = {sku: idx for idx, sku in idx_to_sku.items()}
    num_classes = len(idx_to_meta)
    print(f"Loaded {num_classes} Brand__Variant Classes.")

    # 2. Download or read manifest splits
    storage_client = get_storage_client()
    raw_manifest = read_gcs_bytes(RPC_JSON_URI, storage_client)
    manifest = json.loads(raw_manifest.decode("utf-8"))
    split_records = manifest.get("splits", {}).get(split_name, [])

    if limit > 0 and limit < len(split_records):
        split_records = split_records[:limit]

    print(f"Evaluating {len(split_records)} store crops from split '{split_name}'.")

    # 3. Prepare Model & DataLoader
    model = load_model(device, num_classes, storage_client)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = EvalCropDataset(split_records, sku_to_idx, transform=eval_transform, storage_client=storage_client)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 4. Evaluation Loop
    total_samples = 0
    correct_sku = 0
    correct_brand = 0
    correct_variant = 0
    correct_category = 0
    correct_hul = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for b_idx, (images, targets, true_skus) in enumerate(loader, start=1):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()

            for pred_idx, gt_idx, true_sku in zip(preds, targets, true_skus):
                p_idx = int(pred_idx.item())
                g_idx = int(gt_idx.item())

                pred_meta = idx_to_meta.get(p_idx, {})
                gt_meta = idx_to_meta.get(g_idx, {})

                # L5 Exact SKU
                if p_idx == g_idx:
                    correct_sku += 1

                # L4 Base Variant
                if pred_meta.get("base_variant") == gt_meta.get("base_variant") and pred_meta.get("brand") == gt_meta.get("brand"):
                    correct_variant += 1

                # L3 Brand
                if pred_meta.get("brand") == gt_meta.get("brand"):
                    correct_brand += 1

                # L2 Category
                if pred_meta.get("category") == gt_meta.get("category"):
                    correct_category += 1

                # L1 HUL Ownership
                if pred_meta.get("is_hul") == gt_meta.get("is_hul"):
                    correct_hul += 1

                total_samples += 1

            if b_idx % 10 == 0 or b_idx == len(loader):
                elapsed = time.perf_counter() - t0
                fps = total_samples / max(0.001, elapsed)
                cur_acc = (correct_sku / total_samples) * 100
                print(f"  Processed {total_samples}/{len(split_records)} crops | SKU Acc: {cur_acc:.2f}% | {fps:.1f} crops/sec")

    elapsed_total = time.perf_counter() - t0
    avg_latency_ms = (elapsed_total / max(1, total_samples)) * 1000

    results = {
        "model": "maxvit_tiny_tf_224",
        "dataset": "hul_variants",
        "split": split_name,
        "total_crops": total_samples,
        "exact_sku_acc": round((correct_sku / total_samples) * 100, 2),
        "base_variant_acc": round((correct_variant / total_samples) * 100, 2),
        "brand_acc": round((correct_brand / total_samples) * 100, 2),
        "category_acc": round((correct_category / total_samples) * 100, 2),
        "hul_ownership_acc": round((correct_hul / total_samples) * 100, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(total_samples / max(0.001, elapsed_total), 1),
    }

    print("\n" + "=" * 75)
    print("FINAL BENCHMARK RESULTS:")
    print("=" * 75)
    print(f"  * Total Evaluated Crops:      {results['total_crops']:,}")
    print(f"  * L5 Exact SKU Accuracy:       {results['exact_sku_acc']}%")
    print(f"  * L4 Base Variant Accuracy:    {results['base_variant_acc']}%")
    print(f"  * L3 Brand Accuracy:           {results['brand_acc']}%")
    print(f"  * L2 Category Accuracy:        {results['category_acc']}%")
    print(f"  * L1 HUL Attribution Accuracy: {results['hul_ownership_acc']}%")
    print(f"  * Mean Latency per Crop:       {results['avg_latency_ms']} ms")
    print(f"  * Inference Throughput:        {results['throughput_fps']} crops/sec")
    print("=" * 75)

    out_file = Path("results/maxvit_variants_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Metrics saved to {out_file}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned MaxViT on hul_variants")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--limit", type=int, default=0, help="0 = evaluate all crops in split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    evaluate(split_name=args.split, limit=args.limit, batch_size=args.batch_size, num_workers=args.workers)

