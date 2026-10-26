"""CLEF Contrastive Contact-Sheet Cascade Online Inference Engine.

Provides low-latency, thread-safe online serving for:
1. Individual crop identification (Stage 2A CLM Fast Gate + Stage 2B CLEF Zoom-Strip Contact Sheet)
2. Full-shelf product detection and identification (Stage 1 CPU ONNX YOLO + Stage 2 Contrastive Metadata Cascade)
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from approaches.base import Context, Trace
from approaches.market_share.detection.onnx_yolo_detector import DistilledYoloOnnx
from approaches.market_share.retrieval.clef_standalone import ClefContrastiveMetadata
from utils import dataset
from utils.llm import Gemini, load_config

logger = logging.getLogger("clef_contrastive_inference")

DEFAULT_PIPELINE = "clef_contrastive_contact_sheet"
DEFAULT_APPROACH = "clef_contrastive_metadata"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIER = "priority"
DEFAULT_CROP_DATASET = "hul_golden"
DEFAULT_SHELF_DETECTOR = "distilled_yolo_onnx"


class ClefContrastiveEngine:
    """Thread-safe lazy-initializing inference engine for the CLEF Contrastive Contact-Sheet Cascade."""

    def __init__(
        self,
        crop_appr: Any | None = None,
        shelf_det: Any | None = None,
        llm: Any | None = None,
        catalog: dict[int, dict] | None = None,
    ):
        self._lock = threading.Lock()
        self.crop_appr = crop_appr
        self.shelf_det = shelf_det
        self.llm = llm
        self.catalog = catalog or (getattr(crop_appr, "catalog", {}) if crop_appr else {})
        self.is_ready = bool(crop_appr and shelf_det)
        self._current_model: str = DEFAULT_MODEL
        self._current_tier: str = DEFAULT_TIER

    def ensure_ready(self, model: str = DEFAULT_MODEL, tier: str = DEFAULT_TIER) -> None:
        """Lazily initialize 2,659-SKU gallery vector index and ONNX detector once per process."""
        if self.is_ready and self.crop_appr and self.shelf_det:
            if not self.llm or self._current_model != model or self._current_tier != tier:
                cfg = load_config()
                self.llm = Gemini(model, cfg, tier=tier)
                self._current_model = model
                self._current_tier = tier
            return

        with self._lock:
            if self.is_ready and self.crop_appr and self.shelf_det:
                if not self.llm or self._current_model != model or self._current_tier != tier:
                    cfg = load_config()
                    self.llm = Gemini(model, cfg, tier=tier)
                    self._current_model = model
                    self._current_tier = tier
                return

            cfg = load_config()
            self.llm = Gemini(model, cfg, tier=tier)
            self._current_model = model
            self._current_tier = tier

            setup_ctx = Context(
                model=model,
                llm=self.llm,
                trace=Trace(),
                price=lambda u: {"net_usd": 0.0, "list_usd": 0.0, "credit_usd": 0.0},
                sheet={},
            )

            # Stage 2: CLEF Contrastive Metadata retrieval approach with 2,659-class gallery
            crop_appr = ClefContrastiveMetadata()
            crop_appr.dataset = DEFAULT_CROP_DATASET
            crop_appr.setup(cfg, setup_ctx)
            self.crop_appr = crop_appr
            self.catalog = getattr(crop_appr, "catalog", {}) or dataset.catalog(name=DEFAULT_CROP_DATASET)

            # Stage 1: CPU ONNX shelf detector
            shelf_det = DistilledYoloOnnx()
            shelf_det.dataset = "gtmerch_golden"
            shelf_det.setup(cfg, setup_ctx)
            self.shelf_det = shelf_det

            self.is_ready = True
            logger.info("[ClefContrastiveEngine] Initialized successfully (catalog size=%d)", len(self.catalog))

    def _load_image(self, item: dict | str | bytes | Path) -> tuple[Image.Image, str]:
        """Decode image from URI, file path, raw bytes, or base64."""
        source_label = "unknown"
        raw_bytes: bytes

        if isinstance(item, (bytes, bytearray)):
            raw_bytes = bytes(item)
            source_label = "raw_bytes"
        elif isinstance(item, (str, Path)):
            item_str = str(item)
            source_label = item_str
            if item_str.startswith("data:image/") or ";base64," in item_str:
                b64_part = item_str.split(";base64,", 1)[1] if ";base64," in item_str else item_str
                raw_bytes = base64.b64decode(b64_part)
            else:
                raw_bytes = dataset.read_bytes(item_str)
        elif isinstance(item, dict):
            b64_str = item.get("image_b64") or item.get("b64")
            uri_str = item.get("image_uri") or item.get("uri") or item.get("gcs_uri") or item.get("path")
            if b64_str:
                source_label = str(item.get("id") or item.get("name") or uri_str or "base64")
                if isinstance(b64_str, str) and ";base64," in b64_str:
                    b64_str = b64_str.split(";base64,", 1)[1]
                raw_bytes = base64.b64decode(b64_str)
            elif uri_str:
                source_label = str(uri_str)
                raw_bytes = dataset.read_bytes(source_label)
            else:
                raise ValueError("Item dict must contain 'image_uri' (or 'uri'/'gcs_uri') or 'image_b64' (or 'b64')")
        else:
            raise TypeError(f"Unsupported image input type: {type(item)}")

        im = dataset.open_rgb_normalized(raw_bytes)
        return im, source_label

    def _format_catalog_prediction(
        self, sku_id: int | None, catalog: dict[int, dict], step_info: dict | None = None
    ) -> dict | None:
        """Format resolved SKU ID into clean 5-attribute dictionary."""
        if sku_id is None:
            return None
        cat = catalog.get(int(sku_id)) or {}
        return {
            "sku_id": int(sku_id),
            "is_hul": bool(cat.get("is_hul", False)),
            "category": str(cat.get("category") or ""),
            "packaging_type": str(cat.get("packaging_type") or ""),
            "brand": str(cat.get("brand") or ""),
            "base_variant": str(cat.get("base_variant") or cat.get("variant") or cat.get("product") or ""),
            "product": str(cat.get("product") or cat.get("name") or ""),
            "pack_size": str(cat.get("pack_size") or cat.get("size") or (step_info or {}).get("pack_size") or ""),
        }

    def predict_crop_one(
        self,
        item: dict | str | bytes | Path,
        model: str = DEFAULT_MODEL,
        tier: str = DEFAULT_TIER,
    ) -> dict:
        """Run CLEF Contrastive inference on a single pre-cropped retail product photo."""
        self.ensure_ready(model=model, tier=tier)
        image, source_label = self._load_image(item)
        t0 = time.perf_counter()
        trace = Trace()
        ctx = Context(
            model=model,
            llm=self.llm,
            trace=trace,
            price=lambda u: {"net_usd": 0.0, "list_usd": 0.0, "credit_usd": 0.0},
            sheet={},
        )
        sku_id = self.crop_appr.identify(image, ctx)

        step_info: dict[str, Any] = {}
        for s in reversed(trace.steps):
            if s.get("info"):
                step_info = s["info"]
                break

        stage2a_bypassed = bool(step_info.get("stage2a_bypassed", False))
        stage_name = "stage2a_fast_gate" if stage2a_bypassed else "stage2b_clef_contact_sheet"
        clm_prob = step_info.get("clm_top1_prob")
        clef_prob = step_info.get("clef_prob")
        conf = clm_prob if stage2a_bypassed else clef_prob
        shortlist = step_info.get("shortlist") or []
        top1_cos = shortlist[0].get("cosine") if shortlist else None
        pred_dict = self._format_catalog_prediction(sku_id, self.catalog, step_info=step_info)
        latency_s = time.perf_counter() - t0

        return {
            "image_uri": source_label,
            "width": image.width,
            "height": image.height,
            "sku_id": sku_id,
            "prediction": pred_dict,
            "stage": stage_name,
            "confidence": round(float(conf), 4) if conf is not None else None,
            "top1_cosine": round(float(top1_cos), 4) if top1_cos is not None else None,
            "shortlist_top5": shortlist[:5],
            "latency_s": round(latency_s, 3),
        }

    def predict_crop_payload(self, payload: dict) -> dict:
        """Dispatch single or batched crop prediction requests."""
        model = str(payload.get("model") or DEFAULT_MODEL)
        tier = str(payload.get("tier") or DEFAULT_TIER)
        raw_items = payload.get("images") or payload.get("instances")
        if isinstance(raw_items, list):
            results = [self.predict_crop_one(it, model=model, tier=tier) for it in raw_items]
            return {
                "pipeline": DEFAULT_PIPELINE,
                "approach": DEFAULT_APPROACH,
                "model": model,
                "tier": tier,
                "mode": "crop",
                "count": len(results),
                "predictions": results,
            }

        res = self.predict_crop_one(payload, model=model, tier=tier)
        return {
            "pipeline": DEFAULT_PIPELINE,
            "approach": DEFAULT_APPROACH,
            "model": model,
            "tier": tier,
            "mode": "crop",
            **res,
            "predictions": [res],
        }

    def predict_shelf_one(
        self,
        item: dict | str | bytes | Path,
        model: str = DEFAULT_MODEL,
        tier: str = DEFAULT_TIER,
    ) -> dict:
        """Run Stage-1 CPU ONNX YOLO detector + Stage-2 CLEF Contrastive Cascade on a full shelf photo."""
        self.ensure_ready(model=model, tier=tier)
        image, source_label = self._load_image(item)
        t0 = time.perf_counter()
        trace = Trace()
        ctx = Context(
            model=model,
            llm=self.llm,
            trace=trace,
            price=lambda u: {"net_usd": 0.0, "list_usd": 0.0, "credit_usd": 0.0},
            sheet={},
        )

        boxes: list[tuple[float, float, float, float]] = []
        if isinstance(item, dict) and item.get("boxes"):
            boxes = [tuple(float(v) for v in b) for b in item["boxes"]]
        if not boxes:
            boxes = self.shelf_det.detect(image, ctx)

        sku_ids = self.crop_appr.identify_boxes(image, boxes, ctx) if boxes else []

        box_step_info: dict[int, dict] = {}
        for s in trace.steps:
            info = s.get("info") or {}
            b_idx = info.get("box_index")
            if b_idx is not None:
                box_step_info[int(b_idx)] = info

        detections = []
        stage2a_fast_gate_count = 0
        stage2b_vlm_count = 0
        for idx, (b, s_id) in enumerate(zip(boxes, sku_ids, strict=False)):
            b_info = box_step_info.get(idx, {})
            stage2a_bypassed = bool(b_info.get("stage2a_bypassed", False))
            if stage2a_bypassed:
                stage2a_fast_gate_count += 1
                stage_name = "stage2a_fast_gate"
                conf = b_info.get("clm_top1_prob")
            else:
                stage2b_vlm_count += 1
                stage_name = "stage2b_clef_contact_sheet"
                conf = b_info.get("clef_prob")
            shortlist = b_info.get("shortlist") or []
            top1_cos = shortlist[0].get("cosine") if shortlist else None
            pred_dict = self._format_catalog_prediction(s_id, self.catalog, step_info=b_info)
            detections.append({
                "box_index": idx,
                "box": [round(float(v), 1) for v in b],
                "sku_id": s_id,
                "prediction": pred_dict,
                "stage": stage_name,
                "confidence": round(float(conf), 4) if conf is not None else None,
                "top1_cosine": round(float(top1_cos), 4) if top1_cos is not None else None,
            })

        total_facings = len(detections)
        hul_facings = sum(1 for d in detections if (d.get("prediction") or {}).get("is_hul") is True)
        competitor_facings = sum(
            1 for d in detections if d.get("prediction") is not None and not d["prediction"].get("is_hul")
        )
        hul_facing_sos = round(hul_facings / max(1, total_facings), 4) if total_facings else 0.0

        total_area = sum(
            max(0.0, (d["box"][2] - d["box"][0])) * max(0.0, (d["box"][3] - d["box"][1])) for d in detections
        )
        hul_area = sum(
            max(0.0, (d["box"][2] - d["box"][0])) * max(0.0, (d["box"][3] - d["box"][1]))
            for d in detections
            if (d.get("prediction") or {}).get("is_hul") is True
        )
        hul_area_sos = round(hul_area / max(1e-6, total_area), 4) if total_area > 0 else 0.0

        brand_counter = Counter(
            d["prediction"]["brand"] for d in detections if d.get("prediction") and d["prediction"].get("brand")
        )
        cat_counter = Counter(
            d["prediction"]["category"] for d in detections if d.get("prediction") and d["prediction"].get("category")
        )
        brand_counts = dict(sorted(brand_counter.items(), key=lambda x: -x[1]))
        category_counts = dict(sorted(cat_counter.items(), key=lambda x: -x[1]))
        latency_s = time.perf_counter() - t0

        return {
            "image_uri": source_label,
            "width": image.width,
            "height": image.height,
            "total_facings": total_facings,
            "hul_facings": hul_facings,
            "competitor_facings": competitor_facings,
            "hul_facing_sos": hul_facing_sos,
            "hul_area_sos": hul_area_sos,
            "stage2a_fast_gate_count": stage2a_fast_gate_count,
            "stage2b_vlm_count": stage2b_vlm_count,
            "brand_counts": brand_counts,
            "category_counts": category_counts,
            "detections": detections,
            "latency_s": round(latency_s, 3),
        }

    def predict_shelf_payload(self, payload: dict) -> dict:
        """Dispatch single or batched shelf prediction requests."""
        model = str(payload.get("model") or DEFAULT_MODEL)
        tier = str(payload.get("tier") or DEFAULT_TIER)
        raw_items = payload.get("images") or payload.get("instances")
        if isinstance(raw_items, list):
            results = [self.predict_shelf_one(it, model=model, tier=tier) for it in raw_items]
            return {
                "pipeline": DEFAULT_PIPELINE,
                "approach": DEFAULT_APPROACH,
                "model": model,
                "tier": tier,
                "mode": "shelf",
                "count": len(results),
                "predictions": results,
            }

        res = self.predict_shelf_one(payload, model=model, tier=tier)
        return {
            "pipeline": DEFAULT_PIPELINE,
            "approach": DEFAULT_APPROACH,
            "model": model,
            "tier": tier,
            "mode": "shelf",
            **res,
            "predictions": [res],
        }


# Global singleton instance
_ENGINE_INSTANCE: ClefContrastiveEngine | None = None
_GLOBAL_ENGINE_LOCK = threading.Lock()


def get_engine() -> ClefContrastiveEngine:
    """Return the global singleton ClefContrastiveEngine instance."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        with _GLOBAL_ENGINE_LOCK:
            if _ENGINE_INSTANCE is None:
                _ENGINE_INSTANCE = ClefContrastiveEngine()
    return _ENGINE_INSTANCE


def set_engine(engine: ClefContrastiveEngine | None) -> None:
    """Override the global singleton ClefContrastiveEngine instance (useful for testing/mocking)."""
    global _ENGINE_INSTANCE
    with _GLOBAL_ENGINE_LOCK:
        _ENGINE_INSTANCE = engine
