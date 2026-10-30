"""Production Online Inference Engine for the CLEF Contrastive Contact-Sheet Cascade (SPEC v0.2).

Combines:
- Stage 1: ``distilled_yolo_onnx`` CPU ONNX detector (for full-shelf photos)
- Stage 2A: ``gemini-embedding-2`` CLM Fast-Gate with 0.035 sibling-variant margin guard
- Stage 2B: ``clef_contrastive_metadata`` Contact-Sheet Verifier + Layer-5 ``pack_size`` Sister-SKU Verifier
- Stage 2C: Dustbin/Lidstone confidence calibration in ``[0.02, 0.985]`` and unified ``0.78`` abstention threshold
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
DEFAULT_CONF_THRESHOLD = 0.78


def _clamp_conf(val: float | None) -> float | None:
    if val is None:
        return None
    return round(max(0.02, min(0.985, float(val))), 4)


def _resolve_confidence_and_abstention(
    sku_id: int | None,
    step_info: dict[str, Any],
    *,
    stage2a_bypassed: bool,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    allow_abstain: bool = False,
) -> tuple[int | None, float | None, float | None, float | None, bool, bool, str | None, str]:
    """Compute calibrated v0.2 confidences and evaluate the unified 0.78 abstention contract."""
    clm_prob = step_info.get("clm_top1_prob")
    clef_prob = step_info.get("clef_prob")
    raw_sku_conf = step_info.get("sku_confidence")
    raw_brand_conf = step_info.get("brand_confidence")

    base_conf = clm_prob if stage2a_bypassed else clef_prob
    if base_conf is None:
        base_conf = raw_sku_conf

    # Use calibrated v0.2 sku_confidence whenever available; fall back to clm_top1_prob/clef_prob in mocks
    if raw_sku_conf is not None:
        conf = _clamp_conf(float(raw_sku_conf))
    else:
        conf = _clamp_conf(float(base_conf)) if base_conf is not None else None

    sku_conf = _clamp_conf(float(raw_sku_conf)) if raw_sku_conf is not None else conf
    brand_conf = _clamp_conf(float(raw_brand_conf)) if raw_brand_conf is not None else sku_conf

    guard_mode = str(
        step_info.get("guard_mode") or ("stage2a_fast_gate" if stage2a_bypassed else "panel")
    )
    choice = step_info.get("choice")
    shortlist = step_info.get("shortlist") or []
    top1_cos = float(shortlist[0].get("cosine")) if shortlist and shortlist[0].get("cosine") is not None else None

    reason: str | None = None
    if sku_id is None:
        reason = "open_set_or_unreadable_abstain"
    elif not stage2a_bypassed and choice == 0:
        reason = "stage2b_choice_0_unreadable_or_no_match"
    elif top1_cos is not None and top1_cos < 0.52:
        reason = "low_cosine_open_set"
    elif conf is not None and conf < conf_threshold:
        reason = f"confidence_below_threshold_{conf_threshold:.2f}"
    elif bool(step_info.get("abstain_recommended")):
        reason = "low_calibrated_confidence"

    abstain_recommended = bool(reason is not None or step_info.get("abstain_recommended", False))
    effective_sku_id = sku_id
    if allow_abstain and abstain_recommended:
        effective_sku_id = None
        if "abstain" not in guard_mode:
            guard_mode = f"{guard_mode}+abstain_threshold"
    abstained = effective_sku_id is None
    return (
        effective_sku_id,
        conf,
        sku_conf,
        brand_conf,
        abstain_recommended,
        abstained,
        reason,
        guard_mode,
    )


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

    def _build_segregated_stage_outputs(
        self,
        *,
        step_info: dict[str, Any],
        stage2a_bypassed: bool,
        raw_sku_id: int | None,
        final_sku_id: int | None,
        guard_mode: str,
        model: str,
        sku_conf: float | None,
        brand_conf: float | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build segregated ``embedding_output`` (Stage 2A) and ``gemini_output`` (Stage 2B) JSON objects."""
        from approaches.market_share.retrieval.embedding_retrieval import MODEL as EMB_MODEL

        shortlist_raw = step_info.get("shortlist") or []
        enriched_shortlist: list[dict[str, Any]] = []
        for rank_idx, item in enumerate(shortlist_raw[:5], start=1):
            c_sku = item.get("sku_id")
            c_cos = item.get("cosine")
            c_pred = self._format_catalog_prediction(c_sku, self.catalog) if c_sku is not None else None
            enriched_shortlist.append({
                "rank": rank_idx,
                "sku_id": int(c_sku) if c_sku is not None else None,
                "cosine": round(float(c_cos), 4) if c_cos is not None else None,
                "brand": (c_pred or {}).get("brand", ""),
                "base_variant": (c_pred or {}).get("base_variant", ""),
                "product": (c_pred or {}).get("product", ""),
                "pack_size": (c_pred or {}).get("pack_size", ""),
                "category": (c_pred or {}).get("category", ""),
                "packaging_type": (c_pred or {}).get("packaging_type", ""),
            })

        emb_top1_sku = step_info.get("top1_sku_id")
        if emb_top1_sku is None and shortlist_raw:
            emb_top1_sku = shortlist_raw[0].get("sku_id")
        emb_top1_sku = int(emb_top1_sku) if emb_top1_sku is not None else None
        emb_top1_cos = step_info.get("top1_cosine")
        if emb_top1_cos is None and shortlist_raw:
            emb_top1_cos = shortlist_raw[0].get("cosine")
        emb_top2_cos = step_info.get("top2_cosine")
        if emb_top2_cos is None and len(shortlist_raw) > 1:
            emb_top2_cos = shortlist_raw[1].get("cosine")
        cos_margin = step_info.get("cosine_margin")
        if cos_margin is None and emb_top1_cos is not None and emb_top2_cos is not None:
            cos_margin = round(float(emb_top1_cos) - float(emb_top2_cos), 4)

        emb_sku_conf = step_info.get("stage2a_sku_confidence")
        if emb_sku_conf is None and stage2a_bypassed:
            emb_sku_conf = sku_conf
        emb_brand_conf = step_info.get("stage2a_brand_confidence")
        if emb_brand_conf is None and stage2a_bypassed:
            emb_brand_conf = brand_conf

        embedding_output = {
            "model": EMB_MODEL,
            "stage": "stage2a_clm_embedding",
            "fast_gate_passed": bool(stage2a_bypassed),
            "escalation_reason": (
                None
                if stage2a_bypassed
                else (step_info.get("escalation_reason") or "escalated_to_stage2b_clef")
            ),
            "top1_sku_id": emb_top1_sku,
            "prediction": self._format_catalog_prediction(emb_top1_sku, self.catalog),
            "top1_cosine": round(float(emb_top1_cos), 4) if emb_top1_cos is not None else None,
            "top2_cosine": round(float(emb_top2_cos), 4) if emb_top2_cos is not None else None,
            "cosine_margin": round(float(cos_margin), 4) if cos_margin is not None else None,
            "required_margin": step_info.get("required_margin"),
            "same_brand_top2": bool(step_info.get("same_brand_top2", False)),
            "same_variant_sister": bool(step_info.get("same_variant_sister", False)),
            "clm_top1_prob": (
                round(float(step_info["clm_top1_prob"]), 4)
                if step_info.get("clm_top1_prob") is not None
                else None
            ),
            "coarse_entropy_conf": (
                round(float(step_info["coarse_entropy_conf"]), 4)
                if step_info.get("coarse_entropy_conf") is not None
                else None
            ),
            "z_margin": (
                round(float(step_info["z_margin"]), 4)
                if step_info.get("z_margin") is not None
                else None
            ),
            "lid": (
                round(float(step_info["lid"]), 4)
                if step_info.get("lid") is not None
                else None
            ),
            "sku_confidence": round(float(emb_sku_conf), 4) if emb_sku_conf is not None else None,
            "brand_confidence": round(float(emb_brand_conf), 4) if emb_brand_conf is not None else None,
            "shortlist_top5": enriched_shortlist,
        }

        if stage2a_bypassed:
            gemini_output = {
                "called": False,
                "model": model,
                "stage": "stage2b_clef_contact_sheet",
                "skipped_reason": "stage2a_fast_gate_accepted",
                "panel_choice": None,
                "extracted_pack_size": None,
                "token_prob": None,
                "sku_id": None,
                "prediction": None,
                "sku_confidence": None,
                "brand_confidence": None,
                "guard_mode": "stage2a_fast_gate",
                "layer5_pack_size_verified": False,
                "agreed_with_embedding_top1": None,
            }
        else:
            gem_sku = int(raw_sku_id) if raw_sku_id is not None else final_sku_id
            gem_pred = self._format_catalog_prediction(gem_sku, self.catalog, step_info=step_info)
            raw_choice = step_info.get("choice")
            raw_psize = step_info.get("pack_size")
            raw_tprob = step_info.get("clef_prob")
            gemini_output = {
                "called": True,
                "model": model,
                "stage": "stage2b_clef_contact_sheet",
                "skipped_reason": None,
                "panel_choice": int(raw_choice) if isinstance(raw_choice, int) else raw_choice,
                "extracted_pack_size": str(raw_psize) if raw_psize is not None else "",
                "token_prob": round(float(raw_tprob), 4) if raw_tprob is not None else None,
                "sku_id": gem_sku,
                "prediction": gem_pred,
                "sku_confidence": sku_conf,
                "brand_confidence": brand_conf,
                "guard_mode": guard_mode,
                "layer5_pack_size_verified": "layer5_pack_size_verified" in str(guard_mode),
                "agreed_with_embedding_top1": (
                    bool(gem_sku == emb_top1_sku)
                    if (gem_sku is not None and emb_top1_sku is not None)
                    else False
                ),
            }
        return embedding_output, gemini_output

    def predict_crop_one(
        self,
        item: dict | str | bytes | Path,
        model: str = DEFAULT_MODEL,
        tier: str = DEFAULT_TIER,
        *,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        allow_abstain: bool = False,
    ) -> dict:
        """Run CLEF Contrastive inference on a single pre-cropped retail product photo."""
        self.ensure_ready(model=model, tier=tier)
        if isinstance(item, dict):
            if "conf_threshold" in item and item["conf_threshold"] is not None:
                conf_threshold = float(item["conf_threshold"])
            if "allow_abstain" in item and item["allow_abstain"] is not None:
                allow_abstain = bool(item["allow_abstain"])

        image, source_label = self._load_image(item)
        t0 = time.perf_counter()
        trace = Trace()
        ctx = Context(
            model=model,
            llm=self.llm,
            trace=trace,
            price=lambda u: {"net_usd": 0.0, "list_usd": 0.0, "credit_usd": 0.0},
        )
        appr: Any = self.crop_appr
        prev_abstain = getattr(appr, "allow_abstain", None)
        try:
            if appr is not None:
                appr.allow_abstain = allow_abstain
            raw_sku_id = appr.identify(image, ctx) if appr is not None else None
        finally:
            if appr is not None:
                appr.allow_abstain = prev_abstain

        step_info: dict[str, Any] = {}
        for s in reversed(trace.steps):
            if s.get("info"):
                step_info = s["info"]
                break

        stage2a_bypassed = bool(step_info.get("stage2a_bypassed", False))
        stage_name = "stage2a_fast_gate" if stage2a_bypassed else "stage2b_clef_contact_sheet"
        (
            sku_id,
            conf,
            sku_conf,
            brand_conf,
            abstain_rec,
            abstained,
            abstention_reason,
            guard_mode,
        ) = _resolve_confidence_and_abstention(
            raw_sku_id,
            step_info,
            stage2a_bypassed=stage2a_bypassed,
            conf_threshold=conf_threshold,
            allow_abstain=allow_abstain,
        )
        shortlist = step_info.get("shortlist") or []
        top1_cos = shortlist[0].get("cosine") if shortlist else None
        pred_dict = self._format_catalog_prediction(sku_id, self.catalog, step_info=step_info)
        embedding_output, gemini_output = self._build_segregated_stage_outputs(
            step_info=step_info,
            stage2a_bypassed=stage2a_bypassed,
            raw_sku_id=raw_sku_id,
            final_sku_id=sku_id,
            guard_mode=guard_mode,
            model=model,
            sku_conf=sku_conf,
            brand_conf=brand_conf,
        )
        latency_s = time.perf_counter() - t0

        return {
            "image_uri": source_label,
            "width": image.width,
            "height": image.height,
            "sku_id": sku_id,
            "prediction": pred_dict,
            "stage": stage_name,
            "guard_mode": guard_mode,
            "confidence": conf,
            "sku_confidence": sku_conf,
            "brand_confidence": brand_conf,
            "conf_threshold": round(float(conf_threshold), 4),
            "abstain_recommended": abstain_rec,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "top1_cosine": round(float(top1_cos), 4) if top1_cos is not None else None,
            "shortlist_top5": shortlist[:5],
            "embedding_output": embedding_output,
            "gemini_output": gemini_output,
            "latency_s": round(latency_s, 3),
        }

    def predict_crop_payload(self, payload: dict) -> dict:
        """Dispatch single or batched crop prediction requests."""
        model = str(payload.get("model") or DEFAULT_MODEL)
        tier = str(payload.get("tier") or DEFAULT_TIER)
        raw_ct = payload.get("conf_threshold")
        conf_threshold = float(raw_ct) if raw_ct is not None else DEFAULT_CONF_THRESHOLD
        allow_abstain = bool(payload.get("allow_abstain", False))
        raw_items = payload.get("images") or payload.get("instances")
        if isinstance(raw_items, list):
            results = [
                self.predict_crop_one(
                    it,
                    model=model,
                    tier=tier,
                    conf_threshold=conf_threshold,
                    allow_abstain=allow_abstain,
                )
                for it in raw_items
            ]
            return {
                "pipeline": DEFAULT_PIPELINE,
                "approach": DEFAULT_APPROACH,
                "model": model,
                "tier": tier,
                "mode": "crop",
                "conf_threshold": round(conf_threshold, 4),
                "allow_abstain": allow_abstain,
                "count": len(results),
                "predictions": results,
            }

        res = self.predict_crop_one(
            payload,
            model=model,
            tier=tier,
            conf_threshold=conf_threshold,
            allow_abstain=allow_abstain,
        )
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
        *,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        allow_abstain: bool = False,
    ) -> dict:
        """Run Stage-1 CPU ONNX YOLO detector + Stage-2 CLEF Contrastive Cascade on a full shelf photo."""
        self.ensure_ready(model=model, tier=tier)
        if isinstance(item, dict):
            if "conf_threshold" in item and item["conf_threshold"] is not None:
                conf_threshold = float(item["conf_threshold"])
            if "allow_abstain" in item and item["allow_abstain"] is not None:
                allow_abstain = bool(item["allow_abstain"])

        image, source_label = self._load_image(item)
        t0 = time.perf_counter()
        trace = Trace()
        ctx = Context(
            model=model,
            llm=self.llm,
            trace=trace,
            price=lambda u: {"net_usd": 0.0, "list_usd": 0.0, "credit_usd": 0.0},
        )

        boxes: list[tuple[float, float, float, float]] = []
        if isinstance(item, dict) and item.get("boxes"):
            boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in item["boxes"]]
        det_appr: Any = self.shelf_det
        if not boxes and det_appr is not None:
            boxes = det_appr.detect(image, ctx)

        appr: Any = self.crop_appr
        prev_abstain = getattr(appr, "allow_abstain", None)
        try:
            if appr is not None:
                appr.allow_abstain = allow_abstain
            sku_ids = appr.identify_boxes(image, boxes, ctx) if (boxes and appr is not None) else []
        finally:
            if appr is not None:
                appr.allow_abstain = prev_abstain

        box_step_info: dict[int, dict] = {}
        for s in trace.steps:
            info = s.get("info") or {}
            b_idx = info.get("box_index")
            if b_idx is not None:
                box_step_info[int(b_idx)] = info

        detections = []
        stage2a_fast_gate_count = 0
        stage2b_vlm_count = 0
        gemini_agreed_count = 0
        gemini_overrode_count = 0
        abstained_facings = 0
        abstain_recommended_facings = 0
        for idx, (b, raw_s_id) in enumerate(zip(boxes, sku_ids, strict=False)):
            b_info = box_step_info.get(idx, {})
            stage2a_bypassed = bool(b_info.get("stage2a_bypassed", False))
            if stage2a_bypassed:
                stage2a_fast_gate_count += 1
                stage_name = "stage2a_fast_gate"
            else:
                stage2b_vlm_count += 1
                stage_name = "stage2b_clef_contact_sheet"
            (
                s_id,
                conf,
                sku_conf,
                brand_conf,
                abstain_rec,
                abstained,
                abstention_reason,
                guard_mode,
            ) = _resolve_confidence_and_abstention(
                raw_s_id,
                b_info,
                stage2a_bypassed=stage2a_bypassed,
                conf_threshold=conf_threshold,
                allow_abstain=allow_abstain,
            )
            if abstained:
                abstained_facings += 1
            if abstain_rec:
                abstain_recommended_facings += 1
            shortlist = b_info.get("shortlist") or []
            top1_cos = shortlist[0].get("cosine") if shortlist else None
            pred_dict = self._format_catalog_prediction(s_id, self.catalog, step_info=b_info)
            embedding_output, gemini_output = self._build_segregated_stage_outputs(
                step_info=b_info,
                stage2a_bypassed=stage2a_bypassed,
                raw_sku_id=raw_s_id,
                final_sku_id=s_id,
                guard_mode=guard_mode,
                model=model,
                sku_conf=sku_conf,
                brand_conf=brand_conf,
            )
            if gemini_output["called"]:
                if gemini_output["agreed_with_embedding_top1"]:
                    gemini_agreed_count += 1
                else:
                    gemini_overrode_count += 1
            detections.append({
                "box_index": idx,
                "box": [round(float(v), 1) for v in b],
                "sku_id": s_id,
                "prediction": pred_dict,
                "stage": stage_name,
                "guard_mode": guard_mode,
                "confidence": conf,
                "sku_confidence": sku_conf,
                "brand_confidence": brand_conf,
                "abstain_recommended": abstain_rec,
                "abstained": abstained,
                "abstention_reason": abstention_reason,
                "top1_cosine": round(float(top1_cos), 4) if top1_cos is not None else None,
                "embedding_output": embedding_output,
                "gemini_output": gemini_output,
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
            "abstained_facings": abstained_facings,
            "abstain_recommended_facings": abstain_recommended_facings,
            "hul_facing_sos": hul_facing_sos,
            "hul_area_sos": hul_area_sos,
            "stage2a_fast_gate_count": stage2a_fast_gate_count,
            "stage2b_vlm_count": stage2b_vlm_count,
            "stage_breakdown": {
                "embedding_fast_gate_count": stage2a_fast_gate_count,
                "gemini_vlm_count": stage2b_vlm_count,
                "gemini_agreed_with_embedding_count": gemini_agreed_count,
                "gemini_overrode_embedding_count": gemini_overrode_count,
            },
            "conf_threshold": round(float(conf_threshold), 4),
            "brand_counts": brand_counts,
            "category_counts": category_counts,
            "detections": detections,
            "latency_s": round(latency_s, 3),
        }

    def predict_shelf_payload(self, payload: dict) -> dict:
        """Dispatch single or batched shelf prediction requests."""
        model = str(payload.get("model") or DEFAULT_MODEL)
        tier = str(payload.get("tier") or DEFAULT_TIER)
        raw_ct = payload.get("conf_threshold")
        conf_threshold = float(raw_ct) if raw_ct is not None else DEFAULT_CONF_THRESHOLD
        allow_abstain = bool(payload.get("allow_abstain", False))
        raw_items = payload.get("images") or payload.get("instances")
        if isinstance(raw_items, list):
            results = [
                self.predict_shelf_one(
                    it,
                    model=model,
                    tier=tier,
                    conf_threshold=conf_threshold,
                    allow_abstain=allow_abstain,
                )
                for it in raw_items
            ]
            return {
                "pipeline": DEFAULT_PIPELINE,
                "approach": DEFAULT_APPROACH,
                "model": model,
                "tier": tier,
                "mode": "shelf",
                "conf_threshold": round(conf_threshold, 4),
                "allow_abstain": allow_abstain,
                "count": len(results),
                "predictions": results,
            }

        res = self.predict_shelf_one(
            payload,
            model=model,
            tier=tier,
            conf_threshold=conf_threshold,
            allow_abstain=allow_abstain,
        )
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
