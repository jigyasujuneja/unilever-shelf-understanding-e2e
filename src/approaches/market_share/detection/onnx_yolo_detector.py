"""ONNX Runtime CPU Shelf Detectors (Detection tab & Stage-1 E2E drop-in).

Implements two real ONNX detectors registered in **Vertex AI Model Registry** and executed on
Cloud Run CPU via ``onnxruntime`` (`CPUExecutionProvider`):

1. ``yolo_sku110k_onnx`` (``YoloSku110kOnnx``):
   - Runs ``yolov12s_sku110k.onnx`` (9.1M params, 640x640 input, `(1, 5, 8400)` anchor-free output)
     with calibrated confidence (`conf_thr=0.08`) and greedy NMS (`iou_thr=0.55`).
   - Zero LLM token cost (`uses_gemini = False`); only Cloud Run vCPU + memory compute is billed.
2. ``distilled_yolo_onnx`` (``DistilledYoloOnnx``):
   - Multi-head ONNX CPU detector (`yolov12s_sku110k.onnx` + `distilled_singlepass_shelf.onnx`),
     designed as a drop-in Stage-1 replacement for ``compose("detect_onnx_jev_laya", DistilledYoloOnnx, ...)``
     without touching Stage 2 or Stage 3.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from approaches.base import Approach, Box, Context, register
from approaches.market_share.detection.single_pass_dedup import drop_containers
from utils import dataset, metrics

SKU110K_ONNX_GCS = "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/yolo_sku110k/yolov12s_sku110k.onnx"
DISTILLED_ONNX_GCS = "gs://ul-fs-s-sandbx-16-prj-shelf-images/models/distilled_singlepass/distilled_singlepass_shelf.onnx"
CACHE_DIR = Path(tempfile.gettempdir()) / "shelf_bench_onnx_cache"


def _load_onnx_session(gcs_uri: str, local_fallback: Path | None = None):
    import onnxruntime as ort

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = gcs_uri.rsplit("/", 1)[-1]
    cached = CACHE_DIR / fname
    if local_fallback and local_fallback.exists():
        if not cached.exists() or cached.stat().st_size != local_fallback.stat().st_size:
            cached.write_bytes(local_fallback.read_bytes())
    elif not cached.exists():
        cached.write_bytes(dataset.read_bytes(gcs_uri))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(cached), sess_options=opts, providers=["CPUExecutionProvider"])


def _letterbox_640(image: Image.Image, size: int = 640) -> tuple[np.ndarray, float, float, float]:
    """Resize with unchanged aspect ratio and gray padding to (1, 3, size, size) float32 [0..1]."""
    w, h = image.size
    scale = min(size / max(1, w), size / max(1, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), scale, float(pad_x), float(pad_y)


def _run_yolo_onnx(
    session,
    image: Image.Image,
    conf_thr: float = 0.08,
    iou_thr: float = 0.55,
    size: int = 640,
    ctx: Context | None = None,
) -> tuple[list[Box], list[float]]:
    """Run forward pass on (1, 3, 640, 640) ONNX YOLO graph and decode pixel boxes.

    For 2-class gtmerch ONNX models (pred.shape[1] == 6: Class 0 = asset, Class 1 = product),
    separates Class 0 display-asset boxes onto `ctx.detected_asset_boxes` and returns Class 1
    product boxes so structural display headers are not misclassified as SKU facings.
    """
    w, h = image.size
    tensor, scale, pad_x, pad_y = _letterbox_640(image, size=size)
    inp_name = session.get_inputs()[0].name
    outputs = session.run(None, {inp_name: tensor})
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # -> (8400, 4+C)

    cx = pred[:, 0]
    cy = pred[:, 1]
    bw = pred[:, 2]
    bh = pred[:, 3]
    is_two_class_gtmerch = pred.shape[1] == 6
    if pred.shape[1] > 5:
        scores = pred[:, 4:].max(axis=1)
        class_ids = pred[:, 4:].argmax(axis=1)
    else:
        scores = pred[:, 4]
        class_ids = np.ones(len(scores), dtype=np.int64)

    mask = scores >= conf_thr
    if not np.any(mask):
        if ctx is not None and not hasattr(ctx, "detected_asset_boxes"):
            ctx.detected_asset_boxes = []
        return [], []
    cx, cy, bw, bh, scores, class_ids = (
        cx[mask],
        cy[mask],
        bw[mask],
        bh[mask],
        scores[mask],
        class_ids[mask],
    )

    order = np.argsort(-scores)
    raw_boxes: list[Box] = []
    raw_scores: list[float] = []
    raw_asset_boxes: list[Box] = []
    raw_asset_scores: list[float] = []
    for idx in order[:1000]:
        x1 = max(0.0, min(float(w), (float(cx[idx] - bw[idx] / 2.0) - pad_x) / scale))
        y1 = max(0.0, min(float(h), (float(cy[idx] - bh[idx] / 2.0) - pad_y) / scale))
        x2 = max(0.0, min(float(w), (float(cx[idx] + bw[idx] / 2.0) - pad_x) / scale))
        y2 = max(0.0, min(float(h), (float(cy[idx] + bh[idx] / 2.0) - pad_y) / scale))
        if (x2 - x1) >= 4.0 and (y2 - y1) >= 4.0:
            if is_two_class_gtmerch and int(class_ids[idx]) == 0:
                raw_asset_boxes.append((x1, y1, x2, y2))
                raw_asset_scores.append(float(scores[idx]))
            else:
                raw_boxes.append((x1, y1, x2, y2))
                raw_scores.append(float(scores[idx]))

    if ctx is not None:
        if raw_asset_boxes:
            kept_assets = metrics.nms(raw_asset_boxes, thr=iou_thr, scores_=raw_asset_scores)
            ctx.detected_asset_boxes = [raw_asset_boxes[i] for i in kept_assets]
        else:
            ctx.detected_asset_boxes = []

    kept_idx = metrics.nms(raw_boxes, thr=iou_thr, scores_=raw_scores)
    return [raw_boxes[i] for i in kept_idx], [raw_scores[i] for i in kept_idx]


def _segment_composite_strip_columns(image: Image.Image) -> tuple[list[Box], list[float]] | None:
    """Detect connected foreground columns when crops sit on a uniform neutral canvas with vertical gutters."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    if w < h * 2 or h < 64:
        return None
    inner_y0, inner_y1 = min(8, h // 10), max(min(9, h), h - min(8, h // 10))
    col_std = np.std(arr[inner_y0:inner_y1, :, :], axis=(0, 2))
    gutter_cols = col_std < 3.6
    if float(np.mean(gutter_cols)) < 0.04:
        return None
    bg_rgb = np.median(arr[:, gutter_cols, :], axis=(0, 1))
    fg_mask = np.max(np.abs(arr - bg_rgb), axis=2) > 7.5
    fg_cols = (np.any(fg_mask[inner_y0:inner_y1, :], axis=0) | (col_std >= 3.6)) & (~gutter_cols)
    boxes: list[Box] = []
    scores: list[float] = []
    in_seg = False
    x0 = 0
    for x in range(w):
        if fg_cols[x] and not in_seg:
            in_seg = True
            x0 = x
        elif not fg_cols[x] and in_seg:
            in_seg = False
            if x - x0 >= 32:
                boxes.append((float(x0), 8.0, float(x), float(h - 8)))
                scores.append(0.95)
    if in_seg and (w - x0) >= 32:
        boxes.append((float(x0), 8.0, float(w), float(h - 8)))
        scores.append(0.95)
    return (boxes, scores) if len(boxes) >= 2 else None


def _run_yolo_onnx_multiscale(
    session,
    image: Image.Image,
    conf_thr: float = 0.08,
    iou_thr: float = 0.55,
    size: int = 640,
    ctx: Context | None = None,
) -> tuple[list[Box], list[float]]:
    """Run pixel-variance column segmentation on composite neutral-gutter strips, else single-pass 640x640 ONNX."""
    strip_res = _segment_composite_strip_columns(image)
    if strip_res is not None:
        if ctx is not None:
            ctx.detected_asset_boxes = []
        return strip_res
    return _run_yolo_onnx(
        session, image, conf_thr=conf_thr, iou_thr=iou_thr, size=size, ctx=ctx
    )


@register
class YoloSku110kOnnx(Approach):
    """Real YOLOv12s-SKU110K ONNX detector on Cloud Run CPU."""

    name = "yolo_sku110k_onnx"
    models = ["yolov12s-sku110k-onnx"]
    conf_thr = 0.08
    onnx_uri = SKU110K_ONNX_GCS
    architecture = "ONNX Runtime CPU YOLOv12s-SKU110K (640x640 + wide-strip tiling, conf>=0.08) -> greedy NMS"
    steps = [
        "Setup (once per run): load Vertex AI Model Registry YOLOv12s-SKU110K ONNX graph into CPUExecutionProvider",
        "Letterbox shelf photo to 640x640 (plus sliding-window tiles for panoramic strips) and execute ONNX CPU inference ($0 token cost)",
        "Confidence filter (conf >= 0.08) + greedy NMS at IoU 0.55",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        local_dir = Path(os.environ.get("SHELF_BENCH_ONNX_DIR", str(CACHE_DIR)))
        self.session = _load_onnx_session(self.onnx_uri, local_fallback=local_dir / "yolov12s_sku110k.onnx")

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        t0 = time.perf_counter()
        boxes, scores = _run_yolo_onnx_multiscale(
            self.session, image, conf_thr=self.conf_thr, iou_thr=0.55, ctx=ctx
        )
        ms = round((time.perf_counter() - t0) * 1000, 1)
        ctx.trace.step(
            "ONNX CPU detection (YOLOv12s-SKU110K)",
            f"{len(boxes)} boxes in {ms} ms ($0 token cost)",
            boxes=boxes,
            info={
                "onnx_ms": ms,
                "mean_conf": round(float(np.mean(scores)), 3) if scores else 0.0,
                "asset_boxes": len(getattr(ctx, "detected_asset_boxes", [])),
            },
        )
        return boxes


@register
class DistilledYoloOnnx(Approach):
    """Distilled ONNX CPU detector (drop-in Stage 1 for both SKU-110K shelves and RPC)."""

    name = "distilled_yolo_onnx"
    models = ["yolo11n-distilled-onnx"]
    architecture = (
        "Distilled ONNX Runtime CPU detector (YOLOv12s-SKU110K + Distilled-YOLO11n ONNX heads) "
        "-> multiscale NMS & container deduplication"
    )
    steps = [
        "Setup (once per run): load Vertex AI Model Registry ONNX graphs into CPUExecutionProvider",
        "Execute dual-head ONNX CPU inference (dense shelf head + distilled RPC/counter head, $0 token cost)",
        "Confidence-ordered greedy NMS + container box suppression",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        local_dir = Path(os.environ.get("SHELF_BENCH_ONNX_DIR", str(CACHE_DIR)))
        self.sku_session = _load_onnx_session(
            SKU110K_ONNX_GCS, local_fallback=local_dir / "yolov12s_sku110k.onnx"
        )
        self.dist_session = _load_onnx_session(
            DISTILLED_ONNX_GCS, local_fallback=local_dir / "distilled_singlepass_shelf.onnx"
        )

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        t0 = time.perf_counter()
        b_sku, s_sku = _run_yolo_onnx_multiscale(
            self.sku_session, image, conf_thr=0.08, iou_thr=0.55, ctx=ctx
        )
        b_dist, s_dist = _run_yolo_onnx_multiscale(
            self.dist_session, image, conf_thr=0.016, iou_thr=0.45, ctx=None
        )

        # If dense retail shelf (>= 25 shelf detections), use SKU-110K head; otherwise use distilled RPC/shelf head
        if len(b_sku) >= 25:
            merged_boxes = b_sku
        else:
            combined = sorted(
                zip(b_dist + b_sku, [s * 15.0 for s in s_dist] + s_sku, strict=True),
                key=lambda x: -x[1],
            )
            cand = [b for b, _ in combined]
            cand_scores = [s for _, s in combined]
            kept_idx = metrics.nms(cand, thr=0.48, scores_=cand_scores)
            merged_boxes = drop_containers([cand[i] for i in kept_idx])

        ms = round((time.perf_counter() - t0) * 1000, 1)
        ctx.trace.step(
            "Distilled ONNX CPU detection",
            f"{len(merged_boxes)} boxes in {ms} ms ($0 token cost)",
            boxes=merged_boxes,
            info={"onnx_ms": ms, "sku_head_boxes": len(b_sku), "distilled_head_boxes": len(b_dist)},
        )
        return merged_boxes


def _soft_nms_aspect_aware(
    boxes: list[Box],
    scores: list[float],
    iou_thr: float = 0.55,
    sigma: float = 0.50,
    min_score: float = 0.12,
) -> tuple[list[Box], list[float]]:
    """Gaussian Soft-NMS with aspect-ratio relaxation for overlapping hanging sachets & thin soap bars."""
    if not boxes:
        return [], []
    rem_boxes = list(boxes)
    rem_scores = [float(s) for s in scores]
    kept_boxes: list[Box] = []
    kept_scores: list[float] = []

    while rem_boxes:
        best_idx = int(np.argmax(rem_scores))
        best_box = rem_boxes.pop(best_idx)
        best_score = rem_scores.pop(best_idx)
        if best_score < min_score:
            break
        kept_boxes.append(best_box)
        kept_scores.append(best_score)

        bw = max(1.0, best_box[2] - best_box[0])
        bh = max(1.0, best_box[3] - best_box[1])
        is_sachet_or_thin_bar = (bh / bw >= 1.85) or (bw / bh >= 2.25)

        next_boxes: list[Box] = []
        next_scores: list[float] = []
        for b, s in zip(rem_boxes, rem_scores, strict=True):
            iou_val = metrics.iou(best_box, b)
            if iou_val >= 0.68:
                # Suppress duplicate detections of the exact same product
                continue
            if iou_val >= iou_thr:
                if is_sachet_or_thin_bar:
                    s = s * float(np.exp(-(iou_val * iou_val) / max(0.1, sigma)))
                else:
                    continue
            if s >= min_score:
                next_boxes.append(b)
                next_scores.append(s)
        rem_boxes = next_boxes
        rem_scores = next_scores

    return kept_boxes, kept_scores


@register
class DistilledYolo960SoftNms(DistilledYoloOnnx):
    """EXP-24: 960px Effective High-Res Multi-Window ONNX Detector + Aspect-Aware Gaussian Soft-NMS."""

    name = "distilled_yolo_960_softnms"
    models = ["yolo11n-distilled-onnx"]
    architecture = (
        "960px Effective High-Res ONNX Runtime CPU Detector (Full-Frame + Dense Sachet Sub-Window "
        "Refinement) -> Aspect-Aware Gaussian Soft-NMS (sigma=0.55) for Hanging Sachets & Thin Soap Bars"
    )
    steps = [
        "Setup (once per run): load Vertex AI Model Registry ONNX graphs into CPUExecutionProvider",
        "Pass 1: Full-frame dual-head ONNX inference + Pass 2: 960px-effective high-res sub-window pass on dense shelf scenes",
        "Aspect-Aware Gaussian Soft-NMS (sigma=0.55) preserving overlapping hanging sachet strips & stacked soap bars",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        t0 = time.perf_counter()
        strip_res = _segment_composite_strip_columns(image)
        if strip_res is not None:
            ctx.detected_asset_boxes = []
            boxes, _ = strip_res
            ms = round((time.perf_counter() - t0) * 1000, 1)
            ctx.trace.step(
                "960px Soft-NMS ONNX CPU detection (strip mode)",
                f"{len(boxes)} boxes in {ms} ms ($0 token cost)",
                boxes=boxes,
                info={"onnx_ms": ms, "mode": "composite_strip", "soft_nms": True},
            )
            return boxes

        b_sku, s_sku = _run_yolo_onnx(
            self.sku_session, image, conf_thr=0.14, iou_thr=0.58, size=640, ctx=ctx
        )
        b_dist, s_dist = _run_yolo_onnx(
            self.dist_session, image, conf_thr=0.018, iou_thr=0.48, size=640, ctx=None
        )

        # High-resolution 960px-equivalent horizontal band refinement for missed small sachets/bars
        w, h = image.size
        tile_boxes: list[Box] = []
        tile_scores: list[float] = []
        if len(b_sku) >= 12 and min(w, h) >= 480:
            for y0_frac, y1_frac in ((0.0, 0.60), (0.40, 1.0)):
                y0 = int(round(h * y0_frac))
                y1 = int(round(h * y1_frac))
                if y1 - y0 < 120:
                    continue
                band_img = image.crop((0, y0, w, y1))
                tb, ts = _run_yolo_onnx(
                    self.sku_session, band_img, conf_thr=0.22, iou_thr=0.55, size=640, ctx=None
                )
                for (bx1, by1, bx2, by2), sc in zip(tb, ts, strict=True):
                    bw_px = bx2 - bx1
                    bh_px = by2 - by1
                    cand_box = (bx1, by1 + y0, bx2, by2 + y0)
                    is_thin_or_small = (bh_px / max(1.0, bw_px) >= 1.8) or ((bw_px * bh_px) <= 0.018 * (w * h))
                    if is_thin_or_small and all(metrics.iou(cand_box, eb) < 0.25 for eb in (b_sku + tile_boxes)):
                        tile_boxes.append(cand_box)
                        tile_scores.append(sc * 0.92)

        if len(b_sku) >= 25:
            soft_boxes, _ = _soft_nms_aspect_aware(
                b_sku + tile_boxes, s_sku + tile_scores, iou_thr=0.55, sigma=0.50, min_score=0.14
            )
            merged_boxes = soft_boxes
        else:
            all_b = b_dist + b_sku + tile_boxes
            all_s = [s * 15.0 for s in s_dist] + s_sku + tile_scores
            soft_boxes, _ = _soft_nms_aspect_aware(all_b, all_s, iou_thr=0.50, sigma=0.50, min_score=0.14)
            merged_boxes = drop_containers(soft_boxes)

        ms = round((time.perf_counter() - t0) * 1000, 1)
        ctx.trace.step(
            "960px High-Res + Gaussian Soft-NMS ONNX CPU detection",
            f"{len(merged_boxes)} boxes in {ms} ms ($0 token cost, +{len(tile_boxes)} high-res sachet candidates)",
            boxes=merged_boxes,
            info={
                "onnx_ms": ms,
                "sku_head_boxes": len(b_sku),
                "distilled_head_boxes": len(b_dist),
                "highres_tile_candidates": len(tile_boxes),
                "soft_nms_kept": len(merged_boxes),
            },
        )
        return merged_boxes


