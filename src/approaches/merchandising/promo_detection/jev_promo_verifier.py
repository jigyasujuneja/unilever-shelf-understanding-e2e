"""Track 2 Merchandising: Promo, Multipack & Price-Callout Verifier (`clm_promo_matcher`, `jev_promo_verifier`, `promo_ocr_zoom_verifier`).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Use Case: Merchandising -> Promo & Price Detection (W9, hul_promos)       │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  1. clm_promo_matcher (0 VLM calls):                                       │
  │     • Multi-view + SKU centroid + inner-80% CIELAB + top/bottom promo-band │
  │       colour signature matching for B1G1, B3G1, PO2, PO3, PO4, Rs10 SKUs   │
  │  2. jev_promo_verifier (CLM Gate + CLEF Flash Promo-Band Zoom Verifier):   │
  │     • Stage A: Auto-accepts clear promotional packs via CLM InfoNCE gate   │
  │     • Stage B: Escalates sister promo vs non-promo packs to CLEF Flash     │
  │       Zoom-Strip contact sheets with top-25% + bottom-30% callout bands    │
  │  3. promo_ocr_zoom_verifier (EXP-37 Dual-Band Promo OCR Zoom Rubric):      │
  │     • Renders 2x top-28% neck/header band + 2x bottom-28% price/value band │
  │       alongside candidate reference panels and snaps (brand, base_variant, │
  │       promo_type) to the matching promo sister SKU in the catalog.         │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw

from approaches.base import Context, register
from approaches.market_share.retrieval.clm_info_nce import L8HulUltimate
from approaches.market_share.retrieval.clm_standalone import ClmStandaloneRetrieval
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from utils.embeddings import MODELS

PROMO_TYPES = ("None", "B1G1", "B3G1", "PO2", "PO3", "PO4", "Rs10", "Extra", "Free")

PROMO_OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "visible_promo_text": {"type": "string"},
        "promo_type": {"type": "string", "enum": list(PROMO_TYPES)},
        "choice": {"type": "integer", "minimum": 0, "maximum": 5},
    },
    "required": ["visible_promo_text", "promo_type", "choice"],
}

PROMO_OCR_PROMPT = (
    "Panel Q is a store product crop. Panel TOP is a 2x zoom of the top 28% neck/header band of Q. "
    "Panel BOT is a 2x zoom of the bottom 28% price/value band of Q. Panels 1-{k} are candidate "
    "catalog reference photos.\n"
    "1. Read any visible promotional callout text on TOP, BOT, or Q (such as 'Buy 1 Get 1', 'Buy 3 Get 1', "
    "'Pack of 2', 'Pack of 3', 'Pack of 4', 'Rs 10 Off', '₹10', '20% Extra', 'Free', or '' if none).\n"
    "2. Classify 'promo_type' as one of: None, B1G1, B3G1, PO2, PO3, PO4, Rs10, Extra, Free.\n"
    "3. Pick the numbered panel (1-{k}) matching the product's brand, variant, and promo_type (or 0 if none).\n"
    "{rubric}\n"
    'Return ONLY JSON {{"visible_promo_text": "...", "promo_type": "...", "choice": <int 0..{k}>}}.'
)


def build_promo_dual_zoom_sheet(query: Image.Image, refs: list[Image.Image], cell: int = 220) -> Image.Image:
    """Render [Q | TOP-28% 2x | BOT-28% 2x | Ref 1 .. Ref K] contact sheet."""
    rgb = query.convert("RGB")
    w, h = max(1, rgb.width), max(1, rgb.height)
    top_band = rgb.crop((0, 0, w, max(1, int(round(h * 0.28)))))
    bot_band = rgb.crop((0, max(0, int(round(h * 0.72))), w, h))

    def _fit_cell(im: Image.Image, label: str, border_color: tuple[int, int, int]) -> Image.Image:
        canvas = Image.new("RGB", (cell, cell), (245, 245, 245))
        scale = min((cell - 8) / max(1, im.width), (cell - 26) / max(1, im.height))
        nw, nh = max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale)))
        resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas.paste(resized, ((cell - nw) // 2, 22 + (cell - 26 - nh) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, cell - 1, cell - 1], outline=border_color, width=2)
        draw.rectangle([0, 0, cell - 1, 20], fill=border_color)
        draw.text((6, 3), label, fill=(255, 255, 255))
        return canvas

    panels = [
        _fit_cell(rgb, "Q (Full Crop)", (200, 40, 40)),
        _fit_cell(top_band, "TOP 28% (2x Zoom)", (180, 90, 20)),
        _fit_cell(bot_band, "BOT 28% (2x Zoom)", (180, 90, 20)),
    ]
    for idx, ref in enumerate(refs, start=1):
        panels.append(_fit_cell(ref.convert("RGB"), f"Panel {idx}", (40, 90, 170)))

    sheet = Image.new("RGB", (cell * len(panels), cell), (255, 255, 255))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * cell, 0))
    return sheet


def _normalize_promo_token(raw: str | None) -> str:
    if not raw:
        return "None"
    s = str(raw).strip().upper()
    if s in ("NONE", "NA", "REGULAR", ""):
        return "None"
    for tok in PROMO_TYPES[1:]:
        if tok.upper() in s:
            return tok
    if "BUY 1" in s or "1 GET 1" in s:
        return "B1G1"
    if "BUY 3" in s or "3 GET 1" in s:
        return "B3G1"
    if "PACK OF 2" in s or "PO2" in s:
        return "PO2"
    if "PACK OF 3" in s or "PO3" in s:
        return "PO3"
    if "PACK OF 4" in s or "PO4" in s:
        return "PO4"
    if "10" in s or "RS" in s or "OFF" in s:
        return "Rs10"
    if "EXTRA" in s or "%" in s:
        return "Extra"
    if "FREE" in s:
        return "Free"
    return "None"


@register
class ClmPromoMatcher(ClmStandaloneRetrieval):
    """Zero-VLM CLM InfoNCE matcher for W9 Promotional, Multipack & Price-Tag SKUs."""

    name = "clm_promo_matcher"
    use_case = "merchandising"
    task = "promo_detection"
    dataset = "hul_promos"
    models = MODELS
    architecture = (
        f"Zero-VLM Promo Matcher: CLM InfoNCE ({MODEL} multi-view + SKU centroid + "
        "inner-80% CIELAB promo-band energy, 0 Gemini calls)"
    )
    steps = [
        "Setup: embed promotional SKU reference gallery views, centroids & inner-80% CIELAB descriptors",
        "Stage 0: Estimate global shelf illuminant (p=6) and normalize crop white balance",
        "Stage 1: Rank promotional SKU candidates by CLM InfoNCE energy (base_variant + promo_type)",
        "Return argmax promotional SKU in 0 VLM calls",
    ]


@register
class JevPromoVerifier(L8HulUltimate):
    """CLM InfoNCE Gate + JEV/CLEF Promo-Band Contact-Sheet Verifier for W9 Promo Compliance."""

    name = "jev_promo_verifier"
    use_case = "merchandising"
    task = "promo_detection"
    dataset = "hul_promos"
    architecture = (
        f"CLM InfoNCE Promo Gate ({MODEL}) -> JEV/CLEF Promo-Band Zoom Contact-Sheet Verifier "
        "(verifies B1G1/B3G1/PO3/Rs10/Free callouts)"
    )
    steps = [
        "Setup: embed promotional SKU reference gallery views, centroids & inner-80% CIELAB descriptors",
        "Stage 0: Estimate global shelf illuminant (p=6) and normalize crop white balance",
        "Stage A: Auto-accept promotional SKUs passing CLM InfoNCE posterior & Z-score margin gate (0 VLM calls)",
        "Stage B: Escalate sister promo vs base variants to Gemini contact sheets with bottom-30% zoom strips",
    ]


@register
class PromoOcrZoomVerifier(L8HulUltimate):
    """EXP-37 / EXP-48: Dual-Band Top/Bottom Promo Callout OCR Zoom Verifier."""

    name = "promo_ocr_zoom_verifier"
    use_case = "merchandising"
    task = "promo_detection"
    dataset = "hul_promos"
    architecture = (
        f"EXP-37 Promo OCR Zoom Verifier: {MODEL} top-5 shortlist -> Dual-Band Top-28% Neck + "
        "Bottom-28% Value 2x Zoom Strip + Sister Promo-SKU Catalog Snapping"
    )
    steps = [
        "Setup: embed promotional SKU reference gallery views and index sister (brand, base_variant, promo_type) families",
        "Retrieve top-5 candidate reference photos by CLM InfoNCE energy",
        "Render [Q | TOP-28% 2x | BOT-28% 2x | Panels 1..5] contact sheet with explicit promo_type rubric",
        "Read visible promo callout text + promo_type in Gemini and snap to matching sister promo SKU in catalog",
    ]

    catalog: dict[int, dict[str, Any]]
    jpeg: Any

    def _find_sister_promo_sku(self, base_sku_id: int | None, promo_type: str) -> int | None:
        if base_sku_id is None or not getattr(self, "catalog", None):
            return base_sku_id
        base_meta = self.catalog.get(base_sku_id) or {}
        brand = str(base_meta.get("brand") or "").strip().lower()
        bvar = str(base_meta.get("base_variant") or "").strip().lower()
        if not brand or not bvar:
            return base_sku_id
        target_pt = _normalize_promo_token(promo_type)
        for cid, meta in self.catalog.items():
            if (
                str(meta.get("brand") or "").strip().lower() == brand
                and str(meta.get("base_variant") or "").strip().lower() == bvar
                and _normalize_promo_token(meta.get("promo_type")) == target_pt
            ):
                return int(cid)
        return base_sku_id

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        ranked, _, _ = self._rank_crop(image, ctx, allowed_ids)
        if not ranked:
            return None
        top = ranked[:5]
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        sheet_im = build_promo_dual_zoom_sheet(image, refs)

        rubric_lines: list[str] = ["Candidate Reference Panels:"]
        for idx, (_, pid, _) in enumerate(top, start=1):
            meta: dict[str, Any] = (getattr(self, "catalog", None) or {}).get(pid) or {}
            pt = _normalize_promo_token(meta.get("promo_type"))
            rubric_lines.append(
                f"- Panel {idx}: Brand={meta.get('brand', '')} | Variant={meta.get('base_variant', '')} | Promo_Type={pt}"
            )
        prompt = PROMO_OCR_PROMPT.format(k=len(top), rubric="\n".join(rubric_lines))
        res = ctx.ask(sheet_im, prompt, schema=PROMO_OCR_SCHEMA)
        data = res.data if isinstance(res.data, dict) else {}
        choice = data.get("choice")
        pred_promo = _normalize_promo_token(data.get("promo_type"))
        vis_text = str(data.get("visible_promo_text") or "").strip()

        chosen_id = top[choice - 1][1] if isinstance(choice, int) and 1 <= choice <= len(top) else top[0][1]
        snapped_id = self._find_sister_promo_sku(chosen_id, pred_promo)
        ctx.trace.step(
            "Dual-Band Promo OCR Zoom Verifier",
            f"panel {choice} -> #{snapped_id} (promo_type={pred_promo}, text={vis_text!r})",
            info={
                "choice": choice,
                "chosen_sku_id": chosen_id,
                "snapped_sku_id": snapped_id,
                "promo_type": pred_promo,
                "visible_promo_text": vis_text,
                "shortlist": [{"sku_id": i, "score": round(s, 4)} for s, i, _ in top],
            },
        )
        return snapped_id

