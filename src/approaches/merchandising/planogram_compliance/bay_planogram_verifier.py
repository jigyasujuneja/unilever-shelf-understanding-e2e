"""Track 2 Merchandising: Store-Session Planogram & Share-of-Shelf Verifier (`bay_planogram_verifier`).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Use Case: Merchandising -> Planogram & SOS Compliance (W8, hul_sessions)  │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Stage 0: Global Shelf Illuminant (Shades-of-Gray p=6)                     │
  │  Stage 1: 2-Stage Hierarchical Brand-Gate -> Allowed-ID CLM + JEV          │
  │  Stage 6: Per-Plank Brand-Block, OOS Void & Class 0 Bay Planogram Engine   │
  │    • Routes Class 0 bay vs full-image hanger scope, clusters shelf planks, │
  │      detects per-row OOS void boxes, and scores brand-block & bay purity.  │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from PIL import Image

from approaches.base import Box, Context, register
from approaches.market_share.retrieval.clm_info_nce import HierarchicalClmJev
from approaches.market_share.retrieval.embedding_retrieval import MODEL
from utils import planogram_engine


def compute_planogram_metrics(
    boxes: list[Box],
    pred_ids: list[int | None],
    catalog: dict[int, dict],
    asset_boxes: list[Box] | None = None,
    image_size: tuple[int, int] | None = None,
    planogram_spec: dict | None = None,
) -> dict:
    """Compute Stage 6 shelf planogram compliance, per-row brand-block purity, OOS voids, and bay purity."""
    return planogram_engine.evaluate_shelf_planogram(
        boxes=boxes,
        pred_ids=pred_ids,
        catalog=catalog,
        asset_boxes=asset_boxes,
        image_size=image_size,
        planogram_spec=planogram_spec,
    )


@register
class BayPlanogramVerifier(HierarchicalClmJev):
    """Store-Session Planogram Sequence, Brand-Block Purity, OOS Void & Share-of-Shelf Verifier."""

    name = "bay_planogram_verifier"
    use_case = "merchandising"
    task = "planogram_compliance"
    dataset = "hul_sessions"
    architecture = (
        f"Hierarchical Multi-Prototype Brand-Gate ({MODEL}) + CLM InfoNCE + JEV Contact Sheets -> "
        "Stage 6 Per-Plank Brand-Block, OOS Void & Class 0 Bay Planogram Engine"
    )
    steps = [
        "Setup: build multi-prototype (brand, category, packaging_type) sub-centroids + SKU multi-view & inner-80% CIELAB descriptors",
        "Stage 0: Estimate full-shelf Shades-of-Gray (p=6) illuminant once on shelf session",
        "Stage 1: Identify every shelf facing via Hierarchical Brand-Gate -> CLM InfoNCE + JEV fallback",
        "Stage 6: Route Class 0 bay vs full-image hanger scope, cluster shelf planks, detect per-row OOS void boxes, and score brand-block & bay purity",
    ]

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        from utils.awp_template_index import resolve_awp_planogram_spec

        ids = super().identify_boxes(image, boxes, ctx)
        asset_boxes = getattr(ctx, "detected_asset_boxes", None)
        spec_hint = getattr(ctx, "awp_code", None) or getattr(ctx, "image_id", None)
        planogram_spec = getattr(ctx, "planogram_spec", None) or resolve_awp_planogram_spec(spec_hint)
        plano = compute_planogram_metrics(
            boxes,
            ids,
            getattr(self, "catalog", {}),
            asset_boxes=asset_boxes,
            image_size=(image.width, image.height),
            planogram_spec=planogram_spec,
        )
        spec_check = plano.get("spec_compliance") or {}
        spec_str = (
            f" | AWP={spec_check.get('spec_id')} ({'PASS' if spec_check.get('is_compliant') else 'FAIL'})"
            if spec_check.get("has_spec")
            else ""
        )
        ctx.trace.step(
            "Verify Bay Planogram & Share of Shelf",
            f"Scope={plano['scope_mode']}{spec_str} | HUL SOS={plano['hul_facing_sos']*100:.1f}% "
            f"({plano['hul_facings']}/{plano['total_facings']} facings), "
            f"brand-block purity={plano['brand_block_purity']*100:.1f}%, "
            f"vertical alignment={plano['vertical_block_alignment']*100:.1f}%, "
            f"OOS voids={plano['oos_void_count']} "
            f"(empty share={plano['empty_shelf_share']*100:.1f}%), {plano['distinct_brands']} brands",
            boxes=boxes,
            info=plano,
        )
        return ids
