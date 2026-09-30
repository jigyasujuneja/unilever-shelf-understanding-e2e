"""Stage 5: Perceptual color distance and VLM fine-grained variant disambiguation.

Registers ``tiebreaker`` stage implementations:
  - ``cielab_delta_e_and_systemone`` (default): CIELAB Delta-E2000 color check with constrained VLM decoding.
  - ``cielab_delta_e_only``: CIELAB Delta-E2000 color check only (no VLM call).
  - ``direct_vlm_only``: Direct VLM classification without perceptual color pre-filtering.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from core import catalog as core_catalog
from core import features as core_features
from core.features import compute_ciede2000_approx
from core.retrieval import DjevSystemOneClient
from stages.registry import StageSpec, register_stage


def run_sister_shade_tiebreaker(
    box_xyxy: list[float] | tuple[float, ...] | None = None,
    candidate_skus: list[str] | None = None,
    mode: str = "cielab_delta_e_and_systemone",
    image: Image.Image | None = None,
    ctx: Any | None = None,
) -> dict[str, Any]:
    """Disambiguate near-identical product variants using real crop CIELAB color distance and VLM decoding."""
    if mode not in ("cielab_delta_e_and_systemone", "cielab_delta_e_only", "direct_vlm_only"):
        raise ValueError(f"Unsupported tiebreaker mode: {mode!r}")

    protos = core_catalog._build_real_catalog_prototype_bank()
    proto_by_sku = {p["sku_id"]: p for p in protos}
    candidates = list(candidate_skus) if candidate_skus else [p["sku_id"] for p in protos[:3]]
    if not candidates:
        raise ValueError("stage5_compound_vlm requires at least one candidate SKU")

    box = tuple(float(v) for v in (box_xyxy or (0.0, 0.0, 64.0, 128.0)))
    if len(box) < 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid bounding box coordinates in tiebreaker: {box}")

    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("stage5_compound_vlm requires a valid non-empty PIL.Image.Image")

    crop_feats = core_features.extract_real_crop_features(image, (box[0], box[1], box[2], box[3]))
    observed_lab = tuple(crop_feats["claim_lab"])
    glare_val = float(crop_feats["glare_ratio"])

    scored_by_color: list[tuple[float, str]] = []
    for cand_id in candidates:
        proto = proto_by_sku.get(cand_id)
        ref_lab = tuple(proto["cap_lab"]) if proto and "cap_lab" in proto else (71.1, 8.9, 18.2)
        de = compute_ciede2000_approx(observed_lab, ref_lab)
        scored_by_color.append((de, cand_id))
    scored_by_color.sort(key=lambda t: t[0])
    best_de, best_color_sku = scored_by_color[0]

    if mode == "cielab_delta_e_only":
        return {
            "mode": mode,
            "cielab_delta_e00": best_de,
            "resolved_sku_id": best_color_sku,
            "vlm_invoked": False,
        }

    vlm_client = DjevSystemOneClient()
    ordered_cands = [cand_id for _, cand_id in scored_by_color] if mode == "cielab_delta_e_and_systemone" else candidates
    vlm_result = vlm_client.resolve_crop_systemone(
        box_xyxy=list(box),
        scann_top5=ordered_cands[:5],
        raw_similarity=0.94,
        glare_intensity=glare_val,
        ocr_snippet="",
        image=image,
        ctx=ctx,
    )
    if vlm_result.resolved_base_pack_id not in candidates:
        raise RuntimeError(f"Hallucinated SKU ID {vlm_result.resolved_base_pack_id!r} not in candidate set {candidates}")

    return {
        "mode": mode,
        "cielab_delta_e00": best_de,
        "resolved_sku_id": vlm_result.resolved_base_pack_id,
        "pinned_ratio": vlm_result.pinned_ratio,
        "vlm_invoked": True,
    }


register_stage(
    StageSpec(
        stage_group="tiebreaker",
        name="cielab_delta_e_and_systemone",
        title="CIELAB Color Distance and Constrained VLM Decoding (Default)",
        description="Combines CIELAB Delta-E2000 color comparison on the shade band with constrained VLM decoding.",
        f2_delta=0.0,
        latency_delta_s=0.0,
        cost_delta_inr=0.0,
        default=True,
        fn=run_sister_shade_tiebreaker,
    )
)

register_stage(
    StageSpec(
        stage_group="tiebreaker",
        name="cielab_delta_e_only",
        title="CIELAB Color Distance Only (No VLM Call)",
        description="Selects the closest shade candidate using CIELAB Delta-E2000 color distance without calling a VLM.",
        f2_delta=0.0,
        latency_delta_s=-0.180,
        cost_delta_inr=-0.008,
        default=False,
        fn=lambda **kw: run_sister_shade_tiebreaker(mode="cielab_delta_e_only", **kw),
    )
)

register_stage(
    StageSpec(
        stage_group="tiebreaker",
        name="direct_vlm_only",
        title="Direct VLM Classification Only (No Color Pre-Filter)",
        description="Classifies ambiguous variants directly with the VLM without CIELAB color distance filtering.",
        f2_delta=0.0,
        latency_delta_s=0.220,
        cost_delta_inr=0.011,
        default=False,
        fn=lambda **kw: run_sister_shade_tiebreaker(mode="direct_vlm_only", **kw),
    )
)
