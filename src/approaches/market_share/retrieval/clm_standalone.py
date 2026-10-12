"""Standalone CLM (Contrastive-LM) InfoNCE Retrieval Pipeline (``clm_standalone``).

Architecture:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Pure System-1 Contrastive-LM Retrieval (0 Gemini VLM Calls, Zero JEV)     │
  │  • Inherits directly from EmbeddingRetrieval (zero JevLaya coupling)       │
  │  • Multi-View Max Cosine + L2 SKU Centroid + Inner-80% CIELAB Energy:      │
  │      E(x, y_k) = 0.78*max_cos + 0.22*centroid_cos + w_lab*(s_lab - 0.5)    │
  │  • Temperature-scaled InfoNCE Softmax Posterior P_CLM(y_1|x) at 1/tau=28.0 │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import io

from PIL import Image

from approaches.base import Box, Context, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.retrieval.embedding_retrieval import (
    MODEL,
    EmbeddingRetrieval,
)
from utils.clm import infonce_posterior, infonce_rerank
from utils.embeddings import MODELS, SKUS
from utils.photometry import (
    apply_illuminant,
    estimate_shelf_illuminant,
    inner_lab_zones,
    restore_photometry,
)

CLM_LAB_WEIGHT = 0.035
CLM_INV_TEMP = 28.0
CLM_TOP_K = 5


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@register
class ClmStandaloneRetrieval(EmbeddingRetrieval):
    """Standalone System-1 CLM InfoNCE Retrieval (0 VLM calls, zero JEV inheritance)."""

    name = "clm_standalone"
    task = "retrieval"
    dataset = "rpc"
    models = MODELS
    skus = SKUS
    shortlist_model = MODEL
    architecture = (
        "Standalone CLM (Contrastive-LM) InfoNCE Multi-View + SKU Centroid + Inner-80% CIELAB "
        f"Energy (1/tau={CLM_INV_TEMP}, 0 VLM calls)"
    )
    steps = [
        "Setup: embed reference gallery views, L2 SKU centroids, and inner-80% CIELAB + shoulder-taper descriptors",
        "Per crop: photometric de-glare + multi-view max cosine + SKU centroid cosine + inner-80% CIELAB similarity",
        f"Rank by CLM InfoNCE joint energy E(x, y_k) = 0.78*max_cos + 0.22*centroid_cos + {CLM_LAB_WEIGHT}*(s_lab - 0.5)",
        f"Return top-1 SKU with calibrated InfoNCE posterior P_CLM(y_1|x) at 1/tau={CLM_INV_TEMP} in 0 VLM calls",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        by_sku: dict[int, list[list[float]]] = {}
        for pid, _, vec in self.index:
            by_sku.setdefault(pid, []).append(vec)
        self.sku_centroids: dict[int, list[float]] = {
            pid: unit([sum(col) / len(vecs) for col in zip(*vecs, strict=True)])
            for pid, vecs in by_sku.items()
        }
        self.lab_index: dict[str, list[float]] = {}
        for _, path, _ in self.index:
            if path not in self.lab_index and path in self.jpeg:
                with Image.open(io.BytesIO(self.jpeg[path])) as im:
                    self.lab_index[path] = inner_lab_zones(im.convert("RGB"))

    def ranked(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> list[tuple[float, int, str]]:
        """Rank products by CLM InfoNCE joint energy (energy, sku_id, ref_path)."""
        restored, _ = restore_photometry(image)
        q = unit(self.emb.image(restored, ctx))
        pool = (
            [(pid, path, v) for pid, path, v in self.index if pid in allowed_ids]
            if allowed_ids
            else []
        ) or self.index
        best: dict[int, tuple[float, int, str]] = {}
        for pid, path, v in pool:
            s = _dot(q, v)
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, pid, path)
        coarse = sorted(best.values(), reverse=True)
        if not coarse:
            return []

        gains = getattr(ctx, "shelf_illuminant", None)
        wb_crop = apply_illuminant(restored, gains) if gains else restored
        q_lab = inner_lab_zones(wb_crop)
        return infonce_rerank(
            coarse,
            q,
            q_lab,
            self.lab_index,
            self.sku_centroids,
            lab_weight=CLM_LAB_WEIGHT,
            k=CLM_TOP_K,
        )

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        r = self.ranked(image, ctx, allowed_ids)
        if not r:
            return None
        e1, pid, _ = r[0]
        e2 = r[1][0] if len(r) > 1 else -1.0
        p1, ent_conf = infonce_posterior(r, inv_temp=CLM_INV_TEMP, k=CLM_TOP_K)
        ctx.trace.step(
            "Standalone CLM InfoNCE",
            f"#{pid} (energy {e1:.3f}, margin {e1 - e2:.3f}, P_CLM {p1:.2f}, ent_conf {ent_conf:.2f})",
            info={
                "sku_id": pid,
                "energy": round(e1, 4),
                "margin": round(e1 - e2, 4),
                "clm_top1_prob": p1,
                "clm_entropy_conf": ent_conf,
                "shortlist": [{"sku_id": i, "energy": round(s, 4)} for s, i, _ in r[:CLM_TOP_K]],
            },
        )
        return pid

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        if len(boxes) >= 3:
            ctx.shelf_illuminant = estimate_shelf_illuminant(image, p=6.0)
        return super().identify_boxes(image, boxes, ctx)
