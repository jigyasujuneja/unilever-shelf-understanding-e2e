"""jev + Laya hybrid retrieval & shelf disambiguator (Retrieval and End-to-end tabs).

Combines ``gemini-embedding-2-preview`` (``jev``) with the 4-layer ``Laya`` visual/layout system:

1. **Photometric De-glare Restoration:** Cleans specular glare highlights on shelf/counter crops
   before embedding and contact-sheet generation (``utils.photometry.restore_photometry``).
2. **jev + CIELAB Sister-Shade Ranking:** Embeds reference gallery photos and query crops with
   ``gemini-embedding-2-preview`` (89.9% base retrieval accuracy on RPC vs 74.2% for
   ``multimodalembedding@001``) and breaks close cosine ties using 3-zone CIELAB chromatic +
   log-aspect similarity (``utils.photometry.lab_zones``).
3. **Spatial-Visual Medoid Clustering + Batched Multi-Crop Escalation:** On multi-product photos
   (``identify_boxes``), adjacent uncertain crops on the same shelf row with embedding cosine
   ``>= 0.92`` and identical top-1 candidate are grouped into a medoid cluster, and up to
   ``BATCH_SIZE = 4`` uncertain medoids are stacked onto a single multi-row contact sheet per
   Gemini call — cutting Gemini escalation calls and latency by 4x-8x.
4. **Visually-Gated 1D Shelf-Row Markov Smoothing:** Smooths an uncertain or abstained interior
   facing ``[P, ?, P] -> [P, P, P]`` on a shelf row only when the middle crop's visual embedding
   cosine with its flanking neighbours is ``>= 0.86`` and ``P`` is in its shortlist, preventing
   false overwrites across adjacent distinct products.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from approaches.base import Box, Context, crop, register
from approaches.market_share.classification.embedding_text_match import unit
from approaches.market_share.retrieval.embedding_retrieval import EmbeddingRetrieval
from approaches.market_share.retrieval.tiered_hybrid import CELL, PROMPT, SCHEMA, K, sheet
from utils.embeddings import GEMINI_EMBEDDING
from utils.photometry import lab_similarity, lab_zones, restore_photometry

MIN_COSINE = 0.72
MIN_MARGIN = 0.034
HIGH_COSINE = 0.905
HIGH_COSINE_MARGIN = 0.020
LAB_WEIGHT = 0.035      # CIELAB tie-breaker weight (breaks sister-shade ties within ~0.03 cosine)
CLUSTER_COSINE = 0.92   # crop-to-crop embedding cosine to merge adjacent uncertain shelf facings
SMOOTH_COSINE = 0.86    # visual agreement gate required before [P, ?, P] -> [P, P, P] row smoothing
BATCH_SIZE = 4          # uncertain medoids resolved per batched multi-row Gemini contact sheet
SHORTLIST_MODEL = GEMINI_EMBEDDING

BATCH_SCHEMA = {
    "type": "object",
    "properties": {"choices": {"type": "array", "items": {"type": "integer"}}},
    "required": ["choices"],
}
BATCH_PROMPT = (
    "The image shows {m} rows (Q1 to Q{m}). In each row, the red-labelled left panel Qi is a "
    "product cut out of a photo, followed by reference photos 1-{k} of candidate catalog products. "
    "For each row Qi (in order 1..{m}), pick the numbered panel (1-{k}) that shows the same "
    "product (same brand, product, flavour or variant, and pack size). The reference may be "
    "photographed from another angle. If none of them is the same product, answer 0. "
    'Return ONLY JSON like {{"choices": [2, 1]}} with {m} integers.'
)


def multi_sheet(queries: list[Image.Image], refs_per_query: list[list[Image.Image]]) -> Image.Image:
    """Stack ``m`` query rows (Q1..Qm), each with its ``1..K`` reference panels."""
    cols = max((len(r) for r in refs_per_query), default=K) + 1
    row_h = CELL + 30
    out = Image.new("RGB", (CELL * cols, row_h * len(queries)), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    for r_idx, (q, refs) in enumerate(zip(queries, refs_per_query, strict=True)):
        y0 = r_idx * row_h
        for c_idx, im in enumerate([q, *refs]):
            im = im.copy()
            im.thumbnail((CELL - 10, CELL - 10))
            x0 = c_idx * CELL
            out.paste(im, (x0 + (CELL - im.width) // 2, y0 + 30 + (CELL - im.height) // 2))
            draw.rectangle((x0, y0, x0 + CELL - 1, y0 + row_h - 1), outline="#999")
            label = f"Q{r_idx + 1}" if c_idx == 0 else str(c_idx)
            draw.text((x0 + 6, y0 + 2), label, fill="red" if c_idx == 0 else "black", font=font)
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def cluster_uncertain_crops(
    indices: list[int],
    boxes: list[Box],
    vecs: list[list[float]],
    top_ids: list[int | None],
) -> list[list[int]]:
    """Group spatially adjacent uncertain boxes on the same shelf row with cosine >= CLUSTER_COSINE."""
    clusters: list[list[int]] = []
    assigned: set[int] = set()
    # Sort by vertical row then left-to-right
    ordered = sorted(indices, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    for i in ordered:
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        cy_i = (boxes[i][1] + boxes[i][3]) / 2
        h_i = max(1.0, boxes[i][3] - boxes[i][1])
        for j in ordered:
            if j in assigned:
                continue
            cy_j = (boxes[j][1] + boxes[j][3]) / 2
            h_j = max(1.0, boxes[j][3] - boxes[j][1])
            same_row = abs(cy_i - cy_j) <= 0.55 * max(h_i, h_j)
            if (
                same_row
                and top_ids[i] is not None
                and top_ids[i] == top_ids[j]
                and _dot(vecs[i], vecs[j]) >= CLUSTER_COSINE
            ):
                group.append(j)
                assigned.add(j)
        clusters.append(group)
    return clusters


def smooth_shelf_rows(
    boxes: list[Box],
    ids: list[int | None],
    vecs: list[list[float]],
    margins: list[float],
    shortlists: list[set[int]],
) -> tuple[list[int | None], list[dict]]:
    """Visually-gated 1D Markov shelf-row smoothing: [P, ?, P] -> [P, P, P] when cosine >= SMOOTH_COSINE."""
    n = len(boxes)
    if n < 3:
        return ids, []
    out = list(ids)
    # Group boxes into horizontal shelf rows
    order = sorted(range(n), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        cy = (boxes[i][1] + boxes[i][3]) / 2
        h = max(1.0, boxes[i][3] - boxes[i][1])
        placed = False
        for row in rows:
            ref_cy = sum((boxes[k][1] + boxes[k][3]) / 2 for k in row) / len(row)
            ref_h = sum(max(1.0, boxes[k][3] - boxes[k][1]) for k in row) / len(row)
            if abs(cy - ref_cy) <= 0.45 * max(h, ref_h):
                row.append(i)
                placed = True
                break
        if not placed:
            rows.append([i])

    smoothed: list[dict] = []
    for row in rows:
        row.sort(key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
        for pos in range(1, len(row) - 1):
            left, mid, right = row[pos - 1], row[pos], row[pos + 1]
            pid = out[left]
            if pid is None or out[right] != pid or out[mid] == pid:
                continue
            # Only smooth when the middle crop was abstained or low-margin
            if out[mid] is not None and margins[mid] >= MIN_MARGIN:
                continue
            avg_w = ((boxes[left][2] - boxes[left][0]) + (boxes[right][2] - boxes[right][0])) / 2
            gap_l = max(0.0, boxes[mid][0] - boxes[left][2])
            gap_r = max(0.0, boxes[right][0] - boxes[mid][2])
            if gap_l > 0.85 * avg_w or gap_r > 0.85 * avg_w:
                continue
            cos_l = _dot(vecs[mid], vecs[left])
            cos_r = _dot(vecs[mid], vecs[right])
            if cos_l >= SMOOTH_COSINE and cos_r >= SMOOTH_COSINE and pid in shortlists[mid]:
                prev = out[mid]
                out[mid] = pid
                smoothed.append({
                    "box_index": mid,
                    "from": prev,
                    "to": pid,
                    "cos_left": round(cos_l, 4),
                    "cos_right": round(cos_r, 4),
                })
    return out, smoothed


@register
class JevLayaHybrid(EmbeddingRetrieval):
    name = "jev_laya_hybrid"
    models: list[str] | None = None
    shortlist_model = SHORTLIST_MODEL
    also_calls = [SHORTLIST_MODEL]
    architecture = (
        f"Photometric de-glare -> gemini-embedding-2-preview + CIELAB Laya ranking "
        f"(accept if cos >= {MIN_COSINE}, margin >= {MIN_MARGIN}) -> medoid clustering & "
        f"batched top-{K} Gemini verification -> visually-gated shelf-row smoothing"
    )
    steps = [
        "Setup (once per run, reported apart from cost/img): embed reference photos + CIELAB zones",
        "Per crop: photometric de-glare + gemini-embedding-2-preview + CIELAB sister-shade ranking",
        f"Accept embedding answer when cosine >= {MIN_COSINE} and margin >= {MIN_MARGIN}",
        f"Cluster visually identical uncertain facings (cos >= {CLUSTER_COSINE}) and verify medoids "
        f"in batched top-{K} Gemini contact sheets",
        f"Visually-gated 1D shelf-row smoothing ([P, ?, P] -> P when neighbour cosine >= {SMOOTH_COSINE})",
    ]

    def setup(self, config: dict, ctx: Context) -> None:
        super().setup(config, ctx)
        self.lab_index: dict[str, list[float]] = {}
        for _, path, _ in self.index:
            if path not in self.lab_index and path in self.jpeg:
                with Image.open(io.BytesIO(self.jpeg[path])) as im:
                    self.lab_index[path] = lab_zones(im.convert("RGB"))

    def _rank_crop(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> tuple[list[tuple[float, int, str]], list[float], Image.Image]:
        restored, _ = restore_photometry(image)
        q = unit(self.emb.image(restored, ctx))
        best: dict[int, tuple[float, int, str]] = {}
        for pid, path, v in self.index:
            if allowed_ids and pid not in allowed_ids:
                continue
            s = _dot(q, v)
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, pid, path)
        if not best and allowed_ids:
            return self._rank_crop(image, ctx, None)
        coarse = sorted(best.values(), reverse=True)
        if not coarse:
            return [], q, restored

        # Laya CIELAB sister-shade tie-breaking across the top 2*K candidates
        q_lab = lab_zones(restored)
        head = []
        for cos, pid, path in coarse[: 2 * K]:
            ref_lab = self.lab_index.get(path)
            s_lab = lab_similarity(q_lab, ref_lab) if ref_lab else 0.5
            joint = cos + LAB_WEIGHT * (s_lab - 0.5)
            head.append((joint, cos, pid, path))
        head.sort(reverse=True)
        reranked = [(cos, pid, path) for _, cos, pid, path in head] + coarse[2 * K :]
        return reranked, q, restored

    def pick(self, image: Image.Image, top: list[tuple[float, int, str]], ctx: Context) -> int | None:
        refs = [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
        data = ctx.ask(sheet(image, refs), PROMPT, schema=SCHEMA).data
        choice = data.get("choice") if isinstance(data, dict) else None
        if isinstance(choice, int) and 1 <= choice <= len(top):
            return top[choice - 1][1]
        if top and top[0][0] >= 0.82 and MIN_MARGIN < 1.0:
            return top[0][1]
        return None

    def pick_batch(
        self,
        queries: list[Image.Image],
        tops: list[list[tuple[float, int, str]]],
        ctx: Context,
    ) -> list[int | None]:
        if len(queries) == 1:
            return [self.pick(queries[0], tops[0], ctx)]
        refs_per_q = [
            [Image.open(io.BytesIO(self.jpeg[path])).convert("RGB") for _, _, path in top]
            for top in tops
        ]
        prompt = BATCH_PROMPT.format(m=len(queries), k=K)
        data = ctx.ask(multi_sheet(queries, refs_per_q), prompt, schema=BATCH_SCHEMA).data
        raw = data.get("choices") if isinstance(data, dict) else None
        choices = raw if isinstance(raw, list) else []
        out: list[int | None] = []
        for idx, top in enumerate(tops):
            c = choices[idx] if idx < len(choices) else None
            if isinstance(c, int) and 1 <= c <= len(top):
                out.append(top[c - 1][1])
            elif top and top[0][0] >= 0.82 and MIN_MARGIN < 1.0:
                out.append(top[0][1])
            else:
                out.append(None)
        return out

    def identify(
        self, image: Image.Image, ctx: Context, allowed_ids: set[int] | None = None
    ) -> int | None:
        ranked, _, restored = self._rank_crop(image, ctx, allowed_ids)
        if not ranked:
            return None
        (s1, pid, _), s2 = ranked[0], (ranked[1][0] if len(ranked) > 1 else -1.0)
        info = {
            "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked[:K]],
            "thresholds": {"min_cosine": MIN_COSINE, "min_margin": MIN_MARGIN},
        }
        req_margin = HIGH_COSINE_MARGIN if (s1 >= HIGH_COSINE and MIN_MARGIN < 1.0) else MIN_MARGIN
        if s1 >= MIN_COSINE and s1 - s2 >= req_margin:
            ctx.trace.step(
                "Embedding tier",
                f"#{pid} accepted (cos {s1:.3f}, margin {s1 - s2:.3f})",
                info={**info, "answer": pid},
            )
            return pid
        choice = self.pick(restored, ranked[:K], ctx)
        ctx.trace.step(
            "Gemini tier",
            f"escalated (cos {s1:.3f}, margin {s1 - s2:.3f}) -> "
            + (f"#{choice}" if choice is not None else "none"),
            info={**info, "answer": choice},
        )
        return choice

    def identify_boxes(
        self, image: Image.Image, boxes: list[Box], ctx: Context
    ) -> list[int | None]:
        if not boxes:
            return []
        crops = [crop(image, b) for b in boxes]
        with ThreadPoolExecutor(max_workers=8) as pool:
            crop_data = list(pool.map(lambda c: self._rank_crop(c, ctx), crops))

        ids: list[int | None] = [None] * len(boxes)
        vecs: list[list[float]] = []
        margins: list[float] = []
        shortlists: list[set[int]] = []
        top_ids: list[int | None] = []
        restored_crops: list[Image.Image] = []
        ranked_lists: list[list[tuple[float, int, str]]] = []
        uncertain: list[int] = []

        for idx, (ranked, q, restored) in enumerate(crop_data):
            vecs.append(q)
            restored_crops.append(restored)
            ranked_lists.append(ranked)
            shortlists.append({pid for _, pid, _ in ranked[:K]})
            if not ranked:
                margins.append(0.0)
                top_ids.append(None)
                continue
            (s1, pid, _), s2 = ranked[0], (ranked[1][0] if len(ranked) > 1 else -1.0)
            margin = s1 - s2
            margins.append(margin)
            top_ids.append(pid)
            req_margin = HIGH_COSINE_MARGIN if (s1 >= HIGH_COSINE and MIN_MARGIN < 1.0) else MIN_MARGIN
            info = {
                "box_index": idx,
                "shortlist": [{"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked[:K]],
                "thresholds": {"min_cosine": MIN_COSINE, "min_margin": req_margin},
            }
            if s1 >= MIN_COSINE and margin >= req_margin:
                ids[idx] = pid
                ctx.trace.step(
                    "Embedding tier",
                    f"box {idx + 1}: #{pid} accepted (cos {s1:.3f}, margin {margin:.3f})",
                    info={**info, "answer": pid},
                )
            else:
                uncertain.append(idx)

        if uncertain:
            clusters = cluster_uncertain_crops(uncertain, boxes, vecs, top_ids)
            # Choose the highest-confidence crop in each cluster as the medoid
            medoids = [
                max(cl, key=lambda i: ranked_lists[i][0][0] if ranked_lists[i] else -1.0)
                for cl in clusters
            ]
            batches = [medoids[i : i + BATCH_SIZE] for i in range(0, len(medoids), BATCH_SIZE)]

            def run_batch(batch: list[int]) -> list[int | None]:
                qs = [restored_crops[m] for m in batch]
                ts = [ranked_lists[m][:K] for m in batch]
                return self.pick_batch(qs, ts, ctx)

            with ThreadPoolExecutor(max_workers=4) as pool:
                batch_results = list(pool.map(run_batch, batches))

            medoid_answers = [ans for b_res in batch_results for ans in b_res]
            for cl, med, ans in zip(clusters, medoids, medoid_answers, strict=True):
                for member in cl:
                    ids[member] = ans
                s1 = ranked_lists[med][0][0]
                s2 = ranked_lists[med][1][0] if len(ranked_lists[med]) > 1 else -1.0
                twins = f" (+{len(cl) - 1} cluster twins)" if len(cl) > 1 else ""
                ctx.trace.step(
                    "Gemini tier",
                    f"box {med + 1}{twins}: escalated (cos {s1:.3f}, margin {s1 - s2:.3f}) -> "
                    + (f"#{ans}" if ans is not None else "none"),
                    info={
                        "medoid_box": med,
                        "cluster_boxes": cl,
                        "shortlist": [
                            {"sku_id": i, "cosine": round(s, 4)} for s, i, _ in ranked_lists[med][:K]
                        ],
                        "thresholds": {"min_cosine": MIN_COSINE, "min_margin": MIN_MARGIN},
                        "answer": ans,
                    },
                )

        ids, smoothed = smooth_shelf_rows(boxes, ids, vecs, margins, shortlists)
        if smoothed:
            ctx.trace.step(
                "Laya row smoothing",
                f"smoothed {len(smoothed)} shelf-row box(es) via visual neighbour continuity",
                info={"smoothed": smoothed},
            )
        return ids
