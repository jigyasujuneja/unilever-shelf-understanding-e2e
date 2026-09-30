"""``single_pass_dedup`` on two half-height tiles, joined back at the seam (Detection tab).

The seam stitching comes from Jigyasu's ``hul_domain._merge_seam_split_boxes`` (d96d75b).
SKU-110K shelves are tall photos with 100-250 products. One call on the whole shelf at 2048 px
gives each product only a few dozen pixels and asks Gemini for a very long box list. Two calls,
one on the top half and one on the bottom half, each at 2048 px, give every product twice the
resolution and half the list.

A product that crosses the cut comes back as two pieces: one in the top tile whose bottom edge
is on the seam, and one in the bottom tile whose top edge is on the seam. Pairs that share most
of their columns are merged into one box. Then the ``single_pass_dedup`` rules run on the
merged list.

Changes from the original: the seam band is 1.5% of the image height, not 10%, because the tiles
are cut exactly on the seam. The 0.24-0.68 aspect-ratio condition was dropped, because it kept
wide packs from merging. The x-overlap threshold (0.8 of the narrower box) is unchanged.
"""

from __future__ import annotations

from PIL import Image

from approaches.base import NOT_PRODUCT, Box, Context, label_counts, register
from approaches.market_share.detection.single_pass import PROMPT, SCHEMA, SinglePass, labelled_boxes
from approaches.market_share.detection.single_pass_dedup import NMS_IOU, dedup

SEAM_BAND = 0.015   # a box edge within this share of the image height of the seam touches it
SEAM_X_OVERLAP = 0.8  # share of the narrower piece's width that must overlap to merge


def merge_seam(top: list[Box], bottom: list[Box], seam: float, h: float) -> tuple[list[Box], int]:
    """Join pieces of products cut by the seam. Returns (boxes, number of merges)."""
    band = max(8.0, SEAM_BAND * h)
    top_cut = [i for i, b in enumerate(top) if abs(b[3] - seam) <= band]
    bottom_cut = [j for j, b in enumerate(bottom) if abs(b[1] - seam) <= band]
    pairs: list[tuple[float, int, int]] = []
    for i in top_cut:
        t = top[i]
        for j in bottom_cut:
            b = bottom[j]
            overlap = max(0.0, min(t[2], b[2]) - max(t[0], b[0]))
            share = overlap / (min(t[2] - t[0], b[2] - b[0]) or 1.0)
            if share >= SEAM_X_OVERLAP:
                pairs.append((share, i, j))
    used_t: set[int] = set()
    used_b: set[int] = set()
    merged: list[Box] = []
    for _, i, j in sorted(pairs, reverse=True):  # best-overlapping pairs first, each piece once
        if i in used_t or j in used_b:
            continue
        used_t.add(i)
        used_b.add(j)
        t, b = top[i], bottom[j]
        merged.append((min(t[0], b[0]), t[1], max(t[2], b[2]), b[3]))
    rest = [b for i, b in enumerate(top) if i not in used_t] + \
           [b for j, b in enumerate(bottom) if j not in used_b]
    return merged + rest, len(merged)


@register
class TiledDedup(SinglePass):
    name = "tiled_dedup"
    architecture = "Gemini, one call per half-height tile -> seam stitching -> duplicate removal"
    steps = [
        "Cut the shelf into top and bottom halves; one Gemini call per half (longest side 2048 px)",
        f"Merge the two pieces of products cut by the seam (edges within {SEAM_BAND:.1%} of the "
        f"height of the seam, >= {SEAM_X_OVERLAP:.0%} shared width)",
        "single_pass_dedup rules: drop container boxes, NMS, drop depth ghosts",
    ]

    def detect(self, image: Image.Image, ctx: Context) -> list[Box]:
        w, h = image.size
        seam = h // 2
        halves: list[list[Box]] = []
        for y0, y1 in ((0, seam), (seam, h)):
            res = ctx.ask(image.crop((0, y0, w, y1)), PROMPT, schema=SCHEMA, max_side=2048)
            boxes, labels = labelled_boxes(res.data, w, y1 - y0)
            kept = [(b[0], b[1] + y0, b[2], b[3] + y0)
                    for b, n in zip(boxes, labels, strict=True) if n != NOT_PRODUCT]
            ctx.trace.step(f"Gemini on the {'top' if y0 == 0 else 'bottom'} half",
                           f"1 call, {res.seconds:.1f}s -> {len(kept)} products "
                           f"({label_counts(labels) or 'none'})", boxes=kept)
            halves.append(kept)
        joined, n = merge_seam(halves[0], halves[1], seam, h)
        ctx.trace.step("Seam stitching", f"{n} products cut by the seam merged -> {len(joined)}",
                       boxes=joined)
        a, b, kept = dedup(joined)
        ctx.trace.step("Container boxes removed", f"{len(joined) - len(a)} removed", boxes=a)
        ctx.trace.step(f"NMS at IoU {NMS_IOU}", f"{len(a) - len(b)} removed", boxes=b)
        ctx.trace.step("Depth ghosts removed", f"{len(b) - len(kept)} removed -> {len(kept)}",
                       boxes=kept)
        return kept
