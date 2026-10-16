"""Tests for Residual InfoNCE Projection Head (EXP-32) over reference gallery vectors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.residual_infonce import ResidualInfoNCEHead  # noqa: E402


def test_residual_infonce_head_fits_and_preserves_unit_norm() -> None:
    rng = np.random.default_rng(7)
    dim = 64
    index: list[tuple[int, str, list[float]]] = []
    catalog: dict[int, dict[str, object]] = {}

    for pid in range(1, 25):
        base = rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
        base /= np.linalg.norm(base)
        brand = "Dove" if pid <= 12 else "Lux"
        catalog[pid] = {"sku_id": pid, "brand": brand}
        for v_idx in range(3):
            view = base + rng.normal(0.0, 0.08, size=(dim,)).astype(np.float32)
            view /= np.linalg.norm(view)
            index.append((pid, f"mem://{pid}_{v_idx}.jpg", view.tolist()))

    head = ResidualInfoNCEHead(dim=dim, hidden_dim=32, alpha=0.22, seed=42)
    stats = head.fit(index, catalog=catalog, epochs=15, use_cache=False)

    assert head.is_fitted()
    assert "base_val_r1" in stats
    assert "tuned_val_r1" in stats
    assert stats["tuned_val_r1"] >= stats["base_val_r1"]

    vec = np.asarray(index[0][2], dtype=np.float32)
    out = np.asarray(head.transform_one(vec), dtype=np.float32)
    assert out.shape == (dim,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
