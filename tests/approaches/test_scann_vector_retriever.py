"""Modular tests for `scann_vector_retriever` (`src/approaches/scann_vector_retriever.py`)."""

from __future__ import annotations

import pytest
from conftest import RealShelfFixture, make_real_context
from PIL import Image

import approaches
from utils import hul_domain


def test_scann_vector_retriever_real_workflow_and_zero_vlm_calls(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    clf = approaches.get("scann_vector_retriever")
    clf.setup({})

    ctx_a = make_real_context(model="none", trap_vlm=True)
    ctx_b = make_real_context(model="none", trap_vlm=True)

    preds_a = clf.classify(shelf_fixture_a.image, shelf_fixture_a.boxes[:6], ctx_a)
    preds_b = clf.classify(shelf_fixture_b.image, shelf_fixture_b.boxes[:6], ctx_b)

    assert len(preds_a) == 6 and len(preds_b) == 6
    assert ctx_a.trace.usage.calls == 0 and ctx_b.trace.usage.calls == 0
    assert ctx_a.trace.billed.get("embedding_image", 0) >= 1
    for p in preds_a + preds_b:
        hul_domain.validate_canonical_7dim_prediction(p)
    assert [p["confidence"] for p in preds_a] != [p["confidence"] for p in preds_b]


def test_scann_vector_retriever_hard_fail_on_invalid_inputs(
    shelf_fixture_a: RealShelfFixture,
) -> None:
    clf = approaches.get("scann_vector_retriever")
    ctx = make_real_context(model="none", trap_vlm=True)
    with pytest.raises(ValueError):
        clf.classify(None, shelf_fixture_a.boxes[:2], ctx)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        clf.classify(Image.new("RGB", (0, 0)), shelf_fixture_a.boxes[:2], ctx)
    with pytest.raises(ValueError):
        clf.classify(shelf_fixture_a.image, [(90.0, 20.0, 30.0, 80.0)], ctx)
