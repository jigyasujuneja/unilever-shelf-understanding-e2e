"""Modular tests for `hul_hierarchy_classifier` (`src/approaches/hul_hierarchy_classifier.py`)."""

from __future__ import annotations

import pytest
from conftest import HallucinatingLLM, RealShelfFixture, make_real_context

import approaches
from approaches.base import Context, Trace
from utils import hul_domain


def test_hul_hierarchy_classifier_real_workflow_and_geometry_scoring(
    shelf_fixture_a: RealShelfFixture,
    shelf_fixture_b: RealShelfFixture,
) -> None:
    clf = approaches.get("hul_hierarchy_classifier")
    clf.setup({})

    ctx_a = make_real_context()
    ctx_b = make_real_context()

    preds_a = clf.classify(shelf_fixture_a.image, shelf_fixture_a.boxes[:6], ctx_a)
    preds_b = clf.classify(shelf_fixture_b.image, shelf_fixture_b.boxes[:6], ctx_b)

    assert len(preds_a) == 6 and len(preds_b) == 6
    for p in preds_a + preds_b:
        hul_domain.validate_canonical_7dim_prediction(p)
    assert [p["confidence"] for p in preds_a] != [p["confidence"] for p in preds_b]


def test_hul_hierarchy_classifier_hard_fail_anti_hallucination(
    shelf_fixture_a: RealShelfFixture,
) -> None:
    clf = approaches.get("hul_hierarchy_classifier")
    ctx_halluc = Context(
        model="fake-vlm",
        llm=HallucinatingLLM(),
        trace=Trace(),
        otel_parent=None,
        price=lambda _: {},
    )
    with pytest.raises(ValueError):
        clf.classify(shelf_fixture_a.image, shelf_fixture_a.boxes[:2], ctx_halluc)
    with pytest.raises(ValueError):
        clf.classify(None, shelf_fixture_a.boxes[:2], make_real_context())  # type: ignore[arg-type]
