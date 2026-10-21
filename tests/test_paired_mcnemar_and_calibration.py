"""Unit tests for paired McNemar significance testing and Rogan-Gladen prevalence calibration (PR-15)."""

from __future__ import annotations

import json
from pathlib import Path

import cli
from utils import metrics


def test_paired_mcnemar_test_detects_significant_improvement_and_discordant_pairs():
    rows_a = []
    rows_b = []
    # 80 both correct, 2 regressions (A ok, B wrong), 18 fixes (A wrong, B ok), 0 both wrong => N=100
    for i in range(80):
        iid = f"img_{i:03d}"
        rows_a.append({"image_id": iid, "labels": [{"gt": {"sku_id": 1}, "correct": {"product": True}}]})
        rows_b.append({"image_id": iid, "labels": [{"gt": {"sku_id": 1}, "correct": {"product": True}}]})
    for i in range(80, 82):
        iid = f"img_{i:03d}"
        rows_a.append({"image_id": iid, "labels": [{"gt": {"sku_id": 1}, "correct": {"product": True}}]})
        rows_b.append({"image_id": iid, "labels": [{"gt": {"sku_id": 1}, "correct": {"product": False}}]})
    for i in range(82, 100):
        iid = f"img_{i:03d}"
        rows_a.append({"image_id": iid, "labels": [{"gt": {"sku_id": 1}, "correct": {"product": False}}]})
        rows_b.append({"image_id": iid, "labels": [{"gt": {"sku_id": 1}, "correct": {"product": True}}]})

    res = metrics.paired_mcnemar_test(rows_a, rows_b)
    assert res["paired_n"] == 100
    assert res["a_both_correct"] == 80
    assert res["b_regressions"] == 2
    assert res["c_fixes"] == 18
    assert res["d_both_wrong"] == 0
    assert res["acc_a"] == 0.82
    assert res["acc_b"] == 0.98
    assert res["paired_lift"] == 0.16
    assert res["mcnemar_chi2"] == 12.8
    assert res["p_value"] < 0.001
    assert res["significant"] is True
    assert res["verdict"] == "significant_improvement"
    assert len(res["regressed_ids"]) == 2
    assert len(res["fixed_ids"]) == 18


def test_rogan_gladen_prevalence_correction():
    # Observed 0.916 with TPR=0.96, TNR=0.94 => Youden J = 0.90
    # p_hat = (0.916 + 0.94 - 1.0) / 0.90 = 0.856 / 0.90 = 0.9511
    cal = metrics.rogan_gladen_prevalence(0.916, tpr=0.96, tnr=0.94, n_samples=200)
    assert abs(cal["corrected_rate"] - 0.9511) < 1e-3
    assert cal["ci95_low"] <= cal["corrected_rate"] <= cal["ci95_high"]
    assert cal["youden_j"] == 0.9


def test_cli_compare_subcommand(tmp_path: Path, capsys):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "images.jsonl").write_text(
        json.dumps({"image_id": "1", "gt_count": 1, "tp": 0, "fp": 1, "fn": 1, "f2": 0.0}) + "\n"
        + json.dumps({"image_id": "2", "gt_count": 1, "tp": 1, "fp": 0, "fn": 0, "f2": 1.0}) + "\n"
    )
    (run_b / "images.jsonl").write_text(
        json.dumps({"image_id": "1", "gt_count": 1, "tp": 1, "fp": 0, "fn": 0, "f2": 1.0}) + "\n"
        + json.dumps({"image_id": "2", "gt_count": 1, "tp": 1, "fp": 0, "fn": 0, "f2": 1.0}) + "\n"
    )
    rc = cli.main(["compare", "run_a", "run_b", "--results", str(tmp_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["paired_n"] == 2
    assert out["c_fixes"] == 1
    assert out["b_regressions"] == 0
