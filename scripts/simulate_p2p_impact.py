"""Run the 4-scenario P2P impact simulation on ``gt_hul_sessions`` and ``ms_json_golden``,
generate 12-column ``Past_ROI_Data`` CSVs, write ``docs/P2P_Impact_Readout.md``, and upload
to GCS when ``--upload-gcs`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.p2p_simulation import run_full_p2p_simulation  # noqa: E402


def render_markdown_readout(res: dict) -> str:
    gt_s = res["datasets"]["gt_hul_sessions"]
    ms_s = res["datasets"]["ms_json_golden"]
    lines = [
        "# P2P (Plan-to-Purchase) Impact Simulation Readout",
        "",
        "## 1. Purpose & Downstream Pipeline Alignment",
        "- Mirrors Unilever's exact `ROI_Pipeline.json` (`12-column Past_ROI_Data_*.csv`), Stage A (`NR_Brand_Hair_Beverages.py`), and Stage B (`NR_final_PW2.py`).",
        f"- Formula: `{res['formula']}`.",
        "- Compares 4 concrete scenarios on `gt_hul_sessions` (100 store visit strips) and `ms_json_golden` (100 LabelMe store shelves).",
        "",
        "## 2. Scenario Comparison (`gt_hul_sessions` — 100 Store Visits)",
        "",
        "| Scenario | Usable P2P Facings | Usable Share (%) | Unique Brands | Unique Basepacks | Mean $W_1$ Weight (%) | Mean Association Score |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sc_key, sc_label in [
        ("legacy_h5", "1. Legacy `.h5` Baseline"),
        ("ours_audited_crops", "2. Ours (Audited Crops Drop-in)"),
        ("ours_others_recovered", "3. Ours (+ Competitor `'Others'` Recovery)"),
        ("ours_full_shelf_enriched", "4. Ours (+ Full-Shelf Unannotated Enrichment)"),
    ]:
        s = gt_s["scenarios"][sc_key]
        lines.append(
            f"| {sc_label} | {s['usable_p2p_facings']:,} | {s['usable_facing_share_pct']:.1f}% | "
            f"{s['unique_brands_in_p2p']} | {s['unique_basepacks_in_p2p']} | "
            f"{s['mean_w1_basepack_weight_pct']:.2f}% | {s['mean_association_score']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Scenario Comparison (`ms_json_golden` — 100 Modern Trade Store Shelves)",
        "",
        "| Scenario | Usable P2P Facings | Usable Share (%) | Unique Brands | Unique Basepacks | Mean $W_1$ Weight (%) | Mean Association Score |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for sc_key, sc_label in [
        ("legacy_h5", "1. Legacy `.h5` Baseline"),
        ("ours_audited_crops", "2. Ours (Audited Crops Drop-in)"),
        ("ours_others_recovered", "3. Ours (+ Competitor `'Others'` Recovery)"),
        ("ours_full_shelf_enriched", "4. Ours (+ Full-Shelf Unannotated Enrichment)"),
    ]:
        s = ms_s["scenarios"][sc_key]
        lines.append(
            f"| {sc_label} | {s['usable_p2p_facings']:,} | {s['usable_facing_share_pct']:.1f}% | "
            f"{s['unique_brands_in_p2p']} | {s['unique_basepacks_in_p2p']} | "
            f"{s['mean_w1_basepack_weight_pct']:.2f}% | {s['mean_association_score']:.4f} |"
        )

    gt_up = gt_s["incremental_uplift"]
    ms_up = ms_s["incremental_uplift"]
    lines.extend([
        "",
        "## 4. Key Incremental Business Numbers",
        f"- **Basepack Trust Weight ($W_1 = W_4 \\times W_5 / 100$)**: Rises by **+{ms_up['w1_basepack_weight_delta_pts']:.2f} pts** (**{ms_up['w1_basepack_weight_multiplier']:.2f}x** higher trust in exact SKU adjacency).",
        f"- **Model Upgrade on Audited Boxes (`gt_hul_sessions`)**: **+{gt_up['usable_facings_gain_from_model_upgrade_pct']:.1f}%** more usable P2P facings (eliminating `Model_Output` drop-offs).",
        f"- **Competitor `'Others'` Recovery (`ms_json_golden`)**: **+{ms_up['usable_facings_gain_from_others_recovery_pct']:.1f}%** more usable facings by resolving `Others____Others____NA` into named competitor SKUs instead of dropping them in `NR_Brand_Hair_Beverages.py`.",
        f"- **Full-Shelf Enrichment (`ms_json_golden`)**: **+{ms_up['usable_facings_gain_from_full_shelf_enrichment_pct']:.1f}%** total usable facings vs legacy `.h5` (adding **+{ms_up['unique_brands_added']}** brands and **+{ms_up['unique_basepacks_added']}** basepacks across unannotated shelf rows).",
        "",
    ])
    return "\n".join(lines)


def upload_p2p_to_gcs(out_dir: Path, doc_path: Path) -> list[str]:
    from google.cloud import storage
    import runner

    project = runner.load_config()["gcp"]["project"]
    client = storage.Client(project=project)  # type: ignore[attr-defined]
    b_sb = client.bucket("ul-fs-s-sandbx-16-prj-shelf-images")
    b_ms = client.bucket("marketshare-datasets")
    uploaded: list[str] = []
    for fp in sorted(out_dir.glob("*")):
        if fp.is_file():
            for bucket, prefix in [
                (b_sb, f"results/p2p_simulation/{fp.name}"),
                (b_ms, f"derived_manifests/p2p_simulation/{fp.name}"),
            ]:
                bucket.blob(prefix).upload_from_filename(str(fp))
                uploaded.append(f"gs://{bucket.name}/{prefix}")
    if doc_path.exists():
        b_sb.blob(f"results/p2p_simulation/{doc_path.name}").upload_from_filename(str(doc_path))
        uploaded.append(f"gs://{b_sb.name}/results/p2p_simulation/{doc_path.name}")
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate Unilever P2P impact across 4 scenarios.")
    parser.add_argument(
        "--gt-sessions",
        type=Path,
        default=ROOT / "data" / "gt_hul_sessions" / "test.jsonl",
        help="Path to gt_hul_sessions test.jsonl",
    )
    parser.add_argument(
        "--ms-json",
        type=Path,
        default=ROOT / "data" / "ms_json_golden" / "test.jsonl",
        help="Path to ms_json_golden test.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "p2p_simulation",
        help="Output directory for 12-column CSVs and p2p_impact_summary.json",
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=ROOT / "docs" / "P2P_Impact_Readout.md",
        help="Markdown readout output path",
    )
    parser.add_argument(
        "--upload-gcs",
        action="store_true",
        help="Upload generated CSVs, JSON summary, and Markdown readout to GCS",
    )
    args = parser.parse_args()

    res = run_full_p2p_simulation(args.gt_sessions, args.ms_json, args.out_dir)
    md_text = render_markdown_readout(res)
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(md_text, encoding="utf-8")

    print(f"[OK] Wrote {args.out_dir / 'p2p_impact_summary.json'}")
    print(f"[OK] Wrote {args.doc_out}")
    print(json.dumps(res["datasets"]["ms_json_golden"]["incremental_uplift"], indent=2))

    if args.upload_gcs:
        uris = upload_p2p_to_gcs(args.out_dir, args.doc_out)
        for u in uris:
            print(f"[OK] Uploaded {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
