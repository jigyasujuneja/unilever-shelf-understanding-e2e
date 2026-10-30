#!/usr/bin/env python3
"""Generates high-legibility, low-text-density engineering diagrams and Matplotlib charts.

Design & Legibility Rules for Google Docs (620px column width):
- Logical HTML canvas width = 1000px (rendered at 2x Retina = 2000px PNG).
- At 1000px logical width, 18px-22px fonts scale to 11.2px-13.6px (~8.5pt-10.2pt) in Google Docs.
- ZERO bullet lists inside diagram boxes.
- Maximum 3 short lines per box:
  1. Bold Stage / Task Header (19px-20px)
  2. Monospace Code Module / -a Approach from t.ire4ghmqzbz8 (16px)
  3. Concise Validated Metric (17px bold)
- Standardizes strictly on 500,000 shelves/day (500K/day = 15M/month) across the
  07:30-18:00 IST daytime store operating window (zero 100K/day tier).
"""

from pathlib import Path
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images"
DIAG_DIR = ROOT / "docs" / "diagrams"
CHROME = "/usr/bin/google-chrome"


def render_matplotlib_finops_deepdive():
    """Renders fig_finops_cost_and_slice_deepdive.png (500K/day = 15M/mo + 5K accuracy)."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#94A3B8"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["text.parse_math"] = False

    fig = plt.figure(figsize=(11.5, 7.6), dpi=200, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.48, wspace=0.22,
                          left=0.07, right=0.97, top=0.90, bottom=0.10)

    # Panel A: Horizontal bar chart at 500,000 shelves/day (15M/month)
    ax_a = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.48,
                            left=0.33, right=0.96, top=0.90, bottom=0.10)[0].subgridspec(1, 1)[0]
    ax_a = fig.add_subplot(ax_a)
    configs = [
        "Option 2 + Local CPU ONNX Embedder\n($0.00058 / ₹0.06 | 82.1%–83.8% 5K)",
        "Option 1: Vector Gallery Only (0-LLM)\n($0.00105 / ₹0.10 | 78.9%–80.9% 5K)",
        "Option 2: Hybrid CLM+CLEF (Std 1.0x)\n($0.00132 / ₹0.12 | 82.3%–84.0% 5K)",
        "Option 2: Hybrid CLM+CLEF (EA / PT)\n($0.00160 / ₹0.15 | 82.3%–84.0% 5K)",
        "Option 2 All-In (w/ $11.1K PR01–PR18)\n($0.00234 / ₹0.22 | 500K/day All-In)",
        "Existing 14-GPU Stack (Co-Located L4)\n($0.00280–$0.00341 | 42.32% 5K)",
    ]
    cost_500k = [8.70, 15.75, 19.80, 24.00, 35.10, 51.20]
    colors = ["#0284C7", "#2563EB", "#0D9488", "#059669", "#1E40AF", "#DC2626"]

    y = np.arange(len(configs))
    bars = ax_a.barh(y, cost_500k, height=0.58, color=colors, edgecolor="#0F172A", linewidth=0.8)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(configs, fontsize=10.0, fontweight="bold", color="#0F172A")
    ax_a.set_xlabel("Monthly Spend in USD Thousands ($K/month) at 500,000 Shelves/Day (15M/Month)",
                    fontsize=10.5, fontweight="bold", color="#0F172A")
    ax_a.set_xlim(0, 62)
    ax_a.grid(axis="x", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    for b, val in zip(bars, cost_500k):
        ax_a.text(b.get_width() + 0.8, b.get_y() + b.get_height()/2,
                  f"${val:,.2f}K/mo", va="center", ha="left",
                  fontsize=10.0, fontweight="bold", color="#0F172A" if val < 40 else "#991B1B")

    ax_a.set_title("Panel A: Monthly Cloud Spend at 500,000 Shelves/Day (15M/Month | 1 USD = ₹92.54)",
                   fontsize=12.0, fontweight="bold", color="#0F172A", pad=10)

    # Panel B1: Market Share Task Accuracy
    ax_b1 = fig.add_subplot(gs[1, 0])
    tasks = ["5K SKU\n(Task 3)", "1K Crown\n(Task 4)", "Variants\n(Task 6)", "874 Brands\n(Task 7)", "3D Packs\n(Task 8)"]
    leg_acc = [42.3, 44.5, 48.1, 87.1, 54.9]
    new_acc = [82.3, 84.0, 85.0, 95.9, 93.3]
    x1 = np.arange(len(tasks))
    w = 0.36
    b_l1 = ax_b1.bar(x1 - w/2, leg_acc, width=w, color="#FCA5A5", edgecolor="#991B1B", label="Legacy Cascade")
    b_n1 = ax_b1.bar(x1 + w/2, new_acc, width=w, color="#0D9488", edgecolor="#115E59", label="Option 2 (CLM+CLEF)")
    ax_b1.set_xticks(x1)
    ax_b1.set_xticklabels(tasks, fontsize=9.5, fontweight="bold")
    ax_b1.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax_b1.set_ylim(0, 128)
    ax_b1.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax_b1.set_axisbelow(True)
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)
    ax_b1.legend(loc="upper left", ncol=2, fontsize=8.8, frameon=True)
    ax_b1.set_title("Panel B1: Market Share Task Accuracy", fontsize=11.5, fontweight="bold", pad=8)
    for b in b_l1:
        ax_b1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991B1B")
    for b in b_n1:
        ax_b1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#115E59")

    # Panel B2: Merchandising & Talker Task Accuracy
    ax_b2 = fig.add_subplot(gs[1, 1])
    mtasks = ["In-Bay SKU\n(Task 1)", "Bay Filter\n(Task 2)", "Planogram\n(Task 1/2)", "Promo Brand\n(Task 3)", "Talker F2\n(Task 4)"]
    m_leg = [44.5, 51.1, 58.0, 52.0, 49.2]
    m_new = [97.2, 81.8, 94.7, 97.5, 90.0]
    x2 = np.arange(len(mtasks))
    b_l2 = ax_b2.bar(x2 - w/2, m_leg, width=w, color="#FCA5A5", edgecolor="#991B1B", label="Existing Baseline")
    b_n2 = ax_b2.bar(x2 + w/2, m_new, width=w, color="#2563EB", edgecolor="#1E3A8A", label="Option 2 Merchandising")
    ax_b2.set_xticks(x2)
    ax_b2.set_xticklabels(mtasks, fontsize=9.5, fontweight="bold")
    ax_b2.set_ylabel("Accuracy / F2 (%)", fontsize=10.5, fontweight="bold")
    ax_b2.set_ylim(0, 128)
    ax_b2.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax_b2.set_axisbelow(True)
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)
    ax_b2.legend(loc="upper left", ncol=2, fontsize=8.8, frameon=True)
    ax_b2.set_title("Panel B2: Merchandising Task Accuracy", fontsize=11.5, fontweight="bold", pad=8)
    for b in b_l2:
        ax_b2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991B1B")
    for b in b_n2:
        ax_b2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1E3A8A")

    fig.suptitle("GCP Cost at 500,000 Shelves/Day (15M/Mo) & Validated Task Accuracy",
                 fontsize=14, fontweight="bold", color="#0F172A", y=0.97)
    for out in [OUT_DIR / "fig_finops_cost_and_slice_deepdive.png", DIAG_DIR / "milestone_ablation_progression.png"]:
        fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Rendered fig_finops_cost_and_slice_deepdive.png")


def render_matplotlib_cost_breakdown():
    """Renders fig_cost_service_pie_and_ranges.png (focused purely on 500K/day = 15M/mo cost)."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig = plt.figure(figsize=(11.5, 5.6), dpi=200, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.32,
                          left=0.26, right=0.97, top=0.83, bottom=0.17)

    # Panel A: Per-Shelf Variable AI Cost Receipt ($0.00160 / ₹0.15)
    ax1 = fig.add_subplot(gs[0])
    stages = [
        "Pub/Sub + GCS CMEK\n($0.00002 | ₹0.002)",
        "Cloud Run CPU YOLO11n\n($0.00003 | ₹0.003)",
        "Stage 2A Embed-2 (~11 crops)\n($0.00050 | ₹0.046)",
        "Stage 2B Flash-Lite (32% ties)\n($0.00105 | ₹0.097)",
    ]
    vals = [0.00002, 0.00003, 0.00050, 0.00105]
    cols = ["#64748B", "#0284C7", "#4F46E5", "#0D9488"]
    y = np.arange(len(stages))
    bars = ax1.barh(y, vals, height=0.52, color=cols, edgecolor="#0F172A", linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(stages, fontsize=10.0, fontweight="bold", color="#0F172A")
    ax1.set_xlabel("Per-Shelf Cost in USD ($)", fontsize=10.5, fontweight="bold")
    ax1.set_xlim(0, 0.00145)
    ax1.set_xticks([0.0, 0.0004, 0.0008, 0.0012])
    ax1.set_xticklabels(["$0.0000", "$0.0004", "$0.0008", "$0.0012"], fontsize=9.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("Panel A: Per-Shelf AI Cost Receipt\nTotal Pure AI = $0.00160 / Shelf (₹0.15)",
                  fontsize=11.0, fontweight="bold", pad=10)
    for b, v in zip(bars, vals):
        ax1.text(b.get_width() + 0.00003, b.get_y() + b.get_height()/2,
                 f"${v:.5f}", va="center", ha="left", fontsize=9.8, fontweight="bold", color="#0F172A")

    # Panel B: Stacked Monthly Spend at 500,000 Shelves/Day (15M/Month)
    ax2 = fig.add_subplot(gs[1])
    labels = [
        "Opt 2 +\nLocal CPU",
        "Option 1\n(0-LLM)",
        "Option 2\n(Std 1.0x)",
        "Option 2\n(EA / PT)",
        "Existing\n14× L4 GPU",
    ]
    ai_spend = np.array([8.70, 15.75, 19.80, 24.00, 38.60])
    infra_spend = np.array([12.40, 12.40, 12.40, 12.40, 12.40])
    x = np.arange(len(labels))
    w = 0.50

    b1 = ax2.bar(x, ai_spend, width=w, color=["#0284C7", "#2563EB", "#0D9488", "#059669", "#DC2626"],
                 edgecolor="#0F172A", linewidth=0.8, label="Pure AI Inference ($K/mo)")
    b2 = ax2.bar(x, infra_spend, width=w, bottom=ai_spend, color="#CBD5E1",
                 edgecolor="#334155", linewidth=0.8, hatch="//", label="Fixed PR01–PR18 Infra ($12.4K/mo)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Monthly Spend ($K / Month)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 68)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=8.8, frameon=True)
    ax2.set_title("Panel B: Monthly Spend at 500K/Day (15M/Mo)\n(Pure AI + $12.4K/Mo Shared PR01–PR18 Infra)",
                  fontsize=11.0, fontweight="bold", pad=10)

    for i, (a, inf) in enumerate(zip(ai_spend, infra_spend)):
        tot = a + inf
        ax2.text(x[i], tot + 1.0, f"${tot:.1f}K\n(${a:.1f}K AI)",
                 ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#0F172A")

    fig.suptitle("Bottom-Up Unit Cost & Monthly Cloud Budget at 500,000 Shelves/Day (15M/Month)",
                 fontsize=13.5, fontweight="bold", color="#0F172A", y=0.96)
    for out in [OUT_DIR / "fig_cost_service_pie_and_ranges.png", DIAG_DIR / "tco_burst_economics.png"]:
        fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Rendered fig_cost_service_pie_and_ranges.png")


def render_matplotlib_locust_and_rollout():
    """Renders fig_path_to_prod_locust_and_rollout.png (high-legibility 2-panel chart, 0 tiny tables)."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig = plt.figure(figsize=(11.0, 4.8), dpi=200, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28,
                          left=0.08, right=0.96, top=0.82, bottom=0.16)

    # Panel A: Locust 2.46.7 Load Test (10, 40, 80 workers)
    ax1 = fig.add_subplot(gs[0])
    tiers = [
        "10 Workers\n(Daytime Avg\n~13.2 shelves/s)",
        "40 Workers\n(2.5× Midday Peak\n~33 shelves/s)",
        "80 Workers\n(Max Cloud Run\nBurst Pool)",
    ]
    p50 = [0.12, 1.10, 1.10]
    p95 = [0.34, 2.50, 4.90]
    x = np.arange(len(tiers))
    w = np.float64(0.35)
    b1 = ax1.bar(x - w/2, p50, width=w, color="#0D9488", edgecolor="#0F172A", label="P50 Latency (s)")
    b2 = ax1.bar(x + w/2, p95, width=w, color="#2563EB", edgecolor="#0F172A", label="P95 Latency (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontsize=10.0, fontweight="bold")
    ax1.set_ylabel("Gateway Latency (Seconds)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 6.2)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=9.5, frameon=True)
    ax1.set_title("Panel A: Locust 2.46.7 Load Test (570 Reqs, 0.00% Failures)\n07:30–18:00 IST Store Window vs. 180s Callback SLA",
                  fontsize=11.0, fontweight="bold", pad=10)
    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.12, f"{b.get_height():.2f}s",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#115E59")
    for b in b2:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.12, f"{b.get_height():.2f}s",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E3A8A")

    # Panel B: 35-Pack Shelf Latency Reduction
    ax2 = fig.add_subplot(gs[1])
    steps = [
        "1-Thread\nSequential\nHarness",
        "Row Twin\nGrouping\n(35 → 11)",
        "Option 2\nBatched\nCloud Run",
        "Option 1\n0-LLM\nFast Path",
    ]
    lats = [94.60, 29.50, 2.69, 0.90]
    cols = ["#94A3B8", "#F59E0B", "#0D9488", "#2563EB"]
    x2 = np.arange(len(steps))
    bars2 = ax2.bar(x2, lats, width=0.52, color=cols, edgecolor="#0F172A")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(steps, fontsize=10.0, fontweight="bold")
    ax2.set_ylabel("35-Pack Shelf P50 Latency (s)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 112)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Panel B: 35-Pack Shelf P50 Latency Reduction (-97.2%)\nSequential Harness (94.6s) to Batched Worker (2.69s)",
                  fontsize=11.0, fontweight="bold", pad=10)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2.0, f"{b.get_height():.2f}s",
                 ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#0F172A")

    fig.suptitle("Cloud Run Load-Test Verification at 500,000 Shelves/Day (07:30–18:00 IST Store Window)",
                 fontsize=13.5, fontweight="bold", color="#0F172A", y=0.96)
    for out in [OUT_DIR / "fig_path_to_prod_locust_and_rollout.png", DIAG_DIR / "visual_math_and_eval_science.png"]:
        fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Rendered fig_path_to_prod_locust_and_rollout.png")


COMMON_CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
    background: #FFFFFF;
    color: #0F172A;
    padding: 20px 24px;
    width: 1000px;
  }
  .canvas {
    border: 2px solid #334155;
    background: #FFFFFF;
    padding: 18px 22px;
  }
  .hdr {
    border-bottom: 2px solid #0F172A;
    padding-bottom: 10px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
  }
  .title { font-size: 22px; font-weight: 800; color: #0F172A; letter-spacing: -0.3px; }
  .subtitle { font-size: 15px; font-weight: 600; color: #475569; margin-top: 3px; }
  .badge {
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    color: #1E3A8A;
    background: #EFF6FF;
    border: 1.5px solid #93C5FD;
    padding: 5px 10px;
    white-space: nowrap;
  }
  .lane {
    border: 1.5px dashed #64748B;
    background: #F8FAFC;
    padding: 14px 16px;
    margin-bottom: 14px;
  }
  .lane-lbl {
    font-size: 14px;
    font-weight: 800;
    text-transform: uppercase;
    color: #1E293B;
    background: #E2E8F0;
    border: 1.5px solid #94A3B8;
    display: inline-block;
    padding: 4px 10px;
    margin-bottom: 12px;
    letter-spacing: 0.4px;
  }
  .row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  .row-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; }
  .box {
    border: 2px solid #334155;
    background: #FFFFFF;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .box-h {
    padding: 8px 12px;
    font-size: 17px;
    font-weight: 800;
    color: #FFFFFF;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }
  .bg-navy { background: #1E3A8A; }
  .bg-blue { background: #1D4ED8; }
  .bg-teal { background: #0F766E; }
  .bg-green { background: #15803D; }
  .bg-amber { background: #B45309; }
  .bg-red { background: #B91C1C; }
  .bg-slate { background: #334155; }
  .box-code {
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    background: #F1F5F9;
    border-bottom: 1.5px solid #CBD5E1;
    padding: 7px 12px;
  }
  .box-metric {
    font-size: 17px;
    font-weight: 700;
    color: #0F172A;
    padding: 10px 12px;
    line-height: 1.35;
  }
  .box-foot {
    font-size: 15px;
    font-weight: 800;
    color: #1E3A8A;
    background: #EFF6FF;
    border-top: 1.5px solid #CBD5E1;
    padding: 7px 12px;
  }
  .arrow-down {
    text-align: center;
    font-size: 17px;
    font-weight: 800;
    color: #1E293B;
    padding: 6px 0;
  }
"""


def render_html_to_png(html_body: str, out_paths: list[Path], height: int = 1400):
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{COMMON_CSS}</style></head><body>{html_body}</body></html>"
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp_html = f.name
    tmp_png = tmp_html + ".png"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--window-size=1000,1400",
            f"--screenshot={tmp_png}",
            tmp_html,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from PIL import Image
    im = Image.open(tmp_png).convert("RGB")
    arr = np.array(im)
    non_white = np.where(np.any(arr < 252, axis=2))
    if len(non_white[0]) > 0:
        y_max = min(im.height, int(non_white[0].max()) + 36)
        im = im.crop((0, 0, im.width, y_max))
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p)
    print("Rendered", out_paths[0].name, f"({im.width}x{im.height})")


def build_all_engineering_diagrams():
    # Common CSS for Option 3 (Chevron Step Flow) and Simpler Option 1 (Lucidchart Swimlanes)
    lucid_css = """
    <style>
      .lucid-canvas {
        width: 960px; padding: 22px 24px; background-color: #ffffff;
        border: 2px solid #cbd5e1; border-radius: 12px; box-sizing: border-box;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      .lucid-grid {
        background-image: radial-gradient(#cbd5e1 1.2px, transparent 1.2px);
        background-size: 18px 18px;
      }
      .top-bar {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 16px; background: rgba(255,255,255,0.94);
        padding: 2px 4px; border-radius: 6px;
      }
      .top-title { font-size: 20px; font-weight: 800; color: #0f172a; }
      .step-row { display: flex; align-items: stretch; justify-content: space-between; gap: 6px; }
      .step-card {
        flex: 1; min-width: 0; border-radius: 12px; padding: 16px 10px; text-align: center;
        border: 2.5px solid #cbd5e1; background: #f8fafc;
        display: flex; flex-direction: column; align-items: center; justify-content: space-between;
      }
      .step-num {
        width: 32px; height: 32px; line-height: 32px; border-radius: 50%;
        font-weight: 800; font-size: 15px; color: #fff; margin-bottom: 8px;
      }
      .step-h { font-size: 17px; font-weight: 800; color: #0f172a; margin-bottom: 6px; line-height: 1.2; }
      .step-sub { font-size: 13.5px; font-weight: 600; color: #475569; margin-bottom: 12px; line-height: 1.3; }
      .step-pill {
        display: inline-block; padding: 5px 8px; border-radius: 999px;
        font-size: 11.5px; font-weight: 800; white-space: nowrap;
      }
      .step-arr { display: flex; align-items: center; font-size: 22px; font-weight: 900; color: #64748b; }
      .section-tag {
        font-size: 12.5px; font-weight: 800; letter-spacing: 0.7px;
        text-transform: uppercase; margin: 10px 0 8px 0; color: #334155;
      }
    </style>
    """

    # 1. E2E Figure 1: Option 3 Left-to-Right 5-Step Pipeline Flow (Pure Flow, Explicit Model Names)
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas">
  <div class="top-bar">
    <div class="top-title">End-to-End Shelf Understanding Pipeline Flow</div>
  </div>

  <div class="step-row">
    <div class="step-card" style="border-color:#2563eb; background:#eff6ff;">
      <div class="step-num" style="background:#2563eb;">1</div>
      <div class="step-h">Detect</div>
      <div class="step-sub">Locate Shelf Bays<br>&amp; Product Boxes</div>
      <div class="step-pill" style="background:#dbeafe; color:#1e40af;">Distilled YOLO11n ONNX</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#0d9488; background:#f0fdfa;">
      <div class="step-num" style="background:#0d9488;">2</div>
      <div class="step-h">Filter</div>
      <div class="step-sub">Keep Crops Inside<br>Target Display Bay</div>
      <div class="step-pill" style="background:#ccfbf1; color:#115e59;">Bay Gate (IoA &ge; 0.50)</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#16a34a; background:#f0fdf4;">
      <div class="step-num" style="background:#16a34a;">3</div>
      <div class="step-h">Vector Match</div>
      <div class="step-sub">768-D RAM Gallery<br>(68% Fast Path)</div>
      <div class="step-pill" style="background:#dcfce7; color:#166534;">gemini-embedding-2</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#d97706; background:#fffbeb;">
      <div class="step-num" style="background:#d97706;">4</div>
      <div class="step-h">VLM Tie-Break</div>
      <div class="step-sub">4-Panel Contact Sheet<br>(32% Hard Ties)</div>
      <div class="step-pill" style="background:#fef3c7; color:#92400e;">gemini-3.5-flash-lite</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#7c3aed; background:#f5f3ff;">
      <div class="step-num" style="background:#7c3aed;">5</div>
      <div class="step-h">Aggregate</div>
      <div class="step-sub">Compute Share of<br>Shelf &amp; Planogram</div>
      <div class="step-pill" style="background:#ede9fe; color:#5b21b6;">12 Task Solvers</div>
    </div>
  </div>
</div>
""",
        [
            OUT_DIR / "fig_pipeline_design_marketshare_merch.png",
            DIAG_DIR / "fig_pipeline_design_marketshare_merch.png",
            DIAG_DIR / "tab8_e2e_funnel_code_flow.png",
        ],
        height=420,
    )

    # 2. E2E Figure 2: Option 3 Left-to-Right 5-Step Production Cloud Flow
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas">
  <div class="top-bar">
    <div class="top-title">Production Request &amp; Data Flow (Shared VPC)</div>
  </div>

  <div class="step-row">
    <div class="step-card" style="border-color:#475569; background:#f8fafc;">
      <div class="step-num" style="background:#475569;">1</div>
      <div class="step-h">Store Upload</div>
      <div class="step-sub">Mobile App Uploads<br>Shelf Photo to GCS</div>
      <div class="step-pill" style="background:#e2e8f0; color:#1e293b;">Internal ALB</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#2563eb; background:#eff6ff;">
      <div class="step-num" style="background:#2563eb;">2</div>
      <div class="step-h">Async Queue</div>
      <div class="step-sub">Buffer &amp; Route<br>Upload Events</div>
      <div class="step-pill" style="background:#dbeafe; color:#1e40af;">Cloud Pub/Sub</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#0d9488; background:#f0fdfa;">
      <div class="step-num" style="background:#0d9488;">3</div>
      <div class="step-h">CPU Workers</div>
      <div class="step-sub">Distilled YOLO11n ONNX<br>&amp; RAM Vector Match</div>
      <div class="step-pill" style="background:#ccfbf1; color:#115e59;">Cloud Run Pool</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#d97706; background:#fffbeb;">
      <div class="step-num" style="background:#d97706;">4</div>
      <div class="step-h">Vertex AI</div>
      <div class="step-sub" style="font-size:12.5px;">gemini-embedding-2<br>&amp; gemini-3.5-flash-lite</div>
      <div class="step-pill" style="background:#fef3c7; color:#92400e;">Private Endpoint</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#16a34a; background:#f0fdf4;">
      <div class="step-num" style="background:#16a34a;">5</div>
      <div class="step-h">Store &amp; Callback</div>
      <div class="step-sub">Write Audit JSON<br>&amp; Return Store Score</div>
      <div class="step-pill" style="background:#dcfce7; color:#166534;">Cloud SQL &amp; BQ</div>
    </div>
  </div>
</div>
""",
        [
            OUT_DIR / "p2p_fig1_production_topology.png",
            DIAG_DIR / "p2p_fig1_production_topology.png",
        ],
        height=420,
    )

    # 3. E2E Figure 3: Option 3 Step Cards for Market Share & Merchandising Task Flows
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas">
  <div class="top-bar" style="margin-bottom:6px;">
    <div class="top-title">How Shared Detections Feed Market Share &amp; Merchandising Tasks</div>
  </div>

  <div class="section-tag" style="color:#1d4ed8;">Row A: Market Share Task Flow (8 Tasks)</div>
  <div class="step-row" style="margin-bottom:14px;">
    <div class="step-card" style="border-color:#2563eb; background:#eff6ff; padding:12px 8px;">
      <div class="step-num" style="background:#2563eb; width:28px; height:28px; line-height:28px; font-size:13px;">1–2</div>
      <div class="step-h" style="font-size:16px;">Shelf &amp; Visit Dedup</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">Distilled YOLO11n Boxes<br>&amp; Multi-Photo Dedup</div>
      <div class="step-pill" style="background:#dbeafe; color:#1e40af;">Unique Facings</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#0284c7; background:#f0f9ff; padding:12px 8px;">
      <div class="step-num" style="background:#0284c7; width:28px; height:28px; line-height:28px; font-size:13px;">3–5</div>
      <div class="step-h" style="font-size:16px;">SKU &amp; Crown Match</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">Catalog SKU Match<br>&amp; Top-20% Crown Zoom</div>
      <div class="step-pill" style="background:#e0f2fe; color:#0369a1;">gemini-embedding-2</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#0d9488; background:#f0fdfa; padding:12px 8px;">
      <div class="step-num" style="background:#0d9488; width:28px; height:28px; line-height:28px; font-size:13px;">6–7</div>
      <div class="step-h" style="font-size:16px;">Variants &amp; Rivals</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">Sister Variant &amp; Open-Set<br>Competitor Separation</div>
      <div class="step-pill" style="background:#ccfbf1; color:#115e59;">gemini-3.5-flash-lite</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#16a34a; background:#f0fdf4; padding:12px 8px;">
      <div class="step-num" style="background:#16a34a; width:28px; height:28px; line-height:28px; font-size:13px;">8</div>
      <div class="step-h" style="font-size:16px;">3D Pack &amp; SoS</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">3D Form Factor &amp;<br>Rogan-Gladen SoS</div>
      <div class="step-pill" style="background:#dcfce7; color:#166534;">SoS &amp; Payout</div>
    </div>
  </div>

  <div class="section-tag" style="color:#0f766e;">Row B: Merchandising Task Flow (4 Tasks)</div>
  <div class="step-row">
    <div class="step-card" style="border-color:#0d9488; background:#f0fdfa; padding:12px 8px;">
      <div class="step-num" style="background:#0d9488; width:28px; height:28px; line-height:28px; font-size:13px;">M1</div>
      <div class="step-h" style="font-size:16px;">Bay Isolation</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">Distilled YOLO11n Bay<br>&amp; Counter Clutter Gate</div>
      <div class="step-pill" style="background:#ccfbf1; color:#115e59;">IoA &ge; 0.50 Gate</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#16a34a; background:#f0fdf4; padding:12px 8px;">
      <div class="step-num" style="background:#16a34a; width:28px; height:28px; line-height:28px; font-size:13px;">M2</div>
      <div class="step-h" style="font-size:16px;">Planogram &amp; OOS</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">Row Graph Sequence,<br>Brand Blocks &amp; OOS</div>
      <div class="step-pill" style="background:#dcfce7; color:#166534;">63 Planogram Rules</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#d97706; background:#fffbeb; padding:12px 8px;">
      <div class="step-num" style="background:#d97706; width:28px; height:28px; line-height:28px; font-size:13px;">M3</div>
      <div class="step-h" style="font-size:16px;">Festive Promos</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">Top/Bottom 25% Band<br>B1G1 &amp; Price-Off OCR</div>
      <div class="step-pill" style="background:#fef3c7; color:#92400e;">gemini-3.5-flash-lite</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#7c3aed; background:#f5f3ff; padding:12px 8px;">
      <div class="step-num" style="background:#7c3aed; width:28px; height:28px; line-height:28px; font-size:13px;">M4</div>
      <div class="step-h" style="font-size:16px;">Promo Talkers</div>
      <div class="step-sub" style="font-size:13px; margin-bottom:8px;">Detect Shelf Danglers<br>&amp; Rupee Price Tags</div>
      <div class="step-pill" style="background:#ede9fe; color:#5b21b6;">YOLO11s Talker ONNX</div>
    </div>
  </div>
</div>
""",
        [
            OUT_DIR / "sequence.png",
            DIAG_DIR / "sequence.png",
            DIAG_DIR / "tab8_usecase_x_task_solver_matrix.png",
        ],
        height=540,
    )

    # 4. Market Share Figure 2: Simpler Option 1 Lucidchart Swimlane & Decision Flowchart
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas lucid-grid">
  <div class="top-bar">
    <div class="top-title">Market Share Vector Retrieval &amp; Tie-Breaker Flow</div>
  </div>

  <svg width="910" height="435" viewBox="0 0 910 435" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="ms-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
      </marker>
      <marker id="ms-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a"/>
      </marker>
      <marker id="ms-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#d97706"/>
      </marker>
    </defs>

    <!-- Lane 1 -->
    <rect x="10" y="8" width="890" height="118" rx="8" fill="#f8fafc" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="29" font-size="12" font-weight="800" fill="#64748b" letter-spacing="0.8">LANE 1: OFFLINE CATALOG GALLERY</text>

    <!-- Lane 2 -->
    <rect x="10" y="140" width="890" height="158" rx="8" fill="#f0fdf4" fill-opacity="0.72" stroke="#bbf7d0" stroke-width="1.5"/>
    <text x="24" y="161" font-size="12" font-weight="800" fill="#15803d" letter-spacing="0.8">LANE 2: ONLINE RETRIEVAL</text>

    <!-- Lane 3 -->
    <rect x="10" y="312" width="890" height="115" rx="8" fill="#eff6ff" fill-opacity="0.78" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="24" y="333" font-size="12" font-weight="800" fill="#1d4ed8" letter-spacing="0.8">LANE 3: TASK AGGREGATION</text>

    <!-- Lane 1 Nodes -->
    <rect x="40" y="42" width="195" height="66" rx="33" fill="#ffffff" stroke="#475569" stroke-width="2.5"/>
    <text x="137" y="71" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">Reference Catalog</text>
    <text x="137" y="92" text-anchor="middle" font-size="13" font-weight="600" fill="#475569">Unilever &amp; Rival Packs</text>

    <rect x="310" y="42" width="240" height="66" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2.5"/>
    <text x="430" y="71" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">1. Embed Reference Angles</text>
    <text x="430" y="92" text-anchor="middle" font-size="13" font-weight="600" fill="#2563eb" font-family="monospace">gemini-embedding-2</text>

    <rect x="630" y="42" width="235" height="66" rx="8" fill="#ffffff" stroke="#0d9488" stroke-width="2.5"/>
    <text x="747" y="71" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">2. In-RAM Vector Gallery</text>
    <text x="747" y="92" text-anchor="middle" font-size="13" font-weight="600" fill="#0d9488">K=3 Centroids (&lt;25 MB RAM)</text>

    <!-- Lane 2 Nodes -->
    <rect x="95" y="184" width="220" height="68" rx="8" fill="#ffffff" stroke="#0d9488" stroke-width="2.5"/>
    <text x="205" y="214" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">3. Vector k-NN Lookup</text>
    <text x="205" y="235" text-anchor="middle" font-size="13" font-weight="600" fill="#0d9488">Cosine + Residual InfoNCE</text>

    <polygon points="465,170 565,218 465,266 365,218" fill="#fefce8" stroke="#ca8a04" stroke-width="2.5"/>
    <text x="465" y="214" text-anchor="middle" font-size="15" font-weight="800" fill="#854d0e">Clear Match?</text>
    <text x="465" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#a16207">Margin Gate</text>

    <rect x="645" y="184" width="225" height="68" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2.5"/>
    <text x="757" y="214" text-anchor="middle" font-size="16" font-weight="800" fill="#92400e">4B. VLM Contact Sheet</text>
    <text x="757" y="235" text-anchor="middle" font-size="13" font-weight="600" fill="#b45309" font-family="monospace">gemini-3.5-flash-lite</text>

    <!-- Lane 3 Nodes -->
    <rect x="130" y="346" width="275" height="66" rx="8" fill="#ffffff" stroke="#16a34a" stroke-width="2.5"/>
    <text x="267" y="375" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">4A. Direct SKU Assignment</text>
    <text x="267" y="396" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">Resolved Directly in RAM</text>

    <rect x="495" y="346" width="315" height="66" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2.5"/>
    <text x="652" y="375" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">5. Compute 8 Market Share Tasks</text>
    <text x="652" y="396" text-anchor="middle" font-size="13" font-weight="600" fill="#1d4ed8">Brand, Variant, Pack &amp; Share of Shelf</text>

    <!-- Connectors -->
    <line x1="235" y1="75" x2="306" y2="75" stroke="#475569" stroke-width="2.5" marker-end="url(#ms-arr)"/>
    <line x1="550" y1="75" x2="626" y2="75" stroke="#475569" stroke-width="2.5" marker-end="url(#ms-arr)"/>
    <path d="M 747 108 L 747 133 L 255 133 L 255 180" fill="none" stroke="#475569" stroke-width="2.5" marker-end="url(#ms-arr)"/>
    <line x1="315" y1="218" x2="361" y2="218" stroke="#475569" stroke-width="2.5" marker-end="url(#ms-arr)"/>

    <!-- Yes (68% Direct) -->
    <path d="M 465 266 L 465 302 L 267 302 L 267 342" fill="none" stroke="#16a34a" stroke-width="3" marker-end="url(#ms-green)"/>
    <rect x="335" y="273" width="115" height="23" rx="4" fill="#dcfce7"/>
    <text x="392" y="289" text-anchor="middle" font-size="13" font-weight="800" fill="#15803d">YES (68% Fast)</text>

    <!-- No (32% VLM) -->
    <line x1="565" y1="218" x2="641" y2="218" stroke="#d97706" stroke-width="3" marker-end="url(#ms-amber)"/>
    <rect x="568" y="189" width="68" height="22" rx="4" fill="#fef3c7"/>
    <text x="602" y="204" text-anchor="middle" font-size="12" font-weight="800" fill="#b45309">NO (32%)</text>

    <!-- Merge into 8 Outputs -->
    <line x1="405" y1="379" x2="491" y2="379" stroke="#16a34a" stroke-width="2.5" marker-end="url(#ms-green)"/>
    <path d="M 757 252 L 757 302 L 652 302 L 652 342" fill="none" stroke="#475569" stroke-width="2.5" marker-end="url(#ms-arr)"/>
  </svg>
</div>
""",
        [
            OUT_DIR / "vector_db_generation_and_4_operations.png",
            DIAG_DIR / "vector_db_generation_and_4_operations.png",
        ],
        height=540,
    )

    # 5. Market Share Figure 3: Pure Onboarding Flow (Simpler Option 1, Zero Legacy Lane)
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas lucid-grid">
  <div class="top-bar">
    <div class="top-title">Zero-Downtime New SKU Catalog Onboarding Flow</div>
  </div>

  <svg width="910" height="175" viewBox="0 0 910 175" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="on-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#0d9488"/>
      </marker>
    </defs>

    <rect x="10" y="10" width="890" height="150" rx="8" fill="#f0fdf4" fill-opacity="0.75" stroke="#86efac" stroke-width="1.5"/>
    <text x="24" y="32" font-size="12" font-weight="800" fill="#15803d" letter-spacing="0.8">CATALOG GALLERY UPDATE FLOW (NO MODEL RETRAINING REQUIRED)</text>

    <rect x="25" y="52" width="180" height="82" rx="41" fill="#ffffff" stroke="#0284c7" stroke-width="2.5"/>
    <text x="115" y="88" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">New Pack Shots</text>
    <text x="115" y="110" text-anchor="middle" font-size="13" font-weight="600" fill="#475569">3–5 Reference Views</text>

    <rect x="245" y="52" width="195" height="82" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2.5"/>
    <text x="342" y="88" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">1. Embed Views</text>
    <text x="342" y="110" text-anchor="middle" font-size="13" font-weight="600" fill="#2563eb" font-family="monospace">gemini-embedding-2</text>

    <rect x="480" y="52" width="195" height="82" rx="8" fill="#ffffff" stroke="#0d9488" stroke-width="2.5"/>
    <text x="577" y="88" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">2. Append Centroids</text>
    <text x="577" y="110" text-anchor="middle" font-size="13" font-weight="600" fill="#0d9488">Update Vector Gallery</text>

    <rect x="715" y="52" width="165" height="82" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2.5"/>
    <text x="797" y="88" text-anchor="middle" font-size="16" font-weight="800" fill="#14532d">3. Live in RAM</text>
    <text x="797" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">Active in &lt;60s</text>

    <line x1="205" y1="93" x2="241" y2="93" stroke="#0d9488" stroke-width="2.5" marker-end="url(#on-arr)"/>
    <line x1="440" y1="93" x2="476" y2="93" stroke="#0d9488" stroke-width="2.5" marker-end="url(#on-arr)"/>
    <line x1="675" y1="93" x2="711" y2="93" stroke="#0d9488" stroke-width="2.5" marker-end="url(#on-arr)"/>
  </svg>
</div>
""",
        [
            OUT_DIR / "onboarding.png",
            DIAG_DIR / "onboarding.png",
        ],
        height=320,
    )

    # 6. Merchandising Figure 1: Simpler Option 1 Lucidchart Swimlane & Decision Flowchart
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas lucid-grid">
  <div class="top-bar">
    <div class="top-title">Merchandising &amp; Promo Verification Flow</div>
  </div>

  <svg width="910" height="435" viewBox="0 0 910 435" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="m-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
      </marker>
      <marker id="m-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a"/>
      </marker>
      <marker id="m-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>
      </marker>
    </defs>

    <!-- Lane 1 -->
    <rect x="10" y="8" width="890" height="118" rx="8" fill="#f8fafc" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="29" font-size="12" font-weight="800" fill="#64748b" letter-spacing="0.8">LANE 1: STAGE 1 DETECTION</text>

    <!-- Lane 2 -->
    <rect x="10" y="140" width="890" height="158" rx="8" fill="#f0fdf4" fill-opacity="0.72" stroke="#bbf7d0" stroke-width="1.5"/>
    <text x="24" y="161" font-size="12" font-weight="800" fill="#15803d" letter-spacing="0.8">LANE 2: STAGE 1.5 BAY GATE</text>

    <!-- Lane 3 -->
    <rect x="10" y="312" width="890" height="115" rx="8" fill="#eff6ff" fill-opacity="0.78" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="24" y="333" font-size="12" font-weight="800" fill="#1d4ed8" letter-spacing="0.8">LANE 3: COMPLIANCE SOLVERS</text>

    <!-- Lane 1 Nodes -->
    <rect x="40" y="42" width="180" height="66" rx="33" fill="#ffffff" stroke="#0284c7" stroke-width="2.5"/>
    <text x="130" y="71" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">Display Photo</text>
    <text x="130" y="92" text-anchor="middle" font-size="13" font-weight="600" fill="#475569">Store Display Upload</text>

    <rect x="290" y="42" width="250" height="66" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2.5"/>
    <text x="415" y="71" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">1. Detect Bays &amp; SKUs</text>
    <text x="415" y="92" text-anchor="middle" font-size="13" font-weight="600" fill="#2563eb">Distilled YOLO11n ONNX</text>

    <rect x="615" y="42" width="255" height="66" rx="8" fill="#ffffff" stroke="#7c3aed" stroke-width="2.5"/>
    <text x="742" y="71" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">Task 4: Promo Talkers</text>
    <text x="742" y="92" text-anchor="middle" font-size="13" font-weight="600" fill="#6d28d9">YOLO11s 3-Class ONNX</text>

    <!-- Lane 2 Nodes -->
    <polygon points="415,168 520,218 415,268 310,218" fill="#fefce8" stroke="#ca8a04" stroke-width="2.5"/>
    <text x="415" y="214" text-anchor="middle" font-size="15" font-weight="800" fill="#854d0e">Inside Bay?</text>
    <text x="415" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#a16207">IoA &ge; 0.50</text>

    <rect x="625" y="184" width="245" height="68" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="2.5"/>
    <text x="747" y="214" text-anchor="middle" font-size="16" font-weight="800" fill="#991b1b">Task 2: Prune Clutter</text>
    <text x="747" y="235" text-anchor="middle" font-size="13" font-weight="600" fill="#b91c1c">Out-of-Bay Items Ignored</text>

    <rect x="45" y="184" width="210" height="68" rx="8" fill="#ffffff" stroke="#0d9488" stroke-width="2.5"/>
    <text x="150" y="214" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">2. In-Bay Crops</text>
    <text x="150" y="235" text-anchor="middle" font-size="13" font-weight="600" fill="#0d9488">Scoped to Display Rack</text>

    <!-- Lane 3 Nodes -->
    <rect x="115" y="346" width="315" height="66" rx="8" fill="#ffffff" stroke="#16a34a" stroke-width="2.5"/>
    <text x="272" y="375" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">Task 1: Bay Planogram &amp; OOS</text>
    <text x="272" y="396" text-anchor="middle" font-size="13" font-weight="600" fill="#15803d">Row Graph, Brand Blocks &amp; Stockouts</text>

    <rect x="500" y="346" width="315" height="66" rx="8" fill="#ffffff" stroke="#d97706" stroke-width="2.5"/>
    <text x="657" y="375" text-anchor="middle" font-size="16" font-weight="800" fill="#0f172a">Task 3: Festive Promo Packs</text>
    <text x="657" y="396" text-anchor="middle" font-size="13" font-weight="600" fill="#b45309">gemini-3.5-flash-lite Promo Band OCR</text>

    <!-- Connectors -->
    <line x1="220" y1="75" x2="286" y2="75" stroke="#475569" stroke-width="2.5" marker-end="url(#m-arr)"/>
    <line x1="540" y1="75" x2="611" y2="75" stroke="#475569" stroke-width="2.5" marker-end="url(#m-arr)"/>
    <line x1="415" y1="108" x2="415" y2="164" stroke="#475569" stroke-width="2.5" marker-end="url(#m-arr)"/>

    <!-- No (61% Clutter) -->
    <line x1="520" y1="218" x2="621" y2="218" stroke="#dc2626" stroke-width="3" marker-end="url(#m-red)"/>
    <rect x="532" y="189" width="76" height="22" rx="4" fill="#fee2e2"/>
    <text x="570" y="204" text-anchor="middle" font-size="12" font-weight="800" fill="#b91c1c">NO (61%)</text>

    <!-- Yes (39% Retained) -->
    <line x1="310" y1="218" x2="259" y2="218" stroke="#16a34a" stroke-width="3" marker-end="url(#m-green)"/>
    <rect x="263" y="189" width="44" height="22" rx="4" fill="#dcfce7"/>
    <text x="285" y="204" text-anchor="middle" font-size="12" font-weight="800" fill="#15803d">YES</text>

    <!-- To Lane 3 -->
    <path d="M 150 252 L 150 302 L 272 302 L 272 342" fill="none" stroke="#16a34a" stroke-width="2.5" marker-end="url(#m-green)"/>
    <path d="M 272 302 L 657 302 L 657 342" fill="none" stroke="#16a34a" stroke-width="2.5" marker-end="url(#m-green)"/>
  </svg>
</div>
""",
        [
            OUT_DIR / "merch_fig1_pipeline_architecture.png",
            DIAG_DIR / "merch_fig1_pipeline_architecture.png",
        ],
        height=540,
    )

    # 7. Merchandising Figure 2: Pure Stage 1.5 Bay Overlap Gate Flow (No Legacy Comparison Lane)
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas lucid-grid">
  <div class="top-bar">
    <div class="top-title">Stage 1.5 Display Boundary Filtering Flow</div>
  </div>

  <svg width="910" height="230" viewBox="0 0 910 230" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="bg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
      </marker>
      <marker id="bg-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a"/>
      </marker>
      <marker id="bg-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>
      </marker>
    </defs>

    <!-- Inputs -->
    <rect x="20" y="20" width="215" height="72" rx="8" fill="#ffffff" stroke="#0284c7" stroke-width="2.5"/>
    <text x="127" y="51" text-anchor="middle" font-size="15" font-weight="800" fill="#0f172a">Class 0: Display Bay Box</text>
    <text x="127" y="72" text-anchor="middle" font-size="13" font-weight="600" fill="#0284c7">Distilled YOLO11n (Class 0)</text>

    <rect x="20" y="135" width="215" height="72" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2.5"/>
    <text x="127" y="166" text-anchor="middle" font-size="15" font-weight="800" fill="#0f172a">Class 1: Product Boxes</text>
    <text x="127" y="187" text-anchor="middle" font-size="13" font-weight="600" fill="#2563eb">Distilled YOLO11n (Class 1)</text>

    <!-- Overlap Node -->
    <rect x="290" y="78" width="185" height="72" rx="8" fill="#ffffff" stroke="#475569" stroke-width="2.5"/>
    <text x="382" y="109" text-anchor="middle" font-size="15" font-weight="800" fill="#0f172a">Compute Overlap</text>
    <text x="382" y="130" text-anchor="middle" font-size="13" font-weight="600" fill="#475569">Area(SKU &cap; Bay) / SKU</text>

    <!-- Diamond -->
    <polygon points="585,62 675,114 585,166 495,114" fill="#fefce8" stroke="#ca8a04" stroke-width="2.5"/>
    <text x="585" y="110" text-anchor="middle" font-size="14" font-weight="800" fill="#854d0e">IoA &ge; 0.50?</text>
    <text x="585" y="128" text-anchor="middle" font-size="12" font-weight="700" fill="#a16207">Inside Rack</text>

    <!-- Outputs -->
    <rect x="720" y="20" width="175" height="72" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2.5"/>
    <text x="807" y="51" text-anchor="middle" font-size="15" font-weight="800" fill="#14532d">Keep In-Bay SKUs</text>
    <text x="807" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">39% Retained</text>

    <rect x="720" y="135" width="175" height="72" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="2.5"/>
    <text x="807" y="166" text-anchor="middle" font-size="15" font-weight="800" fill="#991b1b">Drop Clutter</text>
    <text x="807" y="187" text-anchor="middle" font-size="13" font-weight="700" fill="#b91c1c">61% Pruned Early</text>

    <!-- Connectors -->
    <path d="M 235 56 L 262 56 L 262 102 L 286 102" fill="none" stroke="#475569" stroke-width="2.5" marker-end="url(#bg-arr)"/>
    <path d="M 235 171 L 262 171 L 262 126 L 286 126" fill="none" stroke="#475569" stroke-width="2.5" marker-end="url(#bg-arr)"/>
    <line x1="475" y1="114" x2="491" y2="114" stroke="#475569" stroke-width="2.5" marker-end="url(#bg-arr)"/>

    <path d="M 585 62 L 585 56 L 716 56" fill="none" stroke="#16a34a" stroke-width="3" marker-end="url(#bg-green)"/>
    <text x="648" y="46" text-anchor="middle" font-size="13" font-weight="800" fill="#15803d">YES</text>

    <path d="M 585 166 L 585 171 L 716 171" fill="none" stroke="#dc2626" stroke-width="3" marker-end="url(#bg-red)"/>
    <text x="648" y="162" text-anchor="middle" font-size="13" font-weight="800" fill="#b91c1c">NO</text>
  </svg>
</div>
""",
        [
            OUT_DIR / "shelf_bay_vs_sku_example.png",
            DIAG_DIR / "shelf_bay_vs_sku_example.png",
            DIAG_DIR / "_store_bay_crop.png",
        ],
        height=360,
    )

    # 8. E2E Figure 6: Option 3 Left-to-Right 5-Step Canary Rollout Flow
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas">
  <div class="top-bar">
    <div class="top-title">Stratified Cluster Canary &amp; Production Rollout Flow</div>
  </div>

  <div class="step-row">
    <div class="step-card" style="border-color:#475569; background:#f8fafc;">
      <div class="step-num" style="background:#475569;">1</div>
      <div class="step-h">Cluster Lock</div>
      <div class="step-sub">Hash Store Cluster<br>to Fixed Bucket</div>
      <div class="step-pill" style="background:#e2e8f0; color:#1e293b;">30-Day P2P Lock</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#2563eb; background:#eff6ff;">
      <div class="step-num" style="background:#2563eb;">2</div>
      <div class="step-h">Shadow Mode</div>
      <div class="step-sub">Mirror Traffic<br>Silently in Parallel</div>
      <div class="step-pill" style="background:#dbeafe; color:#1e40af;">0% Live Impact</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#0d9488; background:#f0fdfa;">
      <div class="step-num" style="background:#0d9488;">3</div>
      <div class="step-h">10% Canary</div>
      <div class="step-sub">Route Initial<br>Store Clusters</div>
      <div class="step-pill" style="background:#ccfbf1; color:#115e59;">Initial Cutover</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#d97706; background:#fffbeb;">
      <div class="step-num" style="background:#d97706;">4</div>
      <div class="step-h">50% Scale</div>
      <div class="step-sub">Verify Parity &amp;<br>SLA Guardrails</div>
      <div class="step-pill" style="background:#fef3c7; color:#92400e;">Regional Scale</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#16a34a; background:#f0fdf4;">
      <div class="step-num" style="background:#16a34a;">5</div>
      <div class="step-h">100% GA</div>
      <div class="step-sub">Full Production<br>with Auto-Rollback</div>
      <div class="step-pill" style="background:#dcfce7; color:#166534;">Full Cutover</div>
    </div>
  </div>
</div>
""",
        [
            OUT_DIR / "p2p_fig2_canary_routing_flow.png",
            DIAG_DIR / "p2p_fig2_canary_routing_flow.png",
        ],
        height=420,
    )

    # 9. Model Whitelisting Tier Matrix: Pure Option 3 3-Step Production Model Flow
    render_html_to_png(
        lucid_css + """
<div class="lucid-canvas">
  <div class="top-bar">
    <div class="top-title">Production AI Model Execution Flow (Shared VPC)</div>
  </div>

  <div class="step-row">
    <div class="step-card" style="border-color:#2563eb; background:#eff6ff; padding:18px 16px;">
      <div class="step-num" style="background:#2563eb;">1</div>
      <div class="step-h">Distilled YOLO11n &amp; YOLO11s</div>
      <div class="step-sub">960px ONNX Detection of<br>Product Boxes, Bays &amp; Talkers</div>
      <div class="step-pill" style="background:#dbeafe; color:#1e40af;">Cloud Run CPU ONNX</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#0d9488; background:#f0fdfa; padding:18px 16px;">
      <div class="step-num" style="background:#0d9488;">2</div>
      <div class="step-h">gemini-embedding-2</div>
      <div class="step-sub">768-D Vector Retrieval Against<br>In-RAM Catalog Gallery</div>
      <div class="step-pill" style="background:#ccfbf1; color:#115e59;">Vertex AI PSC (68%)</div>
    </div>
    <div class="step-arr">&rarr;</div>
    <div class="step-card" style="border-color:#16a34a; background:#f0fdf4; padding:18px 16px;">
      <div class="step-num" style="background:#16a34a;">3</div>
      <div class="step-h">gemini-3.5-flash-lite</div>
      <div class="step-sub">Contact-Sheet Tie-Breaker for<br>Lookalike Twins &amp; Promo Text</div>
      <div class="step-pill" style="background:#dcfce7; color:#166534;">Vertex AI PSC (32%)</div>
    </div>
  </div>
</div>
""",
        [
            OUT_DIR / "whitelist_tier_matrix.png",
            DIAG_DIR / "whitelist_tier_matrix.png",
        ],
        height=420,
    )


if __name__ == "__main__":
    render_matplotlib_finops_deepdive()
    render_matplotlib_cost_breakdown()
    render_matplotlib_locust_and_rollout()
    build_all_engineering_diagrams()
