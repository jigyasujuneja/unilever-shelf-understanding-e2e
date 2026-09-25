"""``shelf-bench`` command line.

    shelf-bench download                         # fetch + extract SKU-110K (~12 GB)
    shelf-bench list                             # approaches and models
    shelf-bench run -a detect_classify -m gemini-3.8-flash
    shelf-bench run -a single_pass detect_classify -m gemini-3.8-flash gemini-3.5-flash-lite
    shelf-bench cloud-run -a single_pass -m gemini-3.8-flash -t standard priority
    shelf-bench leaderboard
    shelf-bench serve                            # leaderboard UI on http://localhost:8080
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import approaches
import runner
from utils import dataset
from utils.llm import TIERS, load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    d = cfg.get("defaults", {})
    p = argparse.ArgumentParser(prog="shelf-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download and extract SKU-110K")
    s.add_argument("--root", default=dataset.LOCAL_ROOT)

    s = sub.add_parser("upload", help="copy datasets (SKU-110K, HUL labeled, catalog) to Argolis GCS")
    s.add_argument("--root", default=dataset.LOCAL_ROOT)
    s.add_argument("--to", default=cfg.get("gcp", {}).get("data"))
    s.add_argument("--dataset", default="all", choices=["all", "sku110k", "hul_labeled", "catalog"],
                   help="which dataset bundle to sync to Argolis GCS")

    sub.add_parser("splits", help="build and verify the SHA-256 locked Train/Val/Test split manifest")

    sub.add_parser("list", help="list approaches and models")

    s = sub.add_parser("run", help="evaluate approach x model combinations")
    s.add_argument("-a", "--approach", nargs="+", required=True)
    s.add_argument("-m", "--model", nargs="+", required=True)
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["standard"]),
                   help="Gemini PayGo tier(s): standard and/or priority")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")
    s.add_argument("--root", default=dataset.DEFAULT_ROOT, help="local dir or gs:// URI")
    s.add_argument("--results", default=str(runner.RESULTS_DIR), help="local dir or gs:// URI")

    s = sub.add_parser("cloud-run", help="run approach x model combinations on Cloud Run")
    s.add_argument("-a", "--approach", nargs="+", required=True)
    s.add_argument("-m", "--model", nargs="+", required=True)
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["standard"]),
                   help="Gemini PayGo tier(s): standard and/or priority")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")

    s = sub.add_parser("cloud-service", help="deploy the Always-On Cloud Run Web Service on Argolis")
    s.add_argument("--port", type=int, default=8080)

    sub.add_parser("pull", help="copy finished Cloud Run results from GCS into results/")

    s = sub.add_parser("leaderboard", help="print the leaderboard")
    s.add_argument("--results", type=Path, default=runner.RESULTS_DIR)

    s = sub.add_parser("serve", help="serve the leaderboard UI")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--results", type=Path, default=runner.RESULTS_DIR)
    s.add_argument("--root", default=dataset.DEFAULT_ROOT, help="local dir or gs:// URI")

    a = p.parse_args(argv)

    if a.cmd == "download":
        dataset.download(a.root)
    elif a.cmd == "upload":
        dataset.upload(a.root, a.to, dataset_target=a.dataset)
    elif a.cmd == "splits":
        m = dataset.build_and_verify_splits_manifest()
        print(f"Verified SHA-256 Split Manifest ({m['split_sha256'][:16]}): "
              f"train={m['splits']['train']['image_count']} imgs, "
              f"val={m['splits']['val']['image_count']} imgs, "
              f"test={m['splits']['test']['image_count']} imgs (zero_leakage={m['zero_leakage_verified']})")
    elif a.cmd == "list":
        print("Approaches:")
        for name, ap in approaches.all_approaches().items():
            print(f"  {name:<28} {ap.architecture}")
        print("Models (any Vertex Gemini id works):")
        for m in cfg.get("models", []):
            print(f"  {m}")
    elif a.cmd == "run":
        for name in a.approach:
            approaches.get(name)  # fail fast on typos before spending money
        combos = [(n, t, m) for n in a.approach for t in a.tier for m in a.model]
        if int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1")) > 1:  # one combo per Cloud Run task
            combos = [combos[int(os.environ["CLOUD_RUN_TASK_INDEX"])]]
        failed = 0
        for name, tier, model in combos:
            try:
                runner.run(name, model, a.split, a.limit, a.seed, a.workers, a.owner,
                           results_dir=a.results, data_root=a.root, tier=tier)
            except Exception as e:
                failed += 1
                print(f"[{name} x {model} x {tier}] FAILED: {e}", file=sys.stderr)
        return 1 if failed else 0
    elif a.cmd == "cloud-run":
        from utils import cloud

        for name in a.approach:
            approaches.get(name)
        return cloud.run_on_cloud(a.approach, a.model, a.tier, a.split, a.limit, a.seed, a.workers,
                                  a.owner or getpass.getuser())
    elif a.cmd == "cloud-service":
        from utils import cloud

        cloud.deploy_cloud_service(port=a.port)
    elif a.cmd == "pull":
        print(f"Pulled {runner.pull(cfg['gcp']['results'])} new run(s) into results/")
    elif a.cmd == "leaderboard":
        print_leaderboard(runner.leaderboard(a.results))
    elif a.cmd == "serve":
        from utils.server import serve

        serve(a.host, a.port, a.results, a.root)
    return 0



def print_leaderboard(rows: list[dict]) -> None:
    if not rows:
        print("No runs yet. Try: shelf-bench run -a single_pass -m gemini-3.8-flash")
        return
    hdr = f"{'#':>2}  {'run id':<50} {'owner':<10} {'acc':>5} {'rec':>5} {'F2':>5} {'p95':>6} {'p99':>6} {'₹/img':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rank']:>2}  {r['run_id']:<50} {r['owner'][:10]:<10} {r['accuracy']:>5.3f} "
              f"{r['recall']:>5.3f} {r['f2']:>5.3f} {r['p95_latency_s']:>5.1f}s "
              f"{r['p99_latency_s']:>5.1f}s {r['cost_per_image_inr']:>8.3f}")


if __name__ == "__main__":
    sys.exit(main())
