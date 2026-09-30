"""``shelf-bench`` command line.

    shelf-bench bootstrap                        # new GCP project: APIs, bucket, registry, data
    shelf-bench download                         # fetch + extract SKU-110K (~12 GB)
    shelf-bench list                             # approaches and models
    shelf-bench run -a single_pass_dedup -m gemini-3.5-flash-lite
    shelf-bench run -a single_pass_dedup tiled_dedup -m gemini-3.5-flash-lite gemini-3.8-flash
    shelf-bench run -a hierarchy_classify embedding_retrieval -m gemini-3.5-flash-lite gemini-embedding-2-preview
    shelf-bench cloud-run -a shelf_detect_retrieve -m gemini-3.5-flash-lite -t standard flex
    shelf-bench leaderboard
    shelf-bench serve                            # leaderboard UI on http://localhost:8080
    shelf-bench serve-cloud                      # the same UI as a private Cloud Run service
    shelf-bench proxy                            # open that private service on http://localhost:8083

Every approach runs only the models it really calls (``list`` shows them); other
approach x model combinations are skipped.
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

    s = sub.add_parser("upload", help="copy the local dataset to GCS (resumable)")
    s.add_argument("--root", default=dataset.LOCAL_ROOT)
    s.add_argument("--to", default=cfg.get("gcp", {}).get("data"))

    s = sub.add_parser("rpc", help="fetch the RPC checkout subset (~40 MB) and copy it to GCS")
    s.add_argument("--no-upload", action="store_true", help="keep it local only")
    s = sub.add_parser("shelves", help="build the HoloSelecta shelves set from its photos + VOC "
                                       "files and copy it to GCS")
    s.add_argument("raw", help="folder with HoloSelecta's *.jpg and *.xml (its Drive folder)")
    s.add_argument("--no-upload", action="store_true", help="keep it local only")

    sub.add_parser("list", help="list approaches and models")

    s = sub.add_parser("run", help="evaluate approach x model combinations")
    s.add_argument("-a", "--approach", nargs="+", required=True)
    s.add_argument("-m", "--model", nargs="+", required=True)
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["standard"]),
                   help="Gemini PayGo tier(s): standard, priority (1.8x) and/or flex (0.5x)")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")
    s.add_argument("--root", default=dataset.DEFAULT_ROOT, help="SKU-110K: local dir or gs:// URI")
    s.add_argument("--results", default=str(runner.RESULTS_DIR), help="local dir or gs:// URI")

    s = sub.add_parser("cloud-run", help="run approach x model combinations on Cloud Run")
    s.add_argument("-a", "--approach", nargs="+", required=True)
    s.add_argument("-m", "--model", nargs="+", required=True)
    s.add_argument("-t", "--tier", nargs="+", choices=TIERS, default=d.get("tier", ["standard"]),
                   help="Gemini PayGo tier(s): standard, priority (1.8x) and/or flex (0.5x)")
    s.add_argument("--split", default=d.get("split", "test"), choices=dataset.SPLITS)
    s.add_argument("--limit", type=int, default=d.get("limit", 50), help="0 = whole split")
    s.add_argument("--seed", type=int, default=d.get("seed", 0))
    s.add_argument("--workers", type=int, default=d.get("workers", 4))
    s.add_argument("--owner", default=None, help="defaults to $USER")

    sub.add_parser("pull", help="copy finished Cloud Run results from GCS into results/")

    s = sub.add_parser("leaderboard", help="print the leaderboard")
    s.add_argument("--results", type=Path, default=runner.RESULTS_DIR)

    s = sub.add_parser("serve", help="serve the leaderboard UI")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--results", type=Path, default=runner.RESULTS_DIR)
    s.add_argument("--root", default=dataset.DEFAULT_ROOT, help="local dir or gs:// URI")
    s.add_argument("--pull", action="store_true", help="first pull finished runs from GCS")

    sub.add_parser("serve-cloud", help="deploy the leaderboard UI (with results/) to Cloud Run")
    s = sub.add_parser("proxy", help="open the private Cloud Run UI on localhost (authenticated)")
    s.add_argument("--port", type=int, default=8083)
    s.add_argument("--host", default="127.0.0.1")
    sub.add_parser("bootstrap", help="prepare a new GCP project (APIs, bucket, registry, data)")

    a = p.parse_args(argv)

    if a.cmd == "download":
        dataset.download(a.root)
    elif a.cmd == "upload":
        dataset.upload(a.root, a.to)
    elif a.cmd == "rpc":
        dataset.prepare_rpc()
        if not a.no_upload:
            dataset.upload(dataset.RPC_LOCAL, cfg["gcp"]["rpc"])
    elif a.cmd == "shelves":
        dataset.prepare_shelves(a.raw)
        if not a.no_upload:
            dataset.upload(dataset.SHELVES_LOCAL, cfg["gcp"]["shelves"])
    elif a.cmd == "list":
        print("Approaches (use case / task / dataset / models it runs):")
        for name, ap in approaches.all_approaches().items():
            print(f"  {name:<20} {ap.use_case:<13} {ap.task:<15} {ap.dataset:<9} "
                  f"{' + '.join([', '.join(ap.models) if ap.models else 'any Gemini', *ap.also_calls])}  {ap.architecture}")
        print("Gemini models:")
        for m in cfg.get("models", []):
            print(f"  {m}")
    elif a.cmd == "run":
        combos = runner.combos(a.approach, a.tier, a.model)  # fails fast on typos before spending
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
        return 1 if failed or not combos else 0
    elif a.cmd == "cloud-run":
        from utils import cloud

        return cloud.run_on_cloud(a.approach, a.model, a.tier, a.split, a.limit, a.seed, a.workers,
                                  a.owner or getpass.getuser())
    elif a.cmd == "pull":
        print(f"Pulled {runner.pull(cfg['gcp']['results'])} new run(s) into results/")
    elif a.cmd == "leaderboard":
        print_leaderboard(runner.leaderboard(a.results))
    elif a.cmd == "serve":
        from utils.server import serve

        if a.pull:
            print(f"Pulled {runner.pull(cfg['gcp']['results'], a.results)} new run(s) into {a.results}/")
        serve(a.host, a.port, a.results, a.root)
    elif a.cmd == "serve-cloud":
        from utils import cloud

        cloud.deploy_ui()
    elif a.cmd == "proxy":
        from utils import cloud

        cloud.proxy_ui(port=a.port, host=a.host)
    elif a.cmd == "bootstrap":
        from utils import cloud

        cloud.bootstrap()
    return 0



def print_leaderboard(rows: list[dict]) -> None:
    if not rows:
        print("No runs yet. Try: shelf-bench run -a single_pass_dedup -m gemini-3.5-flash-lite")
        return
    hdr = f"{'#':>3}  {'run id':<56} {'owner':<10} {'acc':>5} {'rec':>5} {'F2':>5} {'p95':>6} {'p99':>6} {'₹/img':>8}"
    use_case = task = None
    for r in rows:  # sorted by use case and task, ranked within each
        if r["use_case"] != use_case:
            use_case, task = r["use_case"], None
            print(f"\n=== {use_case.replace('_', ' ').title()} ===")
        if (r["task"], r["dataset"]) != task:
            task = (r["task"], r["dataset"])
            label = f"shelves_{r['task']}" if r["dataset"] == "shelves" else r["task"]
            by = "ranked by F2" if r["task"] in ("detection", "end_to_end") else "ranked by exact product accuracy"
            print(f"\n{label} ({by})")
            print(hdr)
            print("-" * len(hdr))
        print(f"{r['rank']:>3}  {r['run_id']:<56} {r['owner'][:10]:<10} {r['accuracy']:>5.3f} "
              f"{r['recall']:>5.3f} {r['f2']:>5.3f} {r['p95_latency_s']:>5.1f}s "
              f"{r['p99_latency_s']:>5.1f}s {r['cost_per_image_inr']:>8.3f}")


if __name__ == "__main__":
    sys.exit(main())
