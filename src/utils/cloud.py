"""Run benchmarks on Cloud Run (``shelf-bench cloud-run``).

1. Build the container with Cloud Build (source -> Artifact Registry).
2. Create/update the Cloud Run job, with one task per approach x tier x model.
3. Execute it. Every combination runs in its own container, in parallel, reading the
   dataset from GCS and writing results to GCS.
4. Pull the results into ``results/`` and finalize compute cost from real task durations.

Uses Application Default Credentials and REST APIs only, so it doesn't need the gcloud CLI.
"""

from __future__ import annotations

import io
import math
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import google.auth
from google.auth.transport.requests import AuthorizedSession

import runner
from utils import dataset
from utils.llm import load_config

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ["Dockerfile", "pyproject.toml", "README.md", "config.yaml", "configs", "src"]


def _session() -> AuthorizedSession:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds)


def _ok(r):
    if r.status_code >= 400:
        raise RuntimeError(f"{r.request.method} {r.url} -> {r.status_code}: {r.text[:800]}")
    return r.json() if r.content else {}


def _ts(s: str) -> datetime:
    # RFC 3339 with up to nanoseconds, e.g. 2026-09-24T20:24:01.123456789Z
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".")
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def task_seconds(env: dict, session: AuthorizedSession | None = None) -> float:
    """Billed duration of the Cloud Run task that produced a run.

    Cloud Run jobs bill vCPU and memory from task start to task completion, rounded up to the
    nearest 100 ms. Both timestamps come from the Cloud Run Admin API task resource.
    """
    project, region = load_config()["gcp"]["project"], env["region"]
    url = (f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/"
           f"{env['job']}/executions/{env['execution']}/tasks")
    s = session or _session()
    tasks, token = [], ""
    while True:
        d = _ok(s.get(url, params={"pageToken": token} if token else {}))
        tasks += d.get("tasks", [])
        token = d.get("nextPageToken")
        if not token:
            break
    task = next((t for t in tasks if int(t.get("index", 0)) == env["task_index"]), None)
    if not task or not task.get("startTime") or not task.get("completionTime"):
        raise RuntimeError(f"task {env['task_index']} of {env['execution']} has not completed")
    secs = (_ts(task["completionTime"]) - _ts(task["startTime"])).total_seconds()
    return math.ceil(secs * 10) / 10


def build_image(s: AuthorizedSession, project: str, region: str, log=print) -> str:
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    image = f"{region}-docker.pkg.dev/{project}/cloud-run-source-deploy/shelf-bench:{tag}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in SOURCE_FILES:
            tar.add(ROOT / name, arcname=name,
                    filter=lambda t: None if "__pycache__" in t.name or ".egg-info" in t.name else t)
    bucket, obj = f"run-sources-{project}-{region}", f"shelf-bench/source-{tag}.tgz"
    dataset._gcs().bucket(bucket).blob(obj).upload_from_string(buf.getvalue())
    log(f"Building {image} with Cloud Build ...")
    op = _ok(s.post(
        f"https://cloudbuild.googleapis.com/v1/projects/{project}/locations/{region}/builds",
        json={"source": {"storageSource": {"bucket": bucket, "object": obj}},
              "steps": [{"name": "gcr.io/cloud-builders/docker", "args": ["build", "-t", image, "."]}],
              "images": [image],
              "options": {"logging": "CLOUD_LOGGING_ONLY"}}))
    build_id = op["metadata"]["build"]["id"]
    url = f"https://cloudbuild.googleapis.com/v1/projects/{project}/locations/{region}/builds/{build_id}"
    while True:
        b = _ok(s.get(url))
        if b["status"] in ("SUCCESS", "FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED"):
            break
        time.sleep(10)
    if b["status"] != "SUCCESS":
        raise RuntimeError(f"Build {b['status']}: {b.get('logUrl')}")
    log(f"  built in {b.get('timing', {}).get('BUILD', {}).get('endTime', '')[:19]}")
    return image


def deploy_job(s, project, region, image, args, tasks, cfg) -> str:
    cr = cfg.get("cloud_run", {})
    job = cr.get("job", "shelf-bench")
    base = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs"
    body = {
        "template": {
            "taskCount": tasks,
            # Tasks run approach-major / model-minor, so with parallelism = #models the runs
            # executing together each use a different model and don't fight over quota.
            "parallelism": min(tasks, int(cr.get("parallelism", tasks))),
            "template": {
                "maxRetries": 0,
                "timeout": "3600s",
                "containers": [{
                    "image": image,
                    "args": args,
                    "resources": {"limits": {"cpu": str(cr.get("cpu", 2)),
                                             "memory": f"{cr.get('memory_gib', 4)}Gi"}},
                }],
                # Direct VPC egress (e.g. to a private-IP AlloyDB); public traffic still goes direct.
                **({"vpcAccess": {"networkInterfaces": [{"network": cr["network"],
                                                         "subnetwork": cr.get("subnet", cr["network"])}],
                                  "egress": "PRIVATE_RANGES_ONLY"}} if cr.get("network") else {}),
            },
        },
    }
    exists = s.get(f"{base}/{job}").status_code == 200
    op = _ok(s.patch(f"{base}/{job}", json=body) if exists
             else s.post(f"{base}?jobId={job}", json=body))
    _wait_op(s, op["name"])
    return f"{base}/{job}"


def _wait_op(s, name: str, log=None, every: int = 5) -> dict:
    while True:
        op = _ok(s.get(f"https://run.googleapis.com/v2/{name}"))
        if log:
            md = op.get("metadata", {})
            log(f"  tasks: {md.get('succeededCount', 0)} succeeded, "
                f"{md.get('failedCount', 0)} failed, {md.get('runningCount', 0)} running")
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"Cloud Run: {op['error']}")
            return op
        time.sleep(every)


def run_on_cloud(approaches: list[str], models: list[str], tiers: list[str], split: str, limit: int, seed: int,
                 workers: int, owner: str, log=print) -> int:
    cfg = load_config()
    gcp = cfg["gcp"]
    project, region = gcp["project"], gcp["region"]
    s = _session()
    image = build_image(s, project, region, log)
    tasks = len(approaches) * len(tiers) * len(models)
    args = ["run", "-a", *approaches, "-m", *models, "-t", *tiers, "--split", split, "--limit", str(limit),
            "--seed", str(seed), "--workers", str(workers), "--owner", owner,
            "--root", gcp["data"], "--results", gcp["results"]]
    cfg.setdefault("cloud_run", {}).setdefault("parallelism", len(models))
    job_url = deploy_job(s, project, region, image, args, tasks, cfg)
    log(f"Executing {tasks} tasks on Cloud Run job {job_url.rsplit('/', 1)[-1]} ({region}) ...")
    op = _ok(s.post(f"{job_url}:run", json={}))
    log(f"  execution: https://console.cloud.google.com/run/jobs/executions/details/{region}/"
        f"{op.get('metadata', {}).get('name', '').rsplit('/', 1)[-1]}?project={project}")
    try:
        _wait_op(s, op["name"], log=log, every=20)
    finally:
        n = runner.pull(gcp["results"])
        log(f"Pulled {n} new run(s) into results/")
    return 0


def deploy_cloud_service(port: int = 8080, log=print) -> str:
    """Build and deploy the Always-On Cloud Run Web Service (`perfect-store-control-plane`) on Argolis."""
    cfg = load_config()
    gcp = cfg["gcp"]
    cr = cfg.get("cloud_run", {})
    project, region = gcp["project"], gcp["region"]
    service_name = cr.get("service", "perfect-store-control-plane")
    s = _session()
    image = build_image(s, project, region, log)
    base = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services"
    body = {
        "template": {
            "containers": [{
                "image": image,
                "args": ["serve", "--host", "0.0.0.0", "--port", str(port)],
                "ports": [{"containerPort": port}],
                "resources": {"limits": {"cpu": str(cr.get("cpu", 2)),
                                         "memory": f"{cr.get('memory_gib', 4)}Gi"}},
            }],
        },
    }
    exists = s.get(f"{base}/{service_name}").status_code == 200
    op = _ok(s.patch(f"{base}/{service_name}", json=body) if exists
             else s.post(f"{base}?serviceId={service_name}", json=body))
    done_op = _wait_op(s, op["name"])
    uri = done_op.get("response", {}).get("uri", f"https://{service_name}-{region}.a.run.app")
    log(f"Deployed Cloud Run Service: {uri}")
    return uri

