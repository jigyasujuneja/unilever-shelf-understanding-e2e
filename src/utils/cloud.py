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
SOURCE_FILES = ["Dockerfile", "pyproject.toml", "README.md", "config.yaml", "src"]


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
    tasks = len(runner.combos(approaches, tiers, models, log))  # same list each task indexes
    if not tasks:
        raise SystemExit("Nothing to run: no approach runs any of the given models")
    s = _session()
    image = build_image(s, project, region, log)
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


# ---- one-time project setup and the UI service ---------------------------------------------------

APIS = ["aiplatform.googleapis.com", "run.googleapis.com", "cloudbuild.googleapis.com",
        "artifactregistry.googleapis.com", "storage.googleapis.com", "cloudbilling.googleapis.com",
        "cloudtrace.googleapis.com", "logging.googleapis.com"]


def bootstrap(log=print) -> None:
    """Prepare the config.yaml project: enable APIs, create the data bucket, the Cloud Build
    source bucket and the Artifact Registry repo, and upload the labelled products set and the
    RPC set (if ``make rpc`` prepared it).
    Idempotent. SKU-110K (~12 GB) is uploaded separately: ``make data`` then ``shelf-bench upload``.
    """
    gcp = load_config()["gcp"]
    project, region = gcp["project"], gcp["region"]
    s = _session()
    _ok(s.post(f"https://serviceusage.googleapis.com/v1/projects/{project}/services:batchEnable",
               json={"serviceIds": APIS}))
    log(f"APIs enabled in {project}: {', '.join(a.split('.')[0] for a in APIS)}")
    data_bucket = dataset._split_gs(gcp["data"])[0]
    for bucket in (data_bucket, f"run-sources-{project}-{region}"):
        if s.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}").status_code == 404:
            _ok(s.post(f"https://storage.googleapis.com/storage/v1/b?project={project}",
                       json={"name": bucket, "location": region,
                             "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}}}))
            log(f"created gs://{bucket}")
        else:
            log(f"gs://{bucket} exists")
    repos = f"https://artifactregistry.googleapis.com/v1/projects/{project}/locations/{region}/repositories"
    if s.get(f"{repos}/cloud-run-source-deploy").status_code == 404:
        _ok(s.post(f"{repos}?repositoryId=cloud-run-source-deploy", json={"format": "DOCKER"}))
        log(f"created Artifact Registry repo {region}/cloud-run-source-deploy")
    else:
        log("Artifact Registry repo cloud-run-source-deploy exists")
    for local, key in ((dataset.PRODUCTS_LOCAL, "products"), (dataset.RPC_LOCAL, "rpc")):
        if Path(local).is_dir() and gcp.get(key):
            dataset.upload(local, gcp[key])
    log("Next: make data && .venv/bin/shelf-bench upload   (SKU-110K, only needed for detection)")


def deploy_ui(log=print) -> str:
    """Deploy the leaderboard UI as a private Cloud Run service (``cloud_run.service``).
    On start it pulls every finished run from ``gcp.results``; nobody can call it without the
    Cloud Run Invoker role (open it with ``shelf-bench proxy``)."""
    cfg = load_config()
    gcp, cr = cfg["gcp"], cfg.get("cloud_run", {})
    project, region = gcp["project"], gcp["region"]
    name = cr.get("service", "shelf-bench-ui")
    s = _session()
    image = build_image(s, project, region, log)
    base = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services"
    body = {"template": {"containers": [{
        "image": image,
        "args": ["serve", "--pull", "--host", "0.0.0.0", "--port", "8080"],
        "ports": [{"containerPort": 8080}],
        "resources": {"limits": {"cpu": "1", "memory": "2Gi"}},
    }]}}
    exists = s.get(f"{base}/{name}").status_code == 200
    op = _ok(s.patch(f"{base}/{name}", json=body) if exists
             else s.post(f"{base}?serviceId={name}", json=body))
    uri = _wait_op(s, op["name"]).get("response", {}).get("uri", "")
    log(f"Deployed {name}: {uri} (private)\nOpen it: make ui-proxy   (http://localhost:8083)")
    return uri


def proxy_ui(port: int = 8083, host: str = "127.0.0.1", log=print) -> None:
    """Serve the private UI service on ``http://host:port``, adding your identity token to every
    request. Same as ``gcloud run services proxy``, without its extra gcloud component.
    The token comes from ``gcloud auth print-identity-token`` (refreshed every 30 minutes)."""
    import subprocess
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    cfg = load_config()
    gcp, cr = cfg["gcp"], cfg.get("cloud_run", {})
    name = cr.get("service", "shelf-bench-ui")
    svc = f"https://run.googleapis.com/v2/projects/{gcp['project']}/locations/{gcp['region']}/services/{name}"
    target = _ok(_session().get(svc)).get("uri", "").rstrip("/")
    if not target:
        raise RuntimeError(f"Cloud Run service {name} has no URL yet (deploy it with make ui-cloud)")
    token = {"value": "", "expires": 0.0}

    def id_token() -> str:
        if time.time() >= token["expires"]:
            token["value"] = subprocess.check_output(
                ["gcloud", "auth", "print-identity-token"], text=True).strip()
            token["expires"] = time.time() + 1800
        return token["value"]

    class Handler(BaseHTTPRequestHandler):
        def _forward(self, method: str) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            headers = {"Authorization": f"Bearer {id_token()}",
                       "Accept": self.headers.get("Accept", "*/*")}
            if self.headers.get("Content-Type"):
                headers["Content-Type"] = self.headers["Content-Type"]
            req = urllib.request.Request(f"{target}{self.path}", data=self.rfile.read(n) if n else None,
                                         headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    status, resp_headers, body = r.status, r.headers, r.read()
            except urllib.error.HTTPError as e:
                status, resp_headers, body = e.code, e.headers, e.read()
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._forward("GET")

        def do_POST(self) -> None:
            self._forward("POST")

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass

    log(f"Proxying {target} -> http://{host}:{port}  (Ctrl-C to stop)")
    with ThreadingHTTPServer((host, port), Handler) as httpd:
        httpd.serve_forever()
