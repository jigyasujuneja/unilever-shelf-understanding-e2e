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
SOURCE_FILES = ["Dockerfile", "pyproject.toml", "README.md", "config.yaml", "gt-mapping.yml", "src", "experiments", "scripts"]


def _session() -> AuthorizedSession:
    from utils.llm import ensure_gcloud_auth_fallback

    ensure_gcloud_auth_fallback()
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


def build_image(s: AuthorizedSession, project: str, region: str, log=print, extras: str = "") -> str:
    """Build and push the repo image with Cloud Build; ``extras`` = pip extras (e.g. ``cv``)."""
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    image = f"{region}-docker.pkg.dev/{project}/cloud-run-source-deploy/shelf-bench:{tag}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in SOURCE_FILES:
            if not (ROOT / name).exists():
                continue
            tar.add(ROOT / name, arcname=name,
                    filter=lambda t: None if "__pycache__" in t.name or ".egg-info" in t.name else t)
    bucket, obj = f"run-sources-{project}-{region}", f"shelf-bench/source-{tag}.tgz"
    dataset._gcs().bucket(bucket).blob(obj).upload_from_string(buf.getvalue())
    log(f"Building {image} with Cloud Build ...")
    op = _ok(s.post(
        f"https://cloudbuild.googleapis.com/v1/projects/{project}/locations/{region}/builds",
        json={"source": {"storageSource": {"bucket": bucket, "object": obj}},
              "steps": [{"name": "gcr.io/cloud-builders/docker",
                         "args": ["build", *(["--build-arg", f"EXTRAS={extras}"] if extras else []),
                                  "-t", image, "."]}],
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
                "timeout": f"{int(cr.get('timeout_s', 10800))}s",
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
    exists = s.get(f"{base}/{job}", timeout=30).status_code == 200
    op = _ok(s.patch(f"{base}/{job}", json=body, timeout=60) if exists
             else s.post(f"{base}?jobId={job}", json=body, timeout=60))
    _wait_op(s, op["name"])
    return f"{base}/{job}"


def _wait_op(s, name: str, log=None, every: int = 5) -> dict:
    while True:
        try:
            op = _ok(s.get(f"https://run.googleapis.com/v2/{name}", timeout=30))
            if log:
                md = op.get("metadata", {})
                exec_name = md.get("name", "")
                if "/executions/" in exec_name and not any(k in md for k in ("succeededCount", "runningCount", "failedCount")):
                    md = _ok(s.get(f"https://run.googleapis.com/v2/{exec_name}", timeout=30))
                log(f"  tasks: {md.get('succeededCount', 0)} succeeded, "
                    f"{md.get('failedCount', 0)} failed, {md.get('runningCount', 0)} running")
            if op.get("done"):
                if "error" in op:
                    raise RuntimeError(f"Cloud Run: {op['error']}")
                return op
        except RuntimeError:
            raise
        except Exception as poll_err:
            if log:
                log(f"  (poll retry after {poll_err})")
        time.sleep(every)


def run_on_cloud(
    approaches: list[str],
    models: list[str],
    tiers: list[str],
    split: str,
    limit: int,
    seed: int,
    workers: int,
    owner: str,
    dataset_override: str | None = None,
    fields: str | None = None,
    brand_model: str | None = None,
    pack_model: str | None = None,
    image: str | None = None,
    job_name: str | None = None,
    pull_local: bool = False,
    log=print,
    **variant_models: str | None,
) -> int:
    cfg = load_config()
    gcp = cfg["gcp"]
    project, region = gcp["project"], gcp["region"]
    tasks = len(runner.combos(approaches, tiers, models, log))  # same list each task indexes
    if not tasks:
        raise SystemExit("Nothing to run: no approach runs any of the given models")
    s = _session()
    image = image or build_image(s, project, region, log)
    args = [
        "run", "-a", *approaches, "-m", *models, "-t", *tiers,
        *(["-d", dataset_override] if dataset_override else []),
        *(["--fields", fields] if fields else []),
        *(["--brand-model", brand_model] if brand_model else []),
        *(["--pack-model", pack_model] if pack_model else []),
        *[x for k, v in variant_models.items() if v for x in (f"--{k.replace('_', '-')}", v)],
        "--split", split, "--limit", str(limit),
        "--seed", str(seed), "--workers", str(workers), "--owner", owner,
        "--root", gcp["data"], "--results", gcp["results"],
    ]
    cfg.setdefault("cloud_run", {}).setdefault("parallelism", len(models))
    if job_name:
        cfg["cloud_run"]["job"] = job_name
    job_url = deploy_job(s, project, region, image, args, tasks, cfg)
    log(f"Executing {tasks} tasks on Cloud Run job {job_url.rsplit('/', 1)[-1]} ({region}) ...")
    op = _ok(s.post(f"{job_url}:run", json={}))
    log(f"  execution: https://console.cloud.google.com/run/jobs/executions/details/{region}/"
        f"{op.get('metadata', {}).get('name', '').rsplit('/', 1)[-1]}?project={project}")
    try:
        _wait_op(s, op["name"], log=log, every=20)
    finally:
        if pull_local:
            n = runner.pull(gcp["results"])
            log(f"Pulled {n} new run(s) into results/")
        else:
            updated = runner.finalize_remote_compute(gcp["results"], log=log)
            log(f"Finalized compute on {updated} remote run(s) directly in {gcp['results']} (zero local pull)")
    return 0


def run_cli_on_cloud(
    cli_args: list[str],
    image: str | None = None,
    tasks: int = 1,
    cpu: int = 4,
    memory_gib: int = 8,
    job_name: str | None = None,
    log=print,
) -> str:
    """Execute any ``shelf-bench`` subcommand on Cloud Run Jobs in ``us-central1`` with zero local execution."""
    cfg = load_config()
    gcp = cfg["gcp"]
    project, region = gcp["project"], gcp["region"]
    s = _session()
    image = image or build_image(s, project, region, log)
    job_cfg = dict(cfg)
    cr = dict(job_cfg.get("cloud_run", {}))
    cr["cpu"] = cpu
    cr["memory_gib"] = memory_gib
    cr["parallelism"] = tasks
    if job_name:
        cr["job"] = job_name
    job_cfg["cloud_run"] = cr
    job_url = deploy_job(s, project, region, image, cli_args, tasks, job_cfg)
    log(f"Executing `shelf-bench {' '.join(cli_args)}` on Cloud Run job {job_url.rsplit('/', 1)[-1]} ({region}) ...")
    op = _ok(s.post(f"{job_url}:run", json={}))
    log(f"  execution: https://console.cloud.google.com/run/jobs/executions/details/{region}/"
        f"{op.get('metadata', {}).get('name', '').rsplit('/', 1)[-1]}?project={project}")
    _wait_op(s, op["name"], log=log, every=15)
    return image


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


def deploy_ui(image: str | None = None, log=print) -> str:
    """Deploy the leaderboard UI as a private Cloud Run service (``cloud_run.service`` and
    ``perfect-store-control-plane``).
    On start it pulls every finished run from ``gcp.results``; nobody can call it without the
    Cloud Run Invoker role (open it with ``shelf-bench proxy``)."""
    cfg = load_config()
    gcp, cr = cfg["gcp"], cfg.get("cloud_run", {})
    project, region = gcp["project"], gcp["region"]
    primary_name = cr.get("service", "shelf-ui")
    services = list(dict.fromkeys([primary_name, "perfect-store-control-plane"]))
    if gcp.get("results") and Path(runner.RESULTS_DIR).is_dir() and any(Path(runner.RESULTS_DIR).glob("*/summary.json")):
        pushed = runner.push(gcp["results"], overwrite=True, prune_remote=False)
        log(f"Pushed {pushed} local run(s) + experiments_registry.json + qc_audit_report.json to {gcp['results']}")
    s = _session()
    image = image or build_image(s, project, region, log)
    base = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services"
    body = {
        "launchStage": "BETA",
        "iapEnabled": True,
        "template": {
            "scaling": {"minInstanceCount": 1, "maxInstanceCount": 10},
            "containers": [{
                "image": image,
                "args": ["serve", "--pull", "--host", "0.0.0.0", "--port", "8080"],
                "ports": [{"containerPort": 8080}],
                "resources": {"limits": {"cpu": "2", "memory": "4Gi"}},
            }],
        },
    }
    primary_uri = ""
    for svc_name in services:
        exists = s.get(f"{base}/{svc_name}", timeout=30).status_code == 200
        op = _ok(s.patch(f"{base}/{svc_name}?updateMask=template,iapEnabled,launchStage", json=body, timeout=60) if exists
                 else s.post(f"{base}?serviceId={svc_name}", json=body, timeout=60))
        uri = _wait_op(s, op["name"]).get("response", {}).get("uri", "")
        log(f"Deployed {svc_name}: {uri} (Direct IAP enabled)")
        if svc_name == primary_name or not primary_uri:
            primary_uri = uri
    return primary_uri


def _wait_vertex_op(s, region: str, name: str, log=print, every: int = 15) -> dict:
    """Poll a Vertex AI long-running operation; return its response, raise on its error."""
    url = f"https://{region}-aiplatform.googleapis.com/v1/{name}"
    while True:
        op = _ok(s.get(url, timeout=30))
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"Vertex AI {name}: {op['error']}")
            return op.get("response", {})
        log(f"  waiting for {name.rsplit('/operations/', 1)[0].rsplit('/', 1)[-1]} ...")
        time.sleep(every)


_CV_ENDPOINT_NAMES = {
    "brand": "marketshare-brand",
    "pack": "marketshare-packtype",
    "wash_variant": "marketshare-wash-variant",
    "skin_variant": "marketshare-skin-variant",
    "p5_variant": "marketshare-p5-variant",
    "oral_variant": "marketshare-oral-variant",
    "gen_variant": "marketshare-gen-variant",
    "nonhul_p6_variant": "marketshare-nonhul-p6-variant",
    "nonhul_gen_variant": "marketshare-nonhul-gen-variant",
}


def deploy_cv_endpoints(
    brand_model: str | None = None,
    pack_model: str | None = None,
    image: str | None = None,
    roles: list[str] | tuple[str, ...] | None = None,
    log=print,
    **variant_models: str | None,
) -> dict:
    """Register the legacy .h5 brand / pack / variant classifiers in Vertex AI Model Registry and
    deploy each to its Vertex AI Endpoint, served by this repo's image (``shelf-bench serve``,
    /api/cv/predict). A redeploy moves all traffic to the new model and undeploys the old one.
    Put the endpoints in config.yaml ``cv_models.<role>_endpoint``."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = load_config()
    gcp, cv = cfg["gcp"], cfg.get("cv_models") or {}
    project, region = gcp["project"], gcp["region"]
    s = _session()
    image = image or build_image(s, project, region, log, extras="cv")  # the server needs TensorFlow
    api = f"https://{region}-aiplatform.googleapis.com/v1"
    base = f"{api}/projects/{project}/locations/{region}"
    out: dict = {"image": image}
    overrides = {"brand": brand_model, "pack": pack_model, **variant_models}
    target_roles = [r for r in (roles or _CV_ENDPOINT_NAMES) if r in _CV_ENDPOINT_NAMES]
    items = [
        (r, _CV_ENDPOINT_NAMES[r], overrides.get(r) or cv.get(f"{r}_model"))
        for r in target_roles
        if overrides.get(r) or cv.get(f"{r}_model")
    ]

    def _deploy_one(item: tuple[str, str, str], sess=None) -> tuple[str, dict]:
        role, name, model_uri = item
        ws = sess or _session()
        log(f"Uploading {model_uri} to Vertex AI Model Registry ...")
        op = _ok(ws.post(f"{base}/models:upload", timeout=60, json={"model": {
            "displayName": f"{name}-h5",
            "description": f"Legacy InceptionV3 Keras classifier ({model_uri})",
            "artifactUri": model_uri.rsplit("/", 1)[0],  # Vertex copies it to AIP_STORAGE_URI
            "containerSpec": {
                "imageUri": image,
                "command": ["shelf-bench"],
                # --results outside results/: the prediction server must not pull the leaderboard
                "args": ["serve", "--host", "0.0.0.0", "--port", "8080", "--results", "/tmp/cv-serving"],
                "env": [{"name": "SHELF_BENCH_CV_ROLE", "value": role},
                        {"name": f"SHELF_BENCH_CV_{role.upper()}_MODEL", "value": model_uri}],
                "ports": [{"containerPort": 8080}],
                "predictRoute": "/api/cv/predict",
                "healthRoute": "/api/cv/health",
            },
        }}))
        model = _wait_vertex_op(ws, region, op["name"], log)["model"]
        ep_name = f"{name}-vertex-endpoint"
        eps = _ok(ws.get(f"{base}/endpoints", timeout=30)).get("endpoints", [])
        endpoint = next((e["name"] for e in eps if e.get("displayName") == ep_name), None)
        if endpoint is None:
            op = _ok(ws.post(f"{base}/endpoints", json={"displayName": ep_name}, timeout=60))
            endpoint = _wait_vertex_op(ws, region, op["name"], log)["name"]
        old = [d["id"] for d in _ok(ws.get(f"{api}/{endpoint}", timeout=30)).get("deployedModels", [])]
        log(f"Deploying {model} to {endpoint} (15-30 min) ...")
        op = _ok(ws.post(f"{api}/{endpoint}:deployModel", timeout=60, json={
            "deployedModel": {"model": model, "displayName": f"{name}-h5", "dedicatedResources": {
                "machineSpec": {"machineType": "n1-standard-4"}, "minReplicaCount": 1, "maxReplicaCount": 2}},
            "trafficSplit": {"0": 100},
        }))
        _wait_vertex_op(ws, region, op["name"], log, every=30)
        for dm_id in old:
            op = _ok(ws.post(f"{api}/{endpoint}:undeployModel", json={"deployedModelId": dm_id}, timeout=60))
            _wait_vertex_op(ws, region, op["name"], log)
        res = {"model_uri": model_uri, "vertex_model": model, "endpoint": endpoint}
        log(f"  cv_models.{role}_endpoint: {endpoint}")
        return role, res

    if len(items) <= 2:
        for item in items:
            role, res = _deploy_one(item, sess=s)
            out[role] = res
    else:
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            for role, res in pool.map(_deploy_one, items):
                out[role] = res
    return out


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


def submit_hf_gpu_custom_job(
    approaches: list[str],
    hf_model: str = "Cloudflare/clef-flash",
    dataset_name: str = "hul_golden",
    splits: tuple[str, ...] = ("val", "test"),
    limit: int = 1000,
    seed: int = 0,
    workers: int = 8,
    owner: str = "jjuneja",
    *,
    dry_run: bool = False,
    log=print,
) -> dict:
    """Build or submit a Vertex AI GPU CustomJob that loads a Hugging Face open-weight checkpoint
    (``Cloudflare/clef-flash``, ``Cloudflare/clef-27b``, or ``Contrastive-LM/CLM-v0.1-8B``) via vLLM
    and runs ``shelf-bench run`` across ``val`` and ``test`` splits (default ``limit=1000`` per split).
    """
    cfg = load_config()
    gcp = cfg.get("gcp", {})
    project = gcp.get("project", "ul-fs-s-sandbx-16-prj")
    region = gcp.get("region", "us-central1")
    results_gcs = gcp.get("results", "gs://ul-fs-s-sandbx-16-prj-shelf-images/results")

    hf_models = cfg.get("hf_gpu_models") or {}
    spec = hf_models.get(hf_model) or {
        "hf_repo": hf_model,
        "base_arch": "Qwen/Qwen3.5-9B",
        "machine_type": "g2-standard-12",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
        "serving_container": "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:v0.20.0",
    }
    hf_repo = str(spec.get("hf_repo") or hf_model)
    safe_slug = hf_repo.lower().replace("/", "-").replace(".", "-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    display_name = f"shelf-bench-hf-{safe_slug}-{dataset_name}-{ts}"
    src_bucket = dataset._split_gs(results_gcs)[0]
    src_obj = "source/source-hf-gpu.tgz"

    split_cmds = [
        f"PYTHONPATH=src ~/.local/bin/uv run --python 3.12 python -m cli run -a {' '.join(approaches)} -m {hf_repo} -d {dataset_name} "
        f"--split {sp} --limit {limit} --seed {seed} --workers {workers} --owner {owner} "
        f"--results {results_gcs}"
        for sp in splits
    ]
    accel_count = int(spec.get("accelerator_count", 1))
    tp_flag = f" --tensor-parallel-size {accel_count}" if accel_count > 1 else ""
    boot_script = (
        "mkdir -p /tmp/shelf-bench && cd /tmp/shelf-bench\n"
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        f"(vllm serve {hf_repo} --host 127.0.0.1 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --enforce-eager --max-logprobs 16 --trust-remote-code{tp_flag} "
        f"|| python3 -m vllm.entrypoints.openai.api_server --model {hf_repo} --host 127.0.0.1 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --enforce-eager --max-logprobs 16 --trust-remote-code{tp_flag}) &\n"
        f"~/.local/bin/uv run --python 3.12 --with google-cloud-storage python -c \"from google.cloud import storage; storage.Client().bucket('{src_bucket}').blob('{src_obj}').download_to_filename('src.tgz')\"\n"
        "tar -xzf src.tgz\n"
        "~/.local/bin/uv sync --python 3.12\n"
        "for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/health && break || sleep 5; done\n"
        "export HF_VLLM_BASE_URL=http://127.0.0.1:8000/v1\n"
        + "\n".join(split_cmds)
    )

    job_payload = {
        "displayName": display_name,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": spec.get("machine_type", "g2-standard-12"),
                        "acceleratorType": spec.get("accelerator_type", "NVIDIA_L4"),
                        "acceleratorCount": int(spec.get("accelerator_count", 1)),
                    },
                    "replicaCount": 1,
                    "containerSpec": {
                        "imageUri": spec.get(
                            "serving_container",
                            "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:v0.20.0",
                        ),
                        "command": ["bash", "-lc"],
                        "args": [boot_script],
                        "env": [
                            {"name": "HF_HUB_ENABLE_HF_TRANSFER", "value": "1"},
                            {"name": "HF_VLLM_BASE_URL", "value": "http://127.0.0.1:8000/v1"},
                        ],
                    },
                }
            ]
        },
    }

    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/customJobs"
    if dry_run:
        log(f"[DRY-RUN] Vertex AI GPU CustomJob spec prepared for {hf_repo} on {dataset_name} ({splits}, limit={limit}):")
        return {"status": "DRY_RUN", "endpoint_url": url, "custom_job": job_payload}

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in SOURCE_FILES:
            if not (ROOT / name).exists():
                continue
            tar.add(
                ROOT / name,
                arcname=name,
                filter=lambda t: None if "__pycache__" in t.name or ".egg-info" in t.name else t,
            )
    dataset._gcs().bucket(src_bucket).blob(src_obj).upload_from_string(buf.getvalue())

    s = _session()
    resp = _ok(s.post(url, json=job_payload))
    log(f"Submitted Vertex AI GPU CustomJob: {resp.get('name')} ({display_name})")
    return {"status": "SUBMITTED", "endpoint_url": url, "response": resp, "custom_job": job_payload}

