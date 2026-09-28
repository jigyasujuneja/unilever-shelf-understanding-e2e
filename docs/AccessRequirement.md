# GCP Access Requirements, IAM Roles & APIs (`feat/unified-cloud-e2e`)

**Repository:** [`cloud-gtm/unilever-shelf-understanding-with-cv`](https://github.com/cloud-gtm/unilever-shelf-understanding-with-cv)  
**Branch:** `feat/unified-cloud-e2e`  
**Default Region:** `us-central1` (Vertex AI Gemini Endpoint: `global`)

This document provides the complete tabular specification of **GCP APIs** and **IAM Roles** required to bootstrap, deploy, and execute the Unilever Shelf Understanding & Perfect Store Control Plane (`shelf-bench`) in any Google Cloud / Argolis project.

---

## 1. Required GCP APIs to Enable

All 8 core APIs below are automatically enabled when running `python3 -m src.cli --project <PROJECT_ID> bootstrap` (via `ensure_argolis_infra()` in [`src/utils/cloud.py`](../src/utils/cloud.py)).

### 1.1 Core Mandatory GCP APIs (`REQUIRED_ARGOLIS_APIS`)

| # | GCP API Service Name | Display Name | Required For (Branch Component) | CLI / Code Caller | Tier |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `serviceusage.googleapis.com` | **Service Usage API** | Batch-enabling project APIs during `shelf-bench bootstrap` and quota attribution (`x-goog-user-project`) | [`src/utils/cloud.py`](../src/utils/cloud.py) (`ensure_argolis_infra`) | **Mandatory** |
| 2 | `aiplatform.googleapis.com` | **Vertex AI API** | Calling **`Gemini 3.8 Flash`** (`gemini-3.8-flash`), `gemini-3.5-flash-lite`, and **`gemini-embedding-001`** (`512-d` multimodal embeddings) | [`src/utils/llm.py`](../src/utils/llm.py), [`src/utils/embeddings.py`](../src/utils/embeddings.py) | **Mandatory** |
| 3 | `run.googleapis.com` | **Cloud Run Admin API** | Creating & executing the `shelf-bench` parallel Cloud Run Job (`cloud-run`) and deploying the `perfect-store-control-plane` web service (`cloud-service`) | [`src/utils/cloud.py`](../src/utils/cloud.py) (`deploy_job`, `deploy_cloud_service`, `task_seconds`) | **Mandatory** |
| 4 | `cloudbuild.googleapis.com` | **Cloud Build API** | Building the container image from source tarball (`gs://run-sources-<project>-<region>`) and pushing to Artifact Registry | [`src/utils/cloud.py`](../src/utils/cloud.py) (`build_image`) | **Mandatory** |
| 5 | `artifactregistry.googleapis.com` | **Artifact Registry API** | Creating and hosting the Docker repository `us-central1-docker.pkg.dev/<project>/cloud-run-source-deploy/shelf-bench` | [`src/utils/cloud.py`](../src/utils/cloud.py) (`ensure_argolis_infra`, `build_image`) | **Mandatory** |
| 6 | `storage.googleapis.com` | **Cloud Storage API** | Provisioning Uniform-Bucket-Level-Access buckets (`gs://<project>-shelf-images`, `gs://run-sources-<project>-<region>`), uploading `SKU110K_fixed` (`train/val/test`), `HUL_labeled_benchmarks`, `HUL_catalog`, and syncing `results/` | [`src/utils/dataset.py`](../src/utils/dataset.py), [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |
| 7 | `cloudbilling.googleapis.com` | **Cloud Billing API** | Fetching live GCP SKU list prices (Gemini input/output/cached tokens, Cloud Run vCPU-s/GiB-s, GCS ops) and live `USD -> INR` conversion rate | [`src/utils/pricing.py`](../src/utils/pricing.py) | **Mandatory** |
| 8 | `cloudtrace.googleapis.com` | **Cloud Trace API** | Exporting OpenTelemetry distributed traces (`run -> image -> stage -> Gemini/ScaNN call` spans) with deep links in `summary.json["telemetry"]` | [`src/utils/telemetry.py`](../src/utils/telemetry.py) | **Mandatory** |
| 9 | `logging.googleapis.com` | **Cloud Logging API** | Writing structured JSON logs correlated to Cloud Trace IDs and streaming Cloud Build logs (`CLOUD_LOGGING_ONLY`) | [`src/utils/telemetry.py`](../src/utils/telemetry.py), [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |

### 1.2 Optional Production Extension APIs (AlloyDB ScaNN & Gemini Enterprise)

*(Note: By default, `config.yaml` sets `alloydb.fallback_local_scann: true`, so the pipeline runs end-to-end without requiring a live AlloyDB cluster. Enable the APIs below when deploying the managed AlloyDB PGVector/ScaNN instance or Gemini Enterprise Agentspace connector.)*

| # | GCP API Service Name | Display Name | Required For (Branch Component) | CLI / Code Caller | Tier |
| :- | :--- | :--- | :--- | :--- | :--- |
| 10 | `alloydb.googleapis.com` | **AlloyDB for PostgreSQL API** | Managed `pgvector` + `alloydb_scann` cosine index (`products` table) via IAM mTLS (`google-cloud-alloydb-connector`) | [`src/utils/alloydb.py`](../src/utils/alloydb.py), [`config.yaml`](../config.yaml) | Optional (Fallback Enabled) |
| 11 | `compute.googleapis.com` | **Compute Engine API** | Direct VPC Egress (`cloud_run.network` / `subnet`) from Cloud Run to private-IP AlloyDB or L4 GPU `vLLM` endpoints | [`src/utils/cloud.py`](../src/utils/cloud.py) (`deploy_job`), [`scripts/deploy_djev_vllm_argolis.sh`](../scripts/deploy_djev_vllm_argolis.sh) | Optional (VPC / GPU Only) |
| 12 | `discoveryengine.googleapis.com` | **Discovery Engine API (Gemini Enterprise / Agentspace)** | Registering the Perfect Store Grounded Data Store & Agentspace NL-to-Shelf Copilot (`/api/v1/gemini-enterprise-query`) | [`src/utils/server.py`](../src/utils/server.py) | Optional (Agentspace Only) |

---

## 2. Required IAM Roles

### 2.1 Deploying Engineer / Operator Principal (`USER_EMAIL` or CI/CD Runner)

These roles are required for the human user or CI/CD principal executing `shelf-bench bootstrap`, `shelf-bench run`, `shelf-bench cloud-run`, or `shelf-bench cloud-service` via Application Default Credentials (`gcloud auth application-default login`).

| IAM Role ID | Role Title | Scope | Key Permissions Exercised | Used By |
| :--- | :--- | :--- | :--- | :--- |
| `roles/serviceusage.serviceUsageAdmin` | **Service Usage Admin** | Project | `serviceusage.services.enable`, `serviceusage.services.use` | `shelf-bench bootstrap` (`services:batchEnable`) & ADC quota billing |
| `roles/storage.admin` | **Storage Admin** | Project | `storage.buckets.create`, `storage.buckets.get`, `storage.objects.create`, `storage.objects.list`, `storage.objects.get` | Creating `gs://<project>-shelf-images` & `gs://run-sources-<project>-<region>`; uploading datasets and pulling `results/` |
| `roles/artifactregistry.admin` | **Artifact Registry Admin** | Project | `artifactregistry.repositories.create`, `artifactregistry.repositories.get`, `artifactregistry.repositories.uploadArtifacts` | Creating `cloud-run-source-deploy` Docker repo in `us-central1` |
| `roles/cloudbuild.builds.editor` | **Cloud Build Editor** | Project | `cloudbuild.builds.create`, `cloudbuild.builds.get` | Building the `shelf-bench` Docker image via Cloud Build REST API |
| `roles/run.admin` | **Cloud Run Admin** | Project | `run.jobs.create`, `run.jobs.update`, `run.jobs.run`, `run.executions.get`, `run.services.create`, `run.services.update` | Deploying & running the `shelf-bench` Cloud Run Job and `perfect-store-control-plane` Cloud Run Service |
| `roles/iam.serviceAccountUser` | **Service Account User** | Project / Compute SA | `iam.serviceAccounts.actAs` | Required by Cloud Run & Cloud Build when attaching the runtime Service Account to jobs, services, and builds |
| `roles/aiplatform.user` | **Vertex AI User** | Project | `aiplatform.endpoints.predict` | Local execution (`shelf-bench run` / `shelf-bench serve`) calling `Gemini 3.8 Flash` and `gemini-embedding-001` |
| `roles/cloudtrace.agent` | **Cloud Trace Agent** | Project | `cloudtrace.traces.patch` | Exporting OpenTelemetry spans from local benchmark runs |
| `roles/logging.logWriter` | **Logs Writer** | Project | `logging.logEntries.create` | Writing structured JSON benchmark logs from local runs |

> **Note for Argolis Sandbox Users:** In Argolis, your user account typically has `roles/owner` or `roles/editor` on your sandbox project, which already includes all of the above permissions.

---

### 2.2 Cloud Run Runtime Service Account (`shelf-bench` Job & `perfect-store-control-plane` Service)

By default, Cloud Run Jobs and Services execute as the project's default Compute Engine service account (`<PROJECT_NUMBER>-compute@developer.gserviceaccount.com`) or a dedicated custom Service Account configured on the job/service. It requires the following least-privilege runtime roles:

| IAM Role ID | Role Title | Scope | Why the Container Needs It at Runtime | Mandatory / Optional |
| :--- | :--- | :--- | :--- | :--- |
| `roles/aiplatform.user` | **Vertex AI User** | Project | Invokes Vertex AI **`gemini-3.8-flash`**, `gemini-3.5-flash-lite`, and **`gemini-embedding-001`** (`global` / `us-central1`) | **Mandatory** |
| `roles/storage.objectAdmin` | **Storage Object Admin** | `gs://<project>-shelf-images` | Reads `SKU110K_fixed`, `HUL_labeled_benchmarks`, `HUL_catalog`, and writes `summary.json` & `images.jsonl` to `gs://<project>-shelf-images/results/` | **Mandatory** |
| `roles/cloudtrace.agent` | **Cloud Trace Agent** | Project | Exports per-image and per-stage OpenTelemetry trace spans (`src/utils/telemetry.py`) | **Mandatory** |
| `roles/logging.logWriter` | **Logs Writer** | Project | Emits trace-correlated structured JSON logs (`src/utils/telemetry.py`) | **Mandatory** |
| `roles/serviceusage.serviceUsageConsumer` | **Service Usage Consumer** | Project | Required for Vertex AI and Cloud Billing Catalog API quota attribution | **Mandatory** |
| `roles/alloydb.client` | **AlloyDB Client** | Project | Connects to AlloyDB via IAM mTLS (`google.cloud.alloydb.connector`) when `fallback_local_scann: false` | Optional (AlloyDB Only) |
| `roles/alloydb.databaseUser` | **AlloyDB IAM Database User** | AlloyDB Cluster | Authenticates to PostgreSQL as `<sa-name>` without passwords (`enable_iam_auth=True` in [`src/utils/alloydb.py`](../src/utils/alloydb.py)) | Optional (AlloyDB Only) |

---

### 2.3 Cloud Build Service Account (`<PROJECT_NUMBER>@cloudbuild.gserviceaccount.com` or Compute SA)

When `build_image()` submits a build to Cloud Build (`src/utils/cloud.py`), the build service account requires:

| IAM Role ID | Role Title | Scope | Purpose |
| :--- | :--- | :--- | :--- |
| `roles/storage.objectViewer` | **Storage Object Viewer** | `gs://run-sources-<project>-<region>` | Downloads the uploaded source tarball (`shelf-bench/source-<tag>.tgz`) |
| `roles/artifactregistry.writer` | **Artifact Registry Writer** | `cloud-run-source-deploy` repo | Pushes the built container image `us-central1-docker.pkg.dev/<project>/cloud-run-source-deploy/shelf-bench:<tag>` |
| `roles/logging.logWriter` | **Logs Writer** | Project | Streams Docker build logs (`options.logging = "CLOUD_LOGGING_ONLY"`) |

---

## 3. Argolis Organization Policy Compatibility Matrix

Our branch is pre-configured to comply with strict Google Cloud **Argolis** organization policies out of the box:

| Argolis Org Policy Constraint | Default Argolis Enforcement | How `feat/unified-cloud-e2e` Complies Automatically |
| :--- | :--- | :--- |
| `constraints/storage.uniformBucketLevelAccess` | **Enforced** (Legacy ACL buckets rejected) | `ensure_argolis_infra()` in [`src/utils/cloud.py`](../src/utils/cloud.py#L112) explicitly creates all buckets with `"iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}}`. |
| `constraints/iam.disableServiceAccountKeyCreation` | **Enforced** (JSON SA keys blocked) | **100% Keyless Auth**: Uses Application Default Credentials (`google.auth.default()`) locally and attached Service Account metadata identity on Cloud Run, Cloud Build, and AlloyDB IAM Connector. |
| `constraints/compute.requireShieldedVm` | **Enforced** | Cloud Run Gen2 serverless containers + Cloud Build (`CLOUD_LOGGING_ONLY`) require zero non-shielded VMs. |
| `constraints/run.allowedIngress` | **Internal / Authenticated** | Authenticated invocation via `gcloud run services proxy perfect-store-control-plane --port=8080` or Identity-Aware Proxy (IAP). |

---

## 4. Quick-Start CLI Commands to Enable APIs & Grant Roles

Run the following block once for any target GCP project (`PROJECT_ID`) before running `shelf-bench bootstrap`:

```bash
export PROJECT_ID="YOUR_GCP_PROJECT_ID"
export REGION="us-central1"
export USER_EMAIL="$(gcloud config get-value account)"
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
export RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# 1. Enable all required GCP APIs
gcloud services enable \
  serviceusage.googleapis.com \
  aiplatform.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  cloudbilling.googleapis.com \
  cloudtrace.googleapis.com \
  logging.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Grant Deploying Engineer roles (if not already Project Owner/Editor)
for ROLE in \
  roles/serviceusage.serviceUsageAdmin \
  roles/storage.admin \
  roles/artifactregistry.admin \
  roles/cloudbuild.builds.editor \
  roles/run.admin \
  roles/iam.serviceAccountUser \
  roles/aiplatform.user \
  roles/cloudtrace.agent \
  roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="user:${USER_EMAIL}" \
    --role="${ROLE}" \
    --quiet
done

# 3. Grant Cloud Run & Cloud Build Runtime Service Account roles
for ROLE in \
  roles/aiplatform.user \
  roles/storage.objectAdmin \
  roles/artifactregistry.writer \
  roles/cloudtrace.agent \
  roles/logging.logWriter \
  roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="${ROLE}" \
    --quiet
done

# 4. Bootstrap buckets, Artifact Registry, datasets, and splits in one command
python3 -m src.cli --project "${PROJECT_ID}" --region "${REGION}" bootstrap
```
