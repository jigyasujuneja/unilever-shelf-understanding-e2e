# GCP Access, IAM Roles, APIs & Quota Requirements

**Target Environment:** Unilever Client GCP Project (e.g. `unilever-shelf-understanding`) & Sandbox Environments  
**Repository:** [`cloud-gtm/unilever-shelf-understanding-with-cv`](https://github.com/cloud-gtm/unilever-shelf-understanding-with-cv)  
**Branch:** `feat/unified-cloud-e2e`  
**Default Region:** `us-central1` (or `asia-south1` Mumbai for India data residency; Vertex AI Gemini Endpoint: `global`)

This document provides the complete tabular specification of **GCP APIs**, **Least-Privilege IAM Roles** (separated by *Client Cloud Admin*, *FDE / Developer Principal*, and *Runtime Service Account*), **VPC / Security Policy Requirements**, and **Vertex AI / GPU Quotas** required to deploy and run the Unilever Shelf Understanding & Perfect Store Control Plane (`shelf-bench`).

---

## 1. Required GCP APIs to Enable

Unilever Cloud Admins can enable these APIs once via Terraform/CLI (Section 6), or allow `python3 -m src.cli --project <PROJECT_ID> bootstrap` to enable them automatically.

### 1.1 Core Mandatory GCP APIs

| # | GCP API Service Name | Display Name | Purpose in Unilever Shelf Understanding Pipeline | Code / Module Caller | Requirement Tier |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `serviceusage.googleapis.com` | **Service Usage API** | API state verification during bootstrap and Vertex AI / Billing quota attribution (`x-goog-user-project`) | [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |
| 2 | `aiplatform.googleapis.com` | **Vertex AI API** | Managed inference for **`Gemini 3.8 Flash`** (`gemini-3.8-flash`), `gemini-3.5-flash-lite`, and **`gemini-embedding-001`** (`512-d` multimodal embeddings) | [`src/utils/llm.py`](../src/utils/llm.py), [`src/utils/embeddings.py`](../src/utils/embeddings.py) | **Mandatory** |
| 3 | `run.googleapis.com` | **Cloud Run Admin API** | Deploying & executing the `shelf-bench` parallel Cloud Run Job (`cloud-run`) and the `perfect-store-control-plane` web service (`cloud-service`) | [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |
| 4 | `cloudbuild.googleapis.com` | **Cloud Build API** | Building the application container image from source (`gs://run-sources-<project>-<region>`) and pushing to Artifact Registry | [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |
| 5 | `artifactregistry.googleapis.com` | **Artifact Registry API** | Hosting the container repository `<region>-docker.pkg.dev/<project>/cloud-run-source-deploy/shelf-bench` | [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |
| 6 | `storage.googleapis.com` | **Cloud Storage API** | Hosting Uniform-Bucket-Level-Access buckets (`gs://<project>-shelf-images`, `gs://run-sources-<project>-<region>`) for shelf datasets, HUL 245-SKU catalog, and benchmark outputs | [`src/utils/dataset.py`](../src/utils/dataset.py), [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |
| 7 | `cloudbilling.googleapis.com` | **Cloud Billing API** | Fetching live GCP SKU list prices (Gemini token tiers, Cloud Run vCPU/GiB-s, GCS ops) and `USD -> INR` exchange rate for per-image FinOps tracking | [`src/utils/pricing.py`](../src/utils/pricing.py) | **Mandatory** |
| 8 | `cloudtrace.googleapis.com` | **Cloud Trace API** | Exporting OpenTelemetry distributed traces (`run -> image -> stage -> Gemini/ScaNN call` spans) with trace links in `summary.json` | [`src/utils/telemetry.py`](../src/utils/telemetry.py) | **Mandatory** |
| 9 | `logging.googleapis.com` | **Cloud Logging API** | Emitting structured JSON audit logs correlated to Cloud Trace IDs and streaming Cloud Build logs (`CLOUD_LOGGING_ONLY`) | [`src/utils/telemetry.py`](../src/utils/telemetry.py), [`src/utils/cloud.py`](../src/utils/cloud.py) | **Mandatory** |

### 1.2 Production Extension APIs (AlloyDB Vector Search, VPC & Gemini Enterprise)

*(Note: By default, `config.yaml` sets `alloydb.fallback_local_scann: true` so benchmarks and the UI run end-to-end using the embedded in-memory HUL-245 ScaNN index even before AlloyDB is provisioned. Enable the APIs below for full production scale.)*

| # | GCP API Service Name | Display Name | Purpose in Unilever Shelf Understanding Pipeline | Code / Module Caller | Requirement Tier |
| :- | :--- | :--- | :--- | :--- | :--- |
| 10 | `alloydb.googleapis.com` | **AlloyDB for PostgreSQL API** | Managed `pgvector` + `alloydb_scann` index (`products` table) over IAM mTLS (`google-cloud-alloydb-connector`) | [`src/utils/alloydb.py`](../src/utils/alloydb.py), [`config.yaml`](../config.yaml) | Production Extension (Fallback Enabled) |
| 11 | `compute.googleapis.com` | **Compute Engine API** | Direct VPC Egress (`cloud_run.network` / `subnet`) from Cloud Run to private-IP AlloyDB or L4 GPU `vLLM` endpoints | [`src/utils/cloud.py`](../src/utils/cloud.py), [`scripts/deploy_djev_vllm_argolis.sh`](../scripts/deploy_djev_vllm_argolis.sh) | Production Extension (Private VPC / GPU) |
| 12 | `servicenetworking.googleapis.com` | **Service Networking API** | Private Services Access (PSA) peering between the Unilever VPC and the private-IP AlloyDB cluster | VPC / AlloyDB Network Setup | Production Extension (Private-IP AlloyDB) |
| 13 | `discoveryengine.googleapis.com` | **Discovery Engine API (Gemini Enterprise / Agentspace)** | Grounded Data Store & Agentspace NL-to-Shelf Copilot (`/api/v1/gemini-enterprise-query`) | [`src/utils/server.py`](../src/utils/server.py) | Optional (Gemini Enterprise Copilot) |

---

## 2. Required IAM Roles by Persona / Identity

In a client GCP project with strict least-privilege governance, permissions are split across **three identities**:
1. **Unilever Cloud Admin** (one-time project/API/IAM provisioning — or automated via Terraform)
2. **FDE / Engineering Deployer Principal** (the human or CI/CD identity deploying and running `shelf-bench`)
3. **Dedicated Cloud Run Runtime Service Account** (`sa-shelf-bench-runtime@<PROJECT_ID>.iam.gserviceaccount.com` or default Compute SA)

### 2.1 FDE / Engineering Deployer Principal (`user:<ENGINEER_EMAIL>` or CI/CD Workforce Identity)

| IAM Role ID | Role Title | Scope | Key Permissions Exercised | Required If Admin Pre-Provisions Infra? |
| :--- | :--- | :--- | :--- | :--- |
| `roles/run.admin` | **Cloud Run Admin** | Project | `run.jobs.create`, `run.jobs.update`, `run.jobs.run`, `run.executions.get`, `run.services.create`, `run.services.update` | **Yes (Mandatory to deploy/run jobs & UI)** |
| `roles/cloudbuild.builds.editor` | **Cloud Build Editor** | Project | `cloudbuild.builds.create`, `cloudbuild.builds.get` | **Yes (Mandatory to build container)** |
| `roles/artifactregistry.repoAdmin` | **Artifact Registry Repo Admin** *(or `writer` if repo pre-created)* | Project or `cloud-run-source-deploy` | `artifactregistry.repositories.create`, `artifactregistry.repositories.uploadArtifacts` | `writer` sufficient if Admin pre-creates repo |
| `roles/storage.admin` | **Storage Admin** *(or `objectAdmin` on the 2 buckets if pre-created)* | Project or `gs://<project>-shelf-images` + `gs://run-sources-*` | `storage.buckets.create`, `storage.objects.create`, `storage.objects.list`, `storage.objects.get` | `storage.objectAdmin` sufficient if Admin pre-creates both buckets |
| `roles/iam.serviceAccountUser` | **Service Account User** | Runtime SA & Cloud Build SA | `iam.serviceAccounts.actAs` | **Yes (Mandatory to attach SA to Cloud Run/Build)** |
| `roles/serviceusage.serviceUsageConsumer` | **Service Usage Consumer** *(or `serviceUsageAdmin` if running `bootstrap`)* | Project | `serviceusage.services.use` (`serviceusage.services.enable` only if FDE runs `bootstrap` to enable APIs) | `serviceUsageConsumer` sufficient if Admin pre-enables APIs in Table 1 |
| `roles/aiplatform.user` | **Vertex AI User** | Project | `aiplatform.endpoints.predict` | **Yes (For local `shelf-bench run` / `serve`)** |
| `roles/cloudtrace.agent` | **Cloud Trace Agent** | Project | `cloudtrace.traces.patch` | **Yes (For OpenTelemetry trace export)** |
| `roles/logging.logWriter` | **Logs Writer** | Project | `logging.logEntries.create` | **Yes (For structured benchmark logs)** |
| `roles/viewer` | **Viewer** *(or `monitoring.viewer` + `run.viewer`)* | Project | Read-only inspection of Cloud Build logs, Cloud Trace spans, and Cloud Run execution status in GCP Console | Recommended for debugging |

---

### 2.2 Cloud Run Runtime Service Account (`sa-shelf-bench-runtime@<PROJECT_ID>.iam.gserviceaccount.com`)

For Unilever client production deployments, we recommend creating a dedicated least-privilege runtime service account (`sa-shelf-bench-runtime`) rather than using the default Compute Engine service account:

| IAM Role ID | Role Title | Scope | Why the Container Needs It at Runtime | Mandatory / Optional |
| :--- | :--- | :--- | :--- | :--- |
| `roles/aiplatform.user` | **Vertex AI User** | Project | Invokes Vertex AI **`gemini-3.8-flash`**, `gemini-3.5-flash-lite`, and **`gemini-embedding-001`** (`global` / `us-central1`) | **Mandatory** |
| `roles/storage.objectAdmin` | **Storage Object Admin** | `gs://<project>-shelf-images` | Reads `SKU110K_fixed`, `HUL_labeled_benchmarks`, `HUL_catalog`, and writes `summary.json` & `images.jsonl` to `gs://<project>-shelf-images/results/` | **Mandatory** |
| `roles/cloudtrace.agent` | **Cloud Trace Agent** | Project | Exports per-image and per-stage OpenTelemetry trace spans (`src/utils/telemetry.py`) | **Mandatory** |
| `roles/logging.logWriter` | **Logs Writer** | Project | Emits trace-correlated structured JSON logs (`src/utils/telemetry.py`) | **Mandatory** |
| `roles/serviceusage.serviceUsageConsumer` | **Service Usage Consumer** | Project | Required for Vertex AI and Cloud Billing Catalog API quota attribution | **Mandatory** |
| `roles/alloydb.client` | **AlloyDB Client** | Project | Connects to AlloyDB via IAM mTLS (`google.cloud.alloydb.connector`) when `fallback_local_scann: false` | Production Extension (AlloyDB) |
| `roles/alloydb.databaseUser` | **AlloyDB IAM Database User** | AlloyDB Cluster | Authenticates to PostgreSQL as `sa-shelf-bench-runtime` without passwords (`enable_iam_auth=True` in [`src/utils/alloydb.py`](../src/utils/alloydb.py)) | Production Extension (AlloyDB) |

---

### 2.3 Cloud Build Service Account (`<PROJECT_NUMBER>@cloudbuild.gserviceaccount.com` or Custom Build SA)

| IAM Role ID | Role Title | Scope | Purpose |
| :--- | :--- | :--- | :--- |
| `roles/storage.objectViewer` | **Storage Object Viewer** | `gs://run-sources-<project>-<region>` | Downloads the uploaded source tarball (`shelf-bench/source-<tag>.tgz`) |
| `roles/artifactregistry.writer` | **Artifact Registry Writer** | `cloud-run-source-deploy` repo | Pushes the built container image `<region>-docker.pkg.dev/<project>/cloud-run-source-deploy/shelf-bench:<tag>` |
| `roles/logging.logWriter` | **Logs Writer** | Project | Streams Docker build logs (`options.logging = "CLOUD_LOGGING_ONLY"`) |

---

## 3. Vertex AI & Compute Quota Requirements (Unilever Client Project)

| Quota Metric | Service | Minimum Recommended Quota (Benchmark / Pilot) | Production Target (`506K` Images/Day) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **PayGo / Provisioned Throughput (`gemini-3.8-flash`)** | `aiplatform.googleapis.com` | `>= 500 RPM` (`global` endpoint) | `>= 2,500 RPM` (Only `~2%` open-set crops + optional Stage 5a fallback hit Gemini 3.8 Flash) | Complete-Linkage Clustering (`ADR-008`) + ScaNN filters `89%` of crops before VLM calls |
| **Multimodal Embeddings (`gemini-embedding-001`)** | `aiplatform.googleapis.com` | `>= 1,500 RPM` (`us-central1` / `global`) | `>= 10,000 RPM` | Reduced `~3.2x–3.5x` via Stage 3.8 high-purity cluster medoid compression |
| **Cloud Run CPUs per Region (`us-central1`)** | `run.googleapis.com` | `>= 16 vCPU` (`4` parallel tasks × `2 vCPU` + web service) | `>= 64 vCPU` | Configurable in `config.yaml` (`cloud_run.cpu` and `cloud_run.parallelism`) |
| **NVIDIA L4 GPUs (`nvidia-l4` on Cloud Run)** | `run.googleapis.com` | `0` (Default uses serverless fallback) or `1x L4 GPU` | `2–4x L4 GPUs` (if self-hosting `dJev /v1/systemone` container) | Optional: Stage 4.5 + 5a automatically falls back to batched `Gemini 3.8 Flash` when GPU is not provisioned |

---

## 4. Enterprise Security & Organization Policy Compatibility (Unilever Landing Zone & Argolis)

| Organization Policy / Security Control | Enterprise Enforcement | How `feat/unified-cloud-e2e` Complies Out of the Box |
| :--- | :--- | :--- |
| `constraints/storage.uniformBucketLevelAccess` | **Enforced** (Legacy ACLs blocked) | `ensure_argolis_infra()` in [`src/utils/cloud.py`](../src/utils/cloud.py#L112) creates all buckets with `"uniformBucketLevelAccess": {"enabled": True}`. |
| `constraints/iam.disableServiceAccountKeyCreation` | **Enforced** (No JSON SA keys allowed) | **100% Keyless Auth**: Uses Application Default Credentials (`google.auth.default()`) and attached Service Account metadata identity across Cloud Run, Cloud Build, Vertex AI, and AlloyDB IAM mTLS. |
| `constraints/run.allowedIngress` | **Internal / Load Balancer / IAP Only** | Supports authenticated invocation via `gcloud run services proxy perfect-store-control-plane --port=8080` or Internal Application Load Balancer + Identity-Aware Proxy (IAP). |
| **VPC Service Controls (VPC-SC) & Private Google Access** | **Enforced in Client Landing Zones** | All API calls (`aiplatform`, `storage`, `run`, `cloudtrace`, `logging`) use standard Google APIs compatible with `restricted.googleapis.com` VIPs and Direct VPC Egress (`PRIVATE_RANGES_ONLY` in `src/utils/cloud.py`). |

---

## 5. Turnkey Provisioning Script for Unilever Cloud Admin

A Unilever Project Admin can run the script below once to enable all APIs, create the dedicated runtime Service Account, bind least-privilege roles to the FDE/Developer email, and provision the buckets and Artifact Registry repository:

```bash
export PROJECT_ID="unilever-shelf-understanding"   # Replace with target Unilever GCP Project ID
export REGION="us-central1"                        # Or asia-south1 (Mumbai)
export ENGINEER_EMAIL="jjuneja@google.com"         # FDE / Developer principal to grant access
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

# 1. Enable Mandatory GCP APIs
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

# 2. Create Dedicated Least-Privilege Runtime Service Account (Optional; or use default Compute SA)
export RUNTIME_SA_NAME="sa-shelf-bench-runtime"
export RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create "${RUNTIME_SA_NAME}" \
  --display-name="Unilever Shelf Understanding Cloud Run Runtime SA" \
  --project="${PROJECT_ID}" || true

# 3. Grant Runtime Roles to Service Account (and Default Compute SA for Cloud Build/Run compatibility)
for SA in "${RUNTIME_SA}" "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"; do
  for ROLE in \
    roles/aiplatform.user \
    roles/storage.objectAdmin \
    roles/artifactregistry.writer \
    roles/cloudtrace.agent \
    roles/logging.logWriter \
    roles/serviceusage.serviceUsageConsumer; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
      --member="serviceAccount:${SA}" \
      --role="${ROLE}" \
      --quiet
  done
done

# 4. Grant Deployment & Execution Roles to FDE / Developer Principal
for ROLE in \
  roles/run.admin \
  roles/cloudbuild.builds.editor \
  roles/artifactregistry.repoAdmin \
  roles/storage.admin \
  roles/iam.serviceAccountUser \
  roles/serviceusage.serviceUsageConsumer \
  roles/aiplatform.user \
  roles/cloudtrace.agent \
  roles/logging.logWriter \
  roles/viewer; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="user:${ENGINEER_EMAIL}" \
    --role="${ROLE}" \
    --quiet
done

# 5. Bootstrap Buckets, Artifact Registry, Datasets & Splits
python3 -m src.cli --project "${PROJECT_ID}" --region "${REGION}" bootstrap
```
