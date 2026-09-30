# GCP access needed to run shelf-bench

What a GCP project and the people using it need for everything in the [README](../README.md):
local runs, Cloud Run benchmark jobs, and the private UI service. Nothing here is needed for
`make test`, which is offline.

## APIs

`make bootstrap` enables these ([cloud.py](../src/utils/cloud.py) `APIS`); an admin can also
enable them up front.

| API | Used for | Called from |
|-----|----------|-------------|
| `aiplatform.googleapis.com` | Gemini (`gemini-*`), `multimodalembedding@001` and `gemini-embedding-2-preview` | [llm.py](../src/utils/llm.py), [embeddings.py](../src/utils/embeddings.py) |
| `run.googleapis.com` | The `shelf-bench` Cloud Run job and the private UI service | [cloud.py](../src/utils/cloud.py) |
| `cloudbuild.googleapis.com` | Building the container from the working tree | [cloud.py](../src/utils/cloud.py) |
| `artifactregistry.googleapis.com` | The `cloud-run-source-deploy` image repo | [cloud.py](../src/utils/cloud.py) |
| `storage.googleapis.com` | Datasets, results, Cloud Build sources | [dataset.py](../src/utils/dataset.py), [cloud.py](../src/utils/cloud.py) |
| `cloudbilling.googleapis.com` | Live SKU prices for cost per image | [pricing.py](../src/utils/pricing.py) |
| `cloudtrace.googleapis.com`, `logging.googleapis.com` | Run traces and linked logs | [telemetry.py](../src/utils/telemetry.py) |

`serviceusage.googleapis.com` must already be on for `bootstrap` to enable the rest (it is on by
default in new projects).

Only if you turn on the AlloyDB retrieval template
([_detect_retrieve_template.py](../src/approaches/_detect_retrieve_template.py)):
`alloydb.googleapis.com`, plus `compute.googleapis.com` and `servicenetworking.googleapis.com` for a
private-IP instance reached through Direct VPC egress (`cloud_run.network` / `cloud_run.subnet`).

## Roles

### The person running `make run` / `make cloud` / `make ui-cloud`

| Role | Why |
|------|-----|
| `roles/aiplatform.user` | Call Gemini and embeddings (local runs) |
| `roles/storage.objectAdmin` on the data bucket (`roles/storage.admin` to let `bootstrap` create buckets) | Read datasets, read/write results, upload build sources |
| `roles/cloudbuild.builds.editor` | Start the image build |
| `roles/artifactregistry.writer` (`repoAdmin` to let `bootstrap` create the repo) | Push the image |
| `roles/run.developer` | Create/update/execute the job and the UI service |
| `roles/iam.serviceAccountUser` on the runtime service account | Attach it to the job and service |
| `roles/serviceusage.serviceUsageConsumer` (`serviceUsageAdmin` for `bootstrap`) | Quota project for API calls |
| `roles/cloudtrace.agent`, `roles/logging.logWriter` | Export telemetry from local runs |
| `roles/run.invoker` on the UI service | Open it through `make ui-proxy` |

### The Cloud Run runtime service account

The job and the UI service run as the project's default Compute Engine service account unless you
change it. It needs:

| Role | Why |
|------|-----|
| `roles/aiplatform.user` | Gemini and embeddings |
| `roles/storage.objectAdmin` on the data bucket | Read datasets, write results (the UI service writes back re-priced summaries on pull) |
| `roles/cloudtrace.agent`, `roles/logging.logWriter` | Telemetry |
| `roles/serviceusage.serviceUsageConsumer` | Quota project for API calls |
| `roles/alloydb.client` + an IAM database user | Only for the AlloyDB template |

The Cloud Build service account needs to read the source bucket (`run-sources-<project>-<region>`)
and write to the Artifact Registry repo; new projects grant this by default.

## Quota

| Quota | What drives it |
|-------|----------------|
| Vertex AI requests per minute per model | `cloud_run.parallelism` concurrent tasks, each running images concurrently. Tasks are ordered so concurrent ones use different models |
| Cloud Run CPU in the region | `cloud_run.parallelism` x `cloud_run.cpu` (2 vCPU per task by default) + 1 vCPU for the UI service |

## Security

* No service account keys: everything uses Application Default Credentials or the attached
  service account.
* Buckets are created with uniform bucket-level access.
* The UI service is private (no `allUsers` invoker); open it with the authenticated proxy.
