# Unilever Perfect Store AI: Unified Cloud Gondola Intelligence & `shelf-bench` Control Plane

**Owners:** Jigyasu Juneja (`jjuneja@google.com`), Riley Gavigan (`rgavigan@google.com`), Anil (`cloud-gtm`)  
**Repository:** `https://github.com/cloud-gtm/unilever-shelf-understanding-with-cv` (`feat/unified-cloud-e2e`)

## Onboarding Overview

Hindustan Unilever Limited (`HUL`) processes **500,000 retail shelf photographs per day** across Modern Trade hypermarkets and General Trade (`Shikhar` / `Sales EDGE`) outlets under two operational SLAs and FinOps invariants:

* **Marketshare Workflow (`5 to 7` overlapping aisle photos per request):** Returns On-Shelf Availability (`OSA`), closed-catalog HUL 7-dimension SKU identification, open-set Non-HUL competitor identification, and a 4-factor replenishment order within **`<= 30.0s` E2E** (`<= 20.0s` server `P95`).
* **Merchandising Workflow (`1` bay photo per request):** Returns planogram sequence compliance, eye-level (`Red Line`) alignment, brand-block contiguity, and promotional header (`Toker`) audit within **`<= 10.0s` E2E**.
* **EPIC Framework Architecture (`Endpoints`, `Pipelines`, `Identification`, `Compliance`):**
  * [`src/core/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/) decouples SKU localization ([`detection.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/detection.py)), multimodal `gemini-embedding-2-preview` + ScaNN/pgvector retrieval ([`matching.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/matching.py)), and cost-aware Gemini 3.8 Flash / `/v1/systemone` escalation ([`fallback.py`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/core/fallback.py)).
  * [`src/approaches/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/approaches/) and [`src/stages/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/src/stages/) register all 23 benchmark approaches and 6 modular stage groups across all 6 EPICs.
  * [`web/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/web/) hosts the **Perfect Store Control Plane** (`Decision-First` EPIC Validation UI + Leaderboard Arena).
  * [`docs/tdd/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/tdd/) contains Technical Design Documents and ADRs ([`adr_001_unified_e2e_routing.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/tdd/adr_001_unified_e2e_routing.md), [`TDD_GOOGLE_DESIGN_DOC.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/tdd/TDD_GOOGLE_DESIGN_DOC.md)), while [`docs/trials/`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/trials/) isolates experimental validation harnesses and benchmark snapshots.

## Section 8: End-to-End Quickstart (`Local`, `Argolis Cloud Run`, and `Control Plane UI`)

See [`docs/HOW_TO_GUIDE.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/HOW_TO_GUIDE.md) and [`docs/UI_HOW_TO_GUIDE.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/docs/UI_HOW_TO_GUIDE.md) for the complete operational manual, and [`reports/EXECUTIVE_BENCHMARK_AND_KPI_SCORECARD.md`](file:///usr/local/google/home/jjuneja/jjuneja-unilever-shelf-understanding/reports/EXECUTIVE_BENCHMARK_AND_KPI_SCORECARD.md) for full benchmark tables.

### 1. Install Dependencies & Inspect Registered EPIC Approaches
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -e ".[dev]"

.venv/bin/shelf-bench list
.venv/bin/shelf-bench leaderboard
```

### 2. Bootstrap Any Argolis GCP Project and Run End-to-End (`Vertex AI` + `Cloud Run Jobs`)
```bash
# Option A: Turnkey shell automation
./scripts/bootstrap_project.sh <YOUR_ARGOLIS_PROJECT_ID> us-central1

# Option B: Direct CLI provisioning (enables 9 GCP APIs, UBLA buckets, resource IAM, and syncs SKU-110K + HUL catalog)
.venv/bin/shelf-bench --project <YOUR_ARGOLIS_PROJECT_ID> bootstrap --region us-central1

# Run live Vertex AI + Gemini Embedding 2 (gemini-embedding-2-preview) benchmarks across EPICs
.venv/bin/shelf-bench --project <YOUR_ARGOLIS_PROJECT_ID> run \
  -a hul_hierarchy_classifier sister_shade_systemone scann_vector_retriever hul_8stage_gemini38_hybrid \
  -m gemini-3.8-flash \
  --split test \
  --limit 10 \
  --workers 5

# Build container on Cloud Build, execute on Cloud Run Jobs, and pull priced results back to results/
./scripts/deploy_cloud_run.sh <YOUR_ARGOLIS_PROJECT_ID> us-central1
```

### 3. Launch the Perfect Store Control Plane UI (`Decision-First` Reviewer + Leaderboard Arena)
```bash
.venv/bin/shelf-bench serve --port 8080
# Open http://127.0.0.1:8080 for the EPIC Decision-First Audit Workspace & Leaderboard Arena
# REST APIs: GET /api/v1/audits | POST /api/v1/audits/<id>/review | GET /api/leaderboard | GET /api/approaches
```

### 4. Run Automated Unit Tests, Linter, and Trial Diagnostics
```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/python docs/trials/jetski_validation_suite.py
```
