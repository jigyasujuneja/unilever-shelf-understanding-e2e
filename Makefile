.DEFAULT_GOAL := help
PY := .venv/bin/python
SB := .venv/bin/shelf-bench
# Every approach runs only the models it calls; other combinations are skipped.
A ?= single_pass_dedup tiled_dedup hierarchy_classify embedding_text_match djev_classify coarse_jev_gemini \
     embedding_retrieval tiered_hybrid jev_laya_hybrid coarse_jev_laya \
     detect_retrieve detect_tiered detect_jev_laya detect_coarse_jev_gemini \
     shelf_detect_retrieve shelf_detect_tiered shelf_jev_laya shelf_coarse_jev_gemini
# gemini-3.8-flash and multimodalembedding@001 lost to these on every tab (F2 per ₹); add them with M=.
M ?= gemini-3.5-flash-lite gemini-embedding-2
PORT ?= 8080
PROXY_PORT ?= 8083

.PHONY: help setup auth bootstrap test lint qc run cloud pull ui ui-pull ui-deploy ui-cloud ui-proxy legacy-eval data rpc docs clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install (uses uv if present, else python3 -m venv)
	@command -v uv >/dev/null && { uv venv -q --allow-existing .venv && uv pip install -q --python $(PY) -e '.[dev]'; } \
	  || { python3 -m venv .venv && $(PY) -m pip install -q -e '.[dev]'; }
	@touch $(SB)
	@echo "Done. Next: make auth (once), then make test / make ui"

# Any target that needs the venv installs it first (and reinstalls when pyproject.toml changes).
$(SB): pyproject.toml
	@$(MAKE) --no-print-directory setup

auth: ## Log in to GCP (Application Default Credentials) for run / cloud / pull / ui
	gcloud auth application-default login

bootstrap: $(SB) ## New GCP project (config.yaml gcp.project): APIs, bucket, registry, labelled products
	$(SB) bootstrap

test: $(SB) ## Offline tests (no network, no GCP)
	$(PY) -m pytest -q

lint: $(SB) ## Ruff linter check across src, tests, scripts, docs
	$(PY) -m ruff check src tests scripts docs

qc: $(SB) ## Run 12-gate dataset & benchmark quality control verification
	$(SB) qc $(ARGS)

run: $(SB) ## Local dev run: A="approaches" M="models" ARGS="--limit 5"
	$(SB) run -a $(A) -m $(M) $(ARGS)

cloud: $(SB) ## Leaderboard runs on Cloud Run (same A= M= ARGS= overrides), results auto-pulled
	$(SB) cloud-run -a $(A) -m $(M) $(ARGS)

pull: $(SB) ## Re-fetch finished Cloud Run results from GCS (if `make cloud` was interrupted)
	$(SB) pull

ui: $(SB) ## Run local UI on http://localhost:8080 (Cloud Shell Web Preview; live-reloads app.js/cio.js on refresh)
	$(SB) serve --port $(PORT) $(UI_ARGS)

ui-pull: $(SB) ## Pull latest runs from GCS and start local UI on http://localhost:8080
	$(SB) serve --port $(PORT) --pull $(UI_ARGS)

ui-deploy: $(SB) ## Deploy UI changes (app.js / cio.js / server.py) to Cloud Run service (shelf-bench-ui)
	$(SB) serve-cloud

ui-cloud: ui-deploy ## Alias for ui-deploy (deploy the UI as a private Cloud Run service)

ui-proxy: $(SB) ## Open the private Cloud Run UI on http://localhost:8083 (authenticated proxy)
	$(SB) proxy --port $(PROXY_PORT)

legacy-eval: $(SB) ## Export legacy_mapping_table.jsonl or score joined legacy Brand + Pack + Variant predictions
	$(SB) legacy-eval $(ARGS)

data: $(SB) ## Optional: local copy of SKU-110K (~12 GB) in data/, used instead of GCS
	$(SB) download

rpc: $(SB) ## RPC retrieval / end-to-end set from Hugging Face (~20 min once) -> data/rpc + GCS
	$(SB) rpc

docs: ## Regenerate docs/images from docs/diagrams/*.mmd (needs uv)
	@command -v uv >/dev/null || { echo "make docs needs uv: https://docs.astral.sh/uv/"; exit 1; }
	uv run -q --no-project --with playwright playwright install chromium
	uv run -q --no-project --with playwright python docs/render.py

clean: ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
