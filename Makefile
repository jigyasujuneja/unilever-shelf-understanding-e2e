.DEFAULT_GOAL := help
PY := .venv/bin/python
SB := .venv/bin/shelf-bench
# Every approach runs only the models it calls; other combinations are skipped.
A ?= single_pass_dedup tiled_dedup hierarchy_classify embedding_text_match embedding_retrieval \
     tiered_hybrid detect_retrieve detect_tiered shelf_detect_retrieve shelf_detect_tiered
# gemini-3.8-flash and multimodalembedding@001 lost to these on every tab (F2 per ₹); add them with M=.
M ?= gemini-3.5-flash-lite gemini-embedding-2-preview

.PHONY: help setup auth bootstrap test lint run cloud pull ui ui-cloud ui-proxy data rpc docs clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

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

lint: $(SB) ## Ruff
	$(PY) -m ruff check src tests docs

run: $(SB) ## Local dev run: A="approaches" M="models" ARGS="--limit 5"
	$(SB) run -a $(A) -m $(M) $(ARGS)

cloud: $(SB) ## Leaderboard runs on Cloud Run (same A= M= ARGS= overrides), results auto-pulled
	$(SB) cloud-run -a $(A) -m $(M) $(ARGS)

pull: $(SB) ## Re-fetch finished Cloud Run results from GCS (if `make cloud` was interrupted)
	$(SB) pull

ui: $(SB) ## Leaderboard UI on http://localhost:8080 (images stream from GCS)
	$(SB) serve

ui-cloud: $(SB) ## Deploy the UI as a private Cloud Run service (shows every run in GCS)
	$(SB) serve-cloud

ui-proxy: $(SB) ## Open the private Cloud Run UI on http://localhost:8083 (authenticated proxy)
	$(SB) proxy

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
