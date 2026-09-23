.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: help setup test lint typecheck check wheel ui clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install the package with dev extras
	python3 -m venv $(VENV) || true
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e '.[dev]'
	@echo "Done. Next: make test"

test: ## Run the offline test lane (no network, no GCP credentials)
	$(PY) -m pytest -q -m "not live"

lint: ## Check formatting and lint rules
	$(PY) -m ruff check .

lint-fix: ## Auto-fix what ruff can fix
	$(PY) -m ruff check --fix .

typecheck: ## Run mypy
	$(PY) -m mypy

wheel: ## Build a wheel and verify the bundled fixture is packaged
	rm -rf dist
	$(PY) -m build --wheel
	$(PY) -c "import glob,zipfile,sys; \
	names=zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist(); \
	fx=[n for n in names if '_fixtures/' in n]; \
	sys.exit(0) if fx else sys.exit('FAIL: wheel contains no _fixtures/*.png')"
	@echo "Wheel OK (fixture packaged)."

check: lint typecheck test wheel ## Everything CI runs

ui: ## Serve the dashboard locally (run a benchmark first so reports/ is populated)
	$(PY) ui/server.py

clean: ## Remove build and test artifacts
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
