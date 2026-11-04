# FaultTrack: run every target from the repo root.
# Python targets run inside the devcontainer; docker-* and deploy run on the host only.

.DEFAULT_GOAL := help
.PHONY: help lint typecheck test data features train serve docker-build docker-run deploy

# Each target's "## text" comment is what `make help` prints.

help: ## List the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

lint: ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Type check with basedpyright (settings in [tool.pyright])
	uv run basedpyright

test: ## Run the test suite (pytest)
	uv run pytest

data: ## Download RCAEval RE2 from Zenodo into data/raw (metrics only, about 4 GB of zips)
	DEST=data/raw KEEP_ZIPS=0 bash spike/download_rcaeval.sh

features: ## Rebuild data/features/re2_features.parquet + re2_inventory.parquet from data/raw
	uv run python -m faulttrack.build

train: ## Train the final model into models/ and web/results.json (plan line 8)
	@echo "make train: not implemented yet (plan line 8)" >&2; exit 1

serve: ## Serve the API locally on port 8080 (plan line 9)
	uv run uvicorn predict:app --port 8080

docker-build: ## Build the service image; host only (plan line 10)
	@echo "make docker-build: not implemented yet (plan line 10)" >&2; exit 1

docker-run: ## Run the service image; host only (plan line 10)
	@echo "make docker-run: not implemented yet (plan line 10)" >&2; exit 1

deploy: ## Deploy to Fly.io; maintainer only (plan line 11)
	@echo "make deploy: not implemented yet (plan line 11)" >&2; exit 1
