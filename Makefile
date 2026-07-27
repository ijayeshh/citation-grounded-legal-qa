# ============================================================================
# One-command entrypoints. Reproducibility is a first-class feature:
# clone -> make setup -> make index -> make eval -> make run
# ============================================================================

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup index smoke eval run api ui mcp ci test clean

help:
	@echo "Targets:"
	@echo "  setup   - create venv (Python 3.11+) and install pinned deps"
	@echo "  index   - Phase 1: ingest CUAD, chunk, build FAISS + BM25 indexes"
	@echo "  smoke   - Phase 1: retrieval smoke test over persisted indexes (needs 'make index')"
	@echo "  eval    - Phase 3: run faithfulness harness + citation metrics + LegalBench"
	@echo "  run     - Phase 4: start FastAPI backend + Streamlit UI locally"
	@echo "  api     - start FastAPI backend only"
	@echo "  ui      - start Streamlit UI only"
	@echo "  mcp     - start the MCP retriever server"
	@echo "  ci      - run unit tests + promptfoo (as in GitHub Actions)"
	@echo "  test    - run unit tests only"
	@echo "  clean   - remove venv and generated artifacts"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ">> Setup done. Copy .env.example to .env and fill in keys."

index:
	$(PY) -m src.ingest.build_index --config config/config.yaml

# Retrieval smoke test over the persisted FAISS + BM25 indexes built by `make index`.
# Use `--demo` for a self-contained run that needs no prior indexing.
smoke:
	$(PY) -m src.retrieval.smoke --from-disk --config config/config.yaml

eval:
	$(PY) -m src.eval.harness --config config/config.yaml
	$(PY) -m src.eval.legalbench_runner --config config/config.yaml
	@echo ">> Eval reports written to eval_reports/"

# Starts backend and frontend together (backend in background).
run:
	$(PY) -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 & \
	$(VENV)/bin/streamlit run app/streamlit_app.py

api:
	$(PY) -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

ui:
	$(VENV)/bin/streamlit run app/streamlit_app.py

mcp:
	$(PY) -m src.mcp.server

# promptfoo is a Node tool — run via npx (requires Node.js installed).
# This mirrors .github/workflows/ci.yml.
ci: test
	npx --yes promptfoo@latest eval -c promptfoo/promptfooconfig.yaml

test:
	$(PY) -m pytest -q

clean:
	rm -rf $(VENV) data/processed/* eval_reports/* .pytest_cache __pycache__
