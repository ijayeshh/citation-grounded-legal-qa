# CLAUDE.md

## Project

Citation-grounded legal QA over CUAD commercial contracts: answers carry **span-level citations**, and a **faithfulness evaluation harness** measures whether every claim is actually entailed by the source span it cites (scored against CUAD gold spans). A portfolio piece for legal-AI hiring — the differentiator is *measurable, objective faithfulness*, not a RAG demo.

**Single thesis:** grounding + inline verification produces faithful, well-cited answers, provable with objective numbers.

## Current Phase

**Phase 0 — Skeleton.** Root has `PROJECT_PLAN.md`, `config.yaml`, `requirements.txt`, `Makefile`, `env.example`. No `src/` yet. See `PROJECT_PLAN.md` for the full 4-phase build plan.

## Tech Stack

- **Data:** CUAD (510 EDGAR contracts, 41 clause types, gold spans) via HuggingFace `datasets`
- **Retrieval:** BM25 (`rank-bm25`) + dense (`sentence-transformers`) + cross-encoder reranker; FAISS local index; RRF fusion
- **Generation:** Mistral La Plateforme — `mistral-small-latest` (dev), `mistral-large-latest` (final)
- **Verification:** NLI model (`cross-encoder/nli-deberta-v3-base`) for the groundedness gate
- **Orchestration:** LangGraph (verifier loop ONLY)
- **Eval:** faithfulness harness + gold-span citation metrics (P/R/IoU) + LegalBench
- **API/UI:** FastAPI backend, Streamlit frontend, MCP retriever server
- **Infra:** SQLite (logging), Langfuse (tracing), promptfoo (CI), pytest
- Python 3.11+, **CPU-only path must work** (GPU never required)

## Repo Structure (target — see PROJECT_PLAN.md §0.2)

```
config/config.yaml       # all tunables (currently at root as config.yaml)
data/{raw,processed}/    # CUAD download, chunks, gold-span store, FAISS index
src/
  ingest/     # download + parse + chunk CUAD, gold-span store
  retrieval/  # bm25.py, dense.py, reranker.py, hybrid.py
  index/      # faiss_store.py
  generation/ # mistral_client.py (retry/backoff), prompts.py
  graph/      # langgraph retrieve→answer→verify loop
  verify/     # claim_decomposition.py, nli.py, groundedness_gate.py
  eval/       # harness.py, citation_metrics.py, legalbench_runner.py
  schemas/    # pydantic: Answer, Claim, Citation
  persistence/ observability/ api/ mcp/
app/streamlit_app.py
promptfoo/  tests/  eval_reports/  .github/workflows/ci.yml
```

## Coding Conventions

- **Config-driven:** every tunable (models, top-k, chunk size, thresholds) lives in `config.yaml`, never hardcoded. No secrets in config — secrets go in `.env`.
- **Preserve character offsets** end-to-end (chunk → citation → gold-span IoU). Non-negotiable.
- **Load heavy models once.** The NLI model is a singleton reused across gate + harness — never instantiate twice. Streamlit uses `@st.cache_resource`.
- **Pydantic schemas** are the contract between generation, verify, API, MCP, UI. Enforce structured LLM output with Pydantic parsing + retry.
- **All LLM calls go through `mistral_client.py`** with retry + exponential backoff (RPM is the binding constraint).
- **LangGraph only for the verifier loop;** retrieval/citation logic stays plain Python.
- **Reproducibility first:** pin deps; `make setup → index → eval → run` must work from a clean clone.
- **Tests:** pytest, mock the LLM to keep tests fast/offline.

## Do NOT add (explicitly out of scope)

- No baseline/ablation comparison anywhere (frontend, harness, or README).
- No generic safety guardrails (PII/toxicity/jailbreak) — the groundedness gate is the *one* deliberate guardrail.
- No vLLM/self-hosted inference, Docker, Redis, queues, Kubernetes, or cloud deployment. **Local only.**
- Contamination control: split LegalBench eval **by contract, not by clause**; document it.
