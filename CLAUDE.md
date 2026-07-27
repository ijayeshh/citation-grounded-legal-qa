# CLAUDE.md

## Project

Citation-grounded legal QA over CUAD commercial contracts: answers carry **span-level citations**, and a **faithfulness evaluation harness** measures whether every claim is actually entailed by the source span it cites (scored against CUAD gold spans). A portfolio piece for legal-AI hiring — the differentiator is *measurable, objective faithfulness*, not a RAG demo.

**Single thesis:** grounding + inline verification produces faithful, well-cited answers, provable with objective numbers.

## Current Phase

**Phase 3 — Faithfulness harness, LegalBench & CI (next up).** Phases 0–2 are done, committed, and pushed to `origin/main`. See `PROJECT_PLAN.md` for the full 4-phase plan.

Done so far:
- **Phase 1 (1.2–1.7):** CUAD ingest → normalize (`contracts`, `gold_spans` with exact offsets) → sliding-window chunking → FAISS + BM25 → hybrid RRF + cross-encoder reranker. `make index` runs it (`--raw-path` to index from a saved snapshot, `--no-retrieval` to stop after chunking). `make smoke` / `src/retrieval/smoke.py` (`--demo` self-contained, `--from-disk` real artifacts).
- **Phase 2 (2.1–2.8):** `Citation`/`Claim`/`Answer` schemas; `MistralClient` (retry/backoff, `from_config`); cited-answer generation with parse-retry; claim decomposition; shared NLI singleton; groundedness gate; LangGraph verifier loop (`src/graph/pipeline.py`, `answer_question()`).
- **Tests:** 56 passing (`pytest -q`), all offline (LLM + NLI mocked).

Next (Phase 3): `src/eval/harness.py`, `citation_metrics.py`, `legalbench_runner.py`; contamination split; `eval_reports/`; promptfoo + GitHub Actions.

**Env note:** live CUAD download needs pinned `datasets==2.20.0` (script-based loader) in a clean venv; newer `datasets` drops script support. Dataset id is `theatticusproject/cuad-qa`. A real `MISTRAL_API_KEY` is in `.env` (not committed).

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
config/config.yaml       # all tunables
data/{raw,processed}/    # CUAD download, chunks, gold-span store, FAISS index (gitignored)
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
- **Preserve character offsets** end-to-end (chunk → citation → gold-span IoU). Non-negotiable. **Citation offsets come from the retrieved span the model cites (by id), never emitted by the LLM.**
- **Injectable seams for offline tests:** `MistralClient` takes a `transport` + `sleep_fn`; retrievers/reranker/NLI are injected into the gate and graph. Tests mock these — no network, no model downloads.
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
