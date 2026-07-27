# CLAUDE.md

## Project

Citation-grounded legal QA over CUAD commercial contracts: answers carry **span-level citations**, and a **faithfulness evaluation harness** measures whether every claim is actually entailed by the source span it cites (scored against CUAD gold spans). A portfolio piece for legal-AI hiring — the differentiator is *measurable, objective faithfulness*, not a RAG demo.

**Single thesis:** grounding + inline verification produces faithful, well-cited answers, provable with objective numbers.

## Current Phase

**All four phases complete.** The full pipeline (ingest → retrieval → generation →
verification → eval → API/UI/MCP) is built, tested, and pushed to `origin/main`.
Cross-Phase Checklist: 19/19 pass. See `PROJECT_PLAN.md` for the plan.

- **Phase 1 (1.2–1.7):** CUAD ingest → `contracts`/`gold_spans` (exact offsets) →
  sliding-window chunking → FAISS + BM25 → hybrid RRF + cross-encoder reranker. `make index`
  (`--raw-path`, `--no-retrieval`, `--max-chars/--overlap-chars`); `make smoke`.
- **Phase 2 (2.1–2.8):** `Citation`/`Claim`/`Answer` schemas; `MistralClient` (retry/backoff);
  cited-answer generation w/ parse-retry; claim decomposition; shared NLI singleton;
  groundedness gate; LangGraph verifier loop (`graph/pipeline.py`, `answer_question()`).
- **Phase 3 (3.1–3.6):** `eval/harness.py` (NLI faithfulness + LLM-judge cross-check),
  `citation_metrics.py` (gold-span P/R/IoU), `legalbench_runner.py` (non-CUAD contract_nli,
  contamination-controlled), committed `eval_reports/`, promptfoo + GitHub Actions CI.
  `make eval`; `--large` for the Mistral Large headline.
- **Phase 4 (4.1–4.6):** FastAPI backend (single boundary, models loaded once via lifespan,
  Langfuse trace, SQLite log); Streamlit UI (highlighted citations, live score, red flags);
  MCP `search_contracts`; finalized README (headline numbers, mermaid diagrams, scope statement).
- **Tests:** 72 passing (`pytest -q`), all offline (LLM + NLI mocked).

**Headline (sample snapshot, Mistral Large):** faithfulness NLI 0.42 / judge 0.78,
citation recall 1.00 / IoU 0.18, LegalBench 0.875. Numbers are from the 2-contract offline
snapshot; full-corpus run is `make setup && make index && make eval` in a clean venv.

**Env notes:** live CUAD download needs pinned `datasets==2.20.0` (script loader; newer
`datasets` drops script support). Dataset id `theatticusproject/cuad-qa`. Bumped `mcp==1.28.1`
+ `pydantic==2.13.4` for the FastMCP tool API. Real `MISTRAL_API_KEY` in `.env` (not committed);
CI needs it as a GitHub Actions secret for the promptfoo step.

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
