# Citation-Grounded Legal QA with a Faithfulness Evaluation Harness
### Build Plan for Claude Code

---

## 0. Project Summary (read this first)

**What we are building:** A legal question-answering system over commercial contracts that returns answers with **span-level citations**, and — the differentiator — a **faithfulness evaluation harness** that rigorously measures whether every claim the system makes is actually supported by the source span it cites.

**Why it exists:** This is a portfolio piece aimed at getting hired by legal-AI companies (Harvey, DeepJudge, Legora, EvenUp). Those companies live or die on *grounding and verifiable citations*. The scarce, hire-worthy skill is **evaluation** — being able to quantify hallucination and citation faithfulness, not just demo a RAG app. The whole project is built to make trust **measurable and visible**.

**The single thesis to demonstrate:** Grounding + inline verification produces faithful, well-cited answers, and we can prove faithfulness with objective numbers (using CUAD gold spans), not assertions.

**Guiding engineering principles (apply throughout):**
1. **Reproducibility is a first-class feature.** Pinned dependencies, one-command eval, faithfulness numbers front-and-center in the README. A reviewer must be able to `git clone` and run.
2. **Iterate cheap, then measure rigorously.** Build and iterate on the small model; produce final headline numbers on the large model.
3. **Keep the pipeline mechanics visible.** Prefer plain Python for retrieval/citation logic; use frameworks only where they earn their place.
4. **Scope controls to the real threat model.** No generic safety guardrails (no PII/toxicity/jailbreak tooling) — the corpus is public, there is no adversarial user, no private data. The one real risk is ungrounded claims; the groundedness gate handles that.
5. **Load heavy models once, reuse everywhere.** The NLI model in particular does triple duty (offline eval, inline verifier gate, harness entailment) — instantiate it once.

**Explicitly OUT of scope (do not add):**
- No self-hosted inference / vLLM (removed to stay GPU-independent).
- No generic safety guardrails (PII, toxicity, jailbreak, off-topic filters).
- No baseline / ablation comparison anywhere — not in the frontend, not in the harness, not in the README. The system is presented as a single full-method pipeline.
- No Redis, message queues, Kubernetes, Docker, or cloud deployment. **Runs locally only.**
- No production serving infrastructure. This is a research/evaluation project built *with* production-grade discipline, but deliberately not productionized.

---

## 0.1 Final Technology Stack (locked)

| Layer | Choice | Role |
|---|---|---|
| Dataset | **CUAD** | 510 public EDGAR contracts, 41 clause types, expert gold spans |
| Retrieval | **BM25 + dense + cross-encoder reranker** | Hybrid retrieval |
| Vector store | **FAISS** (local) | Dense index, reproducible, no hosted dependency |
| ML libs | **HuggingFace** `datasets`, `sentence-transformers`, an **NLI model** | Data, embeddings, entailment |
| Orchestration | **LangGraph** | The stateful verifier loop only |
| Generation LLM | **Mistral** (La Plateforme free tier) — **Small first, Large for final** | Answer generation |
| Eval (differentiator) | **Faithfulness harness**: claim decomposition → NLI → gold-span citation metrics | Core measurement |
| Benchmark | **LegalBench** | Named headline number (with contamination control) |
| CI | **promptfoo** | Prompt-regression gate in GitHub Actions |
| Frontend | **Streamlit** | Full-method demo: answer + highlighted citations + live faithfulness score |
| Backend | **FastAPI** | One service boundary, three clients (UI, MCP, promptfoo) |
| Integration | **MCP** | Retriever exposed as a governed tool (integration-ready) |
| Persistence | **SQLite** | Log questions + faithfulness scores |
| Observability | **Langfuse** | Per-request tracing (retrieved / answered / flagged) |

**LLM provider note:** Mistral La Plateforme free "Experiment" tier (~1B tokens/month, all models). The binding constraint is **requests-per-minute**, not token volume, because the harness makes many sequential calls per question. Build retry/backoff from day one. Data-training opt-in is acceptable here (public corpus only). If RPM is too tight, fall back to Groq or OpenRouter — verify live limits in the provider console before committing.

---

## 0.2 Repository Layout (create this skeleton first)

```
legal-qa-faithfulness/
├── README.md                  # headline faithfulness numbers up top
├── requirements.txt           # pinned versions
├── .env.example               # MISTRAL_API_KEY, LANGFUSE_* etc. (never commit real keys)
├── Makefile                   # one-command entrypoints (setup, index, eval, run, ci)
├── config/
│   └── config.yaml            # models, paths, top-k, chunk sizes, thresholds
├── data/
│   ├── raw/                   # CUAD download
│   └── processed/             # chunked contracts, gold-span mappings
├── src/
│   ├── ingest/                # download + parse + chunk CUAD, build gold-span store
│   ├── retrieval/             # bm25.py, dense.py, reranker.py, hybrid.py
│   ├── index/                 # faiss_store.py (build/load)
│   ├── generation/            # mistral_client.py (retry/backoff), prompts.py
│   ├── graph/                 # langgraph pipeline (retrieve→answer→verify)
│   ├── verify/                # claim_decomposition.py, nli.py, groundedness_gate.py
│   ├── eval/                  # harness.py, citation_metrics.py, legalbench_runner.py
│   ├── schemas/               # pydantic models (Answer, Claim, Citation, ...)
│   ├── persistence/           # sqlite (log queries + scores)
│   ├── observability/         # langfuse setup
│   ├── api/                   # fastapi app
│   └── mcp/                   # mcp server wrapping the retriever
├── app/
│   └── streamlit_app.py       # frontend
├── promptfoo/
│   └── promptfooconfig.yaml   # CI prompt tests
├── tests/                     # unit tests
├── eval_reports/              # generated results tables (committed)
└── .github/workflows/ci.yml   # runs promptfoo + tests on push
```

---

# PHASE 1 — Data & Retrieval Foundation

**Goal:** A working hybrid retriever over CUAD that, given a question, returns ranked contract spans — and a gold-span store that lets us later score citations objectively. No LLM yet.

**Definition of done:** `make index` builds the FAISS index and BM25 index from CUAD; a retrieval smoke test returns sensible ranked spans for sample questions; gold spans are loaded and queryable by (contract, clause type).

### 1.1 Environment & skeleton
- Create the repo skeleton above. Initialize `git`, `requirements.txt` (pinned), `.env.example`, `Makefile`, `config/config.yaml`.
- Python 3.11+. Create a virtualenv. Everything must run on **CPU** (GPU optional, never required).
- Add `README.md` stub with a "Results" section placeholder at the very top (numbers go here later).

### 1.2 Ingest CUAD
- Load CUAD via HuggingFace `datasets` (or the official CUAD release). Store raw in `data/raw/`.
- CUAD provides, per contract: full contract text, and per clause type the annotated **answer spans** (character offsets / text). Parse into a normalized structure:
  - `contracts[contract_id] = {text, metadata}`
  - `gold_spans[contract_id][clause_type] = [ {start, end, text}, ... ]`
- Persist the gold-span store to `data/processed/` (JSON or SQLite) keyed by `(contract_id, clause_type)`. **This is the asset that makes citation scoring objective — do not skip or approximate it.**

### 1.3 Chunking / segmentation
- CUAD contracts range from a few pages to **~300k characters**. Chunking is a real engineering decision reviewers will ask about — implement it deliberately and document the choice.
- Implement clause-aware / sliding-window chunking with overlap. Each chunk must retain:
  - `chunk_id`, `contract_id`, `char_start`, `char_end`, `text`.
- **Preserve character offsets** end-to-end so a retrieved chunk (and later a cited span) can be mapped back to the exact location in the source contract and compared against gold spans (IoU).
- Store processed chunks in `data/processed/`.

### 1.4 Dense retrieval
- Use `sentence-transformers` for embeddings (a legal-or-general strong embedding model; make it configurable in `config.yaml`).
- Embed all chunks; build a **FAISS** index (`src/index/faiss_store.py`) with build + load functions. Persist the index to disk.
- Store a sidecar mapping `faiss_id -> chunk metadata`.

### 1.5 Lexical retrieval (BM25)
- Implement BM25 over the same chunks (e.g. `rank_bm25` or an equivalent). Legal text needs exact-term matching for defined terms and clause names (e.g. "Change of Control", "Cap on Liability") — BM25 covers what dense misses.

### 1.6 Hybrid fusion + reranker
- `src/retrieval/hybrid.py`: run BM25 and dense in parallel, fuse candidates (e.g. Reciprocal Rank Fusion or weighted score fusion; make the method configurable).
- Add a **cross-encoder reranker** (`sentence-transformers` cross-encoder) over the fused candidate set. This is the highest-ROI retrieval component — the reranker reorders the top-N by query–passage relevance.
- Output: ranked list of chunks with scores + full provenance (contract_id, char offsets).
- Make `top_k` (candidates) and `top_n` (after rerank) configurable.

### 1.7 Retrieval smoke test
- Script that runs a handful of sample contract questions through the full hybrid+rerank pipeline and prints ranked spans. Sanity-check that clause-type questions surface the right clauses.
- Add unit tests for: chunk offset integrity, gold-span loading, fusion determinism.

**Phase 1 deliverables:** ingest + chunk pipeline, gold-span store, FAISS index, BM25 index, hybrid retriever with reranker, smoke test + unit tests, `make index`.

---

# PHASE 2 — Generation, Orchestration & the Groundedness Gate

**Goal:** Turn retrieved spans into a cited answer, then verify each claim against its cited span inline, flagging/regenerating unsupported claims. This is the full-method pipeline as a graph.

**Definition of done:** Given a question, the LangGraph pipeline retrieves → generates a cited answer → verifies each claim → returns a structured `Answer` object where every claim carries a citation and an entailment verdict; unsupported claims are flagged. All heavy models loaded once.

### 2.1 Schemas (do this before wiring anything)
- `src/schemas/`: Pydantic models — `Citation {contract_id, char_start, char_end, text}`, `Claim {text, citation, entailment_label, entailment_score}`, `Answer {question, claims[], raw_text, faithfulness_score, flagged_claims[]}`.
- These schemas are the contract between generation, verification, API, MCP, and the frontend. Get them right early.

### 2.2 Mistral client with rate-limit handling
- `src/generation/mistral_client.py`: wrapper over Mistral La Plateforme.
- **Retry + exponential backoff** on rate-limit (429) errors — mandatory, because RPM is the binding constraint and the harness is call-heavy.
- Model selectable via config: **Mistral Small** for development/iteration, **Mistral Large** for final headline runs.
- Centralize all LLM calls here (so retries, logging, and Langfuse tracing are in one place).

### 2.3 Prompts
- `src/generation/prompts.py`: answer-generation prompt that instructs the model to answer **only** from the provided retrieved spans and to attach a citation (span reference) to each factual claim, returning **structured output** matching the `Answer`/`Claim` schema.
- Enforce citation-output structure with **Pydantic parsing + retry** (this is the appropriate, transparent "guardrail" on output format — not a generic safety library). If the model returns malformed structure, re-prompt.

### 2.4 Claim decomposition
- `src/verify/claim_decomposition.py`: split a generated answer into **atomic claims** (each independently checkable). This is a prerequisite for measuring faithfulness at claim granularity. Implement via a focused LLM call with a strict rubric (and unit-test on fixed examples).

### 2.5 NLI entailment (shared component)
- `src/verify/nli.py`: load an NLI model from HuggingFace **once** (singleton / cached). For a `(claim, cited_span)` pair, return entailment label + score: does the cited span support the claim?
- **This one component is reused three ways:** (a) the inline verifier gate below, (b) the offline faithfulness harness in Phase 3, (c) the harness entailment scoring. Do not instantiate it multiple times.

### 2.6 Groundedness gate (the real guardrail)
- `src/verify/groundedness_gate.py`: for each claim, run NLI against its cited span. If a claim is **not entailed**, either (config-driven): drop it, regenerate it, or flag it. Default: flag + regenerate once, then flag if still unsupported.
- This *is* the project's guardrail — "no answer sentence ships unless it's entailed by a cited source span." Name it as such in code and README. No generic safety guardrails are added.

### 2.7 LangGraph pipeline
- `src/graph/`: model the flow as a graph with conditional edges: **retrieve → generate cited answer → decompose claims → verify (NLI gate) → [if unsupported claims: regenerate those claims → re-verify] → finalize**.
- LangGraph is used **only** here, because this is a genuine stateful loop with conditional ret/regeneration. Do not use LangChain as general glue elsewhere.
- The graph returns the finalized `Answer` object with per-claim entailment verdicts and an overall live faithfulness score (fraction of claims entailed).

### 2.8 Tests
- Unit-test the gate logic (entailed vs. not-entailed routing), schema parsing/retry, and claim decomposition on fixed inputs. Mock the LLM where possible to keep tests fast and offline.

**Phase 2 deliverables:** schemas, rate-limited Mistral client, prompts + structured-output enforcement, claim decomposition, shared NLI component, groundedness gate, LangGraph pipeline, tests.

---

# PHASE 3 — Faithfulness Harness, LegalBench & CI (the differentiator)

**Goal:** The rigorous, offline evaluation that produces the headline numbers. This is what actually gets you hired — treat it as the centerpiece, not an afterthought.

**Definition of done:** `make eval` runs the full harness over a held-out CUAD set, emits a committed results table (faithfulness + gold-span citation metrics), produces a LegalBench headline number with contamination control documented, and promptfoo runs green in CI.

### 3.1 Faithfulness harness
- `src/eval/harness.py`: for a set of evaluation questions, run the full pipeline, then score:
  1. **Claim-level faithfulness:** decompose each answer → for each claim, NLI-check against its cited span → **faithfulness = fraction of claims entailed by their citation.**
  2. Report per-question and aggregate faithfulness.
- Report **both** an NLI-model verdict and (optionally) an LLM-judge verdict with a strict rubric, and cross-check them — using two judge methods is itself a maturity signal.

### 3.2 Gold-span citation metrics (most objective number)
- `src/eval/citation_metrics.py`: compare the system's **cited spans** against CUAD **gold spans** for the relevant clause type:
  - **Precision / Recall / IoU** on character offsets (this is why we preserved offsets end-to-end in Phase 1).
- This is the most defensible, objective metric in the whole project and most people never exploit CUAD's gold spans this way. Feature it prominently.

### 3.3 Contamination control (critical, document it)
- LegalBench's `cuad_*` tasks are **derived from the same CUAD contracts.** To avoid train/test leakage:
  - **Split by contract, not by clause** — hold out a set of CUAD contracts entirely for evaluation; never use them during development/tuning.
  - **Or** evaluate on non-CUAD LegalBench tasks (e.g. contract NLI, definition classification) and/or different clause types than those used to build the system.
- Whichever is chosen, **document the split explicitly** in the README. Showing you understand data leakage is itself a hiring signal.

### 3.4 LegalBench headline number
- `src/eval/legalbench_runner.py`: run the pipeline (or the relevant sub-component) on the chosen LegalBench tasks, honoring the contamination split from 3.3. Produce one clean headline metric to report.
- Optional bonus (zero extra infra): report **Mistral Small vs Mistral Large** faithfulness on the CUAD eval — the "open-model comparison" as a cheap API-to-API study.

### 3.5 Results reporting & reproducibility
- Emit results to `eval_reports/` as committed tables (Markdown/CSV). Surface the headline faithfulness + gold-span numbers at the **top of the README**.
- `make eval` must be a **single command** that reproduces the numbers. Pin all deps.

### 3.6 promptfoo in CI (separate from the harness)
- `promptfoo/promptfooconfig.yaml`: a **small fixed** set of prompt test cases with assertions — every answer includes a citation, output parses as valid structured citations, no answer sentence lacks a supporting reference, latency sane.
- `.github/workflows/ci.yml`: on push, run unit tests + promptfoo; **fail the build** on regression.
- Keep this **separate and distinct** from the faithfulness harness: promptfoo is the cheap always-on smoke alarm (runs every push, out of the deployed app); the harness is the deep on-demand diagnostic. Do not run the heavy harness on every commit.

**Phase 3 deliverables:** faithfulness harness, gold-span citation metrics, documented contamination split, LegalBench runner + headline number, committed eval reports, README results section, promptfoo config + GitHub Actions CI.

---

# PHASE 4 — Interface, Service Boundary, Integration & Observability

**Goal:** Make the full-method system usable and inspectable locally: a Streamlit demo that makes faithfulness *visible*, behind a FastAPI backend, with an MCP wrapper, SQLite logging, and Langfuse tracing. Local only.

**Definition of done:** `make run` starts the FastAPI backend + Streamlit UI locally; asking a question shows the answer with highlighted citations and a live faithfulness score, with unsupported claims flagged in red; queries + scores are logged to SQLite; traces appear in Langfuse; the MCP server exposes the retriever as a tool.

### 4.1 FastAPI backend (the one service boundary)
- `src/api/`: expose the full pipeline as an API — e.g. `POST /answer` returns the structured `Answer` (claims, citations, entailment verdicts, faithfulness score).
- This single backend is consumed by **three clients**: the Streamlit UI, the MCP server, and promptfoo. Keep business logic here, not in the UI.
- Wire **Langfuse** tracing at this layer so every request records what was retrieved, generated, and flagged.

### 4.2 Streamlit frontend (full method only — NO baseline)
- `app/streamlit_app.py`: single-system showcase. For a submitted question, display:
  - the answer,
  - **cited spans highlighted** in the source text,
  - the **live faithfulness score**,
  - **unsupported claims flagged in red** (the money shot — abstract metric made legible at a glance).
- **Critical gotcha:** Streamlit re-runs the whole script on every interaction. Load FAISS index + embedder + reranker + NLI model **once** via `@st.cache_resource`; use `@st.cache_data` for memoizable computed results. Otherwise the demo reloads models every click and feels broken.
- The UI calls the FastAPI backend (do not duplicate pipeline logic in the UI).
- Do **not** implement any tier/baseline comparison — the baseline was dropped from the entire project.

### 4.3 MCP server (integration-ready polish)
- `src/mcp/`: expose the retriever as an MCP tool (e.g. `search_contracts`). Thin wrapper over the same retrieval used by the pipeline.
- This maps directly to the "MCP isn't magic, it's access" framing DeepJudge uses — signals ecosystem awareness. It is polish, not core; keep it thin.

### 4.4 SQLite persistence
- `src/persistence/`: log each submitted question, the returned answer, and its faithfulness score to SQLite. Enough for a demo; would only become Postgres under real concurrent-write production load (out of scope).

### 4.5 Langfuse observability
- `src/observability/`: per-request traces — retrieved spans, generated claims, entailment verdicts, flagged claims. Reinforces the observability-discipline story and makes the pipeline inspectable.

### 4.6 Final wiring, README & reproducibility pass
- `Makefile` targets: `setup`, `index` (Phase 1), `eval` (Phase 3), `run` (Phase 4), `ci`.
- Finalize `README.md`:
  - **Headline faithfulness + gold-span numbers at the very top.**
  - Architecture diagram (retrieval → LangGraph verifier → answer; FastAPI serving UI + MCP + promptfoo; harness offline; promptfoo in CI).
  - The contamination-split explanation.
  - A short **scope-boundary statement**: research/evaluation project built with production-grade discipline; deliberately not productionized; injection/PII guarding and self-hosting (vLLM) named as documented extensions for a public/multi-tenant or private-document deployment; no Redis/queues/K8s (load-driven, added only on measured need).
  - One-command instructions to reproduce the eval.

**Phase 4 deliverables:** FastAPI backend with Langfuse tracing, Streamlit full-method UI (cached models, highlighted citations, live score, red-flagged claims), MCP retriever server, SQLite logging, finalized Makefile + README with headline numbers and scope statement.

---

## Cross-Phase Checklist (verify at the end)

- [ ] Runs entirely locally, CPU-only path works, GPU never required.
- [ ] `git clone` → `make setup` → `make index` → `make eval` → `make run` works end to end.
- [ ] Dependencies pinned; `.env.example` present; no secrets committed.
- [ ] Gold-span store built and used for objective citation precision/recall/IoU.
- [ ] Character offsets preserved from chunk → citation → gold-span comparison.
- [ ] Hybrid retrieval = BM25 + dense + cross-encoder reranker, all configurable.
- [ ] NLI model loaded **once**, reused across gate + harness (no duplicate instantiation).
- [ ] LangGraph used only for the verifier loop; retrieval/citation logic in plain Python.
- [ ] Mistral client has retry/backoff; Small used for iteration, Large for final numbers.
- [ ] Faithfulness harness = claim decomposition → NLI → gold-span metrics; results committed.
- [ ] LegalBench headline number produced **with documented contamination split** (by contract, not clause).
- [ ] promptfoo runs in CI, separate from the harness, fails build on regression.
- [ ] Streamlit uses `@st.cache_resource` for models/index; shows highlighted citations, live faithfulness score, red-flagged unsupported claims.
- [ ] FastAPI is the single backend for UI + MCP + promptfoo; Langfuse tracing wired.
- [ ] MCP server exposes the retriever; SQLite logs questions + scores.
- [ ] **NO baseline/ablation anywhere** (frontend, harness, or README).
- [ ] **NO generic safety guardrails**; groundedness gate named as the one deliberate guardrail.
- [ ] **NO vLLM, Docker, deployment, Redis, queues, or Kubernetes.**
- [ ] README leads with headline numbers and includes the scope-boundary statement.

---

## Suggested Build Order (dependency-aware)

1. **Phase 1** fully (data + retrieval) — everything depends on it.
2. **Phase 2** schemas → Mistral client → prompts → NLI → gate → LangGraph.
3. **Phase 3** harness → citation metrics → contamination split → LegalBench → promptfoo/CI.
4. **Phase 4** FastAPI → Streamlit → MCP → SQLite → Langfuse → README/reproducibility pass.

Build Phase 1 and 2 against **Mistral Small**. Only switch to **Mistral Large** for the final Phase 3 headline run.
