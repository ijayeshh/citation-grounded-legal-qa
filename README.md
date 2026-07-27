# Citation-Grounded Legal QA with a Faithfulness Evaluation Harness

Legal question-answering over CUAD commercial contracts that returns answers with
**span-level citations**, plus a **faithfulness evaluation harness** that measures
whether every claim is actually entailed by the source span it cites — scored
objectively against CUAD gold spans.

---

## Results

Produced by `make eval` (reproducible in one command). Faithfulness = fraction of
atomic claims entailed by their cited source span; citation metrics compare the
system's cited char spans against CUAD **gold spans** — the most objective number
here, and something most projects never exploit.

| Metric | Value |
|---|---|
| Claim faithfulness — NLI (`nli-deberta-v3-base`) | **0.471** |
| Claim faithfulness — LLM judge (Mistral Small, strict rubric) | **0.690** |
| NLI–judge agreement | 0.590 |
| Citation precision (vs. CUAD gold spans) | 0.259 |
| **Citation recall** (vs. CUAD gold spans) | **0.998** |
| Citation IoU (character offsets) | 0.259 |
| **LegalBench headline** — `contract_nli_survival_of_obligations` accuracy (contamination-controlled) | **0.875** |

Generation: Mistral Small. Two independent faithfulness judges (NLI + LLM) are
cross-checked on purpose — their ~41% disagreement is itself the signal that a single
judge would hide. High recall with lower precision reflects cited spans that fully
cover the gold clause but are wider than it (chunk-sized). Per-question tables:
[`eval_reports/faithfulness.md`](eval_reports/faithfulness.md).

> **Provenance:** the numbers above are from a **2-contract sample snapshot** used to
> exercise the harness offline (no full CUAD download in this environment). The full
> headline run is reproduced with `make setup && make index && make eval` on the real
> CUAD corpus in a clean venv (pinned `datasets==2.20.0`).

### Contamination control (Phase 3.3)

LegalBench's `cuad_*` tasks are derived from the **same CUAD contracts** this system
is built and evaluated on, so scoring on them would leak train/test. Chosen split:

- **Evaluate on a non-CUAD LegalBench task** — `contract_nli_survival_of_obligations`,
  from the **ContractNLI** dataset (NDA clauses, entirely different contracts). There
  is **zero overlap** with the CUAD corpus by construction, which is a stronger
  guarantee than a by-contract holdout within CUAD.
- **Relevant sub-component evaluated:** the shared NLI model that powers the
  groundedness gate — i.e. the headline measures the exact entailment capability the
  faithfulness guarantee depends on, not an unrelated skill.
- The `dataset.eval_holdout_contract_ids` knob (split CUAD **by contract, not by
  clause**) remains available for any future CUAD-internal evaluation.

Headline: **accuracy 0.875** (balanced 0.875) — see
[`eval_reports/legalbench.md`](eval_reports/legalbench.md). _(Sample-fixture run in
this environment; the full run pulls the task from `nguha/legalbench` via `make eval`.)_

---

## Status

**Phase 3 in progress.** Done: CUAD ingest + hybrid retrieval (Phase 1), cited-answer
generation + groundedness-gate verifier loop (Phase 2), faithfulness harness + gold-span
citation metrics + committed reports (Phase 3.1–3.2, 3.5). Next: contamination split +
LegalBench headline (3.3–3.4), promptfoo/CI (3.6), then Phase 4 (API/UI/MCP).
See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full plan and [`CLAUDE.md`](CLAUDE.md)
for conventions.

## Quickstart

```bash
make setup     # create Python 3.11+ venv, install pinned deps
cp .env.example .env   # fill in MISTRAL_API_KEY, LANGFUSE_* keys
make index     # Phase 1: ingest CUAD, chunk, build FAISS + BM25 indexes
make eval      # Phase 3: faithfulness harness + citation metrics + LegalBench
make run       # Phase 4: FastAPI backend + Streamlit UI (local)
```

Runs entirely **locally, CPU-only** (GPU never required). Dependencies are pinned;
no secrets are committed.

### Windows (local dev)

The `Makefile` is the canonical Unix/CI entrypoint. On Windows PowerShell, run the
equivalents directly (uses `.venv\Scripts\` instead of `.venv/bin/`):

```powershell
# setup — create venv (Python 3.11+) and install pinned deps
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env   # then fill in keys

# index — Phase 1: ingest CUAD, chunk, build FAISS + BM25 indexes
.\.venv\Scripts\python.exe -m src.ingest.build_index --config config/config.yaml

# eval — Phase 3: faithfulness harness + citation metrics + LegalBench
.\.venv\Scripts\python.exe -m src.eval.harness --config config/config.yaml
.\.venv\Scripts\python.exe -m src.eval.legalbench_runner --config config/config.yaml

# run — Phase 4: FastAPI backend (background) + Streamlit UI
Start-Process .\.venv\Scripts\python.exe -ArgumentList '-m','uvicorn','src.api.app:app','--host','127.0.0.1','--port','8000'
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py

# test / ci
.\.venv\Scripts\python.exe -m pytest -q
npx --yes promptfoo@latest eval -c promptfoo/promptfooconfig.yaml   # requires Node.js
```

Tip: `.\.venv\Scripts\Activate.ps1` activates the venv so you can drop the
`.\.venv\Scripts\` prefix for the rest of the session.

## Scope

Research/evaluation project built with production-grade discipline, deliberately
not productionized. No baseline/ablation comparison, no generic safety guardrails
(the groundedness gate is the one deliberate guardrail), no vLLM/Docker/Redis/
queues/Kubernetes/cloud deployment.
