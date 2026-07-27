"""FastAPI backend — the single service boundary (Phase 4.1).

One backend, three clients (Streamlit UI, MCP server, promptfoo). ALL pipeline
business logic lives here, never in the UI. Heavy models (embedder, reranker, NLI)
and the FAISS/BM25 indexes are loaded ONCE at startup (FastAPI lifespan) and reused
for every request. Langfuse tracing is wired at this layer.

Endpoints:
    POST /answer            -> structured Answer (claims, citations, verdicts, score)
    GET  /contract/{id}     -> full contract text (so the UI can highlight offsets)
    GET  /health            -> readiness

Run:  uvicorn src.api.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.schemas.models import Answer

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AnswerService:
    """Holds the loaded pipeline + stores. Built once; reused across requests."""

    def __init__(self, cfg: dict):
        # Heavy imports kept local so importing this module stays light.
        from src.generation.mistral_client import MistralClient
        from src.graph.pipeline import build_pipeline
        from src.ingest.parse import load_contracts
        from src.observability.langfuse_setup import get_langfuse
        from src.persistence.db import init_db
        from src.retrieval.dense import Embedder
        from src.retrieval.hybrid import HybridRetriever
        from src.verify.groundedness_gate import GroundednessGate
        from src.verify.nli import load_nli

        self.cfg = cfg
        emb = cfg["embeddings"]

        print("[api] loading models ONCE (embedder, reranker, NLI, indexes) ...")
        self.embedder = Embedder(emb["model"], device=emb["device"], batch_size=emb["batch_size"])
        self.retriever = HybridRetriever.from_disk(cfg, self.embedder)  # embedder + reranker + FAISS + BM25
        self.nli = load_nli(cfg["verification"]["nli_model"], device=emb["device"])  # shared singleton
        self.gate = GroundednessGate.from_config(cfg, self.nli)
        self.pipeline = build_pipeline(self.retriever, MistralClient.from_config(cfg), self.gate, cfg)
        self.contracts = load_contracts(cfg["paths"]["contracts"])
        self.langfuse = get_langfuse(cfg)
        self.db_path = cfg["paths"]["sqlite_db"]
        init_db(self.db_path)
        print("[api] ready — models loaded once; serving requests.")

    def answer(self, question: str) -> Answer:
        from src.observability.langfuse_setup import trace_answer
        from src.persistence.db import log_answer

        state = self.pipeline.invoke({"question": question, "round": 0})
        answer, spans = state["answer"], state["spans"]
        trace_answer(self.langfuse, question, spans, answer)  # retrieved / generated / flagged
        try:
            log_answer(self.db_path, question, answer)  # SQLite: question + answer + score
        except Exception:
            pass  # logging must never break a response
        return answer

    def contract_text(self, contract_id: str) -> str:
        rec = self.contracts.get(contract_id)
        if rec is None:
            raise KeyError(contract_id)
        return rec["text"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    app.state.service = AnswerService(cfg)  # loaded once at startup
    yield


app = FastAPI(title="Citation-Grounded Legal QA", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "ready": getattr(app.state, "service", None) is not None}


@app.post("/answer", response_model=Answer)
def answer(req: AnswerRequest):
    return app.state.service.answer(req.question)


@app.get("/contract/{contract_id}")
def contract(contract_id: str):
    try:
        return {"contract_id": contract_id, "text": app.state.service.contract_text(contract_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown contract_id {contract_id!r}")
