"""The stateful verifier pipeline as a LangGraph (Phase 2.7).

Flow (conditional edges):

    retrieve -> generate -> decompose -> verify
                                           |-- unsupported & rounds left --> regenerate --> verify
                                           |-- otherwise --------------------> finalize -> END

LangGraph is used ONLY here, because this is a genuine stateful loop with a
conditional regenerate/re-verify branch. Retrieval, generation, decomposition, and
the groundedness gate are plain Python components injected into the graph; the graph
just wires them and owns the regenerate loop.

The finalized state carries an Answer with per-claim entailment verdicts, the
flagged (unsupported) claims, and an overall live faithfulness score.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.generation.prompts import generate_cited_answer, regenerate_unsupported_claims
from src.schemas.models import Answer
from src.verify.claim_decomposition import decompose_answer


class GraphState(TypedDict, total=False):
    question: str
    spans: list[dict]
    answer: Answer
    round: int


def build_pipeline(retriever, client, gate, cfg: dict, decompose: bool = True):
    """Compile the LangGraph verifier pipeline from injected components.

    - ``retriever`` exposes ``.retrieve(query, dense_top_k, bm25_top_k,
      candidates_before_rerank, top_n)`` (the HybridRetriever).
    - ``client`` is a MistralClient (or anything with ``.complete``).
    - ``gate`` is a GroundednessGate (owns threshold + regenerate policy).
    """
    r = cfg["retrieval"]

    def retrieve_node(state: GraphState) -> dict:
        spans = retriever.retrieve(
            state["question"],
            dense_top_k=r["dense_top_k"],
            bm25_top_k=r["bm25_top_k"],
            candidates_before_rerank=r["candidates_before_rerank"],
            top_n=r["results_after_rerank"],
        )
        return {"spans": spans}

    def generate_node(state: GraphState) -> dict:
        answer = generate_cited_answer(state["question"], state["spans"], client)
        return {"answer": answer}

    def decompose_node(state: GraphState) -> dict:
        return {"answer": decompose_answer(state["answer"], client)}

    def verify_node(state: GraphState) -> dict:
        return {"answer": gate.apply(state["answer"])}

    def regenerate_node(state: GraphState) -> dict:
        answer = state["answer"]
        supported = [c for c in answer.claims if gate.is_entailed(c)]
        unsupported_texts = [c.text for c in answer.flagged_claims]
        regenerated = regenerate_unsupported_claims(
            state["question"], unsupported_texts, state["spans"], client
        )
        new_answer = Answer(
            question=answer.question,
            claims=supported + regenerated,
            raw_text=answer.raw_text,
        )
        return {"answer": new_answer, "round": state.get("round", 0) + 1}

    def finalize_node(state: GraphState) -> dict:
        # No-op passthrough (a LangGraph node must write at least one channel).
        return {"answer": state["answer"]}

    def route_after_verify(state: GraphState) -> str:
        answer = state["answer"]
        rnd = state.get("round", 0)
        if (
            answer.flagged_claims
            and gate.regenerate_unsupported
            and rnd < gate.max_regeneration_rounds
        ):
            return "regenerate"
        return "finalize"

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("decompose", decompose_node)
    g.add_node("verify", verify_node)
    g.add_node("regenerate", regenerate_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "decompose" if decompose else "verify")
    if decompose:
        g.add_edge("decompose", "verify")
    g.add_conditional_edges(
        "verify", route_after_verify, {"regenerate": "regenerate", "finalize": "finalize"}
    )
    g.add_edge("regenerate", "verify")
    g.add_edge("finalize", END)
    return g.compile()


def answer_question(pipeline, question: str) -> Answer:
    """Convenience: run the compiled pipeline on one question and return the Answer."""
    final_state: dict[str, Any] = pipeline.invoke({"question": question, "round": 0})
    return final_state["answer"]
