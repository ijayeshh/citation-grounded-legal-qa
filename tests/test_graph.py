"""LangGraph pipeline routing + end-to-end tests (Phase 2.7). Offline (fake LLM + NLI).

Exercises the real graph (retrieve -> generate -> decompose -> verify -> [regenerate]
-> finalize) and the real GroundednessGate, with only the LLM and NLI mocked.
"""

import json

from src.graph.pipeline import answer_question, build_pipeline
from src.verify.groundedness_gate import GroundednessGate

_SPAN = {
    "chunk_id": "K1::0", "contract_id": "K1", "char_start": 0, "char_end": 20,
    "text": "the source span text",
}

_CFG = {
    "retrieval": {
        "dense_top_k": 10, "bm25_top_k": 10,
        "candidates_before_rerank": 10, "results_after_rerank": 4,
    }
}


class _FakeRetriever:
    def retrieve(self, question, **kwargs):
        return [_SPAN]


class _FakeLLM:
    """Dispatches on the system prompt: generation / decomposition / regeneration."""

    def __init__(self, gen_claims, regen_claims=None):
        self.gen_claims = gen_claims
        self.regen_claims = regen_claims or []
        self.calls = {"generate": 0, "decompose": 0, "regenerate": 0}

    def complete(self, messages, **kw):
        system = messages[0]["content"]
        if "ATOMIC CLAIM DECOMPOSER" in system:
            self.calls["decompose"] += 1
            stmt = next(m["content"] for m in messages if m["content"].startswith("Statement:"))
            text = stmt.split("Statement:\n", 1)[1]
            return json.dumps({"claims": [text]})  # identity decomposition
        if "REGENERATE" in system:
            self.calls["regenerate"] += 1
            return json.dumps({"claims": [{"text": t, "source_id": s} for t, s in self.regen_claims]})
        self.calls["generate"] += 1
        return json.dumps({
            "answer": "raw answer",
            "claims": [{"text": t, "source_id": s} for t, s in self.gen_claims],
        })


class _FakeNLI:
    def check(self, claim, cited_span):
        return ("entailment", 0.95) if "bad" not in claim.lower() else ("contradiction", 0.05)


def _pipeline(llm):
    gate = GroundednessGate(_FakeNLI(), entailment_threshold=0.5,
                            regenerate_unsupported=True, max_regeneration_rounds=1)
    return build_pipeline(_FakeRetriever(), llm, gate, _CFG)


def test_no_regeneration_when_all_entailed():
    llm = _FakeLLM(gen_claims=[("claim one is good", "S1"), ("claim two is good", "S1")])
    state = _pipeline(llm).invoke({"question": "q?", "round": 0})
    assert state["answer"].faithfulness_score == 1.0
    assert state["answer"].flagged_claims == []
    assert state.get("round", 0) == 0
    assert llm.calls["regenerate"] == 0


def test_regenerates_once_and_fixes_unsupported():
    llm = _FakeLLM(
        gen_claims=[("this claim is bad", "S1"), ("this claim is good", "S1")],
        regen_claims=[("now this claim is good", "S1")],
    )
    state = _pipeline(llm).invoke({"question": "q?", "round": 0})
    ans = state["answer"]
    assert llm.calls["regenerate"] == 1
    assert state["round"] == 1
    assert ans.faithfulness_score == 1.0
    assert ans.flagged_claims == []
    # supported original kept + regenerated one
    assert {c.text for c in ans.claims} == {"this claim is good", "now this claim is good"}


def test_flags_when_still_unsupported_after_max_rounds():
    llm = _FakeLLM(
        gen_claims=[("this claim is bad", "S1"), ("this claim is good", "S1")],
        regen_claims=[("this claim is still bad", "S1")],
    )
    state = _pipeline(llm).invoke({"question": "q?", "round": 0})
    ans = state["answer"]
    assert llm.calls["regenerate"] == 1  # only once (max_regeneration_rounds=1)
    assert state["round"] == 1
    assert ans.faithfulness_score == 0.5
    assert [c.text for c in ans.flagged_claims] == ["this claim is still bad"]


def test_answer_question_helper_returns_answer():
    llm = _FakeLLM(gen_claims=[("claim is good", "S1")])
    ans = answer_question(_pipeline(llm), "q?")
    assert ans.faithfulness_score == 1.0
    assert ans.claims[0].citation.contract_id == "K1"
