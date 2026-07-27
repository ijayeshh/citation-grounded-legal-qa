"""Claim decomposition + groundedness gate tests (Phase 2.4-2.6). Offline."""

import json

import pytest

from src.schemas.models import Answer, Citation, Claim
from src.verify.claim_decomposition import (
    DecompositionError,
    decompose_answer,
    decompose_claim,
    decompose_text,
)
from src.verify.groundedness_gate import GroundednessGate


def _cit(text="the source span text"):
    return Citation(contract_id="K1", char_start=0, char_end=len(text), text=text)


# --- Decomposition -----------------------------------------------------------


class _DecompClient:
    """Returns queued raw responses; ignores prompt content."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0

    def complete(self, messages, **kw):
        r = self.responses[self.i]
        self.i += 1
        return r


def test_decompose_text_splits():
    client = _DecompClient([json.dumps({"claims": ["Fact A.", "Fact B."]})])
    assert decompose_text("Fact A and Fact B.", client) == ["Fact A.", "Fact B."]


def test_decompose_empty_falls_back_to_original():
    client = _DecompClient([json.dumps({"claims": []})])
    assert decompose_text("Only fact.", client) == ["Only fact."]


def test_decompose_claim_inherits_citation():
    client = _DecompClient([json.dumps({"claims": ["A.", "B."]})])
    parent = Claim(text="A and B.", citation=_cit())
    subs = decompose_claim(parent, client)
    assert [c.text for c in subs] == ["A.", "B."]
    assert all(c.citation == parent.citation for c in subs)


def test_decompose_answer_flattens():
    client = _DecompClient([
        json.dumps({"claims": ["A1.", "A2."]}),
        json.dumps({"claims": ["B1."]}),
    ])
    ans = Answer(
        question="q?",
        claims=[Claim(text="A.", citation=_cit("s1")), Claim(text="B.", citation=_cit("s2"))],
        raw_text="raw",
    )
    out = decompose_answer(ans, client)
    assert [c.text for c in out.claims] == ["A1.", "A2.", "B1."]
    assert out.raw_text == "raw" and out.question == "q?"


def test_decompose_retries_then_gives_up():
    client = _DecompClient(["bad", "still bad", "nope"])
    with pytest.raises(DecompositionError):
        decompose_text("x", client, max_parse_retries=2)
    assert client.i == 3


# --- Groundedness gate -------------------------------------------------------


class _FakeNLI:
    """Entailment for claims containing 'good'; contradiction otherwise.

    Score is configurable to exercise the threshold.
    """

    def __init__(self, good_score=0.9, bad_score=0.1):
        self.good_score = good_score
        self.bad_score = bad_score

    def check(self, claim, cited_span):
        if "good" in claim.lower():
            return ("entailment", self.good_score)
        return ("contradiction", self.bad_score)


def _answer_with(*texts):
    return Answer(question="q?", claims=[Claim(text=t, citation=_cit()) for t in texts])


def test_gate_flags_unsupported_and_scores():
    gate = GroundednessGate(_FakeNLI(), entailment_threshold=0.5)
    ans = gate.apply(_answer_with("this is good", "this is bad"))
    assert ans.faithfulness_score == 0.5
    assert [c.text for c in ans.flagged_claims] == ["this is bad"]
    # Verdicts written onto claims.
    assert ans.claims[0].entailment_label == "entailment"
    assert ans.claims[1].entailment_label == "contradiction"


def test_gate_all_entailed():
    gate = GroundednessGate(_FakeNLI(), entailment_threshold=0.5)
    ans = gate.apply(_answer_with("good one", "good two"))
    assert ans.faithfulness_score == 1.0
    assert ans.flagged_claims == []


def test_gate_threshold_demotes_low_entailment():
    # Labeled entailment but score below threshold -> not grounded.
    gate = GroundednessGate(_FakeNLI(good_score=0.4), entailment_threshold=0.5)
    ans = gate.apply(_answer_with("good but weak"))
    assert ans.faithfulness_score == 0.0
    assert len(ans.flagged_claims) == 1


def test_gate_no_claims_is_vacuously_faithful():
    gate = GroundednessGate(_FakeNLI())
    ans = gate.apply(Answer(question="q?", claims=[]))
    assert ans.faithfulness_score == 1.0
    assert ans.flagged_claims == []


def test_gate_from_config():
    cfg = {"verification": {"entailment_threshold": 0.7, "regenerate_unsupported": False,
                            "max_regeneration_rounds": 3}}
    gate = GroundednessGate.from_config(cfg, _FakeNLI())
    assert gate.entailment_threshold == 0.7
    assert gate.regenerate_unsupported is False
    assert gate.max_regeneration_rounds == 3
