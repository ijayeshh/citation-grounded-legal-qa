"""Schema, retry/backoff, and structured-output tests (Phase 2.1-2.3). All offline."""

import json

import pytest
from pydantic import ValidationError

from src.generation.mistral_client import MistralClient, MistralError
from src.generation.prompts import (
    StructuredOutputError,
    generate_cited_answer,
    parse_generated,
    resolve_answer,
)
from src.schemas.models import Answer, Citation, Claim

# --- Schemas -----------------------------------------------------------------


def test_citation_rejects_non_positive_span():
    with pytest.raises(ValidationError):
        Citation(contract_id="K1", char_start=10, char_end=10, text="x")
    with pytest.raises(ValidationError):
        Citation(contract_id="K1", char_start=10, char_end=5, text="x")


def test_citation_rejects_negative_offset_and_empty_contract():
    with pytest.raises(ValidationError):
        Citation(contract_id="K1", char_start=-1, char_end=5, text="x")
    with pytest.raises(ValidationError):
        Citation(contract_id="", char_start=0, char_end=5, text="x")


def test_claim_and_answer_defaults():
    c = Claim(text="t", citation=Citation(contract_id="K1", char_start=0, char_end=4, text="abcd"))
    assert c.entailment_label is None and c.entailment_score is None
    a = Answer(question="q?", claims=[c])
    assert a.faithfulness_score is None
    assert a.flagged_claims == []
    assert a.raw_text == ""


def test_answer_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Answer(question="q?", claims=[], bogus=1)


# --- Retry / backoff ---------------------------------------------------------


class _FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _client_with_transport(responses, sleeps, **kw):
    """responses: list where each item is either a str (return) or an Exception (raise)."""
    calls = {"n": 0}

    def transport(**_):
        i = calls["n"]
        calls["n"] += 1
        item = responses[i]
        if isinstance(item, Exception):
            raise item
        return item

    client = MistralClient(
        model="mistral-small-latest",
        transport=transport,
        sleep_fn=lambda s: sleeps.append(s),
        jitter=False,
        backoff_base_seconds=2.0,
        backoff_max_seconds=60.0,
        **kw,
    )
    return client, calls


def test_retry_succeeds_after_429s():
    sleeps = []
    client, calls = _client_with_transport(
        [_FakeHTTPError(429), _FakeHTTPError(429), "ok"], sleeps, max_retries=5
    )
    assert client.complete([{"role": "user", "content": "hi"}]) == "ok"
    assert calls["n"] == 3
    # Exponential backoff (no jitter): 2*2^0, 2*2^1
    assert sleeps == [2.0, 4.0]


def test_backoff_is_capped():
    sleeps = []
    client, _ = _client_with_transport(
        [_FakeHTTPError(429)] * 6 + ["ok"], sleeps, max_retries=10
    )
    client.complete([{"role": "user", "content": "hi"}])
    assert max(sleeps) <= 60.0
    assert sleeps[-1] == 60.0  # 2*2^5 = 64 -> capped to 60


def test_non_retryable_raises_immediately():
    sleeps = []
    client, calls = _client_with_transport([_FakeHTTPError(400)], sleeps, max_retries=5)
    with pytest.raises(MistralError):
        client.complete([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # no retries
    assert sleeps == []


def test_retries_exhausted_raises():
    sleeps = []
    client, calls = _client_with_transport([_FakeHTTPError(429)] * 10, sleeps, max_retries=2)
    with pytest.raises(MistralError):
        client.complete([{"role": "user", "content": "hi"}])
    assert calls["n"] == 3  # initial + 2 retries


# --- Structured output parse + resolve ---------------------------------------

_SPANS = {
    "S1": {"chunk_id": "K1::0", "contract_id": "K1", "char_start": 10, "char_end": 60,
           "text": "Either party may terminate for convenience on 30 days notice."},
    "S2": {"chunk_id": "K1::1", "contract_id": "K1", "char_start": 61, "char_end": 100,
           "text": "Governed by the laws of Delaware."},
}


def test_parse_and_resolve_builds_citations():
    raw = json.dumps({
        "answer": "Yes; either party can terminate and Delaware law governs.",
        "claims": [
            {"text": "Either party may terminate for convenience.", "source_id": "S1"},
            {"text": "The contract is governed by Delaware law.", "source_id": "S2"},
        ],
    })
    ans = resolve_answer("Can it be terminated?", parse_generated(raw), _SPANS)
    assert isinstance(ans, Answer)
    assert len(ans.claims) == 2
    # Citation offsets come from the spans, not the model.
    assert ans.claims[0].citation.contract_id == "K1"
    assert (ans.claims[0].citation.char_start, ans.claims[0].citation.char_end) == (10, 60)
    assert ans.claims[1].citation.char_start == 61


def test_resolve_unknown_source_id_raises():
    raw = json.dumps({"answer": "x", "claims": [{"text": "t", "source_id": "S9"}]})
    with pytest.raises(ValueError):
        resolve_answer("q?", parse_generated(raw), _SPANS)


def test_parse_rejects_malformed_json():
    with pytest.raises(Exception):
        parse_generated("not json at all")


# --- generate_cited_answer with a fake client --------------------------------


class _FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, messages, **kw):
        out = self.outputs[self.calls]
        self.calls += 1
        return out


def test_generate_retries_on_bad_output_then_succeeds():
    good = json.dumps({"answer": "ok", "claims": [{"text": "t", "source_id": "S1"}]})
    client = _FakeClient(["{ broken json", good])
    spans = [_SPANS["S1"], _SPANS["S2"]]
    ans = generate_cited_answer("q?", spans, client, max_parse_retries=2)
    assert client.calls == 2
    assert len(ans.claims) == 1
    assert ans.claims[0].citation.contract_id == "K1"


def test_generate_gives_up_after_retries():
    client = _FakeClient(["nope", "still nope", "nope again"])
    with pytest.raises(StructuredOutputError):
        generate_cited_answer("q?", [_SPANS["S1"]], client, max_parse_retries=2)
    assert client.calls == 3
