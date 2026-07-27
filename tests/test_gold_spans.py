"""Gold-span loading + normalization tests (Phase 1.2).

Uses synthetic SQuAD-style CUAD records so the suite stays fast and offline.
"""

from src.ingest.gold_spans import (
    clause_types,
    count_spans,
    get_spans,
    load_gold_spans,
    save_gold_spans,
)
from src.ingest.parse import (
    clause_type_from_question,
    normalize_records,
    validate_offsets,
)

# A tiny fake contract with two clause types, whose answer_start offsets are exact.
_CONTEXT = (
    "This Agreement is between Acme Corp and Beta LLC. "  # 0..49
    "Either party may terminate with thirty (30) days notice. "
    "This contract shall be governed by the laws of Delaware."
)


def _q(clause: str) -> str:
    return f'Highlight the parts (if any) of this contract related to "{clause}" that should be reviewed by a lawyer.'


def _span(text: str):
    start = _CONTEXT.index(text)
    return {"text": text, "start": start}


def _records():
    termination = "Either party may terminate with thirty (30) days notice."
    governing = "the laws of Delaware"
    parties = "Acme Corp and Beta LLC"
    return [
        {
            "title": "contract_A",
            "context": _CONTEXT,
            "question": _q("Termination For Convenience"),
            "answers": {
                "text": [termination],
                "answer_start": [_CONTEXT.index(termination)],
            },
        },
        {
            "title": "contract_A",
            "context": _CONTEXT,
            "question": _q("Governing Law"),
            "answers": {
                "text": [governing],
                "answer_start": [_CONTEXT.index(governing)],
            },
        },
        {
            "title": "contract_A",
            "context": _CONTEXT,
            "question": _q("Parties"),
            "answers": {"text": [parties], "answer_start": [_CONTEXT.index(parties)]},
        },
        # An impossible / empty answer must be skipped, not crash.
        {
            "title": "contract_A",
            "context": _CONTEXT,
            "question": _q("Cap On Liability"),
            "answers": {"text": [], "answer_start": []},
        },
    ]


def test_clause_type_extraction():
    assert clause_type_from_question(_q("Anti-Assignment")) == "Anti-Assignment"
    assert clause_type_from_question("no quotes here") is None
    assert clause_type_from_question("") is None


def test_normalize_builds_contracts_and_gold():
    contracts, gold = normalize_records(_records())

    assert set(contracts) == {"contract_A"}
    assert contracts["contract_A"]["text"] == _CONTEXT
    assert contracts["contract_A"]["metadata"]["n_chars"] == len(_CONTEXT)
    assert contracts["contract_A"]["metadata"]["source"] == "cuad"

    # Three clause types have spans; the empty one is dropped.
    assert set(gold["contract_A"]) == {
        "Termination For Convenience",
        "Governing Law",
        "Parties",
    }
    assert count_spans(gold) == 3


def test_normalized_offsets_are_exact():
    contracts, gold = normalize_records(_records())
    assert validate_offsets(contracts, gold) == []
    # Spot-check one span re-slices correctly.
    span = get_spans(gold, "contract_A", "Governing Law")[0]
    assert _CONTEXT[span["start"] : span["end"]] == span["text"]


def test_duplicate_spans_deduped():
    recs = _records()
    # Duplicate the governing-law answer within the same clause type.
    gov = "the laws of Delaware"
    recs.append(
        {
            "title": "contract_A",
            "context": _CONTEXT,
            "question": _q("Governing Law"),
            "answers": {"text": [gov, gov], "answer_start": [_CONTEXT.index(gov)] * 2},
        }
    )
    _, gold = normalize_records(recs)
    assert len(get_spans(gold, "contract_A", "Governing Law")) == 1


def test_get_spans_missing_returns_empty():
    _, gold = normalize_records(_records())
    assert get_spans(gold, "nope", "Governing Law") == []
    assert get_spans(gold, "contract_A", "Nonexistent Clause") == []


def test_clause_types_helper():
    _, gold = normalize_records(_records())
    assert clause_types(gold, "contract_A") == [
        "Governing Law",
        "Parties",
        "Termination For Convenience",
    ]


def test_gold_spans_roundtrip(tmp_path):
    _, gold = normalize_records(_records())
    path = str(tmp_path / "gold_spans.json")
    save_gold_spans(gold, path)
    loaded = load_gold_spans(path)
    assert loaded == gold
    # Loaded store is queryable and offsets remain exact.
    span = get_spans(loaded, "contract_A", "Parties")[0]
    assert _CONTEXT[span["start"] : span["end"]] == span["text"]
