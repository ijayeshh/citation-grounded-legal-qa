"""LegalBench runner tests (Phase 3.4). Offline (fixture + fake NLI)."""

import json

from src.eval.legalbench_runner import _balanced_accuracy, run


def _cfg():
    return {
        "dataset": {
            "legalbench_task": "contract_nli_survival_of_obligations",
            "legalbench_source": "nguha/legalbench",
            "legalbench_hypotheses": {
                "contract_nli_survival_of_obligations": "Some obligations survive termination."
            },
        },
        "verification": {"nli_model": "fake", "entailment_threshold": 0.5},
        "embeddings": {"device": "cpu"},
    }


class _FakeNLI:
    """Entailment when the clause mentions 'survive'; else contradiction."""

    def check(self, claim, cited_span):
        if "surviv" in cited_span.lower():
            return ("entailment", 0.9)
        return ("contradiction", 0.1)


def test_balanced_accuracy():
    assert _balanced_accuracy(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"]) == 1.0
    # per-class recall: yes 1/2, no 2/2 -> mean 0.75
    assert _balanced_accuracy(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"]) == 0.75


def test_run_on_fixture(tmp_path):
    rows = [
        {"text": "Confidentiality obligations shall survive termination.", "answer": "Yes"},
        {"text": "These duties survive expiration of the Agreement.", "answer": "Yes"},
        {"text": "All obligations cease upon termination.", "answer": "No"},
        {"text": "This may be executed in counterparts.", "answer": "No"},
    ]
    fx = tmp_path / "lb.jsonl"
    fx.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = run(_cfg(), fixture=str(fx), nli=_FakeNLI())
    assert result["n_examples"] == 4
    assert result["accuracy"] == 1.0  # fake NLI matches the gold pattern exactly
    assert result["balanced_accuracy"] == 1.0
    assert "non-CUAD" in result["contamination_control"]


def test_run_maps_entailment_threshold(tmp_path):
    # A 'survive' clause labeled No should be mispredicted -> accuracy 0.5 on 2 rows.
    rows = [
        {"text": "Obligations survive termination.", "answer": "No"},
        {"text": "Obligations cease at termination.", "answer": "No"},
    ]
    fx = tmp_path / "lb.jsonl"
    fx.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    result = run(_cfg(), fixture=str(fx), nli=_FakeNLI())
    assert result["accuracy"] == 0.5
