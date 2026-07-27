"""SQLite persistence tests (Phase 4.4). Offline."""

from src.persistence.db import init_db, log_answer, recent
from src.schemas.models import Answer, Citation, Claim


def _answer(score, n_flagged):
    cit = Citation(contract_id="K1", char_start=0, char_end=5, text="abcde")
    claims = [Claim(text=f"c{i}", citation=cit) for i in range(2)]
    flagged = claims[:n_flagged]
    return Answer(question="q?", claims=claims, raw_text="r",
                  faithfulness_score=score, flagged_claims=flagged)


def test_init_and_log_roundtrip(tmp_path):
    db = str(tmp_path / "app.db")
    init_db(db)
    rid1 = log_answer(db, "first?", _answer(1.0, 0))
    rid2 = log_answer(db, "second?", _answer(0.5, 1))
    assert rid2 > rid1

    rows = recent(db, limit=10)
    assert len(rows) == 2
    assert rows[0]["question"] == "second?"  # newest first
    assert rows[0]["faithfulness_score"] == 0.5
    assert rows[0]["n_flagged"] == 1
    assert rows[1]["n_claims"] == 2


def test_init_is_idempotent(tmp_path):
    db = str(tmp_path / "app.db")
    init_db(db)
    log_answer(db, "q1", _answer(1.0, 0))
    init_db(db)  # must not wipe or error
    assert len(recent(db)) == 1
