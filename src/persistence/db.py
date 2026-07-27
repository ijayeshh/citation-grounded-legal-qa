"""SQLite logging of each question, answer, and faithfulness score (Phase 4.4).

Enough for a demo and for inspecting what the system was asked and how faithful its
answers were. Plain stdlib sqlite3 — would only become Postgres under real concurrent
write load (out of scope; see the scope statement in the README).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,
    question           TEXT    NOT NULL,
    faithfulness_score REAL,
    n_claims           INTEGER NOT NULL,
    n_flagged          INTEGER NOT NULL,
    answer_json        TEXT    NOT NULL
);
"""


def init_db(db_path: str) -> None:
    """Create the query_log table if needed (and its parent directory)."""
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)


def log_answer(db_path: str, question: str, answer) -> int:
    """Persist one (question, answer, faithfulness) row. Returns the new row id."""
    row = (
        datetime.now(timezone.utc).isoformat(),
        question,
        answer.faithfulness_score,
        len(answer.claims),
        len(answer.flagged_claims),
        answer.model_dump_json(),
    )
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO query_log "
            "(ts, question, faithfulness_score, n_claims, n_flagged, answer_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
        return int(cur.lastrowid)


def recent(db_path: str, limit: int = 20) -> list[dict]:
    """Return the most recent logged queries (newest first)."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, ts, question, faithfulness_score, n_claims, n_flagged "
            "FROM query_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
