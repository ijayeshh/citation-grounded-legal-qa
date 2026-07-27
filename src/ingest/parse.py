"""Normalize raw CUAD records into the two structures the rest of the system uses.

Phase 1.2. Produces:

    contracts[contract_id]                    = {"text": str, "metadata": {...}}
    gold_spans[contract_id][clause_type]      = [{"start": int, "end": int, "text": str}, ...]

Character offsets in ``gold_spans`` are indices into ``contracts[cid]["text"]`` such
that ``text[start:end] == span_text``. These exact offsets are what make later
citation scoring (precision / recall / IoU against system-cited spans) objective,
so this normalization must be exact — never trim, lowercase, or re-encode the
underlying text.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

# CUAD questions read: 'Highlight the parts (if any) of this contract related to
# "Anti-Assignment" that should be reviewed by a lawyer. Details: ...'
# The clause type is the first double-quoted phrase.
_CLAUSE_RE = re.compile(r'"([^"]+)"')


def clause_type_from_question(question: str) -> str | None:
    """Extract the clause-type label (the first quoted phrase) from a CUAD question."""
    if not question:
        return None
    m = _CLAUSE_RE.search(question)
    return m.group(1) if m else None


def normalize_records(records: Iterable[dict]) -> tuple[dict, dict]:
    """Fold raw CUAD QA records into (contracts, gold_spans).

    Each contract appears once per clause-type question in the raw data; we keep a
    single copy of its text. Empty / impossible answers are skipped. Duplicate
    spans (same start/end) within a (contract, clause_type) are de-duplicated while
    preserving first-seen order.
    """
    contracts: dict[str, dict] = {}
    gold_spans: dict[str, dict[str, list[dict]]] = {}
    _seen: dict[tuple[str, str], set[tuple[int, int]]] = {}

    for rec in records:
        contract_id = rec.get("title")
        context = rec.get("context")
        if contract_id is None or context is None:
            continue

        if contract_id not in contracts:
            contracts[contract_id] = {
                "text": context,
                "metadata": {
                    "contract_id": contract_id,
                    "source": "cuad",
                    "n_chars": len(context),
                },
            }

        clause_type = clause_type_from_question(rec.get("question", ""))
        if clause_type is None:
            continue

        answers = rec.get("answers", {}) or {}
        texts = answers.get("text", []) or []
        starts = answers.get("answer_start", []) or []

        for span_text, start in zip(texts, starts):
            if span_text is None or start is None:
                continue
            start = int(start)
            if start < 0:
                continue
            end = start + len(span_text)

            key = (contract_id, clause_type)
            seen = _seen.setdefault(key, set())
            if (start, end) in seen:
                continue
            seen.add((start, end))

            gold_spans.setdefault(contract_id, {}).setdefault(clause_type, []).append(
                {"start": start, "end": end, "text": span_text}
            )

    return contracts, gold_spans


def validate_offsets(contracts: dict, gold_spans: dict) -> list[tuple[str, str, int]]:
    """Return a list of (contract_id, clause_type, span_index) whose stored text does
    not match the substring at its offsets. An empty list means every gold span's
    offsets are exact. Used as a sanity gate during ingest.
    """
    mismatches: list[tuple[str, str, int]] = []
    for contract_id, by_clause in gold_spans.items():
        text = contracts.get(contract_id, {}).get("text", "")
        for clause_type, spans in by_clause.items():
            for i, span in enumerate(spans):
                if text[span["start"] : span["end"]] != span["text"]:
                    mismatches.append((contract_id, clause_type, i))
    return mismatches


def save_contracts(contracts: dict, path: str) -> None:
    """Persist the contracts store as JSONL: one {contract_id, text, metadata} per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for contract_id, rec in contracts.items():
            row = {"contract_id": contract_id, "text": rec["text"], "metadata": rec["metadata"]}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_contracts(path: str) -> dict:
    """Load the contracts store written by :func:`save_contracts`."""
    contracts: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            contracts[row["contract_id"]] = {"text": row["text"], "metadata": row["metadata"]}
    return contracts
