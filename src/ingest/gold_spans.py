"""Persist and query the gold-span store keyed by (contract_id, clause_type).

Phase 1.2. This store is the asset that makes citation scoring objective later:
CUAD's expert annotations, with exact character offsets, are the ground truth
against which the system's cited spans are measured (precision / recall / IoU).
Do not approximate it.

On-disk shape (JSON) mirrors the in-memory structure:

    { contract_id: { clause_type: [ {"start": int, "end": int, "text": str}, ... ] } }
"""

from __future__ import annotations

import json


def save_gold_spans(store: dict, path: str) -> None:
    """Write the gold-span store to ``path`` as UTF-8 JSON (deterministic key order)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, sort_keys=True, indent=2)


def load_gold_spans(path: str) -> dict:
    """Load the gold-span store previously written by :func:`save_gold_spans`."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_spans(store: dict, contract_id: str, clause_type: str) -> list[dict]:
    """Return the gold spans for a (contract_id, clause_type), or [] if none exist."""
    return store.get(contract_id, {}).get(clause_type, [])


def clause_types(store: dict, contract_id: str) -> list[str]:
    """Return the clause types that have at least one gold span for a contract."""
    return sorted(store.get(contract_id, {}).keys())


def count_spans(store: dict) -> int:
    """Total number of gold spans across all contracts and clause types."""
    return sum(len(spans) for by_clause in store.values() for spans in by_clause.values())
