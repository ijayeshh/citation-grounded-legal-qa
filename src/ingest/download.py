"""Download CUAD via HuggingFace `datasets` and snapshot the raw records to data/raw/.

Phase 1.2. CUAD (theatticusproject) is distributed in SQuAD-style QA form: one
record per (contract, clause-type question) with the gold answer spans as
character offsets into the contract `context`. We persist the raw flattened
records verbatim (JSONL) so the whole pipeline is reproducible from disk without
re-hitting HuggingFace.
"""

from __future__ import annotations

import json
import os
from typing import Iterator


def download_cuad(
    raw_dir: str,
    dataset_name: str = "cuad",
    split: str = "train",
    limit: int | None = None,
) -> str:
    """Load CUAD from HuggingFace and write raw records to ``raw_dir`` as JSONL.

    Returns the path to the written JSONL file. ``limit`` caps the number of
    records (handy for smoke runs). Heavy import kept local so the rest of the
    package (and the tests) stay import-light and offline.
    """
    from datasets import load_dataset  # local import: heavy, network-bound

    try:
        try:
            ds = load_dataset(dataset_name, split=split)
        except (ValueError, RuntimeError):
            # Newer `datasets` gates script-based loaders behind trust_remote_code.
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - environment/network dependent
        raise RuntimeError(
            f"Failed to load dataset {dataset_name!r} (split={split!r}). "
            "Ensure `datasets` is installed and the dataset id is reachable. "
            "CUAD lives at 'cuad' / 'theatticusproject/cuad-qa'."
        ) from exc

    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f"cuad_{split}.jsonl")

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, rec in enumerate(ds):
            if limit is not None and i >= limit:
                break
            fh.write(json.dumps(_normalize_record_shape(rec), ensure_ascii=False) + "\n")

    return out_path


def _normalize_record_shape(rec: dict) -> dict:
    """Keep only the fields we depend on, in a stable shape.

    CUAD's ``answers`` is a parallel-arrays feature: {text: [...], answer_start: [...]}.
    """
    answers = rec.get("answers", {}) or {}
    return {
        "id": rec.get("id"),
        "title": rec.get("title"),
        "context": rec.get("context"),
        "question": rec.get("question"),
        "answers": {
            "text": list(answers.get("text", []) or []),
            "answer_start": list(answers.get("answer_start", []) or []),
        },
    }


def load_raw_records(path: str) -> Iterator[dict]:
    """Yield raw CUAD records previously written by :func:`download_cuad`."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
