"""Sliding-window chunking with overlap (Phase 1.3).

Strategy: a pure fixed-size sliding window over the raw character stream of each
contract. Window = ``max_chars``, step = ``max_chars - overlap_chars``. Clause-aware
splitting is a deliberate future refinement (see PROJECT_PLAN.md 1.3); we start
with the simplest correct thing.

The invariant that everything downstream depends on:

    text[chunk["char_start"]:chunk["char_end"]] == chunk["text"]

i.e. offsets are exact indices into the ORIGINAL contract text. We never strip,
normalize, or otherwise mutate the sliced substring, because a retrieved chunk
(and later a cited span) must map back to the exact source location to be compared
against CUAD gold spans by IoU. Preserving offsets end-to-end starts here.

Each chunk carries: chunk_id, contract_id, char_start, char_end, text.
"""

from __future__ import annotations


def sliding_window_chunks(
    contract_id: str,
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[dict]:
    """Chunk one contract's text into overlapping fixed-size windows.

    - Text shorter than ``max_chars`` yields a single chunk covering [0, len(text)].
    - The final window is clamped to len(text) (it may be shorter than max_chars).
    - Empty text yields no chunks.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    n = len(text)
    if n == 0:
        return []

    step = max_chars - overlap_chars
    chunks: list[dict] = []
    start = 0
    idx = 0
    while True:
        end = min(start + max_chars, n)
        chunks.append(
            {
                "chunk_id": f"{contract_id}::{idx:05d}",
                "contract_id": contract_id,
                "char_start": start,
                "char_end": end,
                "text": text[start:end],
            }
        )
        idx += 1
        if end >= n:
            break
        start += step
    return chunks


def chunk_contracts(contracts: dict, max_chars: int, overlap_chars: int) -> list[dict]:
    """Chunk every contract in a ``contracts`` store into a flat list of chunks."""
    all_chunks: list[dict] = []
    for contract_id, rec in contracts.items():
        all_chunks.extend(
            sliding_window_chunks(contract_id, rec["text"], max_chars, overlap_chars)
        )
    return all_chunks


def save_chunks(chunks: list[dict], path: str) -> None:
    """Persist chunks as JSONL, one chunk object per line."""
    import json

    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks(path: str) -> list[dict]:
    """Load chunks written by :func:`save_chunks`."""
    import json

    chunks: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
