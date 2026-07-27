"""Chunk offset integrity tests (Phase 1.3).

The load-bearing invariant for later IoU scoring:
    original_text[chunk.char_start : chunk.char_end] == chunk.text
"""

import pytest

from src.ingest.chunk import chunk_contracts, load_chunks, save_chunks, sliding_window_chunks


def _make_text(n: int) -> str:
    # Deterministic, non-repeating-ish text so offset bugs surface.
    return "".join(chr(ord("A") + (i % 26)) for i in range(n))


def test_offset_integrity_reslicing_reproduces_text():
    text = _make_text(5000)
    chunks = sliding_window_chunks("C1", text, max_chars=1800, overlap_chars=200)
    assert chunks, "expected at least one chunk"
    for ch in chunks:
        assert text[ch["char_start"] : ch["char_end"]] == ch["text"]


def test_chunks_have_required_fields():
    text = _make_text(3000)
    for ch in sliding_window_chunks("C1", text, max_chars=1000, overlap_chars=100):
        assert set(ch.keys()) == {"chunk_id", "contract_id", "char_start", "char_end", "text"}
        assert ch["contract_id"] == "C1"
        assert ch["char_start"] < ch["char_end"]


def test_full_coverage_start_to_end():
    text = _make_text(5000)
    chunks = sliding_window_chunks("C1", text, max_chars=1800, overlap_chars=200)
    assert chunks[0]["char_start"] == 0
    assert chunks[-1]["char_end"] == len(text)


def test_overlap_between_consecutive_chunks():
    text = _make_text(5000)
    max_chars, overlap = 1800, 200
    chunks = sliding_window_chunks("C1", text, max_chars=max_chars, overlap_chars=overlap)
    step = max_chars - overlap
    for prev, nxt in zip(chunks, chunks[1:]):
        # Next window starts exactly one step after the previous (until the final clamp).
        assert nxt["char_start"] == prev["char_start"] + step
        # And they genuinely overlap in the source stream.
        assert nxt["char_start"] < prev["char_end"]


def test_short_text_single_chunk():
    text = _make_text(500)
    chunks = sliding_window_chunks("C1", text, max_chars=1800, overlap_chars=200)
    assert len(chunks) == 1
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == len(text)
    assert chunks[0]["text"] == text


def test_empty_text_yields_no_chunks():
    assert sliding_window_chunks("C1", "", max_chars=1800, overlap_chars=200) == []


def test_chunk_ids_unique():
    text = _make_text(10000)
    chunks = sliding_window_chunks("C1", text, max_chars=1800, overlap_chars=200)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        sliding_window_chunks("C1", "abc", max_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        sliding_window_chunks("C1", "abc", max_chars=0, overlap_chars=0)


def test_chunk_contracts_covers_all_contracts():
    contracts = {
        "C1": {"text": _make_text(2500), "metadata": {}},
        "C2": {"text": _make_text(100), "metadata": {}},
    }
    chunks = chunk_contracts(contracts, max_chars=1000, overlap_chars=100)
    assert {c["contract_id"] for c in chunks} == {"C1", "C2"}
    # Re-slice invariant holds across contracts too.
    for c in chunks:
        assert contracts[c["contract_id"]]["text"][c["char_start"] : c["char_end"]] == c["text"]


def test_chunks_roundtrip_jsonl(tmp_path):
    text = _make_text(4000)
    chunks = sliding_window_chunks("C1", text, max_chars=1800, overlap_chars=200)
    path = str(tmp_path / "chunks.jsonl")
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
