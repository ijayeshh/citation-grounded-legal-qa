"""Provenance integrity tests (Phase 1.7).

Every chunk that comes out of the hybrid pipeline must carry the exact provenance
(contract_id, char_start, char_end, text, chunk_id) of the source chunk it refers
to — nothing dropped, mutated, or swapped through fusion + rerank. This is what
lets a later citation map back to gold spans by character offset (IoU).

Uses fake dense/bm25 retrievers and a fake reranker so the test is fast and offline
(no models, no FAISS).
"""

from src.retrieval.hybrid import HybridRetriever

# Source chunk store with distinct offsets per chunk.
_CHUNKS = {
    "K1::00000": {"chunk_id": "K1::00000", "contract_id": "K1", "char_start": 0, "char_end": 50, "text": "termination for convenience clause text here...."},
    "K1::00001": {"chunk_id": "K1::00001", "contract_id": "K1", "char_start": 51, "char_end": 110, "text": "governing law shall be the state of delaware clause"},
    "K2::00000": {"chunk_id": "K2::00000", "contract_id": "K2", "char_start": 0, "char_end": 40, "text": "anti-assignment change of control clause"},
    "K2::00001": {"chunk_id": "K2::00001", "contract_id": "K2", "char_start": 41, "char_end": 95, "text": "cap on liability limited to direct damages only text"},
}


class _FakeRetriever:
    """Returns preconfigured chunk_ids as ranked candidates (with provenance)."""

    def __init__(self, chunk_ids, method):
        self.chunk_ids = chunk_ids
        self.method = method

    def retrieve(self, query, top_k):
        out = []
        for rank, cid in enumerate(self.chunk_ids[:top_k]):
            base = dict(_CHUNKS[cid])
            base.update({"score": 1.0 / (rank + 1), "rank": rank, "method": self.method})
            out.append(base)
        return out


class _FakeReranker:
    """Deterministic reranker: score by reversed text length (no model)."""

    def rerank(self, query, candidates, top_n):
        scored = []
        for c in candidates:
            item = dict(c)
            item["rerank_score"] = float(len(c["text"]))
            scored.append(item)
        scored.sort(key=lambda c: (-c["rerank_score"], c["chunk_id"]))
        top = scored[:top_n]
        for rank, item in enumerate(top):
            item["rank"] = rank
        return top


def _hybrid():
    dense = _FakeRetriever(["K1::00000", "K1::00001", "K2::00001"], "dense")
    bm25 = _FakeRetriever(["K2::00000", "K1::00000", "K2::00001"], "bm25")
    return HybridRetriever(dense=dense, bm25=bm25, reranker=_FakeReranker(), fusion="rrf", rrf_k=60)


_PROVENANCE_FIELDS = ("chunk_id", "contract_id", "char_start", "char_end", "text")


def test_every_result_matches_source_chunk_exactly():
    results = _hybrid().retrieve(
        "cap on liability", dense_top_k=10, bm25_top_k=10, candidates_before_rerank=10, top_n=4
    )
    assert results
    for r in results:
        src = _CHUNKS[r["chunk_id"]]
        for field in _PROVENANCE_FIELDS:
            assert r[field] == src[field], f"{field} corrupted for {r['chunk_id']}"


def test_results_are_subset_of_source_chunks():
    results = _hybrid().retrieve(
        "governing law", dense_top_k=10, bm25_top_k=10, candidates_before_rerank=10, top_n=4
    )
    assert {r["chunk_id"] for r in results}.issubset(set(_CHUNKS))


def test_offsets_are_ints_and_well_formed():
    results = _hybrid().retrieve(
        "anti-assignment", dense_top_k=10, bm25_top_k=10, candidates_before_rerank=10, top_n=4
    )
    for r in results:
        assert isinstance(r["char_start"], int) and isinstance(r["char_end"], int)
        assert r["char_start"] < r["char_end"]


def test_ranks_are_contiguous_and_top_n_respected():
    results = _hybrid().retrieve(
        "termination", dense_top_k=10, bm25_top_k=10, candidates_before_rerank=10, top_n=2
    )
    assert len(results) == 2
    assert [r["rank"] for r in results] == [0, 1]


def test_pipeline_is_deterministic():
    r1 = _hybrid().retrieve("cap", dense_top_k=10, bm25_top_k=10, candidates_before_rerank=10, top_n=4)
    r2 = _hybrid().retrieve("cap", dense_top_k=10, bm25_top_k=10, candidates_before_rerank=10, top_n=4)
    assert [r["chunk_id"] for r in r1] == [r["chunk_id"] for r in r2]


def test_no_duplicate_chunks_in_output():
    results = _hybrid().retrieve(
        "clause", dense_top_k=10, bm25_top_k=10, candidates_before_rerank=10, top_n=4
    )
    ids = [r["chunk_id"] for r in results]
    assert len(ids) == len(set(ids))
