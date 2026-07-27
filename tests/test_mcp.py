"""MCP retriever tool tests (Phase 4.3). Offline (fake retriever)."""

from src.mcp.server import search_contracts_impl

_CFG = {
    "retrieval": {
        "dense_top_k": 10, "bm25_top_k": 10,
        "candidates_before_rerank": 10, "results_after_rerank": 3,
    }
}


class _FakeRetriever:
    def __init__(self):
        self.last_kwargs = None

    def retrieve(self, query, **kwargs):
        self.last_kwargs = kwargs
        return [
            {"chunk_id": "K1::0", "contract_id": "K1", "char_start": 0, "char_end": 20,
             "text": "termination clause", "rerank_score": 3.2},
            {"chunk_id": "K2::1", "contract_id": "K2", "char_start": 5, "char_end": 40,
             "text": "liability clause", "rerank_score": 1.1},
        ]


def test_search_returns_provenance_spans():
    out = search_contracts_impl(_FakeRetriever(), _CFG, "termination")
    assert [s["chunk_id"] for s in out] == ["K1::0", "K2::1"]
    for s in out:
        assert set(s) == {"chunk_id", "contract_id", "char_start", "char_end", "text", "score"}
    assert out[0]["score"] == 3.2


def test_top_n_override_passed_through():
    r = _FakeRetriever()
    search_contracts_impl(r, _CFG, "q", top_n=5)
    assert r.last_kwargs["top_n"] == 5


def test_top_n_defaults_to_config():
    r = _FakeRetriever()
    search_contracts_impl(r, _CFG, "q")
    assert r.last_kwargs["top_n"] == 3  # results_after_rerank
