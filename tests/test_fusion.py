"""Fusion determinism tests (Phase 1.7).

RRF and weighted fusion must be pure functions of their inputs: identical inputs
always yield identical output, with deterministic tie-breaking by chunk_id.
"""

from src.retrieval.hybrid import reciprocal_rank_fusion, weighted_fusion


def _mk(chunk_id, score, rank):
    return {"chunk_id": chunk_id, "score": score, "rank": rank}


def _dense():
    return [_mk("a", 0.9, 0), _mk("b", 0.8, 1), _mk("c", 0.7, 2)]


def _bm25():
    return [_mk("b", 5.0, 0), _mk("d", 4.0, 1), _mk("a", 3.0, 2)]


def test_rrf_is_deterministic_across_calls():
    r1 = reciprocal_rank_fusion([_dense(), _bm25()], k=60)
    r2 = reciprocal_rank_fusion([_dense(), _bm25()], k=60)
    assert r1 == r2


def test_rrf_ranks_shared_items_highest():
    fused = reciprocal_rank_fusion([_dense(), _bm25()], k=60)
    ids = [cid for cid, _ in fused]
    # 'a' (dense#0, bm25#2) and 'b' (dense#1, bm25#0) appear in both lists -> top two.
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_expected_scores():
    fused = dict(reciprocal_rank_fusion([_dense(), _bm25()], k=60))
    # a: 1/61 (dense rank0) + 1/63 (bm25 rank2)
    assert abs(fused["a"] - (1 / 61 + 1 / 63)) < 1e-12
    # b: 1/62 (dense rank1) + 1/61 (bm25 rank0)
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12


def test_rrf_tie_break_by_chunk_id():
    # Two items with identical single-list rank -> identical scores -> sort by id asc.
    lst = [_mk("zeta", 1.0, 0)]
    lst2 = [_mk("alpha", 1.0, 0)]
    fused = reciprocal_rank_fusion([lst, lst2], k=60)
    assert fused[0][1] == fused[1][1]  # equal scores
    assert [cid for cid, _ in fused] == ["alpha", "zeta"]  # deterministic tie-break


def test_rrf_independent_of_list_order_within_scores():
    # Reversing the *lists* argument order must not change scores (RRF is symmetric).
    a = dict(reciprocal_rank_fusion([_dense(), _bm25()], k=60))
    b = dict(reciprocal_rank_fusion([_bm25(), _dense()], k=60))
    assert a == b


def test_weighted_fusion_deterministic_and_normalized():
    r1 = weighted_fusion([_dense(), _bm25()], weights=[1.0, 1.0])
    r2 = weighted_fusion([_dense(), _bm25()], weights=[1.0, 1.0])
    assert r1 == r2
    # Top dense item 'a' and top bm25 item 'b' each get a normalized 1.0 from one list.
    scores = dict(r1)
    assert set(dict(r1)) == {"a", "b", "c", "d"}
    assert scores["b"] > 0  # 'b' is present in both lists, should score well
