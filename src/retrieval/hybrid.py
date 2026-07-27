"""Hybrid retrieval: fuse BM25 + dense, then cross-encoder rerank (Phase 1.6).

Flow: run dense and BM25 in parallel, fuse their ranked lists (RRF by default, or
weighted), take the top candidates, rerank them with a cross-encoder, and return a
ranked list of chunks with scores and full provenance (contract_id, char offsets,
text). ``top_k`` (candidates) and ``top_n`` (after rerank) are configurable.

All fusion is deterministic: ties break by chunk_id so identical inputs always
produce identical output.
"""

from __future__ import annotations

from src.retrieval.bm25 import BM25Retriever, default_tokenize
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Returns (chunk_id, fused_score) sorted desc, then id asc.

    Uses each item's position within its own list (0-based), not its raw score, so
    dense and lexical scores never need to be made commensurable.
    """
    fused: dict[str, float] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_fusion(
    result_lists: list[list[dict]], weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Min-max normalize each list's scores to [0,1], then weighted-sum per chunk_id."""
    if weights is None:
        weights = [1.0] * len(result_lists)
    fused: dict[str, float] = {}
    for results, w in zip(result_lists, weights):
        if not results:
            continue
        scores = [r["score"] for r in results]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for item in results:
            norm = (item["score"] - lo) / span
            fused[item["chunk_id"]] = fused.get(item["chunk_id"], 0.0) + w * norm
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """Compose dense + BM25 + reranker into one deterministic retrieve() call."""

    def __init__(
        self,
        dense,
        bm25,
        reranker,
        fusion: str = "rrf",
        rrf_k: int = 60,
        weights: list[float] | None = None,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.weights = weights

    def _fuse(self, result_lists: list[list[dict]]) -> list[tuple[str, float]]:
        if self.fusion == "weighted":
            return weighted_fusion(result_lists, self.weights)
        return reciprocal_rank_fusion(result_lists, self.rrf_k)

    def retrieve(
        self,
        query: str,
        dense_top_k: int,
        bm25_top_k: int,
        candidates_before_rerank: int,
        top_n: int,
    ) -> list[dict]:
        dense_hits = self.dense.retrieve(query, dense_top_k)
        bm25_hits = self.bm25.retrieve(query, bm25_top_k)

        # Provenance map: first occurrence wins (both sources carry identical fields).
        chunk_map: dict[str, dict] = {}
        for hit in (*dense_hits, *bm25_hits):
            chunk_map.setdefault(hit["chunk_id"], hit)

        fused = self._fuse([dense_hits, bm25_hits])[:candidates_before_rerank]

        candidates: list[dict] = []
        for fusion_rank, (cid, fusion_score) in enumerate(fused):
            base = chunk_map[cid]
            candidates.append(
                {
                    "chunk_id": base["chunk_id"],
                    "contract_id": base["contract_id"],
                    "char_start": base["char_start"],
                    "char_end": base["char_end"],
                    "text": base["text"],
                    "fusion_score": fusion_score,
                    "fusion_rank": fusion_rank,
                    "method": "hybrid",
                }
            )

        reranked = self.reranker.rerank(query, candidates, top_n)
        return reranked

    @classmethod
    def from_disk(cls, cfg: dict, embedder) -> "HybridRetriever":
        """Build a HybridRetriever from persisted artifacts + config."""
        paths = cfg["paths"]
        rcfg = cfg["retrieval"]
        dense = DenseRetriever.from_disk(paths["faiss_index"], paths["faiss_sidecar"], embedder)
        bm25 = BM25Retriever.load(paths["bm25_store"], tokenizer=default_tokenize)
        reranker = CrossEncoderReranker(
            rcfg["reranker_model"], device=cfg["embeddings"].get("device", "cpu")
        )
        return cls(
            dense=dense,
            bm25=bm25,
            reranker=reranker,
            fusion=rcfg.get("fusion", "rrf"),
            rrf_k=rcfg.get("rrf_k", 60),
        )
