"""Cross-encoder reranker over the fused candidate set (Phase 1.6).

The highest-ROI retrieval component: it scores (query, chunk_text) pairs jointly
and reorders the fused candidates by relevance. The model is loaded ONCE. The
``rerank`` interface is deliberately small so tests/smoke can inject a fake scorer.
"""

from __future__ import annotations


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder, loaded once, reused for every query."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder  # local heavy import

        self.model = CrossEncoder(model_name, device=device)

    def _score(self, query: str, texts: list[str]) -> list[float]:
        scores = self.model.predict([(query, t) for t in texts])
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """Return the top_n candidates reordered by cross-encoder relevance.

        Adds ``rerank_score`` and a final 0-based ``rank`` to each returned dict.
        Ties break deterministically by chunk_id.
        """
        if not candidates:
            return []
        scores = self._score(query, [c["text"] for c in candidates])
        scored = []
        for cand, s in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = s
            scored.append(item)
        scored.sort(key=lambda c: (-c["rerank_score"], c["chunk_id"]))
        top = scored[:top_n]
        for rank, item in enumerate(top):
            item["rank"] = rank
        return top
