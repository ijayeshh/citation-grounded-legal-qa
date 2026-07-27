"""BM25 lexical retrieval over the same chunks (Phase 1.5).

Legal text needs exact-term matching for defined terms and clause names
("Change of Control", "Cap on Liability") that dense retrieval can blur, so BM25
runs alongside the dense retriever and their results are fused later.

Built directly from the chunk store (cheap); optionally pickled for fast reload.
Every candidate carries full provenance (contract_id, char offsets, text).
"""

from __future__ import annotations

import pickle
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer. Deterministic and dependency-free."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """rank-bm25 (Okapi) over chunk texts, returning provenance-carrying candidates."""

    def __init__(self, chunks: list[dict], tokenizer=default_tokenize):
        from rank_bm25 import BM25Okapi  # local import

        self.chunks = chunks
        self.tokenizer = tokenizer
        self._corpus_tokens = [tokenizer(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        scores = self.bm25.get_scores(self.tokenizer(query))
        # Deterministic ordering: score desc, then chunk_id asc for ties.
        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (-scores[i], self.chunks[i]["chunk_id"]),
        )[:top_k]
        results: list[dict] = []
        for rank, i in enumerate(order):
            ch = self.chunks[i]
            results.append(
                {
                    "chunk_id": ch["chunk_id"],
                    "contract_id": ch["contract_id"],
                    "char_start": ch["char_start"],
                    "char_end": ch["char_end"],
                    "text": ch["text"],
                    "score": float(scores[i]),
                    "rank": rank,
                    "method": "bm25",
                }
            )
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"chunks": self.chunks}, fh)

    @classmethod
    def load(cls, path: str, tokenizer=default_tokenize) -> "BM25Retriever":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(data["chunks"], tokenizer=tokenizer)


def build_and_persist(chunks: list[dict], path: str) -> BM25Retriever:
    retriever = BM25Retriever(chunks)
    retriever.save(path)
    return retriever
