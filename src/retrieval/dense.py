"""Dense retrieval with sentence-transformers over a FAISS index (Phase 1.4).

The embedding model is loaded ONCE (``Embedder``) per the "load heavy models once"
principle. ``DenseRetriever`` accepts any object exposing ``.encode(list[str]) ->
2-D float array``, so tests and the smoke demo can inject a lightweight embedder
without downloading a model.

Every returned candidate carries full provenance (contract_id, char offsets, text)
resolved from the FAISS sidecar, so a hit maps back to the exact source location.
"""

from __future__ import annotations

from src.index import faiss_store


class Embedder:
    """Thin wrapper that loads a SentenceTransformer once and returns unit vectors."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # local heavy import

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class DenseRetriever:
    """Query a FAISS index and resolve hits to provenance-carrying candidates."""

    def __init__(self, index, sidecar: list[dict], embedder):
        self.index = index
        self.sidecar = sidecar
        self.embedder = embedder

    @classmethod
    def from_disk(cls, index_path: str, sidecar_path: str, embedder) -> "DenseRetriever":
        index, sidecar = faiss_store.load_faiss_index(index_path, sidecar_path)
        return cls(index, sidecar, embedder)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        q = self.embedder.encode([query])[0]
        hits = faiss_store.search(self.index, q, top_k)
        results: list[dict] = []
        for rank, (faiss_id, score) in enumerate(hits):
            meta = self.sidecar[faiss_id]
            results.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "contract_id": meta["contract_id"],
                    "char_start": meta["char_start"],
                    "char_end": meta["char_end"],
                    "text": meta["text"],
                    "score": score,
                    "rank": rank,
                    "method": "dense",
                }
            )
        return results


def build_and_persist(chunks: list[dict], embedder, index_path: str, sidecar_path: str):
    """Embed chunk texts and build+persist the FAISS index + sidecar."""
    embeddings = embedder.encode([c["text"] for c in chunks])
    return faiss_store.build_faiss_index(embeddings, chunks, index_path, sidecar_path)
