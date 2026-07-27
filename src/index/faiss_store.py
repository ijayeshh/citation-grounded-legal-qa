"""FAISS dense index: build, persist, load, search (Phase 1.4).

Plain-Python wrapper over faiss-cpu. Cosine similarity via inner product over
L2-normalized vectors (IndexFlatIP), which is exact and fully reproducible on CPU.

A JSONL sidecar maps ``faiss_id`` (the row position, 0..N-1, i.e. the order chunks
were added) back to the chunk's provenance so retrieved vectors resolve to exact
source locations:

    {"faiss_id": i, "chunk_id", "contract_id", "char_start", "char_end", "text"}
"""

from __future__ import annotations

import json


def build_faiss_index(embeddings, chunks: list[dict], index_path: str, sidecar_path: str):
    """Build an IndexFlatIP over L2-normalized ``embeddings`` and persist it + sidecar.

    ``embeddings[i]`` must correspond to ``chunks[i]``. Returns the in-memory index.
    """
    import faiss
    import numpy as np

    if len(embeddings) != len(chunks):
        raise ValueError("embeddings and chunks must be aligned 1:1")

    emb = np.asarray(embeddings, dtype="float32")
    if emb.ndim != 2:
        raise ValueError("embeddings must be a 2-D array")
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, index_path)

    with open(sidecar_path, "w", encoding="utf-8") as fh:
        for i, ch in enumerate(chunks):
            row = {
                "faiss_id": i,
                "chunk_id": ch["chunk_id"],
                "contract_id": ch["contract_id"],
                "char_start": ch["char_start"],
                "char_end": ch["char_end"],
                "text": ch["text"],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return index


def load_faiss_index(index_path: str, sidecar_path: str):
    """Load a persisted index and its sidecar. Returns (index, sidecar_list)."""
    import faiss

    index = faiss.read_index(index_path)
    sidecar: list[dict] = []
    with open(sidecar_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                sidecar.append(json.loads(line))
    return index, sidecar


def search(index, query_embedding, top_k: int) -> list[tuple[int, float]]:
    """Return up to ``top_k`` (faiss_id, score) pairs for a single query vector."""
    import faiss
    import numpy as np

    q = np.asarray([query_embedding], dtype="float32")
    faiss.normalize_L2(q)  # normalize query to match the normalized index
    scores, ids = index.search(q, top_k)
    out: list[tuple[int, float]] = []
    for fid, score in zip(ids[0], scores[0]):
        if fid == -1:  # faiss pads with -1 when fewer than top_k results exist
            continue
        out.append((int(fid), float(score)))
    return out
