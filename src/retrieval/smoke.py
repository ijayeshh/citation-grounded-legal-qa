"""Retrieval smoke test (Phase 1.7).

Runs a handful of clause-type questions through the FULL hybrid + rerank pipeline
and prints the ranked spans with provenance, so you can eyeball that clause
questions surface the right clauses.

Modes:
  --demo        Self-contained tiny legal corpus (no CUAD download needed). Uses the
                real embedder + cross-encoder from config over ~a dozen clauses.
  --from-disk   Load the real artifacts built by `make index` (FAISS + BM25) and
                query the actual CUAD corpus.

Usage:
  python -m src.retrieval.smoke --demo
  python -m src.retrieval.smoke --from-disk --config config/config.yaml
"""

from __future__ import annotations

import argparse

import yaml

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, Embedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

# A tiny synthetic contract corpus: each entry is one "chunk" with real offsets.
_DEMO_CLAUSES = [
    ("K1", "Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice to the other party."),
    ("K1", "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."),
    ("K1", "In no event shall either party's aggregate liability exceed the total fees paid under this Agreement in the prior twelve months."),
    ("K1", "Each party shall keep confidential all Confidential Information disclosed by the other party and not use it except as permitted."),
    ("K2", "The Supplier grants the Customer a non-exclusive, non-transferable license to use the Software during the Term."),
    ("K2", "Neither party may assign this Agreement without the prior written consent of the other party, except in a change of control."),
    ("K2", "The Term of this Agreement is three (3) years and shall automatically renew for successive one-year periods unless terminated."),
    ("K2", "The Customer shall indemnify and hold harmless the Supplier from any third-party claims arising from misuse of the Software."),
    ("K3", "All disputes arising under this Agreement shall be resolved by binding arbitration in New York, New York."),
    ("K3", "This Agreement contains a cap on liability limiting damages to direct damages only, excluding consequential damages."),
]

_DEMO_QUESTIONS = [
    'What does the contract say about "Termination For Convenience"?',
    'What is the "Governing Law"?',
    'Is there a "Cap On Liability"?',
    'What are the "Anti-Assignment" restrictions?',
]


def _demo_chunks() -> list[dict]:
    """Turn the synthetic clauses into chunk dicts with exact offsets per contract."""
    chunks: list[dict] = []
    offsets: dict[str, int] = {}
    for i, (cid, text) in enumerate(_DEMO_CLAUSES):
        start = offsets.get(cid, 0)
        end = start + len(text)
        chunks.append(
            {
                "chunk_id": f"{cid}::{i:05d}",
                "contract_id": cid,
                "char_start": start,
                "char_end": end,
                "text": text,
            }
        )
        offsets[cid] = end + 1  # +1 for a notional separator between clauses
    return chunks


def _print_results(question: str, results: list[dict]) -> None:
    print(f"\nQ: {question}")
    print("-" * 78)
    for r in results:
        prov = f"{r['contract_id']} [{r['char_start']}:{r['char_end']}]"
        snippet = r["text"] if len(r["text"]) <= 90 else r["text"][:87] + "..."
        print(f"  #{r['rank']}  rerank={r['rerank_score']:+.3f}  {prov}")
        print(f"      {snippet}")


def run_demo(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    emb_cfg, r_cfg = cfg["embeddings"], cfg["retrieval"]

    print("Loading models (once) ...")
    embedder = Embedder(emb_cfg["model"], device=emb_cfg["device"], batch_size=emb_cfg["batch_size"])
    reranker = CrossEncoderReranker(r_cfg["reranker_model"], device=emb_cfg["device"])

    chunks = _demo_chunks()
    print(f"Indexing {len(chunks)} synthetic clauses ...")
    embeddings = embedder.encode([c["text"] for c in chunks])

    # Build an in-memory FAISS index without touching disk.
    import faiss
    import numpy as np

    arr = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    sidecar = [
        {"faiss_id": i, **{k: c[k] for k in ("chunk_id", "contract_id", "char_start", "char_end", "text")}}
        for i, c in enumerate(chunks)
    ]

    dense = DenseRetriever(index, sidecar, embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(
        dense=dense, bm25=bm25, reranker=reranker,
        fusion=r_cfg.get("fusion", "rrf"), rrf_k=r_cfg.get("rrf_k", 60),
    )

    for q in _DEMO_QUESTIONS:
        results = hybrid.retrieve(
            q,
            dense_top_k=min(r_cfg["dense_top_k"], len(chunks)),
            bm25_top_k=min(r_cfg["bm25_top_k"], len(chunks)),
            candidates_before_rerank=min(r_cfg["candidates_before_rerank"], len(chunks)),
            top_n=3,
        )
        _print_results(q, results)


def run_from_disk(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    emb_cfg, r_cfg = cfg["embeddings"], cfg["retrieval"]

    print("Loading models + persisted indexes ...")
    embedder = Embedder(emb_cfg["model"], device=emb_cfg["device"], batch_size=emb_cfg["batch_size"])
    hybrid = HybridRetriever.from_disk(cfg, embedder)

    for q in _DEMO_QUESTIONS:
        results = hybrid.retrieve(
            q,
            dense_top_k=r_cfg["dense_top_k"],
            bm25_top_k=r_cfg["bm25_top_k"],
            candidates_before_rerank=r_cfg["candidates_before_rerank"],
            top_n=r_cfg["results_after_rerank"],
        )
        _print_results(q, results)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval smoke test (Phase 1.7).")
    ap.add_argument("--config", default="config/config.yaml")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="self-contained synthetic corpus")
    mode.add_argument("--from-disk", action="store_true", help="use artifacts from `make index`")
    args = ap.parse_args()

    if args.from_disk:
        run_from_disk(args.config)
    else:
        run_demo(args.config)


if __name__ == "__main__":
    main()
