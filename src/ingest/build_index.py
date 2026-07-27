"""`make index` entrypoint (Phase 1.2-1.6):

    download CUAD  ->  normalize (contracts + gold spans)  ->  validate offsets
                   ->  persist gold-span store + contracts  ->  chunk  ->  persist chunks
                   ->  build + persist FAISS dense index (+ sidecar)  ->  build + persist BM25

Run:  python -m src.ingest.build_index --config config/config.yaml [--split train]
                                        [--limit N] [--no-retrieval] [--raw-path FILE]

--no-retrieval stops after chunking (skips the heavy embedding/index build).
--raw-path indexes from an existing raw JSONL snapshot instead of downloading
           (reproducibility: re-index without re-hitting HuggingFace).
"""

from __future__ import annotations

import argparse
import os

import yaml

from src.ingest import chunk as chunk_mod
from src.ingest import download as download_mod
from src.ingest import gold_spans as gold_mod
from src.ingest import parse as parse_mod
from src.retrieval import bm25 as bm25_mod
from src.retrieval import dense as dense_mod


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(
    config_path: str,
    split: str = "train",
    limit: int | None = None,
    build_retrieval: bool = True,
    raw_path: str | None = None,
    max_chars: int | None = None,
    overlap_chars: int | None = None,
) -> None:
    cfg = _load_config(config_path)
    paths = cfg["paths"]
    chunking = dict(cfg["chunking"])
    if max_chars is not None:
        chunking["max_chars"] = max_chars
    if overlap_chars is not None:
        chunking["overlap_chars"] = overlap_chars
    total = 6 if build_retrieval else 4

    os.makedirs(paths["raw_dir"], exist_ok=True)
    os.makedirs(paths["processed_dir"], exist_ok=True)

    if raw_path:
        print(f"[1/{total}] Using existing raw snapshot (no download): {raw_path}")
    else:
        print(f"[1/{total}] Downloading CUAD (split={split}, limit={limit}) ...")
        raw_path = download_mod.download_cuad(
            raw_dir=paths["raw_dir"],
            dataset_name=cfg["dataset"]["name"],
            split=split,
            limit=limit,
        )
    print(f"      raw -> {raw_path}")

    print(f"[2/{total}] Normalizing into contracts + gold spans ...")
    contracts, gold = parse_mod.normalize_records(download_mod.load_raw_records(raw_path))
    print(f"      contracts={len(contracts)}  gold_spans={gold_mod.count_spans(gold)}")

    mismatches = parse_mod.validate_offsets(contracts, gold)
    if mismatches:
        print(f"      WARNING: {len(mismatches)} gold spans failed offset validation")
    else:
        print("      offset validation OK (every gold span text matches its offsets)")

    print(f"[3/{total}] Persisting gold-span store + contracts ...")
    gold_mod.save_gold_spans(gold, paths["gold_spans"])
    parse_mod.save_contracts(contracts, paths["contracts"])
    print(f"      gold_spans -> {paths['gold_spans']}")
    print(f"      contracts  -> {paths['contracts']}")

    print(f"[4/{total}] Chunking (sliding window) ...")
    chunks = chunk_mod.chunk_contracts(
        contracts,
        max_chars=chunking["max_chars"],
        overlap_chars=chunking["overlap_chars"],
    )
    chunk_mod.save_chunks(chunks, paths["chunks"])
    print(f"      chunks={len(chunks)} -> {paths['chunks']}")

    if not build_retrieval:
        print("[done] Stopped after chunking (--no-retrieval). Indexes not built.")
        return

    emb = cfg["embeddings"]
    print(f"[5/{total}] Building dense FAISS index (model={emb['model']}, device={emb['device']}) ...")
    embedder = dense_mod.Embedder(emb["model"], device=emb["device"], batch_size=emb["batch_size"])
    dense_mod.build_and_persist(chunks, embedder, paths["faiss_index"], paths["faiss_sidecar"])
    print(f"      faiss index -> {paths['faiss_index']}")
    print(f"      sidecar     -> {paths['faiss_sidecar']}")

    print(f"[6/{total}] Building BM25 index ...")
    bm25_mod.build_and_persist(chunks, paths["bm25_store"])
    print(f"      bm25 -> {paths['bm25_store']}")

    print("[done] Phase 1.2-1.6 complete. Hybrid retriever ready (see src/retrieval/smoke.py).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest + chunk + index CUAD (Phase 1.2-1.6).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-retrieval", action="store_true", help="stop after chunking")
    ap.add_argument("--raw-path", default=None, help="index from an existing raw JSONL snapshot")
    ap.add_argument("--max-chars", type=int, default=None, help="override chunking.max_chars")
    ap.add_argument("--overlap-chars", type=int, default=None, help="override chunking.overlap_chars")
    args = ap.parse_args()
    run(
        args.config,
        split=args.split,
        limit=args.limit,
        build_retrieval=not args.no_retrieval,
        raw_path=args.raw_path,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )


if __name__ == "__main__":
    main()
