"""MCP server exposing the retriever as a governed tool (Phase 4.3).

`search_contracts` is a THIN wrapper over the exact same hybrid retrieval the answer
pipeline uses — "MCP isn't magic, it's access". Integration-ready polish, kept
minimal. The retrieval logic itself stays plain Python (src/retrieval); this only
surfaces it as a tool.

Run:  python -m src.mcp.server
"""

from __future__ import annotations

import os

import yaml
from dotenv import load_dotenv


def search_contracts_impl(retriever, cfg: dict, query: str, top_n: int | None = None) -> list[dict]:
    """Run hybrid retrieval and return ranked spans with full provenance.

    Plain function (no MCP dependency) so it is unit-testable. The MCP tool below is
    a one-line wrapper over this.
    """
    r = cfg["retrieval"]
    n = top_n or r["results_after_rerank"]
    spans = retriever.retrieve(
        query,
        dense_top_k=r["dense_top_k"],
        bm25_top_k=r["bm25_top_k"],
        candidates_before_rerank=r["candidates_before_rerank"],
        top_n=n,
    )
    return [
        {
            "chunk_id": s["chunk_id"],
            "contract_id": s["contract_id"],
            "char_start": s["char_start"],
            "char_end": s["char_end"],
            "text": s["text"],
            "score": s.get("rerank_score", s.get("fusion_score")),
        }
        for s in spans
    ]


def build_server(cfg: dict):
    """Build a FastMCP server with the retriever loaded ONCE and the tool registered."""
    from mcp.server.fastmcp import FastMCP  # local import: optional dependency

    from src.retrieval.dense import Embedder
    from src.retrieval.hybrid import HybridRetriever

    emb = cfg["embeddings"]
    embedder = Embedder(emb["model"], device=emb["device"], batch_size=emb["batch_size"])
    retriever = HybridRetriever.from_disk(cfg, embedder)  # loaded once for the server's lifetime

    server = FastMCP("legal-qa-retriever")

    @server.tool()
    def search_contracts(query: str, top_n: int = 0) -> list[dict]:
        """Search the indexed contracts and return ranked spans with source offsets.

        Args:
            query: natural-language search over the contract corpus.
            top_n: number of spans to return (0 = config default).
        """
        return search_contracts_impl(retriever, cfg, query, top_n or None)

    return server


def main() -> None:
    load_dotenv()
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    build_server(cfg).run()


if __name__ == "__main__":
    main()
