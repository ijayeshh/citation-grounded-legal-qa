"""Streamlit frontend (full method only — NO baseline).

Shows the answer, cited spans highlighted in the source, the live faithfulness
score, and unsupported claims flagged in red. Load FAISS + embedder + reranker +
NLI ONCE via @st.cache_resource. Calls the FastAPI backend (no pipeline logic
duplicated here). Phase 4.2 — not yet implemented.
"""
