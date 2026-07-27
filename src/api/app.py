"""FastAPI app exposing the pipeline (e.g. POST /answer -> structured Answer).

Business logic lives here, not in the UI. Langfuse tracing wired at this layer.
Served as `uvicorn src.api.app:app`. Phase 4.1 — not yet implemented.
"""
