"""Mistral La Plateforme wrapper — the single choke point for all LLM calls.

Retry + exponential backoff on 429 (RPM is the binding constraint). Model
selectable via config (small for dev, large for final). Centralizes retries,
logging, and Langfuse tracing. Phase 2.2 — not yet implemented.
"""
