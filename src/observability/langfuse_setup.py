"""Langfuse observability wiring (Phase 4.1 hook; expanded in 4.5).

Per-request tracing lives at the API layer so every request records what was
retrieved, generated, and flagged. Tracing must NEVER break a response: if Langfuse
is disabled, unconfigured, or not installed, this degrades to a no-op.
"""

from __future__ import annotations

import os


def get_langfuse(cfg: dict):
    """Return a Langfuse client, or None if disabled/unconfigured/unavailable."""
    obs = cfg.get("observability", {})
    if not obs.get("langfuse_enabled", False):
        return None
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None
    try:
        from langfuse import Langfuse  # local import; optional dependency
    except Exception:
        return None
    try:
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        return None


def trace_answer(langfuse, question: str, spans: list, answer) -> None:
    """Record one request: retrieved spans, generated claims, faithfulness, flagged.

    Best-effort — any failure is swallowed so observability never affects the API.
    """
    if langfuse is None:
        return
    try:
        trace = langfuse.trace(name="answer", input={"question": question})
        trace.span(
            name="retrieve",
            output={
                "n_spans": len(spans),
                "chunk_ids": [s.get("chunk_id") for s in spans],
            },
        )
        trace.span(
            name="generate",
            output={
                "n_claims": len(answer.claims),
                "claims": [
                    {"text": c.text, "contract_id": c.citation.contract_id,
                     "char_start": c.citation.char_start, "char_end": c.citation.char_end}
                    for c in answer.claims
                ],
            },
        )
        trace.span(
            name="verify",
            output={
                "faithfulness_score": answer.faithfulness_score,
                "n_claims": len(answer.claims),
                "n_flagged": len(answer.flagged_claims),
                "verdicts": [
                    {"text": c.text, "label": c.entailment_label, "score": c.entailment_score}
                    for c in answer.claims
                ],
                "flagged": [c.text for c in answer.flagged_claims],
            },
        )
        trace.update(output={"raw_text": answer.raw_text, "n_claims": len(answer.claims)})
        langfuse.flush()
    except Exception:
        pass
