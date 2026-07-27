"""Shared NLI entailment component — load the HuggingFace model ONCE (singleton).

Reused three ways: inline verifier gate, offline faithfulness harness, harness
entailment scoring. Never instantiate more than once. Given (claim, cited_span)
returns entailment label + score. Phase 2.5 — not yet implemented.
"""
