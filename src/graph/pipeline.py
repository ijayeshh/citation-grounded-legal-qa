"""Stateful pipeline graph: retrieve -> generate cited answer -> decompose claims
-> verify (NLI gate) -> [regenerate unsupported -> re-verify] -> finalize.

Returns the finalized Answer with per-claim entailment verdicts + live faithfulness
score. Phase 2.7 — not yet implemented.
"""
