"""Offline faithfulness harness (`make eval` entrypoint).

Run the full pipeline over an eval set; decompose answers -> NLI-check each claim
against its cited span -> faithfulness = fraction of claims entailed. Reports NLI
and (optional) LLM-judge verdicts, cross-checked. Phase 3.1 — not yet implemented.
"""
