"""Pydantic models: Citation, Claim, Answer.

Citation {contract_id, char_start, char_end, text}
Claim    {text, citation, entailment_label, entailment_score}
Answer   {question, claims[], raw_text, faithfulness_score, flagged_claims[]}

Phase 2.1 — not yet implemented.
"""
