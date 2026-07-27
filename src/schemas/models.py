"""Pydantic schemas — the contract between generation, verification, API, MCP, and UI.

Phase 2.1. Three models, exactly as the plan specifies:

    Citation {contract_id, char_start, char_end, text}
    Claim    {text, citation, entailment_label, entailment_score}
    Answer   {question, claims[], raw_text, faithfulness_score, flagged_claims[]}

Design notes:
- Citation character offsets are indices into the ORIGINAL contract text (the same
  offsets preserved from chunking through retrieval). They are populated from the
  retrieved span the model cited, NOT emitted by the LLM, so they stay exact and
  are directly comparable to CUAD gold spans (IoU) later.
- ``entailment_label`` / ``entailment_score`` on a Claim and ``faithfulness_score`` /
  ``flagged_claims`` on an Answer are filled in during verification (Phase 2.5-2.6);
  at generation time they are unset (None / empty).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """A reference to an exact character span in a source contract."""

    contract_id: str = Field(..., min_length=1)
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)
    text: str

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_span(self) -> "Citation":
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than char_start ({self.char_start})"
            )
        return self


class Claim(BaseModel):
    """An atomic factual statement from the answer, with the source span it cites.

    ``entailment_label`` / ``entailment_score`` are set by the NLI verifier later.
    """

    text: str = Field(..., min_length=1)
    citation: Citation
    entailment_label: Optional[str] = None
    entailment_score: Optional[float] = None

    model_config = {"extra": "forbid"}


class Answer(BaseModel):
    """A cited answer: every factual claim carries a citation.

    ``faithfulness_score`` (fraction of claims entailed by their citation) and
    ``flagged_claims`` (claims not supported by their citation) are set during
    verification; at generation time faithfulness_score is None and flagged_claims [].
    """

    question: str = Field(..., min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    raw_text: str = ""
    faithfulness_score: Optional[float] = None
    flagged_claims: list[Claim] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
