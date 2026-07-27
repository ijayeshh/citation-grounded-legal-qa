"""Split an answer into atomic, independently-checkable claims (Phase 2.4).

A prerequisite for measuring faithfulness at claim granularity: each atomic claim
can be NLI-checked against its cited span on its own. Decomposition is a focused LLM
call with a strict rubric; it runs per generated claim and the resulting atomic
sub-claims inherit that claim's citation, so grounding is preserved.

All LLM calls go through the injected client (a MistralClient), so this is
config-driven and unit-testable with a mocked client.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, ValidationError

from src.schemas.models import Answer, Claim

# Distinctive header so callers/tests can tell this call apart from generation.
DECOMPOSITION_SYSTEM = (
    "You are an ATOMIC CLAIM DECOMPOSER. Given a single statement, split it into a "
    "list of atomic, independently verifiable factual claims.\n"
    "Rules:\n"
    "1. Each claim must express exactly ONE fact.\n"
    "2. Preserve the original meaning; do NOT add any information not in the statement.\n"
    "3. Do not include opinions, questions, or citations.\n"
    "4. If the statement is already atomic, return it unchanged as a single-element list.\n"
    'Respond with ONLY a JSON object: {"claims": ["<atomic claim>", ...]}'
)


class DecompositionResult(BaseModel):
    claims: list[str] = []

    model_config = {"extra": "ignore"}


class DecompositionError(RuntimeError):
    """Raised when the model cannot produce a valid decomposition."""


def _build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": DECOMPOSITION_SYSTEM},
        {"role": "user", "content": f"Statement:\n{text}"},
    ]


def decompose_text(text: str, client, max_parse_retries: int = 2) -> list[str]:
    """Return the atomic claim strings for one statement (with parse-retry)."""
    messages = _build_messages(text)
    last_err: Optional[Exception] = None
    for _ in range(max_parse_retries + 1):
        raw = client.complete(messages, json_mode=True)
        try:
            result = DecompositionResult.model_validate_json(raw)
            claims = [c.strip() for c in result.claims if c and c.strip()]
            # A well-formed but empty decomposition falls back to the original text
            # so we never silently drop a claim.
            return claims or [text]
        except (ValidationError, ValueError, json.JSONDecodeError) as err:
            last_err = err
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"Your previous response was invalid: {err}. Respond again with "
                    'ONLY a JSON object of the form {"claims": ["..."]}.'
                )},
            ]
    raise DecompositionError(
        f"Failed to decompose statement after {max_parse_retries + 1} attempts: {last_err}"
    )


def decompose_claim(claim: Claim, client, max_parse_retries: int = 2) -> list[Claim]:
    """Decompose one Claim into atomic Claims, each inheriting the citation."""
    texts = decompose_text(claim.text, client, max_parse_retries=max_parse_retries)
    return [Claim(text=t, citation=claim.citation) for t in texts]


def decompose_answer(answer: Answer, client, max_parse_retries: int = 2) -> Answer:
    """Return a new Answer whose claims are the atomic decomposition of the originals.

    Question and raw_text are preserved; entailment fields stay unset (the gate fills
    them). Verification-phase fields are reset since the claim set changed.
    """
    atomic: list[Claim] = []
    for claim in answer.claims:
        atomic.extend(decompose_claim(claim, client, max_parse_retries=max_parse_retries))
    return Answer(question=answer.question, claims=atomic, raw_text=answer.raw_text)
