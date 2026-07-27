"""Answer-generation prompt + structured-output enforcement (Phase 2.3).

The model must answer ONLY from the provided retrieved spans and attach a citation
to each factual claim. It cites a span by its short id (S1, S2, ...); we resolve
that id back to the span's exact provenance (contract_id + char offsets) so the
Citation offsets are objective, never LLM-invented.

Output-format guardrail (NOT a generic safety library): the model is asked for a
strict JSON object; we validate it with Pydantic and, on malformed output, re-prompt
with the error up to a small retry budget. This is the transparent, appropriate
"guardrail" the plan calls for.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, ValidationError

from src.schemas.models import Answer, Citation, Claim

SYSTEM_PROMPT = (
    "You are a legal contract analysis assistant. You answer questions about a "
    "contract using ONLY the numbered source spans provided by the user. "
    "Rules:\n"
    "1. Use ONLY information contained in the provided sources. Never rely on "
    "outside knowledge or assumptions.\n"
    "2. Decompose your answer into atomic factual claims. Each claim must be a "
    "single, independently checkable statement.\n"
    "3. Every claim MUST be grounded in exactly ONE source, referenced by its id "
    "(e.g. \"S1\"). Do not cite a source that does not support the claim.\n"
    "4. If the sources do not contain the answer, say so in 'answer' and return an "
    "empty 'claims' list. Do not fabricate.\n"
    "5. Respond with a SINGLE JSON object and nothing else, matching exactly:\n"
    "{\n"
    '  "answer": "<concise prose answer>",\n'
    '  "claims": [ {"text": "<atomic claim>", "source_id": "<S#>"}, ... ]\n'
    "}"
)


# --- Transport schema: what the LLM returns (before resolving citations) ------


class GeneratedClaim(BaseModel):
    text: str
    source_id: str

    model_config = {"extra": "ignore"}  # lenient toward extra keys from the model


class GeneratedAnswer(BaseModel):
    answer: str = ""
    claims: list[GeneratedClaim] = []

    model_config = {"extra": "ignore"}


class StructuredOutputError(RuntimeError):
    """Raised when the model cannot produce valid, resolvable structured output."""


# --- Prompt construction ------------------------------------------------------


def format_spans(span_by_id: dict[str, dict]) -> str:
    """Render the retrieved spans as a labelled, model-readable block."""
    lines = []
    for sid, span in span_by_id.items():
        loc = f"contract={span['contract_id']} chars {span['char_start']}-{span['char_end']}"
        lines.append(f"[{sid}] ({loc})\n\"{span['text']}\"")
    return "\n\n".join(lines)


def build_messages(question: str, span_by_id: dict[str, dict]) -> list[dict]:
    user = (
        f"Question: {question}\n\n"
        f"Sources:\n{format_spans(span_by_id)}\n\n"
        "Answer the question using only these sources, following the required JSON format."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --- Parsing + resolution -----------------------------------------------------


def parse_generated(raw: str) -> GeneratedAnswer:
    """Parse raw model text into a validated GeneratedAnswer (raises on malformed)."""
    return GeneratedAnswer.model_validate_json(raw)


def resolve_answer(
    question: str, generated: GeneratedAnswer, span_by_id: dict[str, dict]
) -> Answer:
    """Turn a GeneratedAnswer into the canonical Answer, resolving citations.

    Each claim's ``source_id`` is resolved to the retrieved span's exact provenance.
    An unknown source_id is treated as invalid output (raises) so it triggers a
    re-prompt rather than silently dropping grounding.
    """
    claims: list[Claim] = []
    for gc in generated.claims:
        span = span_by_id.get(gc.source_id)
        if span is None:
            raise ValueError(
                f"claim cites unknown source_id {gc.source_id!r}; "
                f"valid ids: {sorted(span_by_id)}"
            )
        citation = Citation(
            contract_id=span["contract_id"],
            char_start=span["char_start"],
            char_end=span["char_end"],
            text=span["text"],
        )
        claims.append(Claim(text=gc.text, citation=citation))
    return Answer(question=question, claims=claims, raw_text=generated.answer)


def generate_cited_answer(
    question: str,
    retrieved_spans: list[dict],
    client,
    max_parse_retries: int = 2,
) -> Answer:
    """Full generation step: prompt -> LLM -> parse+resolve, with retry on bad output.

    ``client`` is a MistralClient (or anything exposing ``complete(messages, json_mode=)``).
    ``retrieved_spans`` are the ranked chunks from the hybrid retriever (each with
    contract_id, char_start, char_end, text). Returns a canonical Answer; entailment
    fields are left unset for the verification phase.
    """
    span_by_id = {f"S{i + 1}": s for i, s in enumerate(retrieved_spans)}
    messages = build_messages(question, span_by_id)

    last_err: Optional[Exception] = None
    for attempt in range(max_parse_retries + 1):
        raw = client.complete(messages, json_mode=True)
        try:
            generated = parse_generated(raw)
            return resolve_answer(question, generated, span_by_id)
        except (ValidationError, ValueError, json.JSONDecodeError) as err:
            last_err = err
            # Re-prompt with the specific error so the model can self-correct.
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {err}. "
                        "Respond again with ONLY a valid JSON object in the required "
                        "format, citing only the provided source ids."
                    ),
                },
            ]

    raise StructuredOutputError(
        f"Model failed to produce valid structured output after "
        f"{max_parse_retries + 1} attempts: {last_err}"
    )


# --- Targeted regeneration (used by the groundedness gate's regenerate step) ---

REGENERATE_SYSTEM = (
    "You are a legal contract analysis assistant. REGENERATE the given statements so "
    "each is DIRECTLY and FULLY supported by exactly one of the numbered source spans. "
    "Rules:\n"
    "1. Use ONLY the provided sources; never rely on outside knowledge.\n"
    "2. Rewrite each unsupported statement into an atomic claim that a source clearly "
    "supports, and cite that source by id.\n"
    "3. If a statement cannot be supported by ANY source, omit it entirely.\n"
    "4. Respond with ONLY a JSON object matching exactly:\n"
    '{ "claims": [ {"text": "<atomic claim>", "source_id": "<S#>"}, ... ] }'
)


def regenerate_unsupported_claims(
    question: str,
    unsupported_texts: list[str],
    retrieved_spans: list[dict],
    client,
    max_parse_retries: int = 2,
) -> list[Claim]:
    """Re-prompt the model to restate unsupported claims so they are grounded.

    Returns resolved Claims (possibly fewer than inputs, or empty if none can be
    grounded). Citations are resolved from the retrieved spans, never the LLM.
    """
    if not unsupported_texts:
        return []
    span_by_id = {f"S{i + 1}": s for i, s in enumerate(retrieved_spans)}
    listing = "\n".join(f"- {t}" for t in unsupported_texts)
    user = (
        f"Question: {question}\n\n"
        f"Sources:\n{format_spans(span_by_id)}\n\n"
        f"These statements were NOT supported by their cited source:\n{listing}\n\n"
        "Rewrite them to be supported, or omit any that cannot be, in the required JSON format."
    )
    messages = [
        {"role": "system", "content": REGENERATE_SYSTEM},
        {"role": "user", "content": user},
    ]

    last_err: Optional[Exception] = None
    for _ in range(max_parse_retries + 1):
        raw = client.complete(messages, json_mode=True)
        try:
            generated = parse_generated(raw)
            return resolve_answer(question, generated, span_by_id).claims
        except (ValidationError, ValueError, json.JSONDecodeError) as err:
            last_err = err
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"Your previous response was invalid: {err}. Respond again with ONLY "
                    "a valid JSON object, citing only the provided source ids."
                )},
            ]
    raise StructuredOutputError(
        f"Regeneration failed to produce valid structured output: {last_err}"
    )
