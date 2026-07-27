"""promptfoo custom provider — runs the real cited-answer output contract.

promptfoo (the cheap, always-on smoke alarm) calls this to exercise the ACTUAL
generation path: prompt -> Mistral -> Pydantic parse/retry -> citation resolution,
given a fixed question + source spans. It returns the structured Answer as JSON for
promptfoo's assertions to check. It does NOT run retrieval or NLI — groundedness
measurement is the harness's job, kept deliberately separate.

Requires MISTRAL_API_KEY in the environment (a GitHub Actions secret in CI).
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def call_api(prompt, options, context):
    """promptfoo entry point. Uses context['vars'] (question + spans), returns {output}."""
    import yaml
    from dotenv import load_dotenv

    from src.generation.mistral_client import MistralClient
    from src.generation.prompts import generate_cited_answer

    load_dotenv(os.path.join(_ROOT, ".env"))
    if not os.environ.get("MISTRAL_API_KEY"):
        return {"error": "MISTRAL_API_KEY not set"}

    vars_ = (context or {}).get("vars", {})
    question = vars_.get("question", prompt)
    spans = vars_.get("spans", [])
    if isinstance(spans, str):
        spans = json.loads(spans)

    cfg = yaml.safe_load(open(os.path.join(_ROOT, "config", "config.yaml"), encoding="utf-8"))
    client = MistralClient.from_config(cfg)  # mistral-small-latest

    try:
        answer = generate_cited_answer(question, spans, client)
    except Exception as exc:  # surface as a provider error so the test fails loudly
        return {"error": f"generation failed: {exc}"}

    return {"output": answer.model_dump_json()}
