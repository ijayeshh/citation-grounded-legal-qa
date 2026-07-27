"""The Groundedness Gate — the project's ONE deliberate guardrail (Phase 2.6).

Policy: no answer sentence ships unless it is entailed by a cited source span.
For each claim, run NLI against its cited span; a claim that is not entailed is
flagged. Whether flagged claims are regenerated (once) before being flagged for
good is config-driven and orchestrated by the LangGraph pipeline (Phase 2.7); this
module owns the *verdict* and the policy knobs.

This is NOT a generic safety guardrail (no PII/toxicity/jailbreak filtering). The
only real risk in this system is ungrounded claims, and this gate is what controls
it. Name it as such wherever it appears.
"""

from __future__ import annotations

from src.schemas.models import Answer, Claim


class GroundednessGate:
    """Scores each claim's entailment against its citation and flags the unsupported.

    ``nli`` is a shared NLI component (see src.verify.nli.load_nli) — the same
    instance is reused by the offline harness, never re-instantiated.
    """

    def __init__(
        self,
        nli,
        entailment_threshold: float = 0.5,
        regenerate_unsupported: bool = True,
        max_regeneration_rounds: int = 1,
    ):
        self.nli = nli
        self.entailment_threshold = entailment_threshold
        self.regenerate_unsupported = regenerate_unsupported
        self.max_regeneration_rounds = max_regeneration_rounds

    @classmethod
    def from_config(cls, cfg: dict, nli) -> "GroundednessGate":
        v = cfg["verification"]
        return cls(
            nli=nli,
            entailment_threshold=v.get("entailment_threshold", 0.5),
            regenerate_unsupported=v.get("regenerate_unsupported", True),
            max_regeneration_rounds=v.get("max_regeneration_rounds", 1),
        )

    def evaluate_claim(self, claim: Claim) -> Claim:
        """Set ``entailment_label`` / ``entailment_score`` on the claim in place."""
        label, score = self.nli.check(claim.text, claim.citation.text)
        claim.entailment_label = label
        claim.entailment_score = score
        return claim

    def is_entailed(self, claim: Claim) -> bool:
        """A claim is grounded iff predicted label is entailment AND score >= threshold."""
        return (
            claim.entailment_label == "entailment"
            and (claim.entailment_score or 0.0) >= self.entailment_threshold
        )

    def apply(self, answer: Answer) -> Answer:
        """Verify every claim, set flagged_claims, and compute the live faithfulness score.

        faithfulness_score = fraction of claims entailed by their citation. With no
        claims it is 1.0 (vacuously faithful — nothing ungrounded was asserted).
        """
        for claim in answer.claims:
            self.evaluate_claim(claim)
        flagged = [c for c in answer.claims if not self.is_entailed(c)]
        total = len(answer.claims)
        entailed = total - len(flagged)
        answer.flagged_claims = flagged
        answer.faithfulness_score = 1.0 if total == 0 else entailed / total
        return answer
