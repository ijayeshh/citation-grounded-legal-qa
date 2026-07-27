"""Shared NLI entailment component (Phase 2.5).

The NLI model is loaded ONCE and reused everywhere via a process-wide cache
(``load_nli``). It is reused three ways — the inline groundedness gate (2.6), the
offline faithfulness harness (3.1), and the harness's entailment scoring (3.2) — so
it must never be instantiated more than once per model name.

For a (claim, cited_span) pair it answers: does the span support (entail) the claim?
premise = cited span (evidence), hypothesis = claim.
"""

from __future__ import annotations

import threading

_INSTANCES: dict[str, "NLIModel"] = {}
_LOCK = threading.Lock()


def load_nli(model_name: str, device: str = "cpu") -> "NLIModel":
    """Return the cached NLIModel for ``model_name`` (creating it once, thread-safe)."""
    with _LOCK:
        inst = _INSTANCES.get(model_name)
        if inst is None:
            inst = NLIModel(model_name, device=device)
            _INSTANCES[model_name] = inst
        return inst


def _softmax(logits):
    import numpy as np

    x = np.asarray(logits, dtype="float64")
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class NLIModel:
    """A HuggingFace cross-encoder NLI model. Prefer ``load_nli`` over direct construction."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder  # local heavy import

        self.model_name = model_name
        self.cross_encoder = CrossEncoder(model_name, device=device)
        # Map class indices -> label names from the model config; find the entail class.
        id2label = self.cross_encoder.model.config.id2label
        self.id2label = {int(k): str(v).lower() for k, v in id2label.items()}
        self.entail_idx = next(
            (i for i, lbl in self.id2label.items() if "entail" in lbl), None
        )
        if self.entail_idx is None:
            raise ValueError(
                f"NLI model {model_name!r} has no 'entailment' label; labels={self.id2label}"
            )

    def check(self, claim: str, cited_span: str) -> tuple[str, float]:
        """Return (predicted_label, entailment_probability) for one (claim, span) pair."""
        return self.check_batch([(claim, cited_span)])[0]

    def check_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Batch variant. Each pair is (claim, cited_span). premise=span, hypothesis=claim."""
        if not pairs:
            return []
        model_inputs = [(span, claim) for (claim, span) in pairs]  # (premise, hypothesis)
        logits = self.cross_encoder.predict(model_inputs, convert_to_numpy=True)
        # predict returns shape (n, num_labels) for multi-class NLI models.
        out: list[tuple[str, float]] = []
        for row in logits:
            probs = _softmax(row)
            label = self.id2label[int(probs.argmax())]
            out.append((label, float(probs[self.entail_idx])))
        return out
