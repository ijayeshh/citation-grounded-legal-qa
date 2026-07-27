"""LegalBench headline number, contamination-controlled (Phase 3.3-3.4).

Contamination control: LegalBench's ``cuad_*`` tasks are derived from the SAME CUAD
contracts this system is built on, so using them would leak. Instead we evaluate on
a NON-CUAD task from the ContractNLI dataset (``contract_nli_*``) — different
contracts entirely — using the pipeline's relevant sub-component: the shared NLI
model that powers the groundedness gate. This directly measures the entailment
capability the faithfulness guarantee rests on, with zero CUAD overlap.

Each example is a contract clause with a Yes/No answer to a fixed task hypothesis.
We apply NLI (premise = clause, hypothesis = task hypothesis) and map entailment ->
Yes. Headline metric = accuracy (balanced accuracy reported alongside).

Run:  python -m src.eval.legalbench_runner --config config/config.yaml
      python -m src.eval.legalbench_runner --fixture data/raw/legalbench_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from src.verify.nli import load_nli


def load_task_examples(cfg: dict, fixture: str | None, limit: int | None) -> list[dict]:
    """Load [{text, answer}] for the configured task, from a fixture or HuggingFace."""
    if fixture:
        rows = []
        with open(fixture, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows[:limit] if limit else rows

    from datasets import load_dataset  # local heavy import

    task = cfg["dataset"]["legalbench_task"]
    source = cfg["dataset"]["legalbench_source"]
    try:
        ds = load_dataset(source, task, split="test", trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - env/network dependent
        raise RuntimeError(
            f"Failed to load LegalBench task {task!r} from {source!r}: {exc}. "
            "Needs `datasets` with script support (pinned datasets==2.20.0)."
        ) from exc
    rows = [{"text": r["text"], "answer": r["answer"]} for r in ds]
    return rows[:limit] if limit else rows


def _balanced_accuracy(gold: list[str], pred: list[str]) -> float:
    """Mean of per-class recall over the classes present in gold."""
    classes = set(gold)
    recalls = []
    for c in classes:
        idx = [i for i, g in enumerate(gold) if g == c]
        correct = sum(1 for i in idx if pred[i] == c)
        recalls.append(correct / len(idx))
    return sum(recalls) / len(recalls) if recalls else 0.0


def run(cfg: dict, fixture: str | None = None, limit: int | None = None, nli=None) -> dict:
    task = cfg["dataset"]["legalbench_task"]
    hypothesis = cfg["dataset"]["legalbench_hypotheses"][task]
    threshold = cfg["verification"].get("entailment_threshold", 0.5)

    examples = load_task_examples(cfg, fixture, limit)

    # SAME shared NLI component as the groundedness gate (loaded once). Injectable for tests.
    if nli is None:
        nli = load_nli(cfg["verification"]["nli_model"], device=cfg["embeddings"]["device"])

    gold: list[str] = []
    pred: list[str] = []
    for ex in examples:
        gold_ans = str(ex["answer"]).strip().lower()
        gold.append("yes" if gold_ans.startswith("y") else "no")
        # premise = clause (evidence), hypothesis = the task statement.
        label, score = nli.check(hypothesis, ex["text"])
        pred.append("yes" if (label == "entailment" and score >= threshold) else "no")

    n = len(gold)
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0
    result = {
        "task": task,
        "source": "fixture" if fixture else cfg["dataset"]["legalbench_source"],
        "contamination_control": "non-CUAD ContractNLI task; no overlap with CUAD corpus",
        "n_examples": n,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(_balanced_accuracy(gold, pred), 4),
        "nli_model": cfg["verification"]["nli_model"],
    }
    return result


def write_report(result: dict, reports_dir: str) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "legalbench.md")
    lines = [
        "# LegalBench Headline (contamination-controlled)",
        "",
        f"- **Task:** `{result['task']}` (non-CUAD, ContractNLI-derived)",
        f"- **Contamination control:** {result['contamination_control']}",
        f"- **Sub-component evaluated:** shared NLI model (`{result['nli_model']}`) — the "
        "groundedness gate's entailment engine",
        f"- **Examples:** {result['n_examples']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy (headline) | {result['accuracy']:.4f} |",
        f"| Balanced accuracy | {result['balanced_accuracy']:.4f} |",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    # machine-readable
    with open(os.path.join(reports_dir, "legalbench.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="LegalBench headline (Phase 3.4).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--fixture", default=None, help="local JSONL of {text, answer} rows")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    result = run(cfg, fixture=args.fixture, limit=args.limit)
    path = write_report(result, cfg["paths"]["eval_reports_dir"])

    print(f"Task: {result['task']}  (source: {result['source']})")
    print(f"Examples: {result['n_examples']}")
    print(f"Accuracy (headline): {result['accuracy']:.4f}")
    print(f"Balanced accuracy:   {result['balanced_accuracy']:.4f}")
    print(f"Report -> {path}")


if __name__ == "__main__":
    main()
