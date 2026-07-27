"""Faithfulness harness (Phase 3.1 + 3.5) — the differentiator, run by `make eval`.

For a set of evaluation questions it runs the FULL pipeline, then scores:

  1. Claim-level faithfulness (NLI): fraction of claims entailed by their cited span,
     using the SAME shared NLI component as the inline gate (never re-instantiated).
  2. Claim-level faithfulness (LLM judge): a strict-rubric second opinion, cross-checked
     against the NLI verdict (agreement rate) — two judge methods is a maturity signal.
  3. Gold-span citation metrics (precision / recall / IoU) via src.eval.citation_metrics.

Results are written to eval_reports/ as committed Markdown + CSV, with the headline
numbers surfaced at the top of the README.

Single full method only — NO baseline, NO ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from src.eval import citation_metrics as cm
from src.generation.mistral_client import MistralClient
from src.graph.pipeline import answer_question, build_pipeline
from src.ingest.gold_spans import load_gold_spans
from src.retrieval.dense import Embedder
from src.retrieval.hybrid import HybridRetriever
from src.verify.groundedness_gate import GroundednessGate
from src.verify.nli import load_nli

# --- LLM judge ---------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a STRICT FAITHFULNESS JUDGE. Decide whether a CLAIM is fully supported "
    "by the SOURCE text. A claim is SUPPORTED only if every part of it is directly "
    "stated or unambiguously implied by the source. If any part is missing, added, "
    "or contradicted, it is NOT_SUPPORTED. Do not use outside knowledge.\n"
    'Respond with ONLY JSON: {"verdict": "supported" | "not_supported"}'
)


class _JudgeResult(BaseModel):
    verdict: str

    model_config = {"extra": "ignore"}


def judge_claim(claim, client) -> bool:
    """Ask the LLM judge whether the cited span supports the claim. Returns True if so."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"SOURCE:\n{claim.citation.text}\n\nCLAIM:\n{claim.text}"},
    ]
    raw = client.complete(messages, json_mode=True)
    try:
        verdict = _JudgeResult.model_validate_json(raw).verdict.strip().lower()
    except Exception:
        return False  # unparseable judge output counts as not-supported (conservative)
    return verdict.startswith("support")


def judge_answer(answer, client) -> list[bool]:
    return [judge_claim(c, client) for c in answer.claims]


# --- Eval set ----------------------------------------------------------------

_QUESTION_TEMPLATE = (
    'Highlight the parts (if any) of this contract related to "{clause}" that should '
    "be reviewed by a lawyer."
)


def build_eval_items(gold_spans: dict, limit: int | None = None) -> list[dict]:
    """Derive one eval item per (contract_id, clause_type) present in the gold store.

    Each item ties a question to the exact gold spans it should cite, so citation
    metrics are objective. Deterministic ordering for reproducibility.
    """
    items: list[dict] = []
    for contract_id in sorted(gold_spans):
        for clause_type in sorted(gold_spans[contract_id]):
            items.append(
                {
                    "id": f"{contract_id}::{clause_type}",
                    "question": _QUESTION_TEMPLATE.format(clause=clause_type),
                    "contract_id": contract_id,
                    "clause_type": clause_type,
                }
            )
    return items[:limit] if limit else items


# --- Core evaluation ---------------------------------------------------------


def evaluate(cfg: dict, limit: int | None = None, use_final_model: bool = False) -> dict:
    """Run the full harness and return {per_question, aggregate}.

    ``use_final_model=True`` runs generation on Mistral Large (the final headline run);
    otherwise Mistral Small (iteration).
    """
    paths, v_cfg, e_cfg, emb_cfg, g_cfg = (
        cfg["paths"], cfg["verification"], cfg["eval"], cfg["embeddings"], cfg["generation"],
    )
    gen_model = g_cfg["model_final"] if use_final_model else g_cfg["model_dev"]

    gold_spans = load_gold_spans(paths["gold_spans"])
    items = build_eval_items(gold_spans, limit=limit)

    # Shared components: NLI loaded ONCE and reused by the gate here.
    embedder = Embedder(emb_cfg["model"], device=emb_cfg["device"], batch_size=emb_cfg["batch_size"])
    retriever = HybridRetriever.from_disk(cfg, embedder)
    nli = load_nli(v_cfg["nli_model"], device=emb_cfg["device"])
    gate = GroundednessGate.from_config(cfg, nli)
    gen_client = MistralClient.from_config(cfg, use_final_model=use_final_model)
    pipeline = build_pipeline(retriever, gen_client, gate, cfg)

    use_judge = e_cfg.get("use_llm_judge", True)
    judge_client = (
        MistralClient.from_config(cfg, model=e_cfg["llm_judge_model"]) if use_judge else None
    )

    per_q: list[dict] = []
    for item in items:
        print(f"  - {item['id']}")
        answer = answer_question(pipeline, item["question"])
        n_claims = len(answer.claims)

        nli_flags = [gate.is_entailed(c) for c in answer.claims]
        nli_faith = answer.faithfulness_score  # fraction entailed (from the gate)

        if use_judge:
            judge_flags = judge_answer(answer, judge_client)
            judge_faith = (sum(judge_flags) / n_claims) if n_claims else 1.0
            agree = [n == j for n, j in zip(nli_flags, judge_flags)]
            agreement = (sum(agree) / n_claims) if n_claims else 1.0
        else:
            judge_faith = agreement = None

        pred = cm.predicted_spans_for_contract(answer, item["contract_id"])
        gold = cm.gold_spans_to_intervals(gold_spans[item["contract_id"]][item["clause_type"]])
        metrics = cm.span_char_metrics(pred, gold)

        per_q.append({
            "id": item["id"],
            "contract_id": item["contract_id"],
            "clause_type": item["clause_type"],
            "n_claims": n_claims,
            "faithfulness_nli": round(nli_faith, 4) if nli_faith is not None else None,
            "faithfulness_judge": round(judge_faith, 4) if judge_faith is not None else None,
            "nli_judge_agreement": round(agreement, 4) if agreement is not None else None,
            "cit_precision": round(metrics["precision"], 4),
            "cit_recall": round(metrics["recall"], 4),
            "cit_iou": round(metrics["iou"], 4),
        })

    aggregate = _aggregate(per_q, use_judge)
    return {
        "per_question": per_q,
        "aggregate": aggregate,
        "n_questions": len(per_q),
        "generation_model": gen_model,
    }


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _aggregate(per_q: list[dict], use_judge: bool) -> dict:
    return {
        "faithfulness_nli": _mean([r["faithfulness_nli"] for r in per_q]),
        "faithfulness_judge": _mean([r["faithfulness_judge"] for r in per_q]) if use_judge else None,
        "nli_judge_agreement": _mean([r["nli_judge_agreement"] for r in per_q]) if use_judge else None,
        "cit_precision": _mean([r["cit_precision"] for r in per_q]),
        "cit_recall": _mean([r["cit_recall"] for r in per_q]),
        "cit_iou": _mean([r["cit_iou"] for r in per_q]),
    }


# --- Reporting (3.5) ---------------------------------------------------------

_COLUMNS = [
    "id", "contract_id", "clause_type", "n_claims",
    "faithfulness_nli", "faithfulness_judge", "nli_judge_agreement",
    "cit_precision", "cit_recall", "cit_iou",
]


def write_reports(results: dict, reports_dir: str, formats: list[str]) -> list[str]:
    os.makedirs(reports_dir, exist_ok=True)
    per_q, agg = results["per_question"], results["aggregate"]
    written: list[str] = []

    if "csv" in formats:
        path = os.path.join(reports_dir, "faithfulness_per_question.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_COLUMNS)
            w.writeheader()
            w.writerows(per_q)
        written.append(path)

    if "md" in formats:
        written.append(_write_summary_md(results, reports_dir))

    # Machine-readable aggregate for the README updater / CI.
    path = os.path.join(reports_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"n_questions": results["n_questions"], "aggregate": agg}, fh, indent=2)
    written.append(path)
    return written


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def _write_summary_md(results: dict, reports_dir: str) -> str:
    per_q, agg, n = results["per_question"], results["aggregate"], results["n_questions"]
    lines = [
        "# Faithfulness & Citation Evaluation",
        "",
        f"Full-method pipeline over **{n}** evaluation questions (single method; no baseline).",
        f"Generation model: **{results.get('generation_model', '?')}**. "
        "Faithfulness = fraction of atomic claims entailed by their cited source span.",
        "",
        "## Headline (aggregate)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Claim faithfulness (NLI) | {_fmt(agg['faithfulness_nli'])} |",
        f"| Claim faithfulness (LLM judge) | {_fmt(agg['faithfulness_judge'])} |",
        f"| NLI–judge agreement | {_fmt(agg['nli_judge_agreement'])} |",
        f"| Citation precision (vs CUAD gold spans) | {_fmt(agg['cit_precision'])} |",
        f"| Citation recall (vs CUAD gold spans) | {_fmt(agg['cit_recall'])} |",
        f"| Citation IoU (char offsets) | {_fmt(agg['cit_iou'])} |",
        "",
        "## Per-question",
        "",
        "| Question (contract :: clause) | Claims | Faith (NLI) | Faith (Judge) | Agree | Cit P | Cit R | Cit IoU |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in per_q:
        lines.append(
            f"| {r['id']} | {r['n_claims']} | {_fmt(r['faithfulness_nli'])} | "
            f"{_fmt(r['faithfulness_judge'])} | {_fmt(r['nli_judge_agreement'])} | "
            f"{_fmt(r['cit_precision'])} | {_fmt(r['cit_recall'])} | {_fmt(r['cit_iou'])} |"
        )
    lines.append("")
    path = os.path.join(reports_dir, "faithfulness.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _print_summary(results: dict) -> None:
    agg = results["aggregate"]
    print("\n===== EVAL SUMMARY (aggregate) =====")
    print(f"  questions            : {results['n_questions']}")
    print(f"  faithfulness (NLI)   : {_fmt(agg['faithfulness_nli'])}")
    print(f"  faithfulness (judge) : {_fmt(agg['faithfulness_judge'])}")
    print(f"  NLI-judge agreement  : {_fmt(agg['nli_judge_agreement'])}")
    print(f"  citation precision   : {_fmt(agg['cit_precision'])}")
    print(f"  citation recall      : {_fmt(agg['cit_recall'])}")
    print(f"  citation IoU         : {_fmt(agg['cit_iou'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Faithfulness + citation harness (Phase 3.1-3.2).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap number of eval questions")
    ap.add_argument("--large", action="store_true",
                    help="run generation on Mistral Large (final headline run)")
    args = ap.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    model = cfg["generation"]["model_final"] if args.large else cfg["generation"]["model_dev"]
    print(f"Running faithfulness harness (limit={args.limit}, model={model}) ...")
    results = evaluate(cfg, limit=args.limit, use_final_model=args.large)

    written = write_reports(results, cfg["paths"]["eval_reports_dir"], cfg["eval"]["report_formats"])
    _print_summary(results)
    print("\nReports written:")
    for p in written:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
