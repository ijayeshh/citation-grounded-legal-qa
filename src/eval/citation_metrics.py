"""Gold-span citation metrics (Phase 3.2) — the most objective number in the project.

Compares the system's cited character spans against CUAD's expert gold spans for a
clause type, on raw character offsets. This is only possible because offsets were
preserved end-to-end (chunk -> citation -> here); most projects never exploit CUAD's
gold spans this way.

Metrics are computed on the SET of character positions each set of spans covers
(spans merged into disjoint intervals first, so overlapping citations aren't
double-counted):

    precision = |pred ∩ gold| / |pred|      (how much of what we cited is gold)
    recall    = |pred ∩ gold| / |gold|      (how much of the gold we covered)
    IoU       = |pred ∩ gold| / |pred ∪ gold|
"""

from __future__ import annotations

Interval = tuple[int, int]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge a list of [start, end) intervals into sorted, disjoint intervals."""
    cleaned = [(int(a), int(b)) for a, b in intervals if b > a]
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for a, b in cleaned[1:]:
        la, lb = merged[-1]
        if a <= lb:  # overlapping or touching
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _total_len(intervals: list[Interval]) -> int:
    return sum(b - a for a, b in intervals)


def _intersection_len(a: list[Interval], b: list[Interval]) -> int:
    """Total overlap length between two lists of disjoint, sorted intervals."""
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def span_char_metrics(pred_spans: list[Interval], gold_spans: list[Interval]) -> dict:
    """Precision / recall / IoU on character offsets for one (contract, clause_type).

    Edge cases: empty pred and empty gold -> all 1.0 (nothing to find, nothing wrongly
    cited). Empty gold with non-empty pred -> precision 0.0. Empty pred with non-empty
    gold -> recall 0.0.
    """
    pred = merge_intervals(pred_spans)
    gold = merge_intervals(gold_spans)
    pred_len, gold_len = _total_len(pred), _total_len(gold)
    inter = _intersection_len(pred, gold)
    union = pred_len + gold_len - inter

    if pred_len == 0 and gold_len == 0:
        precision = recall = iou = 1.0
    else:
        precision = inter / pred_len if pred_len > 0 else 0.0
        recall = inter / gold_len if gold_len > 0 else 0.0
        iou = inter / union if union > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "pred_chars": pred_len,
        "gold_chars": gold_len,
        "overlap_chars": inter,
    }


def predicted_spans_for_contract(answer, contract_id: str) -> list[Interval]:
    """Extract cited character intervals from an Answer for one contract."""
    return [
        (c.citation.char_start, c.citation.char_end)
        for c in answer.claims
        if c.citation.contract_id == contract_id
    ]


def gold_spans_to_intervals(gold_span_dicts: list[dict]) -> list[Interval]:
    """Convert gold-span store entries ({start,end,text}) to [start, end) intervals."""
    return [(g["start"], g["end"]) for g in gold_span_dicts]
