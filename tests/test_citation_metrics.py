"""Gold-span citation metric tests (Phase 3.2). Pure, offline."""

from src.eval.citation_metrics import (
    merge_intervals,
    span_char_metrics,
)


def test_merge_intervals():
    assert merge_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
    assert merge_intervals([(5, 5), (2, 4)]) == [(2, 4)]  # empty span dropped
    assert merge_intervals([]) == []


def test_perfect_overlap():
    m = span_char_metrics([(10, 20)], [(10, 20)])
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["iou"] == 1.0


def test_partial_overlap():
    # pred covers [0,20), gold [10,30): overlap 10, pred 20, gold 20, union 30
    m = span_char_metrics([(0, 20)], [(10, 30)])
    assert m["overlap_chars"] == 10
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert round(m["iou"], 4) == round(10 / 30, 4)


def test_pred_superset_of_gold():
    # Whole-contract citation vs a small gold clause: high recall, low precision.
    m = span_char_metrics([(0, 100)], [(40, 60)])
    assert m["recall"] == 1.0
    assert m["precision"] == 0.2
    assert m["iou"] == 0.2


def test_disjoint_spans():
    m = span_char_metrics([(0, 10)], [(50, 60)])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["iou"] == 0.0


def test_empty_pred_nonempty_gold():
    m = span_char_metrics([], [(0, 10)])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["iou"] == 0.0


def test_empty_both_is_perfect():
    m = span_char_metrics([], [])
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["iou"] == 1.0


def test_overlapping_predictions_not_double_counted():
    # Two overlapping cited spans should merge before scoring.
    m = span_char_metrics([(0, 15), (10, 20)], [(0, 20)])
    assert m["pred_chars"] == 20  # merged, not 25
    assert m["precision"] == 1.0 and m["recall"] == 1.0
