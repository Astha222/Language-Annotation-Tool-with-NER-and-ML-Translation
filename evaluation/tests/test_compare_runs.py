"""compare_runs: sentence-level differences and the D6 gate (identical, or strict F1 within 0.01)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import compare_runs  # noqa: E402


def row(text, gold, pred):
    return {"text": text, "gold": gold, "pred": pred}


A = [row("Rahul in Pune", [["PER", 0, 1], ["LOC", 2, 3]], [["PER", 0, 1, 0.99], ["LOC", 2, 3, 0.90]]),
     row("Infosys news", [["ORG", 0, 1]], [["ORG", 0, 1, 0.80]])]


def test_identical_predictions_pass_even_if_scores_move_slightly():
    b = [row(r["text"], r["gold"], [p[:3] + [p[3] - 0.0001] for p in r["pred"]]) for r in A]
    r = compare_runs.compare(A, b)
    assert r["differing"] == 0 and r["delta"] == 0 and r["passes"] and r["max_score_diff"] == pytest.approx(0.0001)


def test_a_missing_entity_counts_as_a_differing_sentence_and_the_gate_uses_f1():
    b = [A[0], row("Infosys news", [["ORG", 0, 1]], [])]
    r = compare_runs.compare(A, b)
    assert r["differing"] == 1 and r["f1_a"] == 1.0 and r["delta"] < -0.01 and not r["passes"]


def test_misc_differences_are_counted_but_not_scored():
    b = [row(A[0]["text"], A[0]["gold"], A[0]["pred"] + [["MISC", 1, 2, 0.5]]), A[1]]
    r = compare_runs.compare(A, b)
    assert r["differing"] == 1 and r["delta"] == 0 and r["passes"]


def test_runs_over_different_sentences_are_refused():
    with pytest.raises(ValueError):
        compare_runs.compare(A, A[:1])
