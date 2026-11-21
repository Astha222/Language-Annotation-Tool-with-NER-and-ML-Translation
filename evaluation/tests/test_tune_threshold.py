"""Threshold tuning on saved runs: the sweep uses only the tuning files, and predictions below the threshold are ignored."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))
import tune_threshold as tt  # noqa: E402
from bootstrap import counts_per_sentence  # noqa: E402


def write(path, rows):
    path.write_text("".join(json.dumps(r) + chr(10) for r in rows), encoding="utf-8")


def test_low_scoring_wrong_predictions_are_removed_by_a_higher_threshold(tmp_path):
    # gold PER at 0-0; a correct prediction scored 0.8 and a wrong one scored 0.3
    rows = [{"gold": [["PER", 0, 0]], "pred": [["PER", 0, 0, 0.8], ["LOC", 2, 2, 0.3]]}] * 5
    write(tmp_path / "mc_de_tune__de__m.jsonl", rows)
    assert counts_per_sentence(tmp_path / "mc_de_tune__de__m.jsonl", 0.0).sum(axis=0).tolist() == [5, 5, 0]
    assert counts_per_sentence(tmp_path / "mc_de_tune__de__m.jsonl", 0.5).sum(axis=0).tolist() == [5, 0, 0]
    best, table, lowest = tt.choose(tmp_path, "m", ["mc_de_tune"])
    assert lowest == 0.3 and 0.3 < best <= 0.8 and dict(table)[best] == 1.0


def test_wikiann_test_names_find_their_files(tmp_path):
    write(tmp_path / "wikiann__fr__m__n3000.jsonl", [{"gold": [], "pred": []}])
    assert tt.run_file(tmp_path, "wikiann_fr", "m").endswith("wikiann__fr__m__n3000.jsonl")
