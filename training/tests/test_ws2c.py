"""Option B of the CoNLL question: the no-CoNLL run is defined like B_noFIN minus CoNLL, is never started by accident, and its notebook is consistent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import colab_runner  # noqa: E402
import make_ws2c_bundle as ws2c  # noqa: E402


def runs():
    return {name: (student, sources, alpha) for name, student, sources, alpha, _ in colab_runner.RUNS}


def test_no_conll_run_is_b_nofin_without_conll():
    r = runs()
    student, sources, alpha = r["B_noCoNLL_plain"]
    assert (student, alpha) == r["B_noFIN_plain"][0::2]              # same student, same "no teacher"
    assert set(sources.split(",")) == {"wnut2017", "btc"}
    assert set(r["B_noFIN_plain"][1].split(",")) - set(sources.split(",")) == {"conll2003"}


def test_later_runs_are_not_started_by_the_ws2_notebook(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["colab_runner.py", "--dry-run"])
    colab_runner.main()
    out = capsys.readouterr().out
    assert "##### A_distilroberta_distill" in out
    assert "noFIN" not in out and "noCoNLL" not in out


def test_named_run_prints_three_seeded_commands_with_the_right_sources(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["colab_runner.py", "--only", "B_noCoNLL_plain", "--seeds", "0,1,2", "--dev", "dev_mix_nofin.jsonl", "--dry-run"])
    colab_runner.main()
    out = capsys.readouterr().out
    for s in (0, 1, 2):
        assert f"##### B_noCoNLL_plain_s{s}" in out
    assert out.count("--sources wnut2017,btc") == 3 and out.count("--alpha 1.0") == 3 and "dev_mix_nofin.jsonl" in out


def test_notebook_trains_and_scores_the_same_three_models():
    src = "".join("".join(c["source"]) for c in ws2c.notebook()["cells"])
    assert "--only B_noCoNLL_plain --seeds 0,1,2" in src
    for s in (0, 1, 2):
        assert f"B_noCoNLL_plain_s{s}=out/B_noCoNLL_plain_s{s}/best" in src
    assert "conll2003" in src and "MyDrive/ws2c_results" in src
