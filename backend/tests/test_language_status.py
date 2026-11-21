"""Per-language quality labels: the agreed thresholds, the endpoint, and the generated table."""
import json

import pytest

import language_status as ls


# ── the rules ────────────────────────────────────────────────────────────
def test_stable_needs_every_condition():
    status, why = ls.ner_status(0.83, 0.81, 864, open_licence=True, native_test=True)
    assert status == "stable" and why == []


@pytest.mark.parametrize("args, expected_reason", [
    ((0.74, 0.72, 900, True, True), "below 0.75"),                  # F1 just under the bar
    ((0.80, 0.69, 900, True, True), "lower end of the 95% range"),   # good score but a wide range
    ((0.83, 0.81, 499, True, True), "only 499 test sentences"),
    ((0.83, 0.81, 864, True, False), "native speakers"),
    ((0.83, None, 864, True, True), "unknown"),
])
def test_missing_one_condition_gives_beta_and_says_which(args, expected_reason):
    status, why = ls.ner_status(*args)
    assert status == "beta" and any(expected_reason in w for w in why)


def test_below_half_is_experimental_even_with_everything_else():
    assert ls.ner_status(0.499, 0.45, 5000, True, True)[0] == "experimental"
    assert ls.ner_status(0.50, 0.45, 5000, True, True)[0] == "beta"                 # exactly 0.50 is beta


def test_no_open_licence_is_blocked_whatever_the_score():
    assert ls.ner_status(0.95, 0.93, 5000, open_licence=False, native_test=True)[0] == "blocked"


def test_no_result_is_not_measured():
    assert ls.ner_status(None, None, 0, True, False)[0] == "not_measured"


def test_thresholds_are_the_agreed_ones():
    assert ls.THRESHOLDS == {"stable_f1": 0.75, "stable_ci_low": 0.70, "stable_min_sentences": 500, "experimental_below": 0.50}


# ── loading and the endpoint ─────────────────────────────────────────────
def test_missing_file_gives_an_empty_table(monkeypatch, tmp_path):
    monkeypatch.setattr(ls, "STATUS_FILE", tmp_path / "nope.json")
    assert ls.load() == {"thresholds": ls.THRESHOLDS, "languages": []}


def test_damaged_file_gives_an_empty_table(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setattr(ls, "STATUS_FILE", bad)
    assert ls.load()["languages"] == []


def test_endpoint_serves_the_table_without_login(client, monkeypatch, tmp_path):
    table = {"thresholds": ls.THRESHOLDS, "languages": [{"code": "hi", "name": "Hindi", "ner": {"status": "beta"}},
                                                         {"code": "bn", "name": "Bengali", "ner": {"status": "beta"}}]}
    f = tmp_path / "t.json"
    f.write_text(json.dumps(table))
    monkeypatch.setattr(ls, "STATUS_FILE", f)
    r = client.get("/api/languages")
    body = r.json()
    assert r.status_code == 200 and body["thresholds"] == ls.THRESHOLDS
    by_code = {l["code"]: l for l in body["languages"]}
    assert by_code["hi"]["ner"] == {"status": "beta"}                       # the generated fields are passed through unchanged
    assert by_code["hi"]["offline_translation"] is True                     # IndicTrans2 handles Hindi both ways
    assert by_code["bn"]["offline_translation"] is False                    # Bengali needs Azure for now
    assert isinstance(body["azure_configured"], bool)


# ── the generated file that ships with the project ───────────────────────
def test_shipped_table_is_consistent_with_the_rules():
    table = json.loads(ls.STATUS_FILE.read_text(encoding="utf-8"))
    assert table["thresholds"] == ls.THRESHOLDS
    codes = [l["code"] for l in table["languages"]]
    assert codes[0] == "hi" and len(codes) == len(set(codes))                       # Hindi first, no duplicates
    for lang in table["languages"]:
        n = lang["ner"]
        if n["f1"] is not None:             # recomputing the label from the stored numbers must give the stored label (no native-speaker test exists yet)
            again, _ = ls.ner_status(n["f1"], n["ci"][0], n["sentences"], n["licence"] != "none declared", native_test=False)
            assert n["status"] == again, lang["code"]
