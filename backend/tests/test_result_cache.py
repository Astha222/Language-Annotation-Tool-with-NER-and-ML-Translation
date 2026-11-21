"""The result cache: correct, never stale, never able to break a request."""
import sqlite3

import pytest

import result_cache


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "cache" / "results.sqlite"
    monkeypatch.setenv("RESULT_CACHE_PATH", str(path))
    monkeypatch.setattr(result_cache, "_warned", False)
    return path


# ── the cache itself ─────────────────────────────────────────────────────
def test_round_trip_and_miss(cache_file):
    key = result_cache.make_key("ner", "model", "en", "Rahul works in Pune.")
    assert result_cache.get_json(key) is None
    result_cache.put_json(key, [{"word": "Rahul", "start": 0, "end": 5}])
    assert result_cache.get_json(key) == [{"word": "Rahul", "start": 0, "end": 5}]
    assert cache_file.exists()


def test_every_part_of_the_key_matters():
    base = result_cache.make_key("ner", "model-a", "en", "text")
    assert base == result_cache.make_key("ner", "model-a", "en", "text")
    for changed in (("ner", "model-b", "en", "text"), ("ner", "model-a", "fr", "text"), ("ner", "model-a", "en", "text "),
                    ("mt", "model-a", "en", "text")):
        assert result_cache.make_key(*changed) != base


def test_unicode_text_is_fine(cache_file):
    key = result_cache.make_key("mt", "local", "en", "hi", "Rahul Delhi mein rehta hai.")
    result_cache.put_json(key, {"text": "राहुल दिल्ली में रहता है।"})
    assert result_cache.get_json(key)["text"] == "राहुल दिल्ली में रहता है।"


def test_off_means_no_file_and_no_hits(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULT_CACHE_PATH", "off")
    key = result_cache.make_key("x")
    result_cache.put_json(key, [1])
    assert result_cache.get_json(key) is None
    assert not list(tmp_path.iterdir())


def test_a_corrupt_file_is_a_miss_not_an_error(cache_file):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"this is not a sqlite database" * 50)
    key = result_cache.make_key("x")
    assert result_cache.get_json(key) is None                  # does not raise
    result_cache.put_json(key, [1])                            # does not raise either


def test_a_damaged_value_is_a_miss(cache_file):
    key = result_cache.make_key("x")
    result_cache.put(key, "{not json")
    assert result_cache.get_json(key) is None


def test_old_entries_are_dropped_when_the_cache_is_full(cache_file, monkeypatch):
    monkeypatch.setenv("RESULT_CACHE_MAX_ROWS", "100")
    for i in range(150):
        result_cache.put_json(result_cache.make_key("n", i), [i])
    with sqlite3.connect(cache_file) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
    assert count <= 100
    assert result_cache.get_json(result_cache.make_key("n", 149)) == [149]        # the newest survives


# ── NER ──────────────────────────────────────────────────────────────────
class CountingNer:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kw):
        self.calls += 1
        out = []
        for t in texts:
            i = t.find("Rahul")
            out.append([{"entity_group": "PER", "score": 0.99, "word": "Rahul", "start": i, "end": i + 5}] if i >= 0 else [])
        return out


def test_ner_second_run_comes_from_the_cache(main_module, monkeypatch, cache_file):
    ner = CountingNer()
    monkeypatch.setattr(main_module, "get_ner_pipeline", lambda lang="en": ner)
    first = main_module.run_ner("Rahul works in Pune.", "en")
    second = main_module.run_ner("Rahul works in Pune.", "en")
    assert first == second and first[0]["word"] == "Rahul" and ner.calls == 1


def test_cached_entities_are_shifted_to_the_new_position(main_module, monkeypatch, cache_file):
    ner = CountingNer()
    monkeypatch.setattr(main_module, "get_ner_pipeline", lambda lang="en": ner)
    sentence = "Rahul works in Pune."
    main_module.run_ner(sentence, "en")
    text = ("A long first sentence that fills the line with ordinary words only. " * 12) + sentence
    entities = main_module.run_ner(text, "en")
    assert [text[e["start"]:e["end"]] for e in entities] == ["Rahul"]
    assert entities[0]["start"] == text.index("Rahul")


def test_ner_cache_does_not_cross_languages_or_settings(main_module, monkeypatch, cache_file):
    ner = CountingNer()
    monkeypatch.setattr(main_module, "get_ner_pipeline", lambda lang="en": ner)
    main_module.run_ner("Rahul works in Pune.", "en")
    main_module.run_ner("Rahul works in Pune.", "fr")
    monkeypatch.setenv("NER_STRIDE_TOKENS", "64")
    main_module.run_ner("Rahul works in Pune.", "en")
    assert ner.calls == 3


# ── translation ──────────────────────────────────────────────────────────
def test_translation_second_run_comes_from_the_cache(main_module, providers, cache_file):
    first = main_module.perform_translation("en", "fr", "Hello world.")
    second = main_module.perform_translation("en", "fr", "Hello world.")
    assert first == second == "LOC:Hello world." and len(providers["local"]) == 1


def test_translation_cache_follows_language_pair_and_beam_setting(main_module, providers, cache_file, monkeypatch):
    main_module.perform_translation("en", "fr", "Hello world.")
    main_module.perform_translation("en", "de", "Hello world.")
    monkeypatch.setenv("LOCAL_MT_BEAMS", "4")
    main_module.perform_translation("en", "fr", "Hello world.")
    assert len(providers["local"]) == 3


def test_a_fallback_translation_is_not_cached(main_module, providers, azure_on, cache_file, monkeypatch):
    def azure_down(src, tgt, text, *a, **k):
        providers["azure"].append(text)
        return None
    monkeypatch.setattr(main_module.azure_translator, "translate", azure_down)
    used = []
    assert main_module.perform_translation("en", "fr", "Hello world.", used) == "LOC:Hello world." and used == ["local"]
    monkeypatch.setattr(main_module.azure_translator, "translate", lambda s, t, text, *a, **k: "AZ:" + text)
    assert main_module.perform_translation("en", "fr", "Hello world.") == "AZ:Hello world."      # Azure is back: not the stale local one
