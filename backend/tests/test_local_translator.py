"""Offline translation: routing, segmentation and structure-keeping (with a fake engine), plus real-model tests."""
import pytest

import local_translator as lt


class FakeEngine:
    def __init__(self):
        self.batches = []

    def translate_batch(self, texts, beams):
        self.batches.append(list(texts))
        return ["T(" + t + ")" for t in texts]


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(lt, "_engine", lambda src, tgt: engine)
    return engine


# ── support matrix and routing ───────────────────────────────────────────
def test_supported_pairs():
    assert lt.supports("en", "hi") and lt.supports("hi", "en") and lt.supports("fr", "hi") and lt.supports("ja", "ru")
    assert not lt.supports("en", "en") and not lt.supports("en", "ko") and not lt.supports("xx", "en")


def test_pairs_without_english_are_pivoted_through_it(monkeypatch):
    calls = []
    monkeypatch.setattr(lt, "_translate_direct", lambda s, t, text, c: calls.append((s, t)) or f"{s}>{t}:{text}")
    assert lt.translate("fr", "hi", "Bonjour") == "en>hi:fr>en:Bonjour"
    assert calls == [("fr", "en"), ("en", "hi")]
    calls.clear()
    lt.translate("en", "fr", "Hello")
    assert calls == [("en", "fr")]                                   # a direct pair is not pivoted


def test_unsupported_pair_raises(fake_engine):
    with pytest.raises(lt.LocalTranslationError):
        lt.translate("en", "ko", "Hello")


def test_indictrans_models_are_pinned_to_exact_revisions():
    for repo, rev, src_tag, tgt_tag in lt.INDICTRANS.values():
        assert len(rev) == 40 and set(rev) <= set("0123456789abcdef")          # a commit hash, not a moving branch
        assert repo.startswith("prajdabre/rotary-indictrans2-")


def test_beam_width_setting_is_validated(monkeypatch):
    monkeypatch.setenv("LOCAL_MT_BEAMS", "4")
    assert lt._beams() == 4
    monkeypatch.setenv("LOCAL_MT_BEAMS", "banana")
    assert lt._beams() == 2
    monkeypatch.setenv("LOCAL_MT_BEAMS", "99")
    assert lt._beams() == 8


def test_indictrans_can_be_switched_off(monkeypatch):
    built = []

    class Spy:
        def __init__(self, kind, src, tgt):
            built.append(kind)

    monkeypatch.setattr(lt, "_Engine", Spy)
    lt._engines.clear()
    monkeypatch.setenv("LOCAL_INDICTRANS", "0")
    lt._engine("en", "hi")
    lt._engines.clear()
    monkeypatch.setenv("LOCAL_INDICTRANS", "1")
    lt._engine("en", "hi")
    lt._engines.clear()
    assert built == ["marian", "indictrans"]


def test_loaded_models_are_limited_to_save_memory(monkeypatch):
    class Dummy:
        def __init__(self, kind, src, tgt):
            pass

    monkeypatch.setattr(lt, "_Engine", Dummy)
    monkeypatch.setenv("LOCAL_MT_MAX_LOADED", "2")
    lt._engines.clear()
    for pair in (("en", "fr"), ("fr", "en"), ("en", "es")):
        lt._engine(*pair)
    assert list(lt._engines) == [("fr", "en"), ("en", "es")]           # least recently used one was dropped
    lt._engines.clear()


# ── segmentation ─────────────────────────────────────────────────────────
def test_long_lines_split_at_sentence_ends():
    line = ("This is sentence one. " * 80).strip()
    parts = lt._split_line(line, 300)
    assert all(len(p) <= 300 for p in parts) and " ".join(parts) == line


def test_one_enormous_sentence_is_hard_split():
    assert all(len(p) <= 100 for p in lt._split_line("word " * 200, 100))


# ── structure is preserved ───────────────────────────────────────────────
def test_line_structure_is_kept(fake_engine):
    out = lt.translate("en", "hi", "Heading\n\n● First bullet\n---\nLast line.")
    assert out.split("\n") == ["T(Heading)", "", "T(● First bullet)", "---", "T(Last line.)"]


def test_symbol_only_lines_are_never_sent_to_the_model(fake_engine):
    lt.translate("en", "hi", "●\n\n---\n123 ...\nText")
    sent = [t for batch in fake_engine.batches for t in batch]
    assert "●" not in sent and "---" not in sent and "Text" in sent


def test_lines_are_batched_and_sorted_by_length(fake_engine):
    text = "\n".join("word " * n for n in (30, 2, 15, 1, 8, 4, 22, 3, 9, 5))
    lt.translate("en", "hi", text)
    assert [len(b) for b in fake_engine.batches] == [8, 2]
    assert [len(t) for t in fake_engine.batches[0]] == sorted(len(t) for t in fake_engine.batches[0])


def test_translation_can_be_cancelled_between_batches(fake_engine):
    with pytest.raises(lt.Cancelled):
        lt.translate("en", "hi", "\n".join(f"line {i}" for i in range(40)), cancelled=lambda: True)
    assert fake_engine.batches == []


def test_long_line_pieces_are_reassembled_in_order(fake_engine):
    line = " ".join(f"Sentence {i} is here." for i in range(120))
    out = lt.translate("en", "hi", line)
    assert out.count("T(") == len(fake_engine.batches[0]) + sum(len(b) for b in fake_engine.batches[1:])
    assert out.index("Sentence 0") < out.index("Sentence 119")


# ── real models (slow; `pytest --run-slow`) ──────────────────────────────
@pytest.mark.slow
def test_real_hindi_round_trip_keeps_names():
    en = "Sundar Pichai, the CEO of Google, met Narendra Modi in New Delhi last Tuesday."
    hi = lt.translate("en", "hi", en)
    assert sum("ऀ" <= c <= "ॿ" for c in hi) > 20
    back = lt.translate("hi", "en", hi)
    assert "Pichai" in back and "Modi" in back


@pytest.mark.slow
def test_real_french_via_marian():
    fr = lt.translate("en", "fr", "The meeting discussed artificial intelligence and jobs.")
    assert "intelligence artificielle" in fr


# ── Marian configuration: target-language tokens and the English -> Japanese model ───────────────────
def _marian_engine(prefix):
    """An engine object that skips model loading; records what the tokenizer is given."""
    import contextlib
    import types

    seen = {}

    class Tok:
        def __call__(self, texts, **kw):
            seen["texts"] = list(texts)
            return {"input_ids": list(texts)}

        def batch_decode(self, out, skip_special_tokens=True):
            return list(out)

    class Model:
        def generate(self, **kw):
            return kw["input_ids"]

    eng = object.__new__(lt._Engine)
    eng.kind, eng.tok, eng.model, eng.prefix = "marian", Tok(), Model(), prefix
    eng.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    return eng, seen


def test_every_multi_target_marian_model_has_a_target_token():
    # the model cards of opus-mt-en-zh and opus-mt-en-ar say a sentence-initial >>id<< token is required
    assert set(lt._TARGET_TOKEN) >= {("en", "zh"), ("en", "ar")}
    for token in lt._TARGET_TOKEN.values():
        assert token.startswith(">>") and token.endswith("<<")
    assert lt._TARGET_TOKEN[("en", "zh")] == ">>cmn_Hans<<"          # Simplified Chinese, as the WMT24++ references
    assert lt._TARGET_TOKEN[("en", "ar")] == ">>ara<<"


def test_marian_engine_prepends_the_target_token():
    eng, seen = _marian_engine(">>cmn_Hans<<")
    assert eng.translate_batch(["Hello", "Good night"], beams=1) == [">>cmn_Hans<< Hello", ">>cmn_Hans<< Good night"]
    assert seen["texts"] == [">>cmn_Hans<< Hello", ">>cmn_Hans<< Good night"]


def test_marian_engine_without_a_token_leaves_the_text_alone():
    eng, seen = _marian_engine("")
    eng.translate_batch(["Hello"], beams=1)
    assert seen["texts"] == ["Hello"]


def test_english_to_japanese_does_not_use_the_bible_only_jap_model():
    # opus-mt-en-jap is for ISO 639-3 "jap" (not Japanese) and was only evaluated on Bible verses
    assert lt._MARIAN_TARGETS["ja"] != "jap"
    assert lt._MARIAN_MODEL_OVERRIDE[("en", "ja")] == "Helsinki-NLP/opus-tatoeba-en-ja"
