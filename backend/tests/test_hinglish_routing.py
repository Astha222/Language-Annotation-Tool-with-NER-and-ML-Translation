"""Hindi typed in English letters: language detection, NER routing and the Azure source code (hi-Latn)."""
import pytest

HINGLISH = "Rahul kal Mumbai ja raha hai meeting ke liye ."
DEVANAGARI = "राहुल कल मुंबई जा रहा है"


def test_detect_language_reports_hinglish_as_hindi_and_leaves_other_text_alone(main_module):
    assert main_module.detect_language(HINGLISH) == "hi"
    assert main_module.detect_language("The president met the prime minister in Paris on Monday.") == "en"
    assert main_module.detect_language(DEVANAGARI) == "hi"


def test_detect_endpoint_says_whether_the_text_was_romanized(client, auth):
    r = client.post("/api/detect-language", json={"text": HINGLISH}, headers=auth).json()
    assert r == {"detected_language": "hi", "romanized": True}
    r = client.post("/api/detect-language", json={"text": DEVANAGARI}, headers=auth).json()
    assert r["detected_language"] == "hi" and r["romanized"] is False


def test_ner_on_hinglish_uses_the_hindi_model(client, auth, main_module, monkeypatch):
    used = []
    monkeypatch.setattr(main_module, "_ner_pipelines", {main_module.NER_MODEL_HI: lambda b, batch_size=8: used.append("hi") or [[] for _ in b],
                                                        main_module.NER_MODEL_MULTI: lambda b, batch_size=8: used.append("multi") or [[] for _ in b],
                                                        main_module.NER_MODEL_EN: lambda b, batch_size=8: used.append("en") or [[] for _ in b]})
    r = client.post("/api/ner", json={"text": HINGLISH, "lang": "auto"}, headers=auth)
    assert r.status_code == 200 and used == ["hi"]


def test_azure_gets_hi_latn_for_romanized_hindi_and_hi_for_devanagari(main_module, providers, azure_on):
    main_module.perform_translation("hi", "en", HINGLISH)
    main_module.perform_translation("hi", "en", DEVANAGARI)
    assert [c[0] for c in providers["azure"]] == ["hi-Latn", "hi"]


def test_offline_and_mymemory_are_not_used_for_romanized_hindi(main_module, providers, monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROVIDERS", "local,mymemory")
    monkeypatch.setattr(main_module, "_provider_order", lambda: ["local", "mymemory"])
    with pytest.raises(Exception):
        main_module.perform_translation("hi", "en", HINGLISH)
    assert providers["local"] == [] and providers["mymemory"] == []


def test_devanagari_hindi_still_uses_the_offline_model(main_module, providers, monkeypatch):
    monkeypatch.setattr(main_module, "_provider_order", lambda: ["local"])
    assert main_module.perform_translation("hi", "en", DEVANAGARI).startswith("LOC:")
