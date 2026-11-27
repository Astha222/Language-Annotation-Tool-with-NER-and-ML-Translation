"""Language from the script (backend/script_lang.py) and its use in detect_language."""
import script_lang


def test_each_indian_script_gives_its_language():
    samples = {"or": "ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ ଅଟେ।", "pa": "ਅੰਮ੍ਰਿਤਸਰ ਪੰਜਾਬ ਵਿੱਚ ਹੈ।", "gu": "અમદાવાદ ગુજરાતમાં છે.",
               "ta": "சென்னை தமிழ்நாட்டின் தலைநகரம்.", "te": "హైదరాబాద్ తెలంగాణ రాజధాని.", "kn": "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.",
               "ml": "തിരുവനന്തപുരം കേരളത്തിന്റെ തലസ്ഥാനമാണ്.", "bn": "কলকাতা পশ্চিমবঙ্গের রাজধানী।"}
    for lang, text in samples.items():
        assert script_lang.script_language(text) == lang, lang


def test_shared_scripts():
    assert script_lang.script_language("दिल्ली भारत की राजधानी है") == "deva"
    assert script_lang.script_language("গুৱাহাটী অসমৰ আটাইতকৈ ডাঙৰ চহৰ") == "as"        # contains ৰ / ৱ
    assert script_lang.script_language("ঢাকা বাংলাদেশের রাজধানী") == "bn"


def test_latin_mixed_and_empty_text_are_left_to_the_caller():
    assert script_lang.script_language("The capital of India is New Delhi.") is None
    assert script_lang.script_language("Meeting with राहुल tomorrow at the Mumbai office about the budget") is None   # mostly Latin
    assert script_lang.script_language("") is None and script_lang.script_language("123 !!!") is None


def test_detect_language_uses_the_script_first(main_module):
    assert main_module.detect_language("ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ ଅଟେ।") == "or"           # langdetect failed on Odia before
    assert main_module.detect_language("গুৱাহাটী অসমৰ আটাইতকৈ ডাঙৰ চহৰ") == "as"
    assert main_module.detect_language("चेन्नई") in main_module.DEVANAGARI_LANGS                    # Devanagari: never a non-Devanagari language


def test_devanagari_falls_back_to_hindi_when_langdetect_says_something_else(main_module, monkeypatch):
    import langdetect
    monkeypatch.setattr(langdetect, "detect", lambda text: "id")
    assert main_module.detect_language("दिल्ली भारत की राजधानी है") == "hi"
    monkeypatch.setattr(langdetect, "detect", lambda text: "mr")
    assert main_module.detect_language("मुंबई महाराष्ट्राची राजधानी आहे") == "mr"
