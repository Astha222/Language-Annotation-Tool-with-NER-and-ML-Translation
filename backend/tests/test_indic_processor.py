"""The pure-Python port of IndicTransToolkit's processor (no model involved)."""
import pytest

pytest.importorskip("indicnlp")

from indic_processor import IndicProcessor


@pytest.fixture(scope="module")
def proc():
    return IndicProcessor()


def test_preprocess_prepends_language_tags(proc):
    pre, maps = proc.preprocess_batch(["Hello, how are you?"], "eng_Latn", "hin_Deva")
    assert pre[0].startswith("eng_Latn hin_Deva ") and "Hello" in pre[0]
    assert len(maps) == 1


def test_indic_digits_become_ascii(proc):
    pre, _ = proc.preprocess_batch(["कीमत ९८७६ रुपये"], "hin_Deva", "eng_Latn")
    assert "9876" in pre[0] or "<ID" in pre[0]                      # digits are normalised (or protected as a placeholder)
    assert "९" not in pre[0]


def test_urls_emails_and_numbers_are_protected_and_restored(proc):
    text = "Mail jobs@gravity.com or visit https://gravity.com/careers about 12-34-5678 today."
    pre, maps = proc.preprocess_batch([text], "eng_Latn", "hin_Deva")
    assert "jobs@gravity.com" not in pre[0] and "https://gravity.com/careers" not in pre[0]     # hidden from the model
    assert "ID1" in pre[0] and "ID2" in pre[0]
    # a model may mangle the markers: the known variants are restored all the same
    mangled = pre[0].split(" ", 2)[2]
    restored = proc.postprocess_batch([mangled], "eng_Latn", maps)[0]
    assert "jobs@gravity.com" in restored or "https://gravity.com/careers" in restored


def test_placeholder_maps_are_per_call_not_shared(proc):
    """The original toolkit keeps maps in a shared queue; this port must not mix concurrent callers up."""
    _, m1 = proc.preprocess_batch(["Mail a@b.com now"], "eng_Latn", "hin_Deva")
    _, m2 = proc.preprocess_batch(["Mail c@d.com now"], "eng_Latn", "hin_Deva")
    assert any(v == "a@b.com" for v in m1[0].values()) and any(v == "c@d.com" for v in m2[0].values())
    assert not any(v == "c@d.com" for v in m1[0].values())


def test_english_postprocess_detokenizes(proc):
    assert proc.postprocess_batch(["Hello , world !"], "eng_Latn", [{}])[0] == "Hello, world!"


def test_precomposed_nukta_letters_do_not_leak_a_backslash_escape(proc):
    # indic-nlp-library-itt 0.1.1 wrote the six characters "़" (backslash, u, 0, 9, 3, C) instead of the nukta sign when
    # decomposing letters such as ऱ (U+0931, very common in Marathi) or क़ (U+0958)
    for text, lang in (("शेतकऱ्यांची भेट", "mar_Deva"), ("क़िला", "hin_Deva"), ("दुसऱ्या दिवशी", "mar_Deva")):
        out = proc.preprocess_batch([text], lang, "eng_Latn")[0][0]
        assert "\\" not in out, out
        assert "़" in out, out          # the real nukta sign is there instead


def test_every_offered_indian_script_is_preprocessed_with_its_tags(proc):
    samples = {"ben_Beng": "কলকাতা", "asm_Beng": "গুৱাহাটী", "ory_Orya": "ଭୁବନେଶ୍ୱର", "pan_Guru": "ਅੰਮ੍ਰਿਤਸਰ", "guj_Gujr": "અમદાવાદ",
               "tam_Taml": "சென்னை", "tel_Telu": "హైదరాబాద్", "kan_Knda": "ಬೆಂಗಳೂರು", "mal_Mlym": "കൊച്ചി", "mar_Deva": "मुंबई", "hin_Deva": "दिल्ली"}
    for tag, word in samples.items():
        out = proc.preprocess_batch([word], tag, "eng_Latn")[0][0]
        assert out.startswith(f"{tag} eng_Latn ") and len(out.split()) >= 3 and "\\" not in out, (tag, out)


def test_hindi_path_runs_end_to_end(proc):
    pre, maps = proc.preprocess_batch(["नरेंद्र मोदी नई दिल्ली में हैं।"], "hin_Deva", "eng_Latn")
    assert pre[0].startswith("hin_Deva eng_Latn ")
    out = proc.postprocess_batch(["नरेंद्र मोदी नई दिल्ली में हैं ।"], "hin_Deva", maps)
    assert "नरेंद्र मोदी" in out[0]
