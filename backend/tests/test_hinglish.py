"""Romanized-Hindi detector (backend/hinglish.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hinglish  # noqa: E402


def test_typical_hinglish_is_detected():
    assert hinglish.is_romanized_hindi("Rahul kal Mumbai ja raha hai meeting ke liye .")
    assert hinglish.is_romanized_hindi("kya tum kal office aa rahe ho ?")


def test_english_and_other_latin_languages_are_not_detected():
    for text in ["The president met the prime minister in Paris on Monday.",
                 "I will meet you at the main office tomorrow, so do not be late.",
                 "Il ne sait pas ce que je pense de la situation à Paris.",
                 "Mein Hund geht heute mit mir in den Park.",
                 "No me gusta la comida de este restaurante."]:
        assert not hinglish.is_romanized_hindi(text), text


def test_words_that_are_also_ordinary_english_are_not_markers():
    for w in ["to", "the", "me", "main", "par", "do", "in", "so", "is", "hi"]:
        assert w not in hinglish.MARKERS


def test_devanagari_and_empty_text_are_left_to_langdetect():
    assert not hinglish.is_romanized_hindi("राहुल कल मुंबई जा रहा है")
    assert not hinglish.is_romanized_hindi("")
    assert not hinglish.is_romanized_hindi("12345 !!!")


def test_one_marker_is_not_enough_and_the_share_matters():
    assert not hinglish.is_romanized_hindi("My friend said bhai is a good word to use here in the office today")
    assert hinglish.hinglish_score("aaj ka din bahut achha tha") == (5, 6, 5 / 6)
