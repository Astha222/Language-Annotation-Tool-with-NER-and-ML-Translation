"""The Hinglish test-set builder: markup -> words and BIO tags."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))
import make_hinglish_set as m  # noqa: E402


def test_markup_becomes_words_and_bio_tags():
    words, tags = m.parse_line("[Virat Kohli|PER] ne [Delhi|LOC] mein match khela .")
    assert words == ["Virat", "Kohli", "ne", "Delhi", "mein", "match", "khela", "."]
    assert tags == ["B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "O"]


def test_plain_sentences_have_no_entities_and_broken_markup_is_rejected():
    assert set(m.parse_line("aaj ka din bahut achha tha")[1]) == {"O"}
    try:
        m.parse_line("[Rahul|XYZ] ja raha hai")
    except ValueError:
        return
    raise AssertionError("unknown entity type should be rejected")


def test_the_real_file_parses_and_every_entity_type_occurs():
    rows = m.parse_file()
    assert len(rows) >= 100
    tags = {t for r in rows for t in r["ner"]}
    assert {"B-PER", "B-ORG", "B-LOC", "I-PER", "O"} <= tags
    assert all(len(r["words"]) == len(r["ner"]) for r in rows)
