"""Converters for the Russian / Arabic / Japanese / German / Chinese test sets (no downloads)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))
import foreign_data as fd  # noqa: E402


def test_iob2_comments_are_skipped_sentences_split_on_blank_lines_and_oth_dropped():
    text = "# id = 1\n# text = x\n1\tKori\tB-PER\t-\t-\n2\tSchulman\tI-PER\t-\t-\n3\tsagte\tO\t-\t-\n\n# id = 2\n1\tGoogle\tB-ORG\t-\n2\tWindows\tB-OTH\t-\n"
    rows = fd.parse_iob2(text)
    assert rows == [{"words": ["Kori", "Schulman", "sagte"], "ner": ["B-PER", "I-PER", "O"]},
                    {"words": ["Google", "Windows"], "ner": ["B-ORG", "O"]}]


def test_anercorp_pers_becomes_per_and_misc_is_dropped():
    assert fd.map_tags(["B-PERS", "I-PERS", "B-MISC", "O", "B-LOC"], fd.ANER_TYPES) == ["B-PER", "I-PER", "O", "O", "B-LOC"]


def test_japanese_character_tags_from_spans_with_the_documented_type_mapping():
    row = fd.char_bio("山田は東京駅へ", [{"name": "山田", "span": [0, 2], "type": "人名"},
                                       {"name": "東京駅", "span": [3, 6], "type": "施設名"},
                                       {"name": "東京", "span": [3, 5], "type": "地名"}])       # overlaps the facility: skipped
    assert row["words"] == list("山田は東京駅へ")
    assert row["ner"] == ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O"]


def test_products_and_events_are_not_scored():
    assert set(fd.char_bio("ABC", [{"name": "ABC", "span": [0, 3], "type": "製品名"}])["ner"]) == {"O"}


def test_prediction_label_aliases_map_pers_to_per():
    import eval_ner
    assert eval_ner.LABEL_ALIASES["PERS"] == "PER"
