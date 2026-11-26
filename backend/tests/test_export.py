"""Unit tests for training format exports: CoNLL / BIO and JSONL.

Verifies:
1. Known sentence: "Rahul works at Infosys in Pune ." produces the exact expected BIO tags.
2. JSONL export structure: {"text":"...","entities":[[start,end,"PER"],[start,end,"ORG"]],"lang":"..."}
3. English annotation export and Hindi annotation export correspond to stored entities.
4. Sentence splitting with blank lines between sentences.
5. Character-level tokenization for Chinese (zh) and Japanese (ja).
6. Multi-token entity BIO labeling (B-TYPE followed by I-TYPE).
7. FastAPI endpoint GET /api/export?format=conll and GET /api/export?format=jsonl.
"""

import json
import pytest

from training_export import (
    tokenize_for_conll,
    annotation_to_conll,
    export_conll,
    annotation_to_jsonl_dict,
    export_jsonl,
)


class TestCoNLLExport:
    def test_known_sentence_bio_labels(self):
        """Verify the exact known benchmark sentence from task specification:

        Input: 'Rahul works at Infosys in Pune .'
        Expected:
        Rahul    B-PER
        works    O
        at       O
        Infosys  B-ORG
        in       O
        Pune     B-LOC
        .        O
        """
        text = "Rahul works at Infosys in Pune ."
        entities = [
            {"entity_group": "PER", "start": 0, "end": 5},
            {"entity_group": "ORG", "start": 15, "end": 22},
            {"entity_group": "LOC", "start": 26, "end": 30},
        ]
        conll_out = annotation_to_conll(text, entities, lang="en")
        lines = conll_out.strip().split("\n")

        expected_pairs = [
            ("Rahul", "B-PER"),
            ("works", "O"),
            ("at", "O"),
            ("Infosys", "B-ORG"),
            ("in", "O"),
            ("Pune", "B-LOC"),
            (".", "O"),
        ]

        actual_pairs = [tuple(line.split("\t")) for line in lines]
        assert actual_pairs == expected_pairs

    def test_multi_token_entities_bio_tags(self):
        """A token beginning an entity gets B-TYPE, subsequent tokens get I-TYPE."""
        text = "Sundar Pichai leads Google Cloud in New York City ."
        entities = [
            {"entity_group": "PER", "start": 0, "end": 13},       # Sundar Pichai
            {"entity_group": "ORG", "start": 20, "end": 32},      # Google Cloud
            {"entity_group": "LOC", "start": 36, "end": 49},      # New York City
        ]
        conll_out = annotation_to_conll(text, entities, lang="en")
        lines = [line.split("\t") for line in conll_out.strip().split("\n")]

        tag_map = {tok: tag for tok, tag in lines}
        assert tag_map["Sundar"] == "B-PER"
        assert tag_map["Pichai"] == "I-PER"
        assert tag_map["leads"] == "O"
        assert tag_map["Google"] == "B-ORG"
        assert tag_map["Cloud"] == "I-ORG"
        assert tag_map["New"] == "B-LOC"
        assert tag_map["York"] == "I-LOC"
        assert tag_map["City"] == "I-LOC"

    def test_blank_line_between_sentences(self):
        """Sentences must be separated by a blank line."""
        text = "Rahul works at Infosys. He lives in Pune."
        entities = [
            {"entity_group": "PER", "start": 0, "end": 5},
            {"entity_group": "ORG", "start": 15, "end": 22},
            {"entity_group": "LOC", "start": 36, "end": 40},
        ]
        conll_out = annotation_to_conll(text, entities, lang="en")
        # There should be an empty line separating the sentences
        blocks = conll_out.split("\n\n")
        assert len(blocks) == 2, f"Expected 2 sentence blocks separated by blank line, got: {repr(conll_out)}"

        sent1_lines = blocks[0].split("\n")
        assert sent1_lines[0] == "Rahul\tB-PER"
        assert sent1_lines[-1] == ".\tO"

        sent2_lines = blocks[1].split("\n")
        assert sent2_lines[0] == "He\tO"
        assert sent2_lines[-2] == "Pune\tB-LOC"
        assert sent2_lines[-1] == ".\tO"

    def test_chinese_japanese_one_char_per_token(self):
        """For Chinese and Japanese, use one character per token."""
        zh_text = "我爱北京天安门。"
        zh_entities = [
            {"entity_group": "LOC", "start": 2, "end": 4},   # 北京
            {"entity_group": "LOC", "start": 4, "end": 7},   # 天安门
        ]
        zh_out = annotation_to_conll(zh_text, zh_entities, lang="zh")
        lines = [tuple(l.split("\t")) for l in zh_out.strip().split("\n")]
        expected = [
            ("我", "O"),
            ("爱", "O"),
            ("北", "B-LOC"),
            ("京", "I-LOC"),
            ("天", "B-LOC"),
            ("安", "I-LOC"),
            ("门", "I-LOC"),
            ("。", "O"),
        ]
        assert lines == expected

        ja_text = "田中さんは東京に行く。"
        ja_entities = [
            {"entity_group": "PER", "start": 0, "end": 2},   # 田中
            {"entity_group": "LOC", "start": 5, "end": 7},   # 東京
        ]
        ja_out = annotation_to_conll(ja_text, ja_entities, lang="ja")
        ja_lines = [tuple(l.split("\t")) for l in ja_out.strip().split("\n")]
        assert ja_lines[0] == ("田", "B-PER")
        assert ja_lines[1] == ("中", "I-PER")
        assert ja_lines[5] == ("東", "B-LOC")
        assert ja_lines[6] == ("京", "I-LOC")

    def test_hindi_annotation_conll(self):
        """Hindi annotation export with Devanagari script and danda punctuation."""
        hi_text = "राहुल शर्मा इंफोसिस में काम करता है।"
        p_rahul = hi_text.index("राहुल शर्मा")
        p_infosys = hi_text.index("इंफोसिस")
        hi_entities = [
            {"entity_group": "PER", "start": p_rahul, "end": p_rahul + len("राहुल शर्मा")},
            {"entity_group": "ORG", "start": p_infosys, "end": p_infosys + len("इंफोसिस")},
        ]
        hi_out = annotation_to_conll(hi_text, hi_entities, lang="hi")
        lines = [tuple(l.split("\t")) for l in hi_out.strip().split("\n")]
        assert lines[0] == ("राहुल", "B-PER")
        assert lines[1] == ("शर्मा", "I-PER")
        assert lines[2] == ("इंफोसिस", "B-ORG")
        assert lines[3] == ("में", "O")
        assert lines[-1] == ("।", "O")


class TestJSONLExport:
    def test_known_sentence_jsonl(self):
        """Test JSONL output for the known sentence."""
        text = "Rahul works at Infosys in Pune ."
        entities = [
            {"entity_group": "PER", "start": 0, "end": 5},
            {"entity_group": "ORG", "start": 15, "end": 22},
            {"entity_group": "LOC", "start": 26, "end": 30},
        ]
        obj = annotation_to_jsonl_dict(text, entities, lang="en")
        assert obj == {
            "text": "Rahul works at Infosys in Pune .",
            "entities": [[0, 5, "PER"], [15, 22, "ORG"], [26, 30, "LOC"]],
            "lang": "en",
        }

        # Check serialized string
        raw_jsonl = export_jsonl([{"original_text": text, "ner_tags": entities, "source_lang": "en"}])
        parsed = json.loads(raw_jsonl.strip())
        assert parsed == obj

    def test_hindi_annotation_jsonl(self):
        """Test JSONL output for Hindi annotation."""
        hi_text = "राहुल शर्मा इंफोसिस में काम करता है।"
        p_rahul = hi_text.index("राहुल शर्मा")
        p_infosys = hi_text.index("इंफोसिस")
        hi_entities = [
            {"entity_group": "PER", "start": p_rahul, "end": p_rahul + len("राहुल शर्मा")},
            {"entity_group": "ORG", "start": p_infosys, "end": p_infosys + len("इंफोसिस")},
        ]
        obj = annotation_to_jsonl_dict(hi_text, hi_entities, lang="hi")
        assert obj["text"] == hi_text
        assert obj["lang"] == "hi"
        assert obj["entities"] == [
            [p_rahul, p_rahul + len("राहुल शर्मा"), "PER"],
            [p_infosys, p_infosys + len("इंफोसिस"), "ORG"],
        ]
        # Verify offsets slice the exact words
        for start, end, label in obj["entities"]:
            word = hi_text[start:end]
            assert word in ("राहुल शर्मा", "इंफोसिस")


class TestExportAPIEndpoints:
    def test_export_conll_and_jsonl_api(self, client, auth):
        """Verify GET /api/export?format=conll and GET /api/export?format=jsonl endpoints."""
        # 1. Create an English annotation
        en_text = "Rahul works at Infosys in Pune ."
        en_entities = [
            {"entity_group": "PER", "start": 0, "end": 5, "word": "Rahul"},
            {"entity_group": "ORG", "start": 15, "end": 22, "word": "Infosys"},
            {"entity_group": "LOC", "start": 26, "end": 30, "word": "Pune"},
        ]
        client.post(
            "/api/annotations",
            json={
                "original_text": en_text,
                "translated_text": "Rahul travaille chez Infosys à Pune.",
                "ner_tags": json.dumps(en_entities),
                "source_lang": "en",
                "target_lang": "fr",
            },
            headers=auth,
        )

        # 2. Create a Hindi annotation
        hi_text = "राहुल शर्मा इंफोसिस में काम करता है।"
        p_rahul = hi_text.index("राहुल शर्मा")
        p_infosys = hi_text.index("इंफोसिस")
        hi_entities = [
            {"entity_group": "PER", "start": p_rahul, "end": p_rahul + len("राहुल शर्मा"), "word": "राहुल शर्मा"},
            {"entity_group": "ORG", "start": p_infosys, "end": p_infosys + len("इंफोसिस"), "word": "इंफोसिस"},
        ]
        client.post(
            "/api/annotations",
            json={
                "original_text": hi_text,
                "translated_text": "Rahul Sharma works in Infosys.",
                "ner_tags": json.dumps(hi_entities),
                "source_lang": "hi",
                "target_lang": "en",
            },
            headers=auth,
        )

        # 3. Test CoNLL export endpoint
        conll_res = client.get("/api/export?format=conll", headers=auth)
        assert conll_res.status_code == 200
        assert "text/plain" in conll_res.headers.get("content-type", "")
        conll_text = conll_res.text
        assert "Rahul\tB-PER" in conll_text
        assert "Infosys\tB-ORG" in conll_text
        assert "Pune\tB-LOC" in conll_text
        assert "राहुल\tB-PER" in conll_text
        assert "शर्मा\tI-PER" in conll_text

        # 4. Test JSONL export endpoint
        jsonl_res = client.get("/api/export?format=jsonl", headers=auth)
        assert jsonl_res.status_code == 200
        assert "application/x-ndjson" in jsonl_res.headers.get("content-type", "")
        jsonl_lines = [json.loads(line) for line in jsonl_res.text.strip().split("\n") if line.strip()]
        assert len(jsonl_lines) >= 2

        # Verify English record
        en_record = next(r for r in jsonl_lines if r["text"] == en_text)
        assert en_record["lang"] == "en"
        assert en_record["entities"] == [[0, 5, "PER"], [15, 22, "ORG"], [26, 30, "LOC"]]

        # Verify Hindi record
        hi_record = next(r for r in jsonl_lines if r["text"] == hi_text)
        assert hi_record["lang"] == "hi"
        assert hi_record["entities"] == [
            [p_rahul, p_rahul + len("राहुल शर्मा"), "PER"],
            [p_infosys, p_infosys + len("इंफोसिस"), "ORG"],
        ]

        # 5. Backward compatibility: verify raw GET /api/export without format returns JSON list
        raw_res = client.get("/api/export", headers=auth)
        assert raw_res.status_code == 200
        assert isinstance(raw_res.json(), list)

        # 6. Error handling: invalid export format
        bad_res = client.get("/api/export?format=unsupported_format", headers=auth)
        assert bad_res.status_code == 400
