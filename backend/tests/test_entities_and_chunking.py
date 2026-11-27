"""Entity clean-up (whole tokens, no punctuation, no duplicates), text chunking and NER over long text."""
import pytest


def ent(txt, word, nth=1, group="ORG", score=0.9):
    i = -1
    for _ in range(nth):
        i = txt.index(word, i + 1)
    return {"entity_group": group, "score": score, "start": i, "end": i + len(word)}


# ── chunking ─────────────────────────────────────────────────────────────
def test_chunks_are_lossless_ordered_and_bounded(main_module):
    text = ("Hello world. " * 100) + "x" * 1000 + "\n" + ("नमस्ते दुनिया। " * 50)
    chunks = main_module.chunk_text(text, 450)
    assert max(len(c) for _, c in chunks) <= 450
    assert all(text[o:o + len(c)] == c for o, c in chunks)              # offsets point at the right place
    assert "".join(c for _, c in chunks) == text                         # nothing lost, nothing duplicated


def test_chunking_empty_text(main_module):
    assert main_module.chunk_text("", 100) == []


# ── entity clean-up ──────────────────────────────────────────────────────
TXT = "We use MERN (MongoDB, Express.js, React.js, Node.js) and Google. Google's Meta-owned Vue.js apps."


def test_dotted_tokens_cut_by_the_model_become_whole(main_module):
    pieces = [ent(TXT, "js", 1), ent(TXT, "Node"), ent(TXT, "js", 3), ent(TXT, "js", 2), ent(TXT, "React"),
              ent(TXT, "Vue"), ent(TXT, "js", 4), ent(TXT, "MongoDB")]
    words = [e["word"] for e in main_module._clean_entities(TXT, pieces, 0)]
    assert words == ["MongoDB", "Express.js", "React.js", "Node.js", "Vue.js"]


def test_sentence_final_dot_and_possessive_are_not_swallowed(main_module):
    out = main_module._clean_entities(TXT, [ent(TXT, "Google", 1), ent(TXT, "Google", 2), ent(TXT, "Meta")], 0)
    assert [e["word"] for e in out] == ["Google", "Google", "Meta"]


def test_entities_never_overlap_and_offsets_match(main_module):
    pieces = [ent(TXT, "js", 1), ent(TXT, "Express"), ent(TXT, "Express.js")]
    out = main_module._clean_entities(TXT, pieces, 0)
    assert len(out) == 1 and TXT[out[0]["start"]:out[0]["end"]] == "Express.js"


def test_overlap_keeps_the_most_confident_label(main_module):
    out = main_module._clean_entities(TXT, [ent(TXT, "Express", group="ORG", score=0.4), ent(TXT, "js", 1, "MISC", 0.95)], 0)
    assert out[0]["entity_group"] == "MISC" and out[0]["score"] == 0.95


def test_hindi_entities_are_untouched(main_module):
    hi = "नरेंद्र मोदी ने गूगल में बात की।"
    out = main_module._clean_entities(hi, [{"entity_group": "PER", "score": .99, "start": 0, "end": 11},
                                           {"entity_group": "ORG", "score": .9, "start": 15, "end": 19}], 0)
    assert [e["word"] for e in out] == ["नरेंद्र मोदी", "गूगल"]


def test_edge_punctuation_is_trimmed(main_module):
    txt = "Meet Google, today"
    out = main_module._clean_entities(txt, [{"entity_group": "ORG", "score": .9, "start": 5, "end": 12}], 0)
    assert out[0]["word"] == "Google" and out[0]["end"] == 11


def test_offset_is_added_for_later_chunks(main_module):
    out = main_module._clean_entities("Pichai", [{"entity_group": "PER", "score": .9, "start": 0, "end": 6}], 1000)
    assert (out[0]["start"], out[0]["end"]) == (1000, 1006)


# ── run_ner (with the stand-in pipeline from conftest) ───────────────────
def test_run_ner_maps_offsets_back_across_chunks(main_module):
    sentence = "Pichai met Nadella in Seattle. "
    text = sentence * 120                                                # several 800-char chunks
    ents = main_module.run_ner(text, "en")
    assert len(ents) == 120 * 3
    assert all(text[e["start"]:e["end"]] == e["word"] for e in ents)


def test_run_ner_reports_progress_and_can_be_cancelled(main_module):
    seen = []
    main_module.run_ner("Pichai " * 600, "en", progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == seen[-1][1]
    with pytest.raises(main_module.JobCancelled):
        main_module.run_ner("Pichai " * 600, "en", cancelled=lambda: True)


def test_entity_summary_counts_unique_entities(main_module):
    ents = [{"word": "Google", "entity_group": "ORG", "score": .9}, {"word": "Google", "entity_group": "ORG", "score": .99},
            {"word": "Modi", "entity_group": "PER", "score": .8}]
    summary = main_module._summarize_entities(ents)
    assert summary[0] == {"word": "Google", "entity_group": "ORG", "score": .99, "count": 2}
    assert len(summary) == 2
