"""E1 seen/unseen split on a 5-sentence fixture where the right answers are easy to count by hand."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))

import seen_unseen as su  # noqa: E402


def row(text, gold, pred):
    """Rows as run_all.py saves them: token offsets, gold spans (type, first token, last token), predictions (type, a, b, score)."""
    offsets, pos = [], 0
    for tok in text.split():
        offsets.append([pos, pos + len(tok)])
        pos += len(tok) + 1
    return {"text": text, "offsets": offsets, "gold": [list(g) for g in gold], "pred": [list(p) + [0.9] for p in pred]}


# training data: Rahul Sharma (PER) and Pune (LOC) were seen; nothing else
TRAIN = [(["Rahul", "Sharma", "lives", "in", "Pune"], ["B-PER", "I-PER", "O", "O", "B-LOC"])]
SEEN = su.keys_from_sentences(TRAIN)

ROWS = [
    row("Rahul Sharma lives here", [("PER", 0, 1)], [("PER", 0, 1)]),            # seen entity, found
    row("Visit Pune today", [("LOC", 1, 1)], []),                                   # seen entity, missed
    row("Anita Rao works at Infosys", [("PER", 0, 1), ("ORG", 4, 4)], [("PER", 0, 1)]),   # two unseen: one found, one missed
    row("Meera met Anita", [("PER", 0, 0), ("PER", 2, 2)], [("PER", 0, 0), ("PER", 2, 2), ("ORG", 1, 1)]),   # two unseen found + one false alarm
    row("nothing to see", [], []),                                                  # no entity at all
]


def test_entity_keys_are_typed_and_lower_cased():
    assert SEEN == {("PER", "rahul sharma"), ("LOC", "pune")}
    assert su.keys_from_sentences([(["a", "b", "c"], ["B-ORG", "I-ORG", "O"])], joiner="") == {("ORG", "ab")}      # zh/ja style
    assert su.keys_from_sentences([(["x"], ["B-MISC"])]) == set()                                               # only PER / ORG / LOC count


def test_counts_of_seen_and_unseen_entities():
    rep = su.report(ROWS, SEEN)
    assert rep["sentences"] == 5
    assert (rep["gold_seen"], rep["gold_unseen"]) == (2, 4)                 # Rahul Sharma + Pune seen; Anita Rao, Infosys, Meera, Anita not
    assert rep["recall_seen"] == 1 / 2                                       # Rahul found, Pune missed
    assert rep["recall_unseen"] == 3 / 4                                     # Anita Rao, Meera, Anita found; Infosys missed
    assert abs(rep["share_seen"] - 2 / 6) < 1e-9


def test_unseen_subset_keeps_only_sentences_without_a_seen_gold_entity():
    rep = su.report(ROWS, SEEN)
    assert rep["unseen_size"] == 3                                           # sentences 3, 4 and 5 (the last has no entity at all)
    # tp 3 (Anita Rao, Meera, Anita), fp 1 (the false ORG), fn 1 (Infosys): F1 = 2*3 / (2*3 + 1 + 1)
    assert abs(rep["f1_unseen"] - 6 / 8) < 1e-9
    assert rep["f1_unseen_ci"] is None                                       # too few sentences for a meaningful interval


def test_all_sentence_f1_matches_strict_scoring():
    rep = su.report(ROWS, SEEN)
    # tp 4 (Rahul, Anita Rao, Meera, Anita), fp 1, fn 2 (Pune, Infosys)
    assert abs(rep["f1_all"] - 8 / (8 + 1 + 2)) < 1e-9


def test_everything_seen_leaves_an_empty_unseen_subset():
    everything = {("PER", "rahul sharma"), ("LOC", "pune"), ("PER", "anita rao"), ("ORG", "infosys"), ("PER", "meera"), ("PER", "anita")}
    rep = su.report(ROWS, everything)
    assert rep["gold_unseen"] == 0 and rep["recall_unseen"] is None
    assert rep["unseen_size"] == 1 and rep["f1_unseen"] == 0.0               # only the entity-free sentence remains, nothing to score


def test_annotation_artefacts_are_set_aside_from_the_split():
    rows = [row("@ Rahul", [("PER", 0, 0), ("PER", 1, 1)], [("PER", 1, 1)]),       # a bare "@" gold entity the model does not tag
            row("the Borrower pays", [("PER", 1, 1)], [])]
    seen = {("PER", "@"), ("PER", "borrower")}
    rep = su.report(rows, seen)
    assert rep["gold_artefacts"] == 2 and rep["gold_seen"] == 0 and rep["gold_unseen"] == 1   # only "Rahul" is a real, unseen name
    assert rep["unseen_size"] == 2                                                              # neither sentence has a real seen entity
    assert su.is_artefact("@") and su.is_artefact("borrower") and not su.is_artefact("pune")


def test_has_seen_entity_ignores_artefacts():
    r = row("@ Rahul", [("PER", 0, 0), ("PER", 1, 1)], [])
    assert su.has_seen_entity(r, {("PER", "@")}) is False
    assert su.has_seen_entity(r, {("PER", "rahul")}) is True
