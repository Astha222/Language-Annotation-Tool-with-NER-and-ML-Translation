"""The pure parts of training/distill.py: label grouping, span decoding and the distillation loss."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import distill as d                                   # noqa: E402

L = {name: i for i, name in enumerate(d.LABELS)}


def test_student_groups_match_the_label_names():
    for type_name, ids in zip(d.TYPES, d.STUDENT_GROUPS):
        names = {d.LABELS[i] for i in ids}
        assert names == ({"O"} if type_name == "O" else {f"B-{type_name}", f"I-{type_name}"})


def test_teacher_label_schemes_are_grouped_by_type():
    plain = {0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}                                # roberta-large-ner-english
    assert d.type_groups(plain) == [[0], [1], [2], [3], [4]]
    bio = dict(enumerate(d.LABELS))                                                           # dslim/bert-base-NER
    assert d.type_groups(bio)[d.TYPES.index("PER")] == [L["B-PER"], L["I-PER"]]
    with pytest.raises(ValueError):
        d.type_groups({0: "O", 1: "PER"})                                                     # a type is missing
    with pytest.raises(ValueError):
        d.type_groups({**plain, 5: "B-DATE"})


def test_spans_follow_bio_rules_and_ignore_misc():
    tags = [L["B-PER"], L["I-PER"], L["O"], L["B-ORG"], L["B-ORG"], L["I-LOC"], L["B-MISC"]]
    assert d.spans(tags) == {("PER", 0, 1), ("ORG", 3, 3), ("ORG", 4, 4), ("LOC", 5, 5)}      # I-LOC without B- starts an entity
    assert d.spans([L["O"]] * 3) == set()


def test_variants_change_case_only():
    assert d.variant(["Rahul", "in", "Pune"], "lower") == ["rahul", "in", "pune"]
    assert d.variant(["Rahul", "in", "Pune"], "upper") == ["RAHUL", "IN", "PUNE"]
    assert d.variant(["Rahul"], "orig") == ["Rahul"]


def _logits_for(label, strength=8.0):
    z = torch.zeros(1, 1, len(d.LABELS))
    z[0, 0, L[label]] = strength
    return z


def test_loss_is_small_when_student_matches_gold_and_teacher():
    gold = torch.tensor([[L["B-PER"]]])
    teacher = torch.tensor([[[0.0, 9.0, 0.0, 0.0, 0.0]]])                                     # teacher: PER
    good = d.distill_loss(_logits_for("B-PER"), gold, teacher, torch.ones(1, 1), alpha=0.5, temp=2.0)
    wrong = d.distill_loss(_logits_for("B-LOC"), gold, teacher, torch.ones(1, 1), alpha=0.5, temp=2.0)
    assert good < 0.1 and wrong > 3 * good and wrong > 1.0


def test_teacher_term_ignores_boundaries_b_versus_i():
    teacher = torch.tensor([[[0.0, 9.0, 0.0, 0.0, 0.0]]])
    gold = torch.tensor([[-100]])                                                             # no gold: only the teacher term counts
    b = d.distill_loss(_logits_for("B-PER"), gold, teacher, torch.ones(1, 1), alpha=0.0, temp=1.0)
    i = d.distill_loss(_logits_for("I-PER"), gold, teacher, torch.ones(1, 1), alpha=0.0, temp=1.0)
    assert math.isclose(b.item(), i.item(), rel_tol=1e-4)                                      # both are "PER" to the teacher


def test_alpha_one_is_plain_cross_entropy_and_masked_words_are_skipped():
    logits = _logits_for("B-PER")
    gold = torch.tensor([[L["B-PER"]]])
    ce = torch.nn.functional.cross_entropy(logits.view(-1, 9), gold.view(-1))
    assert math.isclose(d.distill_loss(logits, gold, torch.zeros(1, 1, 5), torch.ones(1, 1), 1.0, 2.0).item(), ce.item(), rel_tol=1e-6)
    only_ce = d.distill_loss(logits, gold, torch.zeros(1, 1, 5), torch.zeros(1, 1), 0.5, 2.0)    # no word has a teacher score
    assert math.isclose(only_ce.item(), ce.item(), rel_tol=1e-6)
