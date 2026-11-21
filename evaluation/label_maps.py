"""One table that maps every dataset's own entity types to the four labels the app's English model uses.

PER, ORG, LOC and MISC. MISC is kept for training (it stops the model from forcing products, events and creative works into ORG);
the evaluation scripts still score only PER / ORG / LOC (see eval_ner.KEEP). The same table is used for training and evaluation.

    label_maps.TRAIN_MAP[dataset][original label] -> "PER" | "ORG" | "LOC" | "MISC" | None (dropped, becomes O)
"""

LABELS = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]   # same order as dslim/bert-base-NER

TRAIN_MAP = {
    "conll2003": {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"},
    "wnut2017": {"person": "PER", "corporation": "ORG", "group": "ORG", "location": "LOC",
                 "product": "MISC", "creative-work": "MISC"},
    "btc": {"PER": "PER", "ORG": "ORG", "LOC": "LOC"},
    "fin": {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"},
}


def to_tag(label: str, dataset: str) -> str:
    """'B-corporation' -> 'B-ORG'; unknown or dropped types become 'O'."""
    if label == "O":
        return "O"
    kind = TRAIN_MAP[dataset].get(label[2:])
    return "O" if kind is None else f"{label[:2]}{kind}"
