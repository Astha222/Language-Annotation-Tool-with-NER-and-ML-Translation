"""How much of a test set's entity inventory was already seen in the training mix?

For every test set it reports the share of gold PER / ORG / LOC entity strings (case-insensitive) that also occur as an entity of
the SAME type in the training data of the same source. A high share means a model trained on that source can score well partly by
remembering names, so its score there says less about how it handles new text.

    python evaluation/overlap_check.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from common import DATA                                 # noqa: E402
from eval_ner import DATASETS, build, load_split         # noqa: E402
from label_maps import LABELS                            # noqa: E402
from seen_unseen import is_artefact                      # noqa: E402

KEEP = {"PER", "ORG", "LOC"}


def entities(tokens, tags):
    """(type, lower-cased string) for each PER / ORG / LOC entity of a BIO-tagged sentence."""
    out, cur = [], None
    for t, tag in zip(tokens, tags):
        if tag == "O" or (cur and tag.startswith("B-")) or (cur and cur[0] != tag[2:]):
            if cur and cur[0] in KEEP:
                out.append((cur[0], " ".join(cur[1]).lower()))
            cur = None
        if tag != "O":
            cur = cur or [tag[2:], []]
            cur[1].append(t)
    if cur and cur[0] in KEEP:
        out.append((cur[0], " ".join(cur[1]).lower()))
    return out


def main():
    train = [json.loads(l) for l in (DATA / "train_mix.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {}
    for r in train:
        seen.setdefault(r["source"], set()).update(entities(r["tokens"], [LABELS[i] for i in r["tags"]]))
    print(f"{'test set':10s} {'gold entities':>14s} {'seen in training (same source)':>32s} {'  per type (PER / ORG / LOC)'}")
    for ds in ("wnut2017", "btc", "fin"):
        test = load_split(ds, "en", DATASETS[ds][2][1], None)
        ents = []
        for tokens, tags in test:
            ents += entities(tokens, tags)
        tot = len(ents)
        hit = sum(1 for e in ents if e in seen[ds])
        names = [e for e in ents if not is_artefact(e[1])]
        hit_names = sum(1 for e in names if e in seen[ds])
        print(f"{ds:10s} excluding annotation artefacts (bare symbols such as @, 'borrower', 'lender'): {hit_names} of {len(names)} = {hit_names / max(1, len(names)):.1%} seen")
        per = {}
        for t in ("PER", "ORG", "LOC"):
            sub = [e for e in ents if e[0] == t]
            per[t] = sum(1 for e in sub if e in seen[ds]) / max(1, len(sub))
        print(f"{ds:10s} {tot:14d} {hit:20d} = {hit / tot:5.1%}      " + " / ".join(f"{per[t]:.0%}" for t in per))


if __name__ == "__main__":
    main()
