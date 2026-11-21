"""Build the English NER training mix: train splits of CoNLL-2003, WNUT-2017, BTC and FIN, one label set, no test leakage.

    python training/build_mix.py            # writes evaluation/data/train_mix.jsonl and prints a report
    python training/build_mix.py --dev      # writes evaluation/data/dev_mix.jsonl (WNUT, BTC, FIN dev splits: only for choosing checkpoints)

Rules (evaluation plan, sections 2.3 and 8.2):
  * only TRAIN splits are used; dev splits are kept for tuning and test splits for reporting
  * labels come from evaluation/label_maps.py (PER, ORG, LOC, MISC)
  * every training sentence whose normalized text, or its "skeleton" (lower-case letters and digits only, so URLs, @mentions
    and punctuation do not hide a copy), also occurs in ANY test set we report on is removed
Note: using BTC and FIN training data makes BTC and FIN in-domain for the resulting model; report them separately from the
zero-shot columns.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from common import DATA                                   # noqa: E402
from label_maps import LABELS, TRAIN_MAP, to_tag          # noqa: E402

TRAIN_SETS = ["conll2003", "wnut2017", "btc", "fin"]
TEST_FILES = ["ner/conll2003/en_test.jsonl", "ner/wnut2017/en_test.jsonl", "ner/btc/en_test.jsonl",
              "ner/fin/en_test.jsonl", "ner/wikiann/en_test.jsonl"]


def normalize(tokens):
    return re.sub(r"\s+", " ", " ".join(tokens)).strip().lower()


def skeleton(tokens):
    text = re.sub(r"https?://\S+|@\w+", " ", " ".join(tokens).lower())
    return re.sub(r"[^a-z0-9]+", "", text)


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="build the dev mix (validation splits) instead of the training mix")
    args = ap.parse_args()
    split, sets = ("valid", ["wnut2017", "btc", "fin"]) if args.dev else ("train", TRAIN_SETS)
    OUT = DATA / ("dev_mix.jsonl" if args.dev else "train_mix.jsonl")
    banned_exact, banned_skel = set(), set()
    for f in TEST_FILES:
        for r in rows(DATA / f):
            banned_exact.add(normalize(r["tokens"]))
            sk = skeleton(r["tokens"])
            if len(sk) >= 8:
                banned_skel.add(sk)
    print(f"test sentences indexed: {len(banned_exact)} exact, {len(banned_skel)} skeletons")

    mix, report = [], []
    label_id = {l: i for i, l in enumerate(LABELS)}
    for ds in sets:
        id2label = {v: k for k, v in json.loads((DATA / "ner" / ds / "label.json").read_text(encoding="utf-8")).items()}
        kept = dropped_exact = dropped_skel = dups = 0
        entities, seen = Counter(), set()
        for r in rows(DATA / "ner" / ds / f"en_{split}.jsonl"):
            n = normalize(r["tokens"])
            sk = skeleton(r["tokens"])
            if n in banned_exact:
                dropped_exact += 1
                continue
            if len(sk) >= 8 and sk in banned_skel:
                dropped_skel += 1
                continue
            if n in seen:                                   # exact duplicates inside the training data add nothing
                dups += 1
                continue
            seen.add(n)
            tags = [to_tag(id2label[t], ds) for t in r["tags"]]
            entities.update(t[2:] for t in tags if t.startswith("B-"))
            mix.append({"source": ds, "tokens": r["tokens"], "tags": [label_id[t] for t in tags]})
            kept += 1
        report.append((ds, kept, dropped_exact, dropped_skel, dups, dict(entities)))
    OUT.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in mix) + "\n", encoding="utf-8")
    print(f"\n{'dataset':10s} {'kept':>6s} {'in test (exact)':>16s} {'in test (near)':>15s} {'train dups':>11s}  entities")
    for ds, kept, de, ds_, dups, ent in report:
        print(f"{ds:10s} {kept:6d} {de:16d} {ds_:15d} {dups:11d}  {ent}")
    print(f"\ntotal {len(mix)} sentences -> {OUT}")


if __name__ == "__main__":
    main()
