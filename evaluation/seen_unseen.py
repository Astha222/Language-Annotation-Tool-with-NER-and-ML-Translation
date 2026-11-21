"""E1: split every test result into entities a model could have MEMORISED from its training data and entities it has never seen.

A test entity is "seen" when an entity with the same type and the same lower-cased text occurs in the training data of the model. For a saved run
(run_all.py writes one) this reports:

  all-sentence F1      strict F1 on every test sentence, as everywhere else
  unseen subset        strict F1 on the sentences that contain NO seen gold entity, with its size and a 95% bootstrap range
  recall seen/unseen   the share of seen (and of unseen) gold entities that the model found exactly
Annotation artefacts (a bare "@", the contract role words "Borrower" and "Lender") are set aside from the split: see ARTEFACT_SURFACES.

If the unseen-subset F1 is far below the all-sentence F1 the headline number is partly memorisation.

    python evaluation/seen_unseen.py --run runs/btc__en__bert-base-NER.jsonl --train conll2003
    python evaluation/seen_unseen.py --table english          # the four English models on WNUT, BTC and FIN
    python evaluation/seen_unseen.py --table indic            # the Indian-language runs, with Hindi training data as "seen"
    python evaluation/run_all.py ... --train-sets conll2003   # the same report straight after a scoring run

Training sets: conll2003, conll2003+test (the teacher was also trained on CoNLL test), wnut2017, btc, fin, train_mix, train_mix:<source>,
naamapadam_hi (the Hindi training split, streamed from its zip), none. Join several with "+".
"""
import argparse
import json
import zipfile
from pathlib import Path

from bootstrap import counts_per_sentence_rows, interval
from common import DATA, use_console_utf8
from eval_ner import KEEP, load_split
from label_maps import LABELS

HERE = Path(__file__).resolve().parent


# Gold "entities" that are annotation conventions rather than names: a bare symbol such as "@" (the Twitter set labels it as an entity) and the
# role words of contracts (the finance set labels "Borrower" and "Lender" as persons). They are common in the training data and rarely tagged by
# any model, so they would make "seen entities" look hard. They are set aside from the seen/unseen split (but still count in the all-sentence F1).
ARTEFACT_SURFACES = {"borrower", "lender"}


def is_artefact(surface, drop=ARTEFACT_SURFACES):
    return not any(ch.isalpha() for ch in surface) or surface in drop


def keys_from_sentences(sentences, joiner=" "):
    """(type, lower-cased surface) of every PER / ORG / LOC entity in BIO-tagged sentences [(tokens, tags)]."""
    keys = set()
    for tokens, tags in sentences:
        cur = None
        for tok, tag in zip(tokens + [None], tags + ["O"]):
            if cur and (tag == "O" or tag.startswith("B-") or cur[0] != tag[2:]):
                keys.add((cur[0], joiner.join(cur[1]).lower()))
                cur = None
            if tag != "O" and tok is not None:
                if cur is None:
                    cur = [tag[2:], []]
                cur[1].append(tok)
    return {k for k in keys if k[0] in KEEP}


def _mix_sentences(source=None):
    out = []
    for line in (DATA / "train_mix.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if source is None or r["source"] == source:
                out.append((r["tokens"], [LABELS[i] for i in r["tags"]]))
    return out


def _naamapadam_hi_keys():
    """Entity inventory of the Hindi training split, streamed from the zip so the 490 MB file is never written out."""
    from huggingface_hub import hf_hub_download
    archive = hf_hub_download("ai4bharat/naamapadam", "data/hi_IndicNER_v1.0.zip", repo_type="dataset")
    keys = set()
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith("hi_train.json"))
        with z.open(name) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keys |= keys_from_sentences([(r["words"], r["ner"])])
    return keys


def training_keys(spec, joiner=" "):
    keys = set()
    for part in spec.split("+"):
        part = part.strip()
        if part in ("", "none"):
            continue
        if part == "conll2003":
            keys |= keys_from_sentences(load_split("conll2003", "en", "train", None), joiner)
        elif part == "test" or part == "conll2003test":                       # "conll2003+test"
            keys |= keys_from_sentences(load_split("conll2003", "en", "test", None), joiner)
        elif part in ("wnut2017", "btc", "fin"):
            keys |= keys_from_sentences(load_split(part, "en", "train", None), joiner)
        elif part == "train_mix":
            keys |= keys_from_sentences(_mix_sentences(), joiner)
        elif part.startswith("train_mix:"):
            keys |= keys_from_sentences(_mix_sentences(part.split(":", 1)[1]), joiner)
        elif part == "naamapadam_hi":
            keys |= _naamapadam_hi_keys()
        else:
            raise SystemExit(f"unknown training set {part!r}")
    return keys


def has_seen_entity(r, seen, drop=ARTEFACT_SURFACES):
    """True when a sentence contains a gold entity (not an annotation artefact) that occurs in the training data."""
    for t, a, b in map(tuple, r["gold"]):
        surface = r["text"][r["offsets"][a][0]:r["offsets"][b][1]].lower()
        if not is_artefact(surface, drop) and (t, surface) in seen:
            return True
    return False


def report(rows, seen, resamples=1000, drop=ARTEFACT_SURFACES):
    """The E1 numbers for a list of saved result rows (as written by run_all.py)."""
    counts_all, counts_unseen = [], []
    found = {"seen": [0, 0], "unseen": [0, 0]}                        # [found, total] gold entities
    artefacts = 0
    for r in rows:
        gold = {tuple(g) for g in r["gold"]}
        pred = {(t, a, b) for t, a, b, s in r["pred"] if t in KEEP}
        has_seen = False
        for t, a, b in gold:
            surface = r["text"][r["offsets"][a][0]:r["offsets"][b][1]].lower()
            if is_artefact(surface, drop):
                artefacts += 1
                continue
            kind = "seen" if (t, surface) in seen else "unseen"
            has_seen |= kind == "seen"
            found[kind][1] += 1
            found[kind][0] += (t, a, b) in pred
        c = (len(gold & pred), len(pred - gold), len(gold - pred))
        counts_all.append(c)
        if not has_seen:
            counts_unseen.append(c)

    def f1(cs):
        tp = sum(c[0] for c in cs)
        fp = sum(c[1] for c in cs)
        fn = sum(c[2] for c in cs)
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    out = {"sentences": len(rows), "f1_all": f1(counts_all), "unseen_size": len(counts_unseen), "gold_artefacts": artefacts,
           "gold_seen": found["seen"][1], "gold_unseen": found["unseen"][1],
           "share_seen": found["seen"][1] / max(1, found["seen"][1] + found["unseen"][1]),
           "recall_seen": found["seen"][0] / found["seen"][1] if found["seen"][1] else None,
           "recall_unseen": found["unseen"][0] / found["unseen"][1] if found["unseen"][1] else None,
           "f1_unseen": f1(counts_unseen) if counts_unseen else None, "f1_unseen_ci": None}
    if len(counts_unseen) >= 20:
        out["f1_unseen_ci"] = interval(counts_per_sentence_rows(counts_unseen), resamples)
    return out


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def line(name, dataset, rep):
    ci = rep["f1_unseen_ci"]
    unseen = "-" if rep["f1_unseen"] is None else f"{rep['f1_unseen']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else " [too few sentences]")
    return (f"| {name} | {dataset} | {rep['sentences']} | {rep['f1_all']:.3f} | {unseen} | {rep['unseen_size']} | "
            f"{rep['share_seen']:.0%} | {fmt(rep['recall_seen'])} | {fmt(rep['recall_unseen'])} |")


HEADER = ("| model | test set | sentences | F1 all | F1 unseen subset [95% CI] | unseen sentences | gold entities seen in training | recall seen | recall unseen |\n"
          "|---|---|---|---|---|---|---|---|---|")


def load_rows(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


ENGLISH_SETS = ("wnut2017", "btc", "fin", "uner_en_ewt")      # uner_en_ewt: E5, English web text no model was trained on

ENGLISH_MODELS = [  # (name, run-file model name, documented training data)
    ("bert-base-NER (default)", "bert-base-NER", "conll2003"),
    ("roberta-large (teacher; trained on CoNLL train and test)", "roberta-large-ner-english", "conll2003+test"),
    ("student A (CoNLL + WNUT + BTC + FIN)", "A_distilroberta_distill", "train_mix"),
    ("student B (same data as A, NO teacher)", "B_distilroberta_plain", "train_mix"),
    ("student C (CoNLL only)", "C_distilroberta_zeroshot", "train_mix:conll2003"),
]
INDIC_RUNS = [  # (name, run file stem, language, training data that counts as "seen")
    ("indicbert-hi on Hindi (Naamapadam)", "naamapadam__hi__indicbert_hi", "hi", "naamapadam_hi"),
    ("indicbert-hi on Hindi (HiNER)", "hiner__hi__indicbert_hi", "hi", "naamapadam_hi"),
    ("indicbert-hi on Marathi", "nm_mr__mr__indicbert_hi", "mr", "naamapadam_hi"),
    ("indicbert-hi on Bengali", "nm_bn__bn__indicbert_hi", "bn", "naamapadam_hi"),
    ("indicbert-hi on Tamil", "nm_ta__ta__indicbert_hi", "ta", "naamapadam_hi"),
    ("indicbert-hi on Telugu", "nm_te__te__indicbert_hi", "te", "naamapadam_hi"),
    ("indicbert-hi on Gujarati", "nm_gu__gu__indicbert_hi", "gu", "naamapadam_hi"),
    ("indicbert-hi on Kannada", "nm_kn__kn__indicbert_hi", "kn", "naamapadam_hi"),
    ("indicbert-hi on Malayalam", "nm_ml__ml__indicbert_hi", "ml", "naamapadam_hi"),
    ("indicbert-hi on Punjabi", "nm_pa__pa__indicbert_hi", "pa", "naamapadam_hi"),
]


def common_unseen_table(models, datasets, runs):
    """Score every model on the SAME sentences: those with no gold entity seen by ANY of the models' training data."""
    seen_by = {train: training_keys(train) for _, _, train in models}
    union = set().union(*seen_by.values())
    lines = [("| test set | common unseen sentences | " + " | ".join(name for name, _, _ in models) + " |"),
             "|---|---|" + "---|" * len(models)]
    for ds in datasets:
        per_model = {stem: load_rows(runs / f"{ds}__en__{stem}.jsonl") for _, stem, _ in models}
        first = next(iter(per_model.values()))
        keep = [i for i, r in enumerate(first) if not has_seen_entity(r, union)]
        cells = []
        for _, stem, _ in models:
            sub = [per_model[stem][i] for i in keep]
            counts = [(len({tuple(g) for g in r["gold"]} & {(t, a, b) for t, a, b, _ in r["pred"] if t in KEEP}),
                       len({(t, a, b) for t, a, b, _ in r["pred"] if t in KEEP} - {tuple(g) for g in r["gold"]}),
                       len({tuple(g) for g in r["gold"]} - {(t, a, b) for t, a, b, _ in r["pred"] if t in KEEP})) for r in sub]
            tp, fp, fn = (sum(c[k] for c in counts) for k in range(3))
            f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            lo, hi = interval(counts_per_sentence_rows(counts)) if len(counts) >= 20 else (None, None)
            cells.append(f"{f:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else ""))
        lines.append(f"| {ds} | {len(keep)} of {len(first)} | " + " | ".join(cells) + " |")
    return chr(10).join(lines)


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a saved run file (runs/*.jsonl)")
    ap.add_argument("--train", default="none", help="training data of the model that made the run, for example conll2003 or train_mix:conll2003")
    ap.add_argument("--lang", default="en", help="zh and ja are joined without spaces")
    ap.add_argument("--table", choices=["english", "english-common", "english-names", "indic"], help="build a whole table")
    args = ap.parse_args()
    runs = HERE / "runs"
    if args.run:
        rep = report(load_rows(args.run), training_keys(args.train, "" if args.lang in ("zh", "ja") else " "))
        print(HEADER)
        print(line(Path(args.run).stem, "", rep))
        return
    if args.table == "english-names":
        print(names_only_table(NAMES_MODELS, ENGLISH_SETS, runs))
        return
    if args.table == "english-common":
        print(common_unseen_table(ENGLISH_MODELS, ENGLISH_SETS, runs))
        return
    print(HEADER)
    if args.table == "english":
        cache = {}
        for name, stem, train in ENGLISH_MODELS:
            seen = cache.setdefault(train, training_keys(train))
            for ds in ENGLISH_SETS:
                print(line(name, ds, report(load_rows(runs / f"{ds}__en__{stem}.jsonl"), seen)))
    if args.table == "indic":
        seen = training_keys("naamapadam_hi")
        for name, stem, lang, _ in INDIC_RUNS:
            path = runs / f"{stem}.jsonl"
            if path.exists():
                print(line(name, "", report(load_rows(path), seen)))



def surface_of(r, a, b):
    return r["text"][r["offsets"][a][0]:r["offsets"][b][1]].lower()


def names_only_counts(rows, drop=ARTEFACT_SURFACES):
    """Per-sentence (tp, fp, fn) after removing annotation artefacts from BOTH the gold labels and the predictions."""
    counts = []
    for r in rows:
        gold = {tuple(g) for g in r["gold"] if not is_artefact(surface_of(r, g[1], g[2]), drop)}
        pred = {(t, a, b) for t, a, b, _ in r["pred"] if t in KEEP and not is_artefact(surface_of(r, a, b), drop)}
        counts.append((len(gold & pred), len(pred - gold), len(gold - pred)))
    return counts


def names_only_table(models, datasets, runs):
    """F1 on names only: how much of a model's score depends on labelling conventions such as tagging a bare '@' or the word 'Borrower'."""
    lines = ["| model | test set | F1 all (standard) | F1 names only [95% CI] | change |", "|---|---|---|---|---|"]
    for name, stem in models:
        for ds in datasets:
            path = runs / f"{ds}__en__{stem}.jsonl"
            if not path.exists():
                continue
            rows = load_rows(path)
            std = report(rows, set())["f1_all"]
            counts = names_only_counts(rows)
            tp, fp, fn = (sum(c[k] for c in counts) for k in range(3))
            f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            lo, hi = interval(counts_per_sentence_rows(counts))
            lines.append(f"| {name} | {ds} | {std:.3f} | {f:.3f} [{lo:.3f}, {hi:.3f}] | {f - std:+.3f} |")
    return chr(10).join(lines)


NAMES_MODELS = [("bert-base-NER (default)", "bert-base-NER"), ("roberta-large (teacher)", "roberta-large-ner-english"),
                ("student A (in-domain, with teacher)", "A_distilroberta_distill"), ("student B (in-domain, NO teacher)", "B_distilroberta_plain"),
                ("student C (CoNLL only)", "C_distilroberta_zeroshot")]


if __name__ == "__main__":
    main()
