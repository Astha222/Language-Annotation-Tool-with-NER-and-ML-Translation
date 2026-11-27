"""Language identification: the app's detect_language() against langdetect alone, sentence by sentence, on every frozen test set.

The expected language of each set is its own language (Hinglish -> hi). Accuracy = share of sentences detected as that language; the most frequent
wrong answers are listed. Short sentences are the hard case, so the share of sentences under 30 characters is shown too.

    python evaluation/eval_language_id.py
"""
import collections
import sys

from common import load_backend, use_console_utf8
import eval_ner

SETS = [("naamapadam", "hi"), ("hiner", "hi"), ("mc_hi", "hi"), ("hinglish", "hi")] + \
       [(f"nm_{l}", l) for l in ("mr", "bn", "as", "or", "pa", "gu", "ta", "te", "kn", "ml")] + \
       [("wnut2017", "en"), ("uner_en_ewt", "en"), ("mc_de", "de"), ("uner_de", "de"), ("mc_es", "es"), ("mc_fr", "fr"),
        ("uner_ru", "ru"), ("uner_zhs", "zh"), ("uner_ja", "ja"), ("anercorp", "ar")]
LIMIT = 1000           # sentences per set (the first N), to keep the run short


def texts(dataset, lang):
    split = eval_ner.LOCAL[dataset][4] if dataset in eval_ner.LOCAL else eval_ner.DATASETS[dataset][2][1]
    rows = eval_ner.load_split(dataset, lang if dataset in eval_ner.LOCAL or dataset == "wikiann" else "en", split, LIMIT)
    sep = "" if lang in ("zh", "ja") else " "
    return [sep.join(words) for words, _ in rows]


def main():
    use_console_utf8()
    app = load_backend()
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0

    def old(t):
        try:
            return app.normalize_lang(detect(t))
        except Exception:
            return None

    print(f"{'test set':14s}{'lang':>5s}{'n':>6s}{'langdetect alone':>18s}{'app (new)':>11s}   most frequent wrong answers (app)")
    total = collections.Counter()
    for dataset, lang in SETS:
        try:
            xs = texts(dataset, lang)
        except Exception as e:                      # a set that is not downloaded on this machine
            print(f"{dataset:14s} skipped ({type(e).__name__})")
            continue
        o = [old(t) for t in xs]
        n = [app.detect_language(t) for t in xs]
        acc_o = sum(x == lang for x in o) / len(xs)
        acc_n = sum(x == lang for x in n) / len(xs)
        wrong = collections.Counter(x for x in n if x != lang).most_common(3)
        print(f"{dataset:14s}{lang:>5s}{len(xs):>6d}{acc_o:>17.1%} {acc_n:>10.1%}   {wrong}")
        total["n"] += len(xs); total["old"] += sum(x == lang for x in o); total["new"] += sum(x == lang for x in n)
    print(f"{'all':14s}{'':>5s}{total['n']:>6d}{total['old'] / total['n']:>17.1%} {total['new'] / total['n']:>10.1%}")


if __name__ == "__main__":
    sys.exit(main())
