"""How well does backend/hinglish.py tell Romanized Hindi from other Latin-script text?

Positives: the 125 sentences of the Hinglish test set. Negatives: informal English (Twitter: BTC and WNUT-2017 test sentences), Wikipedia English (WikiANN en),
and German / Spanish / French (MultiCoNER dev), all from the frozen data folder. langdetect's answer on the same sentences is shown for comparison.

    python evaluation/eval_hinglish_detect.py
"""
import json
import sys

from common import BACKEND, DATA, use_console_utf8

sys.path.insert(0, str(BACKEND))
import hinglish                              # noqa: E402


def load(path, words="tokens", n=None):
    rows = [json.loads(l) for l in (DATA / path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [" ".join(r[words]) for r in rows[:n] if r.get(words)]


def safe_detect(detect, text):
    try:
        return detect(text)
    except Exception:                                  # "No features in text" (for example a lone @)
        return None


def main():
    use_console_utf8()
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
    except ImportError:
        detect = None
    sets = {
        "Hinglish (positive)": load("ner_hi/hinglish/hi_test.json", "words"),
        "English tweets (BTC)": load("ner/btc/en_test.jsonl"),
        "English tweets (WNUT)": load("ner/wnut2017/en_test.jsonl"),
        "English Wikipedia (WikiANN)": load("ner/wikiann/en_test.jsonl", n=1500),
        "German (MultiCoNER)": load("ner_multi/mc_de_dev.json", "words"),
        "Spanish (MultiCoNER)": load("ner_multi/mc_es_dev.json", "words"),
        "French (MultiCoNER)": load("ner_multi/mc_fr_dev.json", "words"),
    }
    print(f"{'set':30s}{'sentences':>10s}{'flagged as Hinglish':>22s}{'langdetect says hi':>22s}")
    shown = {}
    for name, texts in sets.items():
        flagged = [t for t in texts if hinglish.is_romanized_hindi(t)]
        ld = sum(safe_detect(detect, t) == "hi" for t in texts) if detect else -1
        print(f"{name:30s}{len(texts):>10d}{len(flagged):>12d} ({len(flagged) / len(texts):6.1%}){ld:>12d} ({ld / len(texts):6.1%})")
        shown[name] = flagged if name != "Hinglish (positive)" else [t for t in texts if t not in flagged]
    pos = sets["Hinglish (positive)"]
    tp = sum(hinglish.is_romanized_hindi(t) for t in pos)
    fp = sum(hinglish.is_romanized_hindi(t) for k, v in sets.items() if k != "Hinglish (positive)" for t in v)
    neg_total = sum(len(v) for k, v in sets.items() if k != "Hinglish (positive)")
    print(f"\nrecall {tp}/{len(pos)} = {tp / len(pos):.3f};  false positives {fp}/{neg_total} = {fp / neg_total:.4f};  "
          f"precision (if Hinglish were as common as in this mix) {tp / (tp + fp) if tp + fp else 0:.3f}")
    print("\nmissed Hinglish sentences (first 15):")
    for t in shown["Hinglish (positive)"][:15]:
        print("  ", t, "->", hinglish.hinglish_score(t))
    for name, flagged in shown.items():
        if name != "Hinglish (positive)" and flagged:
            print(f"\nfalse positives in {name} (first 6):")
            for t in flagged[:6]:
                print("  ", t[:140], "->", hinglish.hinglish_score(t))


if __name__ == "__main__":
    main()
