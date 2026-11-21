"""Generate backend/language_status.json from the evaluation results (the single source for the Stable / Beta / Experimental labels).

    python evaluation/make_language_status.py

For each language it looks, among the candidate models with an OPEN licence, for the best strict F1 on the language's human-labelled test set
(the file with the most sentences per model is used, so a full-set run replaces a 300-sentence one), computes the 95% bootstrap range, and applies the
rules in backend/language_status.py. Models without an open licence are listed as reference only and can never be the offered model.
Translation is recorded separately; it has no agreed thresholds yet, so it is "not_evaluated" except where a measurement exists.
"""
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
import language_status as rules                                    # noqa: E402
from bootstrap import counts_per_sentence, interval               # noqa: E402
from common import use_console_utf8                                # noqa: E402
from run_all import summarize                                      # noqa: E402

RUNS = HERE / "runs"

# the owner asked for Hindi first; the rest follows the order agreed in the plan
LANGUAGES = [("hi", "Hindi"), ("mr", "Marathi"), ("bn", "Bengali"), ("ta", "Tamil"), ("te", "Telugu"), ("gu", "Gujarati"),
             ("kn", "Kannada"), ("ml", "Malayalam"), ("pa", "Punjabi"), ("or", "Odia"), ("as", "Assamese")]

# candidate models: id, licence, open?, names the run files may have
MODELS = {
    "indicbert_hi": {"id": "neuronsbyisshu/indicbert-hi-ner-naamapadam", "licence": "CC0-1.0", "open": True,
                     "aliases": ["indicbert_hi", "indicbert_hi_naamapadam"]},
    "indicner": {"id": "ai4bharat/IndicNER", "licence": "MIT", "open": True, "aliases": ["indicner"]},
    "davlan_multi": {"id": "Davlan/xlm-roberta-base-wikiann-ner", "licence": "none declared", "open": False,
                     "aliases": ["davlan_multi", "current_multi"]},
}
# the app uses these today; the other languages are not offered yet
# the app uses these today: one open model for nine languages (Odia and Assamese are not offered: 0.27 and a 51-sentence test)
OFFERED = {l: "indicbert_hi" for l in ("hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa")}
NATIVE_TEST = {}                                  # language -> True once a test made by native speakers exists (none yet)
TRANSLATION = {"hi": {"status": "beta", "note": "offline IndicTrans2 scored chrF++ 35.2 against Azure's 36.0 on 40 WMT24++ segments (a small sample)"}}
AZURE_TESTED = {"hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa", "or", "as"}      # a test sentence was translated correctly on 2026-11-17


def test_set(lang):
    return ("naamapadam", "Naamapadam Hindi gold test") if lang == "hi" else (f"nm_{lang}", f"Naamapadam {lang} gold test")


def best_file(dataset, lang, aliases):
    """The run file with the most sentences among all files for this model."""
    best, size = None, -1
    for alias in aliases:
        for path in RUNS.glob(f"{dataset}__{lang}__{alias}*.jsonl"):
            stem = path.stem[len(f"{dataset}__{lang}__{alias}"):]
            if stem and not stem.startswith("__n"):
                continue                                    # a different model whose name starts with this alias
            n = sum(1 for _ in open(path, encoding="utf-8"))
            if n > size:
                best, size = path, n
    return best


def measure(path):
    counts = counts_per_sentence(path)
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    lo, hi = interval(counts)
    return {"f1": round(summarize(rows)["strict"][2], 3), "ci": [round(lo, 3), round(hi, 3)], "sentences": len(rows)}


def main():
    use_console_utf8()
    out = []
    for rank, (lang, name) in enumerate(LANGUAGES, 1):
        dataset, label = test_set(lang)
        results = {}
        for key, m in MODELS.items():
            path = best_file(dataset, lang, m["aliases"])
            if path:
                results[key] = {**measure(path), "model": m["id"], "licence": m["licence"], "open": m["open"]}
        open_results = {k: v for k, v in results.items() if v["open"]}
        chosen = max(open_results, key=lambda k: open_results[k]["f1"]) if open_results else None
        offered = OFFERED.get(lang)
        shown = offered if offered in results else chosen
        if shown:
            r = results[shown]
            status, reasons = rules.ner_status(r["f1"], r["ci"][0], r["sentences"], r["open"], NATIVE_TEST.get(lang, False))
        else:
            r, status, reasons = None, ("blocked" if results else "not_measured"), (["only models without an open licence were measured"] if results else ["no test result yet"])
        entry = {
            "code": lang, "name": name, "priority": rank, "offered_in_app": lang in OFFERED,
            "ner": {"status": status, "reasons": reasons, "test_set": label,
                    "model": r["model"] if r else None, "licence": r["licence"] if r else None, "f1": r["f1"] if r else None,
                    "ci": r["ci"] if r else None, "sentences": r["sentences"] if r else 0,
                    "best_open_candidate": chosen, "alternatives": [{"key": k, **v} for k, v in results.items() if k != shown]},
            "translation": TRANSLATION.get(lang, {"status": "not_evaluated",
                                                  "note": "Azure translated a test sentence correctly; offline IndicTrans2 has not been enabled or measured for this language"}),
            "azure_translation_works": lang in AZURE_TESTED,
        }
        out.append(entry)
        print(f"{rank:2d} {name:10s} {status:13s} " + (f"{r['model'].split('/')[-1]:42s} F1 {r['f1']:.3f} {r['ci']} n={r['sentences']}" if r else "-"))
    payload = {"generated": date.today().isoformat(), "thresholds": rules.THRESHOLDS, "languages": out}
    path = HERE.parent / "backend" / "language_status.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
