"""WS0 accuracy runner: score one or more NER models on the frozen test sets and keep every prediction.

For each (dataset, model) it writes   evaluation/runs/<dataset>__<lang>__<model>.jsonl   (one line per sentence: text,
gold spans, predicted spans with scores) and prints precision / recall / F1 with the numbers the plan asks for:
  * strict    entity-level match: type AND both boundaries (the headline number, same rule as eval_ner.py)
  * partial   the type matches and the spans overlap (boundary errors stop counting as misses)
  * per type  P / R / F1 for PER, ORG and LOC separately

Because predictions are saved, any two runs can be compared later (bootstrap.py, error_analysis.py, pareto.py) without
running a model again.

    python evaluation/run_all.py --models app --datasets wnut2017,btc,fin --n 500   # the sample used in the reports
    python evaluation/run_all.py --models dslim/bert-base-NER --full               # every sentence of every test split
    python evaluation/run_all.py --models app --datasets btc --n 50 --seqeval-check
"""
import argparse
import json
import os
import time
from pathlib import Path

from common import DATA, load_backend, use_console_utf8
from download_data import MANIFEST, sha256
from eval_ner import DATASETS, KEEP, LOCAL, load_split, make_pipeline, predict

RUNS = Path(__file__).resolve().parent / "runs"
TYPES = ["PER", "ORG", "LOC"]


def variant():
    """Name suffix for runs made with the ONNX build of the app's English model (NER_ONNX_DIR set)."""
    if not os.getenv("NER_ONNX_DIR", "").strip():
        return ""
    return "-onnx-fp32" if os.getenv("NER_ONNX_FILE", "model_int8.onnx") == "model.onnx" else "-onnx-int8"


def suffix(n):
    """File-name suffix recording how many sentences a run covers, so a small check never replaces a bigger run."""
    return f"__n{n}" if n else ""


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def kept(preds, min_score=0.0):
    return {(t, a, b) for t, a, b, s in preds if t in KEEP and s >= min_score}


def strict_counts(sent_gold, sent_preds, only=None):
    tp = fp = fn = 0
    for gold, preds in zip(sent_gold, sent_preds):
        g = {x for x in gold if only is None or x[0] == only}
        p = {x for x in preds if only is None or x[0] == only}
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return tp, fp, fn


def partial_counts(sent_gold, sent_preds):
    """Greedy one-to-one matching of predictions to gold spans of the same type that overlap."""
    tp = fp = fn = 0
    for gold, preds in zip(sent_gold, sent_preds):
        unmatched = set(gold)
        for t, a, b in sorted(preds):
            hit = next((g for g in sorted(unmatched) if g[0] == t and g[1] <= b and a <= g[2]), None)
            if hit:
                unmatched.discard(hit)
                tp += 1
            else:
                fp += 1
        fn += len(unmatched)
    return tp, fp, fn


def seqeval_f1(rows):
    """Independent check of the strict score using the seqeval library (token-tag sequences, IOB2)."""
    from seqeval.metrics import f1_score
    y_true, y_pred = [], []
    for r in rows:
        n = len(r["offsets"])
        for spans, out in ((r["gold"], y_true), (r["pred"], y_pred)):
            tags = ["O"] * n
            for t, a, b, *_ in sorted(spans, key=lambda x: x[1]):
                if t not in KEEP or any(tags[i] != "O" for i in range(a, b + 1)):
                    continue
                tags[a] = f"B-{t}"
                for i in range(a + 1, b + 1):
                    tags[i] = f"I-{t}"
            out.append(tags)
    return f1_score(y_true, y_pred)


def summarize(rows, min_score=0.0):
    gold = [{tuple(g) for g in r["gold"]} for r in rows]
    pred = [{(t, a, b) for t, a, b, s in r["pred"] if t in KEEP and s >= min_score} for r in rows]
    out = {"sentences": len(rows), "gold": sum(len(g) for g in gold)}
    out["strict"] = prf(*strict_counts(gold, pred))
    out["partial"] = prf(*partial_counts(gold, pred))
    out["per_type"] = {t: prf(*strict_counts(gold, pred, only=t)) for t in TYPES}
    return out


def verify_manifest(dataset, lang, split):
    if dataset == "hinglish_deva":              # derived from the frozen hinglish set by make_hinglish_deva.py, not frozen itself
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for e in manifest["files"]:
        if (e["dataset"], e["lang"], e["split"]) == (dataset, lang, split):
            if sha256(DATA / e["file"]) != e["sha256"]:
                raise SystemExit(f"{e['file']} no longer matches MANIFEST.json; re-run download_data.py --verify")
            return
    raise SystemExit(f"{dataset}/{lang}/{split} is not in MANIFEST.json; run download_data.py")


def run_model(app, dataset, lang, model, n):
    test_split = LOCAL[dataset][4] if dataset in LOCAL else DATASETS[dataset][2][1]
    verify_manifest(dataset, lang, test_split)
    sentences = load_split(dataset, lang, test_split, n)
    nlp, shown = make_pipeline(app, lang, model)
    t0 = time.time()
    results = predict(app, nlp, lang, sentences)
    seconds = time.time() - t0
    if getattr(nlp, "recovered", 0) or getattr(nlp, "unresolved", 0):       # GLiNER adapter: entities whose character span was missing
        print(f"  note: {dataset}/{lang}: {nlp.recovered} entity spans recovered from their text, {nlp.unresolved} dropped (span not found)", flush=True)
        for word, context in getattr(nlp, "unresolved_examples", [])[:5]:
            print(f"    not placed: {word!r} in {context!r}", flush=True)
        nlp.recovered = nlp.unresolved = 0            # counts are per dataset
        nlp.unresolved_examples = []
    rows = [{"text": text, "offsets": offsets, "gold": sorted(gold), "pred": preds}
            for text, offsets, gold, preds in results]
    return rows, shown, seconds


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="wnut2017,btc,fin")
    ap.add_argument("--langs", default="en", help="only wikiann has more than English")
    ap.add_argument("--models", default="app", help="comma-separated; 'app' = the model the app uses for that language; "
                                                    "'name=path' names a local model folder")
    ap.add_argument("--n", type=int, default=None, help="first N sentences of each test split (default: all)")
    ap.add_argument("--full", action="store_true", help="every sentence (the default; kept to match the plan's wording)")
    ap.add_argument("--seqeval-check", action="store_true", help="also print the strict F1 as computed by seqeval")
    args = ap.parse_args()
    n = None if args.full else args.n

    RUNS.mkdir(exist_ok=True)
    specs = [m.strip().rpartition("=")[2] for m in args.models.split(",")]
    app = load_backend() if "app" in specs else None            # the web app is only needed to evaluate the app's own model
    lines = ["| dataset | model | sentences | strict P | strict R | strict F1 [95% CI] | partial F1 | PER F1 | ORG F1 | LOC F1 | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dataset in (d.strip() for d in args.datasets.split(",")):
        for lang in ([l.strip() for l in args.langs.split(",")] if dataset == "wikiann" else [LOCAL[dataset][0]] if dataset in LOCAL else ["en"]):
            for spec in (m.strip() for m in args.models.split(",")):
                label, _, model = spec.rpartition("=") if "=" in spec else ("", "", spec)        # "name=path" gives a folder a readable name
                try:
                    rows, shown, seconds = run_model(app, dataset, lang, model, n)
                    if label:
                        shown = label
                except Exception as e:                                      # one failure must not stop the rest
                    print(f"{dataset}/{lang}/{spec}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
                    continue
                name = shown.split("/")[-1] + (variant() if model == "app" and lang == "en" and not label else "")
                out_path = RUNS / f"{dataset}__{lang}__{name}{suffix(n)}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                s = summarize(rows)
                from bootstrap import counts_per_sentence, interval               # 95% interval of this run's own F1
                lo, hi = interval(counts_per_sentence(out_path))
                extra = f"  seqeval {seqeval_f1(rows):.3f}" if args.seqeval_check else ""
                print(f"{dataset}/{lang}/{name}: strict F1 {s['strict'][2]:.3f} [{lo:.3f}, {hi:.3f}] partial {s['partial'][2]:.3f} "
                      f"({s['sentences']} sentences, {seconds:.0f}s){extra}", flush=True)
                lines.append(f"| {dataset} | `{name}` | {s['sentences']} | {s['strict'][0]:.3f} | {s['strict'][1]:.3f} | "
                             f"{s['strict'][2]:.3f} [{lo:.3f}, {hi:.3f}] | {s['partial'][2]:.3f} | " +
                             " | ".join(f"{s['per_type'][t][2]:.3f}" for t in TYPES) + f" | {seconds:.0f} |")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
