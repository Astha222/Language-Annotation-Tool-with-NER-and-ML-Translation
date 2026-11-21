"""Where exactly does a model go wrong? Turn a saved run into a CSV with one row per gold or predicted span.

error_kind is one of
  correct         the prediction equals a gold span (type and boundaries)
  missed          a gold span with no prediction overlapping it
  wrong_type      a prediction covers the same tokens as a gold span but has another type
  wrong_boundary  a prediction of the right type overlaps a gold span but with different boundaries
  spurious        a prediction that overlaps no gold span

found_elsewhere_in_doc (on missed / wrong_type rows) says whether the SAME surface text was predicted correctly in
another sentence of the run. That share is how much a document-level consistency sweep (WS4) could recover without
any new model. The public NER sets are sentence-level, so the "document" here is the whole saved run; on real uploads
it will be one document.

    python evaluation/error_analysis.py runs/btc__en__bert-base-NER.jsonl
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import use_console_utf8
from run_all import KEEP


def surface(row, a, b):
    return row["text"][row["offsets"][a][0]:row["offsets"][b][1]]


def analyse(rows):
    correct_forms = set()                    # surface forms predicted correctly somewhere
    for r in rows:
        gold = {tuple(g) for g in r["gold"]}
        for t, a, b, s in r["pred"]:
            if (t, a, b) in gold:
                correct_forms.add(surface(r, a, b).lower())
    out = []
    for i, r in enumerate(rows):
        gold = sorted(tuple(g) for g in r["gold"])
        preds = sorted((t, a, b, s) for t, a, b, s in r["pred"] if t in KEEP)
        used = set()
        for gt, ga, gb in gold:
            text = surface(r, ga, gb)
            exact = next((p for p in preds if (p[0], p[1], p[2]) == (gt, ga, gb)), None)
            if exact:
                used.add(exact)
                out.append((i, text, gt, text, gt, "correct", ""))
                continue
            overlap = [p for p in preds if p[1] <= gb and ga <= p[2]]
            same_span = next((p for p in overlap if (p[1], p[2]) == (ga, gb)), None)
            same_type = next((p for p in overlap if p[0] == gt), None)
            elsewhere = text.lower() in correct_forms
            if same_span:
                used.add(same_span)
                out.append((i, text, gt, surface(r, same_span[1], same_span[2]), same_span[0], "wrong_type", elsewhere))
            elif same_type:
                used.add(same_type)
                out.append((i, text, gt, surface(r, same_type[1], same_type[2]), same_type[0], "wrong_boundary", elsewhere))
            else:
                out.append((i, text, gt, "", "", "missed", elsewhere))
        for p in preds:
            if p not in used and not any(p[1] <= g[2] and g[1] <= p[2] for g in gold):
                out.append((i, "", "", surface(r, p[1], p[2]), p[0], "spurious", ""))
    return out


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", help="CSV path (default: next to the run file, .errors.csv)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.run, encoding="utf-8") if line.strip()]
    result = analyse(rows)
    dest = Path(args.out) if args.out else Path(args.run).with_suffix(".errors.csv")
    with open(dest, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "gold_text", "gold_type", "pred_text", "pred_type", "error_kind", "found_elsewhere_in_doc"])
        w.writerows(result)
    kinds = Counter(r[5] for r in result)
    gold_total = kinds["correct"] + kinds["missed"] + kinds["wrong_type"] + kinds["wrong_boundary"]
    print(f"{dest.name}: {gold_total} gold entities")
    for k in ("correct", "missed", "wrong_type", "wrong_boundary", "spurious"):
        print(f"  {k:15s} {kinds[k]:5d}" + (f"  ({kinds[k] / gold_total:.0%} of gold)" if k != "spurious" and gold_total else ""))
    recoverable = sum(1 for r in result if r[5] in ("missed", "wrong_type", "wrong_boundary") and r[6] is True)
    errors = kinds["missed"] + kinds["wrong_type"] + kinds["wrong_boundary"]
    if errors:
        print(f"  of the {errors} gold entities not found correctly, {recoverable} ({recoverable / errors:.0%}) "
              f"were found correctly elsewhere in the run (what a consistency sweep could recover)")


if __name__ == "__main__":
    main()
