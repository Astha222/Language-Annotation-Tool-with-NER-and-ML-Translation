"""Compare two saved runs of the same test sets sentence by sentence (for example PyTorch against an ONNX build of the same model).

    python evaluation/compare_runs.py B_noFIN_plain_s1 B_noFIN_s1_onnx_fp32 --datasets wnut2017,btc,fin,uner_en_ewt

For each dataset it reads runs/<dataset>__en__<name>.jsonl for both names and reports: how many sentences have a different set of predicted
entities (type and word span; ALL types, MISC included, because the app shows them), the strict F1 of each (PER/ORG/LOC, the usual scoring),
the F1 difference, and the largest score difference on entities both runs predicted. Gate (D6, the plan's rule for speed-ups): identical
predictions, or a strict F1 change of at most 0.01 on every dataset.
"""
import argparse
import json
from pathlib import Path

from eval_ner import KEEP

RUNS = Path(__file__).resolve().parent / "runs"
GATE = 0.01


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strict_f1(rows):
    tp = fp = fn = 0
    for r in rows:
        gold = {tuple(g[:3]) for g in r["gold"] if g[0] in KEEP}
        pred = {tuple(p[:3]) for p in r["pred"] if p[0] in KEEP}
        tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * rec / (p + rec) if p + rec else 0.0


def compare(a_rows, b_rows):
    if len(a_rows) != len(b_rows) or any(x["text"] != y["text"] for x, y in zip(a_rows, b_rows)):
        raise ValueError("the two runs do not cover the same sentences in the same order")
    differing, max_score_diff = 0, 0.0
    for x, y in zip(a_rows, b_rows):
        sx = {tuple(p[:3]): p[3] for p in x["pred"]}
        sy = {tuple(p[:3]): p[3] for p in y["pred"]}
        differing += set(sx) != set(sy)
        for key in set(sx) & set(sy):
            max_score_diff = max(max_score_diff, abs(sx[key] - sy[key]))
    fa, fb = strict_f1(a_rows), strict_f1(b_rows)
    return {"sentences": len(a_rows), "differing": differing, "f1_a": fa, "f1_b": fb, "delta": fb - fa,
            "max_score_diff": max_score_diff, "passes": differing == 0 or abs(fb - fa) <= GATE}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference run name, e.g. B_noFIN_plain_s1")
    ap.add_argument("b", help="run to check, e.g. B_noFIN_s1_onnx_fp32")
    ap.add_argument("--datasets", default="wnut2017,btc,fin,uner_en_ewt")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    print(f"| dataset | sentences | sentences with different entities | strict F1 {args.a} | strict F1 {args.b} | difference | largest score difference | gate |")
    print("|---|---|---|---|---|---|---|---|")
    ok = True
    for ds in args.datasets.split(","):
        r = compare(read(RUNS / f"{ds}__{args.lang}__{args.a}.jsonl"), read(RUNS / f"{ds}__{args.lang}__{args.b}.jsonl"))
        ok &= r["passes"]
        print(f"| {ds} | {r['sentences']} | {r['differing']} | {r['f1_a']:.3f} | {r['f1_b']:.3f} | {r['delta']:+.4f} | {r['max_score_diff']:.4f} | "
              f"{'pass' if r['passes'] else 'FAIL'} |")
    print("gate:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
