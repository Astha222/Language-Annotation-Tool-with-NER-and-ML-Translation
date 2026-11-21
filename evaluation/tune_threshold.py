"""Choose the score threshold of a zero-shot model (GLiNER, Otter) on a TUNING split, then apply it to test runs, without running the model again.

Run the model once with a low threshold (GLINER_THRESHOLD=0.1) on the tuning sets and the test sets; every saved prediction keeps its score.
This script sweeps thresholds on the tuning runs (pooled micro-F1), picks the best, and reports every test set at the default 0.5 and at the
chosen threshold, with 95% bootstrap intervals and the paired difference. The tuning sets are MultiCoNER TRAIN samples (mc_de_tune, ...): our
MultiCoNER test scores use the dev split, so tuning there would be tuning on the test.

    python evaluation/tune_threshold.py --runs runs --model gliner_x_base --tune mc_de_tune,mc_es_tune,mc_fr_tune --test mc_de,mc_es,mc_fr,uner_de,wikiann_fr,wikiann_es
Test names "wikiann_fr" etc. refer to run files wikiann__fr__<model>*.jsonl.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from bootstrap import compare, counts_per_sentence, f1, interval
from common import use_console_utf8
from run_all import KEEP

GRID = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]


def run_file(runs, dataset, model):
    if dataset.startswith("wikiann_"):
        pattern = f"wikiann__{dataset.split('_', 1)[1]}__{model}*.jsonl"
    else:
        pattern = f"{dataset}__*__{model}.jsonl"
    found = sorted(glob.glob(str(Path(runs) / pattern)))
    return found[0] if found else None


def min_score_in(path):
    scores = [s for line in open(path, encoding="utf-8") if line.strip() for t, a, b, s in json.loads(line)["pred"] if t in KEEP]
    return min(scores) if scores else 1.0


def choose(runs, model, tune_sets):
    files = [run_file(runs, d, model) for d in tune_sets]
    files = [f for f in files if f]
    if not files:
        raise SystemExit(f"no tuning runs for {model}")
    lowest = max(min_score_in(f) for f in files)
    table = []
    for t in GRID:
        if t + 1e-9 < lowest:
            continue                                   # predictions below this score were never saved
        pooled = np.vstack([counts_per_sentence(f, t) for f in files])
        table.append((t, f1(pooled)))
    best = max(table, key=lambda x: x[1])
    return best[0], table, lowest


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tune", default="mc_de_tune,mc_es_tune,mc_fr_tune")
    ap.add_argument("--test", required=True)
    ap.add_argument("--default", type=float, default=0.5)
    args = ap.parse_args()
    t_best, table, lowest = choose(args.runs, args.model, args.tune.split(","))
    print(f"{args.model}: lowest saved score {lowest:.2f}; tuning F1 by threshold: " + ", ".join(f"{t:.2f}:{v:.3f}" for t, v in table))
    print(f"chosen threshold {t_best:.2f} (tuning sets: {args.tune})")
    print(f"| test set | sentences | F1 at {args.default} [95% CI] | F1 at {t_best:.2f} [95% CI] | difference [95% CI] |")
    print("|---|---|---|---|---|")
    for ds in args.test.split(","):
        f = run_file(args.runs, ds, args.model)
        if not f:
            print(f"| {ds} | - | no run | | |")
            continue
        a, b = counts_per_sentence(f, args.default), counts_per_sentence(f, t_best)
        r = compare(b, a)
        lo_a, hi_a = interval(a)
        lo_b, hi_b = interval(b)
        print(f"| {ds} | {len(a)} | {f1(a):.3f} [{lo_a:.3f}, {hi_a:.3f}] | {f1(b):.3f} [{lo_b:.3f}, {hi_b:.3f}] | {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] |")


if __name__ == "__main__":
    main()
