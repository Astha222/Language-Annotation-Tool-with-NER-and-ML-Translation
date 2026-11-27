"""Is system A really better than system B? Paired bootstrap and permutation test on two saved runs.

Both runs must come from run_all.py on the same dataset and the same sentences. Sentences are resampled with
replacement (the same resample for both systems), the strict micro-F1 difference A - B is recomputed each time, and
the 2.5th / 97.5th percentiles give a 95% confidence interval. A "wins" only if the whole interval is above zero.
For small test sets (FIN has 305 sentences) the bootstrap is less reliable, so an approximate randomization test
(swap each sentence's two outputs at random) gives a p-value as well.

    python evaluation/bootstrap.py runs/btc__en__roberta-large-ner-english.jsonl runs/btc__en__bert-base-NER.jsonl
"""
import argparse
import json
import random

import numpy as np

from common import use_console_utf8
from run_all import KEEP


def counts_per_sentence(path, min_score=0.0):
    """One row per sentence: (tp, fp, fn) under strict matching; predictions scoring below min_score are ignored (threshold tuning)."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    out = np.zeros((len(rows), 3), dtype=np.int64)
    for i, r in enumerate(rows):
        gold = {tuple(g) for g in r["gold"]}
        pred = {(t, a, b) for t, a, b, s in r["pred"] if t in KEEP and s >= min_score}
        out[i] = (len(gold & pred), len(pred - gold), len(gold - pred))
    return out


def counts_per_sentence_rows(counts):
    """The same array counts_per_sentence() builds, from a list of (tp, fp, fn) tuples (for subsets of a run)."""
    return np.array(counts, dtype=np.int64).reshape(-1, 3)


def f1(c):
    tp, fp, fn = c.sum(axis=0)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def compare(a, b, resamples=1000, seed=0):
    if len(a) != len(b):
        raise SystemExit(f"runs have different sentence counts ({len(a)} vs {len(b)}); they must cover the same sentences")
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(resamples)
    for k in range(resamples):
        idx = rng.integers(0, n, n)
        diffs[k] = f1(a[idx]) - f1(b[idx])
    low, high = np.percentile(diffs, [2.5, 97.5])
    observed = f1(a) - f1(b)
    # approximate randomization: swap the two systems' outputs sentence by sentence
    count = 0
    swaps = 5000
    for _ in range(swaps):
        flip = rng.random(n) < 0.5
        x = np.where(flip[:, None], b, a)
        y = np.where(flip[:, None], a, b)
        count += abs(f1(x) - f1(y)) >= abs(observed) - 1e-12
    return {"f1_a": f1(a), "f1_b": f1(b), "diff": observed, "ci": (float(low), float(high)),
            "p_permutation": (count + 1) / (swaps + 1), "sentences": n}


def interval(c, resamples=1000, seed=0):
    """95% bootstrap confidence interval for one system's own F1."""
    rng = np.random.default_rng(seed)
    n = len(c)
    vals = [f1(c[rng.integers(0, n, n)]) for _ in range(resamples)]
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()
    r = compare(counts_per_sentence(args.run_a), counts_per_sentence(args.run_b), args.resamples)
    verdict = "A is better" if r["ci"][0] > 0 else "B is better" if r["ci"][1] < 0 else "no reliable difference"
    print(f"A {r['f1_a']:.3f}   B {r['f1_b']:.3f}   A-B {r['diff']:+.3f}   95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]   "
          f"permutation p={r['p_permutation']:.4f}   ({r['sentences']} sentences)  ->  {verdict}")


if __name__ == "__main__":
    main()
