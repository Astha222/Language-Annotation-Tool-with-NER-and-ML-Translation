"""WS0 Pareto report: accuracy (F1) against speed (p95 latency), one point per model, so a model is judged on both.

Reads the files written by run_all.py (runs/<dataset>__en__<model>.jsonl) and bench_latency.py
(runs/latency__<model>__b<batch>.json) and writes evaluation/pareto.png plus a text table. A model is on the Pareto
frontier if no other model is both faster and more accurate; only those are candidates.

    python evaluation/pareto.py --dataset btc --size medium --batch 1
"""
import argparse
import json
from pathlib import Path

from common import use_console_utf8
from run_all import summarize, suffix

RUNS = Path(__file__).resolve().parent / "runs"


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="btc")
    ap.add_argument("--n", type=int, default=None, help="use the runs made with --n N (default: full-split runs)")
    ap.add_argument("--size", default="medium", help="latency input size: short, medium or long")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "pareto.png"))
    args = ap.parse_args()

    points = []
    for lat in sorted(RUNS.glob(f"latency__*__b{args.batch}.json")):        # *__t<N>.json experiment files do not match
        report = json.loads(lat.read_text(encoding="utf-8"))
        name = lat.name.split("__")[1]
        run = RUNS / f"{args.dataset}__en__{name}{suffix(args.n)}.jsonl"
        if not run.exists() or args.size not in report["sizes"]:
            continue
        rows = [json.loads(l) for l in run.read_text(encoding="utf-8").splitlines() if l.strip()]
        points.append((name + (" (" + report["runtime"] + ")" if "runtime" in report else ""),
                       report["sizes"][args.size]["p95_ms"], summarize(rows)["strict"][2], len(rows)))
    if not points:
        raise SystemExit("nothing to plot: need both a run_all.py result and a bench_latency.py result for the same model")

    frontier = [p for p in points if not any(q[1] <= p[1] and q[2] >= p[2] and q != p for q in points)]
    print(f"{args.dataset} (F1) vs p95 latency of a {args.size} text, batch {args.batch}")
    for name, ms, f1, n in sorted(points, key=lambda p: p[1]):
        print(f"  {name:55s} F1 {f1:.3f} ({n} sent.)  p95 {ms:8.0f} ms  {'<- frontier' if (name, ms, f1, n) in frontier else ''}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(points, key=lambda p: p[1])
    fig, (ax, side) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for n, (name, ms, f1, _) in enumerate(ordered, 1):
        on = (name, ms, f1, _) in frontier
        ax.scatter(ms, f1, s=140, color="#1d4ed8" if on else "#9ca3af", zorder=3)
        ax.annotate(str(n), (ms, f1), ha="center", va="center", fontsize=7.5, color="white", zorder=4)
    if len(frontier) > 1:
        fs = sorted(frontier, key=lambda p: p[1])
        ax.plot([p[1] for p in fs], [p[2] for p in fs], color="#1d4ed8", linewidth=1, linestyle="--", zorder=2)
    ax.margins(x=0.1, y=0.12)
    ax.set_xlabel(f"p95 latency, {args.size} text, batch {args.batch} (ms, lower is better)")
    ax.set_ylabel(f"{args.dataset} strict F1 (higher is better)")
    ax.set_title("Accuracy against speed (blue = Pareto frontier)")
    ax.grid(alpha=0.3)
    side.axis("off")
    nl = chr(10)
    side.text(0, 1, nl.join(f"{n}  {name}{nl}     F1 {f1:.3f}, p95 {ms:.0f} ms" for n, (name, ms, f1, _) in enumerate(ordered, 1)),
              va="top", fontsize=7.8, family="monospace")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
