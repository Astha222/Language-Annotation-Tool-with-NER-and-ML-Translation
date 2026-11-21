"""Run the planned WS2 distillation experiments one after another (used by the Colab notebook, also works locally).

    python colab_runner.py --drive-dir /content/drive/MyDrive/ws2_results          # all experiments
    python colab_runner.py --only A_distilroberta_distill --dry-run                # print the command only

Each finished experiment is copied to --drive-dir at once, so a disconnected session loses at most the run in progress.
A summary table of the best dev F1 of every experiment is printed at the end (the dev set is only for choosing checkpoints;
the real accuracy is measured later on the laptop, on the test sets).
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

TEACHER = "Jean-Baptiste/roberta-large-ner-english"
DEBERTA = "microsoft/deberta-v3-small"
STUDENT = "distilroberta-base"

# name, student, training sources, alpha (1.0 = no teacher), why
RUNS = [
    ("A_distilroberta_distill", STUDENT, "all", 0.5, "main experiment: gold labels + teacher, all four datasets"),
    ("B_distilroberta_plain", STUDENT, "all", 1.0, "ablation: same data, NO teacher (shows what the teacher adds)"),
    ("C_distilroberta_zeroshot", STUDENT, "conll2003", 0.5, "zero-shot variant: trained on news only, so Twitter/WNUT/FIN stay unseen"),
    ("D_deberta_distill", DEBERTA, "all", 0.5, "the plan's first-choice backbone"),
    # licence-safe variants (2026-11-18): FIN's original licence is unverified, so neither A nor B can ship. Same recipes without FIN;
    # run them with a dev file that has no FIN sentences either (dev_mix_nofin.jsonl), so FIN is not used even to pick the checkpoint.
    ("A_noFIN_distill", STUDENT, "conll2003,wnut2017,btc", 0.5, "licence-safe A: gold labels + teacher, without FIN"),
    ("B_noFIN_plain", STUDENT, "conll2003,wnut2017,btc", 1.0, "licence-safe B: no teacher, without FIN (B was best on English-EWT)"),
]


def command(name, student, sources, alpha, args, seed=0):
    cmd = [sys.executable, "distill.py", "--teacher", TEACHER, "--student", student,
           "--train", args.train, "--dev", args.dev, "--out", str(Path(args.out_dir) / name),
           "--sources", sources, "--alpha", str(alpha), "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--lr", str(args.lr), "--seed", str(seed), "--teacher-cache", str(Path(args.out_dir) / f"teacher_{sources}.pt")]
    return cmd + shlex.split(args.extra)


def summary(out_dir):
    print("\n=== summary (best dev F1 = mean of WNUT / BTC / FIN dev; NOT test scores) ===")
    for log in sorted(Path(out_dir).glob("*/training_log.json")):
        meta = json.loads(log.read_text(encoding="utf-8"))
        last = meta["log"][-1]["dev_f1"] if meta["log"] else {}
        print(f"{log.parent.name:28s} best dev {meta['best_dev_mean_f1']:.4f}  last {({k: round(v, 3) for k, v in last.items()})}  "
              f"{meta['seconds'] / 60:.0f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="train_mix.jsonl")
    ap.add_argument("--dev", default="dev_mix.jsonl")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--drive-dir", help="copy every finished experiment here (e.g. a Google Drive folder)")
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2 (names get a _s<seed> suffix when more than one)")
    ap.add_argument("--extra", default="", help="extra arguments passed to distill.py (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    seeds = [int(s) for s in args.seeds.split(",")]
    for name, student, sources, alpha, why in RUNS:
        if wanted and name not in wanted:
            continue
        if not wanted and "noFIN" in name:              # added later; run only when asked for by name (the WS2 notebook runs everything)
            continue
        for seed in seeds:
            run_name = name if len(seeds) == 1 else f"{name}_s{seed}"
            cmd = command(run_name, student, sources, alpha, args, seed)
            print(f"\n##### {run_name}: {why}\n{' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            t0 = time.time()
            if subprocess.run(cmd).returncode != 0:
                print(f"!!! {run_name} failed; continuing with the next experiment", flush=True)
                continue
            print(f"##### {run_name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
            if args.drive_dir:
                dest = Path(args.drive_dir) / run_name
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(Path(args.out_dir) / run_name, dest)
                print(f"copied to {dest}", flush=True)
    if not args.dry_run:
        summary(args.out_dir)


if __name__ == "__main__":
    main()
