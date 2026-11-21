"""WS0 latency runner: how long does NER take, measured the same way every time?

Protocol (evaluation plan, section 8.5), with the knobs exposed so a slow machine can run a shorter version:
  * fixed corpus built from the frozen WNUT-2017 + BTC test sentences: short (~200 chars), medium (~2,000), long (~20,000)
  * load the model and record cold-start time, then warm-up calls, then time every call with time.perf_counter()
  * threads fixed explicitly, peak memory via psutil, model size on disk, two batch settings (1 = interactive, 32 = documents)
  * the whole run can be repeated; the median of each statistic is reported
What is timed is the app's real path (app.run_ner: chunking, model, entity clean-up), so the number is what a user waits.

    python evaluation/bench_latency.py --models app --sizes short,medium --reps 30
    python evaluation/bench_latency.py --models app --sizes short,medium,long --reps 100 --repeats 3   # full protocol
    python evaluation/bench_latency.py --lang hi --models app --sizes short,medium,long --reps 100 --repeats 3  # Hindi (WS3, E3)
With --lang hi the texts are built from the frozen Naamapadam + HiNER Hindi test sentences and the app's Hindi model (NER_MODEL_HI) is timed;
a "name=path" model then replaces the Hindi model instead of the English one. Result files get a __hi suffix.
Results are written to evaluation/runs/latency__<model>__b<batch>.json for pareto.py (with a __t<threads> suffix when
the thread count is not the number of physical cores).
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import psutil

from common import DATA, load_backend, use_console_utf8

RUNS = Path(__file__).resolve().parent / "runs"
FREE_AT_START = None
TARGET = {"short": 200, "medium": 2000, "long": 20000}


SOURCES = {"en": [("ner/wnut2017/en_test.jsonl", "tokens"), ("ner/btc/en_test.jsonl", "tokens")],
           "hi": [("ner_hi/naamapadam/hi_test.json", "words"), ("ner_hi/hiner/hi_test.json", "words")]}
LANG = {"value": "en"}


def corpus(size, count):
    """Deterministic texts of about TARGET[size] characters, made of consecutive frozen test sentences of the chosen language."""
    sentences = []
    for name, field in SOURCES[LANG["value"]]:
        for line in (DATA / name).read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)[field]:
                sentences.append(" ".join(json.loads(line)[field]))
    texts, pos = [], 0
    for _ in range(count):
        text = ""
        while len(text) < TARGET[size]:
            text += sentences[pos % len(sentences)] + " "
            pos += 1
        texts.append(text.strip())
    return texts


def model_disk_gb(model_id):
    try:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(model_id, local_files_only=True))
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e9
    except Exception:
        return None


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def one_pass(app, size, reps, warmup, batch, proc):
    texts = corpus(size, reps)
    lang = LANG["value"]
    for t in corpus("short", warmup):
        app.run_ner(t, lang, batch_size=batch)
    times, peak = [], 0
    for t in texts:
        t0 = time.perf_counter()
        app.run_ner(t, lang, batch_size=batch)
        times.append(time.perf_counter() - t0)
        peak = max(peak, proc.memory_info().rss)
    chars = sum(len(t) for t in texts)
    return {"p50_ms": pct(times, 0.5) * 1000, "p95_ms": pct(times, 0.95) * 1000,
            "chars_per_s": chars / sum(times), "peak_rss_gb": peak / 1e9}


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="app")
    ap.add_argument("--sizes", default="short,medium")
    ap.add_argument("--reps", type=int, default=30, help="timed calls per size (plan: 100)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=1, help="repeat the whole run; medians are reported (plan: 3)")
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--no-save", action="store_true", help="print only; do not write a result file (for experiments)")
    ap.add_argument("--threads", type=int, default=psutil.cpu_count(logical=False))
    ap.add_argument("--merge", action="store_true",
                    help="add these sizes to an existing result file of the same model/batch/threads (run the sizes one at a time on a slow machine)")
    ap.add_argument("--lang", default="en", choices=sorted(SOURCES), help="language of the texts and of the app model that is timed")
    args = ap.parse_args()
    LANG["value"] = lang = args.lang

    import torch
    torch.set_num_threads(args.threads)
    app = load_backend()
    proc = psutil.Process()
    global FREE_AT_START
    FREE_AT_START = round(psutil.virtual_memory().available / 1e9, 2)
    RUNS.mkdir(exist_ok=True)
    print(f"threads={args.threads} physical_cores={psutil.cpu_count(logical=False)} ram={psutil.virtual_memory().total / 1e9:.1f} GB "
          f"torch={torch.__version__}", flush=True)
    for spec in args.models.split(","):
        given, _, model = spec.rpartition("=") if "=" in spec else ("", "", spec)         # "name=path" names a local model folder
        if model != "app":                        # get_ner_pipeline() reads these, so run_ner now uses the candidate
            if lang == "hi":
                app.NER_MODEL_HI = model
            else:
                app.NER_MODEL_EN = model
        model_id = app.ner_model_name(lang)
        from run_all import variant
        label = given or (model_id.split("/")[-1] + (variant() if model == "app" else ""))
        t0 = time.perf_counter()
        app.get_ner_pipeline(lang)
        app.run_ner("Warm up call for Rahul in Delhi." if lang == "en" else "राहुल दिल्ली में है।", lang)
        cold = time.perf_counter() - t0
        disk = model_disk_gb(model_id)
        for batch in (int(b) for b in args.batches.split(",")):
            report = {"model": model_id, "lang": lang, "runtime": ("ONNX " + label.rsplit("-", 1)[-1].upper() if "-onnx-" in label else "ONNX") if "onnx" in label.lower() else "PyTorch fp32", "threads": args.threads, "batch": batch,
                      "cold_start_s": cold, "disk_gb": disk, "reps": args.reps, "repeats": args.repeats, "sizes": {}}
            for size in args.sizes.split(","):
                passes = [one_pass(app, size, args.reps if size != "long" else max(5, args.reps // 5), args.warmup, batch, proc)
                          for _ in range(args.repeats)]
                report["sizes"][size] = {k: statistics.median(p[k] for p in passes) for k in passes[0]}
                s = report["sizes"][size]
                print(f"{model_id} batch={batch} {size:6s} p50 {s['p50_ms']:8.0f} ms  p95 {s['p95_ms']:8.0f} ms  "
                      f"{s['chars_per_s']:7.0f} chars/s  peak {s['peak_rss_gb']:.2f} GB", flush=True)
            # a run with a non-default thread count gets its own file, so experiments never replace the baseline
            tag = "" if args.threads == psutil.cpu_count(logical=False) else f"__t{args.threads}"
            out = RUNS / f"latency__{label}__b{batch}{tag}{'' if lang == 'en' else '__' + lang}.json"
            report["free_ram_gb_at_start"] = FREE_AT_START
            if args.merge and out.exists():
                old = json.loads(out.read_text(encoding="utf-8"))
                if (old.get("model"), old.get("threads"), old.get("batch")) == (model_id, args.threads, batch):
                    report["sizes"] = {**old.get("sizes", {}), **report["sizes"]}
                    report["cold_start_s"] = old.get("cold_start_s", cold)
            if not args.no_save:
                out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("cold start (load + first call):", f"{cold:.1f}s", "| model on disk:", f"{disk:.2f} GB" if disk else "unknown")


if __name__ == "__main__":
    main()
