"""NER on LONG documents: does the way text is cut up before the model (chunks vs overlapping windows) change accuracy?

The public test sets are single sentences, so they never reach a chunk boundary. This builds documents by joining N
consecutive test sentences with a space (gold entities keep their character offsets), runs the app's real run_ner() on each
document and scores entities by exact character span and type (PER / ORG / LOC). Run it once per setting and compare:

    python evaluation/eval_long_docs.py --dataset fin --sentences 15                         # sentence-aware 800-char chunks
    NER_STRIDE_TOKENS=64 python evaluation/eval_long_docs.py --dataset fin --sentences 15    # overlapping 512-token windows
Per-document predictions are saved to evaluation/runs/long__<dataset>__<setting>.jsonl so bootstrap-style comparisons are possible.
"""
import argparse
import json
import os
import time

from common import load_backend, use_console_utf8
from eval_ner import DATASETS, KEEP, build, load_split
from run_all import prf, variant


def make_documents(dataset, sentences_per_doc, max_docs):
    sentences = load_split(dataset, "en", DATASETS[dataset][2][1], None)
    docs = []
    for i in range(0, len(sentences) - sentences_per_doc + 1, sentences_per_doc):
        text, gold = "", set()
        for tokens, tags in sentences[i:i + sentences_per_doc]:
            sent, offsets, spans = build(tokens, tags, "en")
            base = len(text) + (1 if text else 0)
            text += (" " if text else "") + sent
            for t, a, b in spans:
                if t in KEEP:
                    gold.add((t, base + offsets[a][0], base + offsets[b][1]))
        docs.append((text, gold))
        if len(docs) >= max_docs:
            break
    return docs


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fin")
    ap.add_argument("--sentences", type=int, default=15, help="sentences per document")
    ap.add_argument("--max-docs", type=int, default=60)
    args = ap.parse_args()

    app = load_backend()
    docs = make_documents(args.dataset, args.sentences, args.max_docs)
    stride = os.getenv("NER_STRIDE_TOKENS", "0")
    setting = ("windows" + stride if stride not in ("", "0") else "chunks" + (os.getenv("NER_CHUNK_CHARS") or "800")) + variant()
    tp = fp = fn = 0
    chars = 0
    out = []
    t0 = time.time()
    for text, gold in docs:
        preds = {(e["entity_group"], e["start"], e["end"]) for e in app.run_ner(text, "en") if e["entity_group"] in KEEP}
        tp += len(gold & preds)
        fp += len(preds - gold)
        fn += len(gold - preds)
        chars += len(text)
        out.append({"chars": len(text), "gold": sorted(gold), "pred": sorted(preds)})
    p, r, f = prf(tp, fp, fn)
    path = os.path.join(os.path.dirname(__file__), "runs", f"long__{args.dataset}__{setting}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"{args.dataset} {setting}: {len(docs)} documents, average {chars // len(docs)} chars | "
          f"P {p:.3f} R {r:.3f} F1 {f:.3f} (TP {tp} FP {fp} FN {fn}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
