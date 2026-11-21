"""Word-level scoring of a token-classification model, the way most NER papers do it (and the Naamapadam / IndicNER paper):
each gold word gets the label predicted for its FIRST sub-word, and the BIO tags are turned into entities with the same rule as the gold.

This bypasses the app's path (Hugging Face pipeline aggregation + entity clean-up), so it separates "what the model knows" from
"what our decoding makes of it". Strict entity F1 with a 95% bootstrap range, same scorer as run_all.py.

    python evaluation/word_level_eval.py --model ai4bharat/IndicNER --datasets naamapadam,nm_bn
"""
import argparse
import json

import torch

import eval_ner
from bootstrap import counts_per_sentence_rows, f1, interval
from common import use_console_utf8
from run_all import KEEP, RUNS


def spans(tags):
    return eval_ner.build(["x"] * len(tags), tags, "hi")[2]


def predict_words(model, tok, words, id2label, batch=16):
    out = []
    for i in range(0, len(words), batch):
        chunk = words[i:i + batch]
        enc = tok(chunk, is_split_into_words=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).logits
        pred = logits.argmax(-1)
        for b, ws in enumerate(chunk):
            labels, seen = ["O"] * len(ws), set()
            for t, w in enumerate(enc.word_ids(b)):
                if w is not None and w not in seen:
                    seen.add(w)
                    labels[w] = id2label[int(pred[b, t])]
            out.append([l if l == "O" or l[2:] in KEEP else "O" for l in labels])
    return out


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", help="short name for the saved file")
    ap.add_argument("--datasets", default="naamapadam")
    args = ap.parse_args()
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
    except ValueError as e:                     # tokenizer saved by a newer transformers: build it from tokenizer.json (as eval_ner does)
        if "TokenizersBackend" not in str(e):
            raise
        tok = eval_ner.fast_tokenizer(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    name = args.name or args.model.split("/")[-1]
    for ds in args.datasets.split(","):
        lang, split = eval_ner.LOCAL[ds][0], eval_ner.LOCAL[ds][4]
        sentences = eval_ner.load_split(ds, lang, split, None)
        preds = predict_words(model, tok, [w for w, _ in sentences], id2label)
        counts, rows = [], []
        for (words, gold_tags), p in zip(sentences, preds):
            g, q = spans(gold_tags), spans(p)
            counts.append((len(g & q), len(q - g), len(g - q)))
            rows.append({"text": " ".join(words), "gold": sorted(g), "pred": [[t, a, b, 1.0] for t, a, b in sorted(q)]})
        c = counts_per_sentence_rows(counts)
        lo, hi = interval(c)
        (RUNS / f"{ds}__{lang}__{name}__wordlevel.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
        print(f"{ds}/{lang}/{name} word-level: strict F1 {f1(c):.3f} [{lo:.3f}, {hi:.3f}] ({len(sentences)} sentences)", flush=True)


if __name__ == "__main__":
    main()
