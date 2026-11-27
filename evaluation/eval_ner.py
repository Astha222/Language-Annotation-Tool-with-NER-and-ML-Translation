"""Entity-level precision / recall / F1 of named-entity recognition, for the app's models or any candidate.

Everything is scored on the three types the app is built around - PERSON, ORGANIZATION, LOCATION - with exact
entity matching: a prediction counts only if its type AND first and last token match. Gold entities of other types
(MISC, product, creative-work, ...) are ignored, and so are predictions of other types.

Benchmarks (downloaded on first use from the Hugging Face Hub, cached in evaluation/data, never committed):
    wikiann    Wikipedia, many languages          (Pan et al. 2017; mirror tner/wikiann)       silver-standard labels
    conll2003  Reuters news                       (Tjong Kim Sang & De Meulder 2003; tner)     models trained on it are in-domain
    wnut2017   noisy social media                 (Derczynski et al. 2017; tner)
    btc        Twitter                            (Derczynski et al. 2016; tner)
    fin        financial agreements               (Salinas Alvarado et al. 2015; tner)

Usage (from the project root):
    # the app's current models on WikiANN, per language
    python evaluation/eval_ner.py --datasets wikiann --langs en,hi,fr --n 500
    # compare English candidates across domains
    python evaluation/eval_ner.py --datasets conll2003,wnut2017,btc,fin,wikiann --models app,dslim/bert-large-NER

Reading notes:
  * Davlan/xlm-roberta-base-wikiann-ner (every non-English language) was fine-tuned on WikiANN, so its WikiANN
    scores are optimistic. dslim/bert-base-NER was trained on CoNLL-2003, so CoNLL scores are optimistic for it
    and for every other CoNLL-trained model. Out-of-domain sets (wnut2017, btc, fin) are the fairer comparison.
  * With a few hundred sentences the scores move by about +-0.02-0.03; do not read much into smaller differences.
"""
import argparse
import json
import sys
import time

from common import BACKEND, DATA, download, load_backend, use_console_utf8

sys.path.insert(0, str(BACKEND))
from ner_clean import _clean_entities                 # noqa: E402  the app's own cleaning, without importing the web app

TNER = "https://huggingface.co/datasets/tner"
NO_SPACE_LANGS = {"zh", "ja"}                        # WikiANN tokens are single characters; do not join them with spaces
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
KEEP = {"PER", "ORG", "LOC"}
LABEL_ALIASES = {"PERS": "PER", "PERSON": "PER", "ORGANIZATION": "ORG", "LOCATION": "LOC",
                 # llm-book/bert-base-japanese-v3-ner-wikipedia-dataset uses the stockmark type names (same mapping as foreign_data.JA_TYPES)
                 "人名": "PER", "法人名": "ORG", "政治的組織名": "ORG", "その他の組織名": "ORG", "地名": "LOC", "施設名": "LOC"}   # e.g. the CAMeL Arabic model says PERS

# dataset -> (url of a split, label file url, splits (dev, test), label -> PER/ORG/LOC or None to drop)
IDENTITY = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}
DATASETS = {
    "wikiann":   (lambda lang, split: f"{TNER}/wikiann/resolve/main/dataset/{lang}/{split}.jsonl",
                  f"{TNER}/wikiann/resolve/main/dataset/label.json", ("dev", "test"), IDENTITY),
    "conll2003": (lambda lang, split: f"{TNER}/conll2003/resolve/main/dataset/{split}.json",
                  f"{TNER}/conll2003/resolve/main/dataset/label.json", ("valid", "test"), IDENTITY),
    "wnut2017":  (lambda lang, split: f"{TNER}/wnut2017/resolve/main/dataset/{split}.json",
                  f"{TNER}/wnut2017/resolve/main/dataset/label.json", ("valid", "test"),
                  {"person": "PER", "location": "LOC", "corporation": "ORG", "group": "ORG"}),
    "btc":       (lambda lang, split: f"{TNER}/btc/resolve/main/dataset/{split}.json",
                  f"{TNER}/btc/resolve/main/dataset/label.json", ("valid", "test"), IDENTITY),
    "fin":       (lambda lang, split: f"{TNER}/fin/resolve/main/dataset/{split}.json",
                  f"{TNER}/fin/resolve/main/dataset/label.json", ("valid", "test"), IDENTITY),
}


# Datasets stored locally (downloaded by download_data.py): name -> (language, file pattern, field with the words, field with the BIO tags)
# (language, file, words field, tags field, split name used in the manifest)
LOCAL = {"naamapadam": ("hi", "ner_hi/naamapadam/hi_{split}.json", "words", "ner", "test"),
         "hiner": ("hi", "ner_hi/hiner/hi_{split}.json", "words", "ner", "test"),
         # Hindi in English letters, 125 sentences written by the assistant (native-speaker review pending); hinglish_deva = the same words transliterated
         # to Devanagari by IndicXlit (built on Colab by the WS3e notebook, not frozen in the manifest)
         "hinglish": ("hi", "ner_hi/hinglish/hi_{split}.json", "words", "ner", "test"),
         "hinglish_deva": ("hi", "ner_hi/hinglish/deva_{split}.json", "words", "ner", "test"),
         # Russian, Arabic, Japanese and ordinary-text German / Chinese (evaluation/foreign_data.py)
         **{name: (lang, f"ner_foreign/{name}_{{split}}.json", "words", "ner", "test")
            for name, lang in (("uner_en_ewt", "en"), ("uner_ja", "ja"), ("uner_ru", "ru"), ("uner_de", "de"), ("uner_zh", "zh"), ("uner_zhs", "zh"), ("anercorp", "ar"), ("stockmark_ja", "ja"))},
         # MultiCoNER v2 dev splits (short, lower-cased, low-context sentences; 33 types mapped to PER / ORG / LOC)
         **{f"mc_{l}": (l, f"ner_multi/mc_{l}_dev.json", "words", "ner", "dev") for l in ("de", "es", "fr", "zh", "hi")},
         # MultiCoNER TRAIN samples: only for tuning zero-shot thresholds (never report them as test scores)
         **{f"mc_{l}_tune": (l, f"ner_multi/mc_{l}_tune.json", "words", "ner", "tune") for l in ("de", "es", "fr")},
         # Naamapadam gold test sets for the other Indian languages (ai4bharat, CC0)
         **{f"nm_{l}": (l, f"ner_indic/{l}_test.json", "words", "ner", "test") for l in ("as", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te")}}


def load_local(dataset, split, n):
    """Human-labelled Hindi sentences with BIO tags already as text. Empty sentences are dropped (a pipeline cannot read them)."""
    _, pattern, words, tags, _ = LOCAL[dataset]
    path = DATA / pattern.format(split=split)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sentences = [(r[words], [t if t == "O" or t[2:] in KEEP else "O" for t in r[tags]]) for r in rows if r[words]]
    return sentences[:n]


def load_split(dataset, lang, split, n):
    if dataset in LOCAL:
        return load_local(dataset, split, n)
    url_for, label_url, _, mapping = DATASETS[dataset]
    labels = json.loads(download(label_url, DATA / "ner" / dataset / "label.json").read_text(encoding="utf-8"))
    id2label = {v: k for k, v in labels.items()}
    path = download(url_for(lang, split), DATA / "ner" / dataset / f"{lang}_{split}.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sentences = []
    for r in rows[:n]:
        tags = []
        for t in r["tags"]:
            label = id2label[t]
            if label == "O":
                tags.append("O")
            else:
                kind = mapping.get(label[2:])
                tags.append("O" if kind is None else f"{label[:2]}{kind}")      # drop types we do not score
        sentences.append((r["tokens"], tags))
    return sentences


def build(tokens, tags, lang):
    """Sentence text, per-token character offsets and gold entity spans (type, first token, last token)."""
    sep = "" if lang in NO_SPACE_LANGS else " "
    text, offsets = "", []
    for i, tok in enumerate(tokens):
        if i:
            text += sep
        offsets.append((len(text), len(text) + len(tok)))
        text += tok
    spans, cur = set(), None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur:
                spans.add(tuple(cur))
                cur = None
        elif tag.startswith("B-") or cur is None or cur[0] != tag[2:]:
            if cur:
                spans.add(tuple(cur))
            cur = [tag[2:], i, i]
        else:
            cur[2] = i
    if cur:
        spans.add(tuple(cur))
    return text, offsets, spans


def to_token_span(offsets, start, end):
    covered = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    return (covered[0], covered[-1]) if covered else None


_PIPELINES = {}


def make_pipeline(app, lang, model):
    if model == "app":
        return app.get_ner_pipeline(lang), app.ner_model_name(lang)
    if model.startswith(("gliner:", "otter:")) and model not in _PIPELINES:   # zero-shot models go through an adapter (gliner package / trust_remote_code)
        from gliner_adapter import GlinerPipeline
        kind, _, model_id = model.partition(":")
        _PIPELINES[model] = GlinerPipeline(model_id, kind=kind)
    if model not in _PIPELINES:                                   # load each candidate once, not once per dataset
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1                  # a GPU (Colab) is used when there is one
        try:
            _PIPELINES[model] = pipeline("ner", model=model, aggregation_strategy="first", device=device)
        except ValueError as e:
            if "TokenizersBackend" not in str(e):
                raise
            # tokenizer saved by a newer transformers than the pinned 4.57: build it from the plain tokenizer.json instead
            _PIPELINES[model] = pipeline("ner", model=model, tokenizer=fast_tokenizer(model), aggregation_strategy="first", device=device)
    return _PIPELINES[model], model


def fast_tokenizer(model):
    import json
    from pathlib import Path
    from transformers import PreTrainedTokenizerFast
    if Path(model).is_dir():
        file, cfg_file = Path(model) / "tokenizer.json", Path(model) / "tokenizer_config.json"
    else:
        from huggingface_hub import hf_hub_download
        file, cfg_file = hf_hub_download(model, "tokenizer.json"), hf_hub_download(model, "tokenizer_config.json")
    cfg = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
    specials = {k: cfg[k] for k in ("unk_token", "pad_token", "cls_token", "sep_token", "mask_token", "bos_token", "eos_token") if k in cfg}
    return PreTrainedTokenizerFast(tokenizer_file=str(file), model_max_length=512, **specials)


def predict(app, nlp, lang, sentences):
    """Run NER over each sentence on its own; returns [(text, offsets, gold, predictions)]."""
    built = [build(t, g, lang) for t, g in sentences]
    out = []
    for i in range(0, len(built), 16):
        batch = built[i:i + 16]
        raw = nlp([b[0] for b in batch], batch_size=16, **({"aggregation_strategy": "simple"} if lang in NO_SPACE_LANGS else {}))
        for (text, offsets, gold), ents in zip(batch, raw):
            preds = []
            for e in _clean_entities(text, ents, 0, snap=lang not in NO_SPACE_LANGS):
                span = to_token_span(offsets, e["start"], e["end"])
                if span:
                    preds.append((LABEL_ALIASES.get(e["entity_group"], e["entity_group"]), span[0], span[1], e["score"]))
            out.append((text, offsets, gold, preds))
    return out


def score(results, threshold=0.0):
    tp = fp = fn = 0
    for _, _, gold, preds in results:
        kept = {(t, a, b) for t, a, b, s in preds if s >= threshold and t in KEEP}
        tp += len(kept & gold)
        fp += len(kept - gold)
        fn += len(gold - kept)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="wikiann")
    ap.add_argument("--langs", default="en", help="languages (only wikiann has more than English)")
    ap.add_argument("--models", default="app", help="comma-separated; 'app' = the model the app uses for that language")
    ap.add_argument("--n", type=int, default=500, help="sentences per split")
    ap.add_argument("--sweep", action="store_true", help="also print the effect of a minimum confidence score")
    ap.add_argument("--out", help="also write the tables to this markdown file")
    args = ap.parse_args()

    app = load_backend()
    rows = ["| dataset | lang | model | sentences | gold entities | precision | recall | F1 |", "|---|---|---|---|---|---|---|---|"]
    sweep = ["| dataset | lang | model | min score | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        langs = args.langs.split(",") if dataset == "wikiann" else ["en"]
        for lang in (l.strip() for l in langs):
            for model in (m.strip() for m in args.models.split(",")):
                t0 = time.time()
                try:
                    nlp, shown = make_pipeline(app, lang, model)
                    dev_split, test_split = DATASETS[dataset][2]
                    test = predict(app, nlp, lang, load_split(dataset, lang, test_split, args.n))
                    dev = predict(app, nlp, lang, load_split(dataset, lang, dev_split, args.n)) if args.sweep else []
                except Exception as e:                                # one failure must not stop the rest
                    print(f"{dataset}/{lang}/{model}: skipped ({type(e).__name__}: {str(e)[:100]})", flush=True)
                    continue
                gold_total = sum(len(g) for _, _, g, _ in test)
                p, r, f = score(test)
                rows.append(f"| {dataset} | {lang} | `{shown.split('/')[-1]}` | {len(test)} | {gold_total} | {p:.3f} | {r:.3f} | {f:.3f} |")
                if args.sweep:
                    best = max(THRESHOLDS, key=lambda t: score(dev, t)[2])          # chosen on dev, reported on test
                    for t in THRESHOLDS:
                        tp_, tr_, tf_ = score(test, t)
                        sweep.append(f"| {dataset} | {lang} | `{shown.split('/')[-1]}` | {t:.2f}{' (best on dev)' if t == best else ''} "
                                     f"| {tp_:.3f} | {tr_:.3f} | {tf_:.3f} |")
                print(f"{dataset}/{lang}/{shown}: F1 {f:.3f} (P {p:.3f} R {r:.3f}) in {time.time() - t0:.0f}s", flush=True)

    report = ["## Entity recognition (PER / ORG / LOC, exact entity match)", "", *rows]
    if args.sweep:
        report += ["", "### Effect of a minimum confidence score (cut-off picked on the dev split, reported on test)", "", *sweep]
    print("\n" + "\n".join(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
