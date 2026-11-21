# Evaluation

Scripts that measure how accurate the app really is, on public benchmarks, so claims can be checked and regressions
caught. Recorded results are in [RESULTS.md](RESULTS.md).

| Script | Measures | Data |
|---|---|---|
| `eval_ner.py` | entity recognition: precision, recall, F1 for people / organizations / locations | WikiANN (many languages), CoNLL-2003, WNUT-2017, Broad Twitter Corpus, FIN |
| `eval_translation.py` | translation quality (chrF++, BLEU) of each provider against human references | WMT24++ (English -> X) |

## Setup

```bash
cd backend
pip install -r requirements-dev.txt        # includes sacrebleu
cd ..
```

The scripts import the real backend code (`backend/main.py`) but use an in-memory SQLite database, so your real databases
are never touched. Benchmark files are downloaded on first use into `evaluation/data/` (git-ignored). The Azure rows need
`AZURE_TRANSLATOR_KEY` in `backend/.env`.

## Run

```bash
# entity recognition: the app's models, per language
python evaluation/eval_ner.py --datasets wikiann --langs en,hi,fr,es,de,ru,ar,zh,ja --n 400 --sweep

# compare candidate English models across domains
python evaluation/eval_ner.py --datasets conll2003,wnut2017,btc,fin,wikiann --models app,dslim/bert-large-NER

# translation quality of each provider, English -> Hindi
python evaluation/eval_translation.py --langs hi --providers local,azure --n 60
```

Run `--help` on either script for every option. Expect a few minutes per language; large NER models and the local
translation models are slow on a CPU.

## How to read the numbers

* **Sample sizes are small** (a few hundred sentences, 40-60 translation segments), so differences under roughly 0.02 F1
  or 2 chrF++ points are noise. Use `--n` to raise the sample when comparing close candidates.
* **Entity scoring is exact and entity-level:** a prediction counts only if its type and both boundaries match. This is
  stricter than token-level accuracy and gives lower numbers.
* **Some scores are optimistic by construction.** `Davlan/xlm-roberta-base-wikiann-ner` was trained on WikiANN, and
  `dslim/bert-base-NER` on CoNLL-2003; their scores on those datasets are in-domain. WNUT-2017, Broad Twitter and FIN are
  the fair out-of-domain tests.
* **WikiANN labels are silver-standard** (derived automatically from Wikipedia), so even strong models score modestly.
* **Translation scores** compare against one human reference per segment; a correct translation worded differently loses
  points. Use them to compare providers on the same sample, not as an absolute quality grade.
* **This is not a guarantee for your documents.** Accuracy depends on the language, domain and quality of the source text.
  Always spot-check results that matter.
