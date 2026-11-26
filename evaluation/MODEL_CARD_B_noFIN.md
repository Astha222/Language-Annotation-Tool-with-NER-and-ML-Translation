---
language: en
license: apache-2.0
base_model: distilroberta-base
pipeline_tag: token-classification
library_name: transformers
tags:
  - ner
  - named-entity-recognition
  - token-classification
  - english
datasets:
  - tner/conll2003
  - tner/wnut2017
  - tner/btc
metrics:
  - f1
---

# English NER, small student "B-noFIN" (DistilRoBERTa)

A small English named-entity recognition model: it marks **people (PER), organisations (ORG), locations (LOC) and miscellaneous names (MISC)** in text.
It is the default English model of the open-source
[Language Annotation Tool](https://github.com/Astha222/Language-Annotation-Tool-with-NER-and-ML-Translation) (UPES), where it is called `english-b-nofin`.

Short version: on English web text that it was not trained on (emails, reviews, forum posts) it finds names better than `dslim/bert-base-NER`
(strict F1 **0.762 +- 0.009** against 0.703), with a smaller model (82 M against about 108 M parameters) that was 1.4x-2x faster on a laptop CPU, and 2.3x-2.9x faster with its ONNX build (identical output). It is weaker on organisations, and it was never tested on legal, medical or non-English text.

## Use

```python
from transformers import pipeline

ner = pipeline("ner", model="muskan-424/distilroberta-ner-english-b-nofin", aggregation_strategy="first")
ner("Satya Nadella told Reuters that Microsoft will open an office in Hyderabad.")
# -> [{'entity_group': 'PER', 'word': 'Satya Nadella', ...}, {'entity_group': 'ORG', 'word': 'Reuters', ...}, ...]
```

Inside the Language Annotation Tool nothing needs to be done: the tool loads it by itself (see "Changing or adding an NER model" in its README) and, if this
model cannot be downloaded, falls back to `dslim/bert-base-NER` and says so in its log.

It was trained on sentences of up to 128 tokens; for long text, cut it into sentences or chunks of about 800 characters, as the tool does.

## Model details

| | |
|---|---|
| Task | English named-entity recognition (token classification) |
| Architecture | `distilroberta-base` (82 M parameters, 6 layers; Apache-2.0) with a token-classification head |
| Labels (9, BIO) | `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC` (same order as `dslim/bert-base-NER`, so it can replace it without other changes) |
| Language | English only |
| Type | Plain fine-tuning with human labels (**no teacher model, no distillation**). The project's name for it is `B_noFIN_plain`. |

### Training data

Training sentences of three public datasets, mapped to one label set (`PER`, `ORG`, `LOC`, `MISC`):

| Dataset | Text | Sentences used | How its labels were mapped |
|---|---|---|---|
| [CoNLL-2003](https://huggingface.co/datasets/tner/conll2003) | news (Reuters) | 12,589 | PER, ORG, LOC, MISC as they are |
| [WNUT-2017](https://huggingface.co/datasets/tner/wnut2017) | social media, forums | 2,296 | person -> PER; corporation, group -> ORG; location -> LOC; product, creative-work -> MISC |
| [Broad Twitter Corpus](https://huggingface.co/datasets/tner/btc) | tweets | 6,312 | PER, ORG, LOC as they are |

Total 21,197 sentences. Sentences that also appear in any test set we report on (exactly, or after removing URLs, @mentions and punctuation) were removed from training first
(326 CoNLL and 10 BTC sentences). The financial-contract dataset FIN was deliberately **not** used (its original licence could not be verified); hence "noFIN".
The validation sentences used to choose the best checkpoint are 1,997 WNUT-2017 and BTC validation sentences, no FIN either.

### Training procedure

`training/distill.py` in the project repository, with `--alpha 1.0` (gold labels only), `--sources conll2003,wnut2017,btc`:
4 epochs, batch 32, learning rate 5e-5, AdamW, maximum 128 tokens, mixed precision, one Colab T4 GPU; the checkpoint with the best mean validation F1 is kept.
**Case augmentation:** every epoch each sentence is shown as the original (65%), all lower-case (30%) or all upper-case (5%), so the model also copes with text typed without capital letters.
Three training runs with different random starts (seeds 0, 1, 2) were made; **the published weights are seed 1**, chosen on validation F1 (see "Published weights" below).

## Evaluation

Strict entity-level F1 (an entity counts only if its type **and** both boundaries are exactly right), PER/ORG/LOC only, with 95% bootstrap intervals; code and frozen test sets in
`evaluation/` of the project repository. MISC is predicted, but the evaluation scores only PER, ORG and LOC.

### The fair test: English Web Treebank (UNER, 2,077 sentences, emails / reviews / newsgroups / answers)

No model in this table was trained on it.

| Model | Strict F1 |
|---|---|
| `dslim/bert-base-NER` (previous default of the tool) | 0.703 [0.674, 0.729] |
| **This model, 3 training runs (seeds 0 / 1 / 2)** | 0.755 / 0.772 / 0.760, **mean 0.762 +- 0.009** |
| Difference to bert-base-NER, per run (paired, 95% interval) | +0.052 [+0.027, +0.075], +0.069 [+0.044, +0.095], +0.057 [+0.035, +0.081] |
| `Jean-Baptiste/roberta-large-ner-english` (355 M parameters, 4x larger; reference) | 0.793 [0.771, 0.816] |

All three runs are better than bert-base-NER with every interval above 0, and the spread between runs is small (sd 0.009), so the result does not depend on a lucky seed.
By type, published run (seed 1), EWT: PER 0.862 (P 0.835, R 0.891), ORG 0.604 (P 0.572, R 0.640), LOC 0.818 (P 0.771, R 0.871). **Organisations are the weakest type.**

### Social-media test sets (names only, three runs, mean +- sd)

"Names only" removes two labelling quirks of these datasets (a bare "@" tagged as an entity in BTC, and "Borrower"/"Lender" in FIN) from gold labels and predictions.

| Test set | This model | bert-base-NER |
|---|---|---|
| WNUT-2017 (1,287 sentences), names only | 0.590 +- 0.015 | 0.493 (standard scoring; names-only not recorded) |
| WNUT-2017, standard scoring | 0.483 +- 0.012 | 0.493 |
| BTC (2,000 tweets), names only | 0.792 +- 0.007 | 0.401 (standard scoring) |

Read these carefully. The standard WNUT score (0.483) is **below** bert-base-NER because the model learned BTC's convention of tagging "@" in tweets. BTC is partly **in-domain**:
the model trained on BTC's training split, so its BTC score is not a measure of how it does on new kinds of text. Use the EWT table above for that.
FIN (financial contracts, not used in training) scores 0.186 +- 0.006: **do not use this model on financial or legal contracts without checking.**

### Speed

Timed on a laptop CPU (4 threads, batch 1, the tool's own path including text chunking; median of 30 calls; the laptop was busy, so the 95th percentiles swung and are not quoted):

| | ~200 characters | ~2,000 characters | peak memory |
|---|---|---|---|
| this model, PyTorch | 152 ms | 872-1,298 ms | 0.73 GB |
| this model, ONNX fp32 (`onnx_ner.py` in the project) | 75 ms | 651-811 ms | 0.87 GB |
| `dslim/bert-base-NER`, PyTorch, same session | 304 ms | 1,875 ms | 0.88 GB |

The ONNX fp32 build gives **identical predictions** to PyTorch on all 5,669 test sentences above (0 sentences differ; largest score difference 0.0001). An INT8 build (82 MB) exists but was not evaluated.

## Intended use and limits

**Intended use:** finding names in English text for annotation and review: the tool shows the suggestions to a person who accepts or corrects them. Use the scores above as a guide to how many
suggestions will need correcting (roughly one in four names is missed or mistyped on ordinary web text, more for organisations).

**Not suitable, or not tested:**
- Anything other than English (no multilingual ability was tested; the tool uses other models for other languages).
- Legal, medical, scientific, financial text (finance scored 0.19) or other text unlike news, forums, tweets and web pages.
- Fully automatic decisions about people or organisations without human review.
- Entity types other than PER, ORG, LOC, MISC (no dates, numbers or products as separate types).
- Uses where a missed name matters, such as redaction of personal data: recall is not high enough.

**Biases and gaps.** The training data is mostly Western news and English social media of the 2010s; names, places and organisations from other regions, and recent ones, are recognised less reliably.
The model was not checked for differences between groups of names.

## Published weights

| | |
|---|---|
| Seed published | **1** (`B_noFIN_plain_s1`) |
| Why this seed | Highest validation F1 (rule fixed in advance: highest, and the lower seed number if two are within 0.002); test scores were not used. s0 0.7544, s1 0.7646, s2 0.7635: s1 and s2 are within noise of each other |
| Best validation F1 of that run | 0.7646 (mean of WNUT-2017 0.7293 and BTC 0.8000 validation F1; checkpoint at epoch 2.5 of 4) |
| Strict F1 of that run | English-EWT 0.772 [0.744, 0.798]; WNUT-2017 0.486 (names only 0.597); BTC 0.818 (names only 0.800); FIN 0.189. The three-run means above describe the recipe; this row describes the file you download. |
| File format | PyTorch (`model.safetensors`), 326 MB |

## Licences and provenance

- **Weights and card:** Apache-2.0, like the base model `distilroberta-base` (the owner of the repository sets this when uploading).
- **Training data:** WNUT-2017 and the Broad Twitter Corpus are CC-BY-4.0 (attribution required: please cite them, below). CoNLL-2003: the annotations are free to use, the underlying Reuters news text is
  copyrighted, which is the usual accepted practice for models trained on CoNLL; **the project owner has still to confirm the terms** before the weights are published.
- The model is not derived from `Jean-Baptiste/roberta-large-ner-english` or any other fine-tuned NER model: it is trained from the plain `distilroberta-base` with the labels of the datasets above.

## Reproduce

```bash
python training/build_mix.py && python training/build_mix.py --dev      # training mix and validation mix (both still contain FIN sentences)
# keep only non-FIN validation sentences (what the project did), then train one run; the project used a Colab T4 GPU for seeds 0, 1 and 2
# (notebook WS2_Colab/WS2b_noFIN_Seeds_Colab.ipynb does all of this)
python training/distill.py --student distilroberta-base --train evaluation/data/train_mix.jsonl --dev <dev_mix_without_fin.jsonl>        --out out/B_noFIN --sources conll2003,wnut2017,btc --alpha 1.0 --epochs 4 --seed 0
python evaluation/run_all.py --models "B=out/B_noFIN/best" --datasets wnut2017,btc,fin,uner_en_ewt --full
```

Full tables, the bootstrap method and every caveat: `evaluation/RESULTS.md` (sections E1, E5 and "WS2b"); decision record: `IMPROVEMENT_LOG.md` ("D1").

## Citation

- Base model: `distilroberta-base` (Sanh et al., *DistilBERT, a distilled version of BERT*, 2019; Liu et al., *RoBERTa*, 2019).
- Tjong Kim Sang and De Meulder, *Introduction to the CoNLL-2003 Shared Task*, 2003.
- Derczynski et al., *Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition*, 2017.
- Derczynski et al., *Broad Twitter Corpus: A Diverse Named Entity Recognition Resource*, COLING 2016.
- Evaluation text: Universal NER, English-EWT (Mayhew et al., 2023; CC-BY-SA-4.0).
