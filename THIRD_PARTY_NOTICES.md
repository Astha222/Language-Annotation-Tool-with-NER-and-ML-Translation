# Third-party models, data and code

This project's own code is MIT-licensed (see [LICENSE](LICENSE)). It **downloads** the models below at run time from
the Hugging Face Hub; none of them is bundled in this repository. Each keeps its own license, which you accept by using
it. Please read the model cards linked here before redistributing a built copy of the application.

## Models used by default

| Purpose | Model | License | Notes |
|---|---|---|---|
| English entity recognition (default, decision D1) | `english-b-nofin`: `muskan-424/distilroberta-ner-english-b-nofin` (to be published on the Hugging Face Hub) | Apache-2.0 for the base model [distilroberta-base](https://huggingface.co/distilroberta-base); the licence of the published weights is set when they are uploaded | Fine-tuned (no teacher model, no FIN data) on CoNLL-2003, WNUT-2017 and BTC training sentences. WNUT-2017 and BTC are CC-BY-4.0 (attribution required); the owner still has to confirm the CoNLL-2003 terms before the weights are published. Until the weights are available the app uses the fallback below. |
| English entity recognition (fallback and previous default) | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | MIT | Fine-tuned on CoNLL-2003 news text. |
| Entity recognition, Hindi | [neuronsbyisshu/indicbert-hi-ner-naamapadam](https://huggingface.co/neuronsbyisshu/indicbert-hi-ner-naamapadam) | CC0-1.0 | BERT-style model fine-tuned on the Naamapadam Hindi data (CC0-1.0). Replaced Davlan/xlm-roberta-base-wikiann-ner for Hindi (strict F1 0.827 against 0.650 on the Naamapadam gold test). Detects PER, ORG, LOC only (no dates). |
| Entity recognition, every language except English and the Indian languages (default, decision D2(a)) | [Davlan/xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl) | AFL-3.0 (Academic Free License 3.0) | XLM-RoBERTa base fine-tuned (published on Hugging Face by Davlan) on named-entity data of ten high-resource languages (Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, Chinese; labels PER, ORG, LOC, DATE). AFL-3.0 is permissive but requires keeping the copyright and licence notice with any copy of the weights; this project only downloads them on the user's machine. It replaced `Davlan/xlm-roberta-base-wikiann-ner`, which has no declared licence (it can still be selected with `NER_MODEL_MULTI=multi-davlan-wikiann` to reproduce old results, but must not be redistributed). |
| Entity recognition, optional (registered, used for no language by default) | [whoisjones/otter-cross-mmbert](https://huggingface.co/whoisjones/otter-cross-mmbert) | Apache-2.0 | Zero-shot model. **It runs Python code from its model repository** (`trust_remote_code`), allowed by the project owner on 2026-11-25 only at the reviewed commit `8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d`; `NER_ALLOW_REMOTE_CODE=0` switches it off. |
| Hindi <-> English translation | [prajdabre/rotary-indictrans2-en-indic-dist-200M](https://huggingface.co/prajdabre/rotary-indictrans2-en-indic-dist-200M) and [...-indic-en-dist-200M](https://huggingface.co/prajdabre/rotary-indictrans2-indic-en-dist-200M) | MIT | IndicTrans2 variants by Varun Gumma, Pranjal A. Chitale and Kalika Bali. Run with `trust_remote_code`, so the exact commits are pinned in `backend/local_translator.py`. Citation: Gumma et al., NAACL 2025, "Towards Inducing Long-Context Abilities in Multilingual Neural Machine Translation Models". Built on IndicTrans2 (AI4Bharat). |
| Other language pairs | [Helsinki-NLP/opus-mt-*](https://huggingface.co/Helsinki-NLP) (Marian) | Apache-2.0 or CC-BY-4.0 (per model) | When last checked, `opus-mt-en-de`, `opus-mt-zh-en` and `opus-mt-ru-en` were CC-BY-4.0 (attribution required) and the others Apache-2.0. Check each model card. |
| English to Japanese | [Helsinki-NLP/opus-tatoeba-en-ja](https://huggingface.co/Helsinki-NLP/opus-tatoeba-en-ja) (Marian) | Apache-2.0 | Default for en to ja. Optional, not used by default: [staka/fugumt-en-ja](https://huggingface.co/staka/fugumt-en-ja) (CC-BY-SA-4.0, share-alike), selected with `LOCAL_MT_JA_MODEL`. |
| Faster NER runtime (optional) | [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Only if `NER_ONNX_DIR` is set; exporting the model also uses the `onnx` package (Apache-2.0). |

Models that were evaluated and **deliberately not used**: `facebook/nllb-200-*` (CC-BY-NC-4.0, non-commercial),
`ai4bharat/indictrans2-*` and `ai4bharat/IndicNER` (gated: require a Hugging Face login to download).

## Optional cloud services

* **Azure AI Translator** - a commercial Microsoft service, used only if you configure your own key.
* **Google Translate via `deep-translator`** and **MyMemory** - free, rate-limited web services, off by default.
  The Google path is an unofficial scraper; check its terms before relying on it.

## Code

* `backend/indic_processor.py` is a pure-Python port of `IndicProcessor` from
  [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit) - MIT License, Copyright (c) Varun Gumma.
* Python and JavaScript dependencies are listed in `backend/requirements.txt` and `frontend/package.json`; each is
  distributed under its own license (mostly MIT, BSD or Apache-2.0).

## Evaluation data (downloaded by `evaluation/`, never committed)

* [WMT24++](https://huggingface.co/datasets/google/wmt24pp) - Apache-2.0 (Deutsch et al., 2025).
* WikiANN (Pan et al., 2017), CoNLL-2003, WNUT-2017, Broad Twitter Corpus and FIN, through the
  [tner](https://huggingface.co/tner) mirrors. Several carry their own research-use terms; they are only used locally to
  compute scores.
