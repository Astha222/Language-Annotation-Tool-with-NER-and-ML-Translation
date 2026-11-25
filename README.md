# Language Annotation Tool

Detect the **named entities** in text and documents and **translate** them, with open-source models that run on your own
machine. Upload a PDF, Word, Excel or PowerPoint file (or paste text), get people / organizations / locations
highlighted and a translation into Hindi, French, Spanish, German, Chinese, Arabic, Japanese, Russian or English, keep a
private history, and export everything.

* **Open source (MIT)** - and open by default: no API key, no quota, and your text never leaves your computer.
  A cloud translator (Azure) can be plugged in for speed and quality.
* **Documents, not just sentences** - PDF, DOCX, XLSX, PPTX, ODT/ODS/ODP, CSV/TSV, TXT/MD, JSON, XML, HTML. Long files run
  as background jobs with a progress bar and a Cancel button.
* **Measured, not promised** - [`evaluation/`](evaluation/) scores the models on public benchmarks;
  results and their limits are in [`evaluation/RESULTS.md`](evaluation/RESULTS.md).

## Contents

[Features](#features) · [Requirements](#requirements) · [Quick start](#quick-start) · [Configuration](#configuration) ·
[Changing or adding an NER model](#changing-or-adding-an-ner-model) · [Translation providers](#translation-providers) · [Documents](#documents) · [Accuracy](#accuracy) · [Project layout](#project-layout) ·
[Development](#development) · [Known limitations](#known-limitations) · [License](#license)

## Features

| | |
|---|---|
| Named-entity recognition | People, organizations and locations, with confidence scores (dates are not detected) |
| Translation | 9 languages in any direction; offline open models by default, Azure optional |
| Language detection | Automatic, or choose the source language yourself |
| Documents | Upload, extract text (PDF line-break repair included), analyse and translate as a background job |
| History and export | Per-user history; export to JSON, CSV, Excel, TXT, XML or PDF |
| Accounts | Registration, login (JWT), per-user data |

## Requirements

* **Python 3.10+** and **Node.js 20+**
* **MongoDB** (stores each annotation's entities) - e.g. `docker run -d -p 27017:27017 mongo`
* **PostgreSQL**, *or* SQLite for a no-setup trial (`POSTGRES_URL=sqlite:///./annotationtool.db`)
* A CPU is enough. Plan for **about 3 GB of disk** for the models and **4 GB of RAM** (8 GB is comfortable).
  The models download on first use: entity models about 1.5 GB, Hindi translation about 1.8 GB (both directions),
  every other language about 0.3 GB per direction.

## Quick start

```bash
git clone <this repository>
cd Language-Annotation-Tool-with-NER-and-ML-Translation-main

# 1. backend
cd backend
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                  # then edit .env: at least POSTGRES_URL (or SQLite) and JWT_SECRET
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 2. frontend (new terminal)
cd frontend
npm install
npx expo start --web --port 8081
```

Open <http://localhost:8081>, register an account and try it. API docs: <http://localhost:8000/docs>.
On Windows, `start-project.ps1` starts MongoDB/PostgreSQL checks, the backend and the frontend in one go.

The first translation or entity request downloads its models and can take a few minutes; later requests are fast.

## Configuration

Everything is set in `backend/.env`; [`backend/.env.example`](backend/.env.example) documents every option. The important ones:

| Setting | Purpose |
|---|---|
| `POSTGRES_URL`, `MONGO_URL` | Databases |
| `JWT_SECRET` | Login token secret - generate one: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `TRANSLATION_PROVIDERS` | Which translators to use, in order (default `azure,local`) |
| `AZURE_TRANSLATOR_KEY`, `AZURE_TRANSLATOR_REGION` | Optional Azure AI Translator |
| `LOCAL_MT_BEAMS` | Offline translation quality/speed (1 = fastest, 4 = best) |
| `CORS_ORIGINS` | Browser origins allowed to call the API |

Check which translators work from your machine: `cd backend && python check_translation.py`.

## Changing or adding an NER model

Which entity-recognition model each language uses is **data, not code**: it is listed in [`backend/ner_models.json`](backend/ner_models.json)
(`models` = what is available, with licence and measured accuracy; `routes` = which model each language uses). Today:

| Route | Languages | Model | Licence |
|---|---|---|---|
| `en` | English | `english-b-nofin` (a distilroberta-base student, decision D1); falls back to `english-bert-base` if its weights cannot be loaded | Apache-2.0 base; see THIRD_PARTY_NOTICES |
| `indic` | Hindi and eight other Indian languages | `indic-indicbert-hi` | CC0-1.0 |
| `*` | every other language (French, Spanish, German, Russian, Chinese, Arabic, Japanese, ...) | `multi-davlan-hrl` (decision D2(a)) | AFL-3.0 |

See what the running server uses with `GET /health/models` (`ner_routes`, `ner_fallbacks`), or without starting it: `python backend/ner_registry.py`.

**Switch a default** (the first two need no code change):

1. In `backend/.env`: `NER_MODEL_EN=english-bert-base` (or `NER_MODEL_HI`, `NER_MODEL_MULTI`). The value can be a name from the registry,
   a Hugging Face model id, or a local folder.
2. For one language, or to keep your choices in your own file: write a file shaped like `ner_models.json` and set `NER_MODELS_FILE=<that file>` in `.env`.
   Its `models` are added to the bundled ones and its `routes` replace the bundled ones, for example `{"routes": {"fr": "my-french-model"}}` sends only French to another model.
3. Edit `routes` in `backend/ner_models.json` itself (changes the default for everyone).

**Add a model you fine-tuned** (a Hugging Face token-classification model with PER/ORG/LOC labels, optionally MISC):

1. Train it (for example with `training/distill.py`) and keep the `best` folder, or upload it to the Hugging Face Hub.
2. Measure it on the frozen test sets before it becomes a default, for example
   `python evaluation/run_all.py --models "mymodel=<folder or id>" --datasets wnut2017,btc,uner_en_ewt --full`, and compare with the model it would replace in `evaluation/RESULTS.md`.
3. Add it under `models` (`source` = the folder or id, plus `licence`, `evidence` and an optional `fallback`), then point a route at it (see above). Restart the backend.
4. Record the licence of the weights and of the training data in `THIRD_PARTY_NOTICES.md`.

If a model cannot be loaded (folder missing, repository not found, no internet) and it has a `fallback`, the app logs a warning, uses the fallback and shows it in
`/health/models`; without a `fallback` the request fails with the error. Cached entity results are keyed by the model that really answered, so switching
models never shows another model's old results. Each loaded model takes about 1 GB of memory; `NER_MAX_MODELS` (default 2) limits how many stay loaded.
**Models that run code from their own repository.** Otter (`multi-otter`, Apache-2.0) is better on lower-case text (Spanish 0.70, French 0.69 against 0.36 and 0.46 for the default), but it
is not a normal Hugging Face NER model: loading it runs Python code that lives in the model's repository (`trust_remote_code`). The project allows this only under strict rules: the registry entry must say
`"trust_remote_code": true` and pin a full 40-character `revision` (the commit that was reviewed, never a branch), a model given only by a Hugging Face id or a folder never gets remote code, and
`NER_ALLOW_REMOTE_CODE=0` in `.env` switches all such models off. It is registered but **not used for any language yet** (routing waits for the in-house mixed test set); to try it for Spanish add
`{"routes": {"es": "multi-otter"}}` to your `NER_MODELS_FILE`. Before trusting a new revision, read its code and put the new hash in the registry.

**Faster CPU inference with ONNX.** An ONNX build of a model (made with `backend/export_onnx_ner.py <model id or folder>`; the fp32 file gives identical predictions and was 1.7-1.9x faster, the INT8 file is smaller but lost accuracy on one test set)
belongs to ONE model. Attach it to that model's registry entry: `"onnx_dir": "models/my-model-onnx"` (relative to `backend/`, or an absolute path) and optionally `"onnx_file"` (default `model.onnx`, the fp32 build).
It then serves every language that uses the model. If the file is missing or cannot be loaded (for example `onnxruntime` is not installed) the app logs a warning and uses the normal model, so a request never fails because of it.
The export script records the source model in `export_info.json` and the app **refuses an export made from a different model**. The old setting `NER_ONNX_DIR` / `NER_ONNX_FILE` still works for English, but it does not say
which model it belongs to, so it is deprecated (a warning is logged; folders without `export_info.json` cannot be checked): when you change the English model, export the new one again and use the registry instead.
`GET /health/models` (`ner_onnx`) shows which loaded models run from an ONNX build.

## Translation providers

`TRANSLATION_PROVIDERS` is an ordered list; the first provider that works translates each piece of text, and a provider
that fails is skipped for a while.

| Provider | Quality | Speed | Cost / limits | Privacy |
|---|---|---|---|---|
| `azure` - Azure AI Translator | Best of the four, in our tests (see [results](evaluation/RESULTS.md)) | Fast | Your own key; the free tier is limited (about 2 M characters/month and an hourly rate cap, per Microsoft - check current terms) | Text is sent to Microsoft |
| `local` - open-source models | Good for the 11 Indian languages (IndicTrans2, close to Azure); acceptable for European languages (Marian); weaker for Arabic, Chinese, Japanese | About 30-60 characters/second on a laptop CPU | Free, unlimited | **Text stays on your machine** |
| `google` - unofficial endpoint | Good | Fast until throttled | Rate-limited, unofficial: use at your own risk | Text is sent to Google |
| `mymemory` | Fair | Slow | About 5,000 characters/day | Text is sent to MyMemory |

The default `azure,local` uses Azure when you set a key and quietly falls back to the offline models otherwise. For a fully
offline setup use `TRANSLATION_PROVIDERS=local`. Offline translation between English and the 11 Indian languages (Hindi, Marathi,
Bengali, Assamese, Odia, Punjabi, Gujarati, Tamil, Telugu, Kannada, Malayalam) uses the MIT-licensed IndicTrans2 distilled models,
one per direction for all 11 (about 0.9 GB each); they ship their own Python code, so the exact reviewed commits are pinned
(see [SECURITY.md](SECURITY.md)). Set `LOCAL_INDICTRANS=0` to refuse that (Hindi then uses Marian; the other Indian languages need Azure).

## Documents

Supported: **PDF** (text-based), **DOCX**, **XLSX/XLSM**, **PPTX**, **ODT/ODS/ODP**, **CSV/TSV**, **TXT/MD/LOG**, **JSON**,
**XML**, **HTML**. Up to 50 MB and 1,000,000 characters of text per file.

Not supported: scanned PDFs (they contain pictures, not text; there is no OCR) and the old binary `.doc`, `.xls`, `.ppt`
(save them as `.docx`, `.xlsx`, `.pptx` first). Upload works in the web app.

Upload a file, choose the languages and press **Analyze & Translate**. The whole document is processed in the background;
the result is saved to History, and Export gives you the complete translation and entity list.

## Accuracy

What we measured (full tables, method and caveats in [`evaluation/RESULTS.md`](evaluation/RESULTS.md)):

* **Translation (English to X, chrF++ on 40 WMT24++ segments per language):** Azure scores 32-66 across Hindi, French,
  German, Spanish, Russian, Chinese, Arabic and Japanese. The offline models are close to Azure for Hindi (35.2 vs 36.0)
  and within 3-8 points for French, German, Spanish and Arabic. They are clearly weaker for Russian and Chinese, and
  **offline Japanese is weak** (17.9 chrF++ against Azure's 38.3, up from 4.4 after replacing the model; some outputs are still garbled, so use Azure for Japanese).
* **Offline translation for the Indian languages (English to X, chrF++ on 40 WMT24++ segments each):** IndicTrans2 is about as good as Azure:
  better for Marathi, Gujarati, Punjabi, Tamil, Telugu and Kannada (+1 to +8), slightly worse for Hindi and Malayalam (-1 to -3) and Bengali (-4).
  It is much slower on a laptop CPU (7-34 characters per second). Odia and Assamese work but are not measured.
* **English entity recognition (strict F1):** the default is the B-noFIN student, a small model trained on news, WNUT-2017 and BTC text. Over three training runs it scores 0.762 +- 0.009 on English-EWT
  (text it was not trained on), against 0.703 for the previous default `dslim/bert-base-NER`; names only: 0.59 on WNUT-2017 and 0.79 on BTC; 0.19 on financial text, which it was not trained on.
  The figures that follow are for the previous default: about 0.56-0.61 on Wikipedia text, 0.49 on noisy social media (full WNUT-2017 test set), 0.40 on Twitter and
  0.18 on financial text. Informal and financial text are hard: on Twitter most names are missed, on financial text many get the wrong type.
  Larger candidate models score 0.02-0.19 higher but are slower.
* **Hindi entity recognition (strict F1):** 0.83 on 864 human-labelled Hindi sentences (Naamapadam) and 0.76 on 3,094 sentences of a second dataset (HiNER),
  up from 0.65 and 0.53 with the model used before. Person names are found best (about 0.8-0.9), organizations worst (about 0.64-0.73). Dates are not detected (the previous model
  did not detect them either).
* **Hindi typed in English letters (Hinglish):** recognised by a word-list check (`backend/hinglish.py`) and sent to the Hindi model, which scores 0.94 on 125 sentences written by the project team's assistant
  and reviewed by a Hindi-speaking team member (one label corrected; the old route scored 0.71). Translation uses Azure's `hi-Latn` source for such text (the offline model and MyMemory cannot read it). Names typed without capital letters are found less well (0.76 on 10 such sentences),
  and a short message with a single Hindi word is not recognised.
* **Other Indian languages (offered in the app, marked "beta"):** the same open (CC0) model, trained only on Hindi, finds names in Marathi (F1 0.83), Telugu (0.85), Malayalam (0.85),
  Gujarati (0.83), Kannada (0.83), Bengali (0.81), Tamil (0.75) and Punjabi (0.73) on human-labelled news-style test sentences. Odia (0.27) does not work and Assamese (0.50 on only 51 sentences) is uncertain.
  These tests come from the same data family as the model's training data, and for Hindi the score fell from 0.83 to 0.76 and 0.60 on independent sets, so treat the numbers as optimistic. The language buttons carry a quality badge and a note; the labels come from `backend/language_status.json` (`GET /api/languages`).
  Offline translation (IndicTrans2) covers these languages and Odia and Assamese too (see "Translation providers"); language detection recognises all of them
  by their script (Odia 99.9%, Assamese 92%, the others 96-100% on test sentences).
* **Other languages (strict F1 with the default `Davlan/xlm-roberta-base-ner-hrl`, human-labelled test sets, ordinary cased text):** German 0.82, Arabic 0.81 (its training data may include part of that test set), Chinese 0.71-0.78,
  Russian 0.74, Japanese 0.64-0.72, against 0.35-0.51 for the model used before. On the hard benchmark with all text lower-cased, short and without punctuation it scores German 0.44, Spanish 0.36, French 0.46, Chinese 0.53;
  organisations are the weakest type everywhere. Spanish and French cased text was measured only on automatically labelled WikiANN data, where the model used before scores higher (0.85 against 0.56-0.61) because it was trained on that data, so that comparison says little. Chinese and Japanese entity highlighting used to cover whole clauses and was fixed.

Accuracy depends on the language, domain and quality of the source text. Spot-check anything that matters.

## Project layout

```
backend/      FastAPI app, text extraction, offline translation, Azure client, tests
frontend/     Expo / React Native (web) app
evaluation/   benchmark scripts and recorded results
```

Architecture, API reference and design decisions: [`project_documentation.md`](project_documentation.md).

## Development

```bash
cd backend  && pip install -r requirements-dev.txt && python -m pytest     # ~1 minute, no network or model downloads
cd frontend && npx tsc --noEmit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, how to add a language or a translation provider, and how to measure
an accuracy change before proposing it.

## Known limitations

* Offline translation for Indian languages is slow on a CPU (roughly 3,500 characters a minute), so very long documents take a
  long time; use Azure for those. Hindi typed in English letters (Hinglish) is translated by Azure only.
* Windows: if loading a translation model fails with `operator torchvision::nms does not exist`, the installed `torchvision`
  package is broken or blocked (Smart App Control reports `WinError 4551` for its `_C.pyd`). The project does not use
  `torchvision`: `pip uninstall torchvision` fixes it.
* Machine translation can still translate names (for example the word "Gravity" in a heading) or word things literally.
* Entity recognition makes mistakes, especially on technical terms and names it has not seen (see the accuracy numbers).
* The server is for a single machine: uploads are kept on local disk (`DOC_DIR`). Document jobs are saved in the SQL database, so a
  restart does not lose them: finished jobs stay readable and an interrupted job is restarted once automatically.
* `npm audit` reports findings in the frontend dependency tree. They are in Expo's development tooling (not part of what
  the browser runs) and in `xlsx`, which is used only to *write* spreadsheets, never to read untrusted files.
* The model for languages other than English and the Indian languages (`Davlan/xlm-roberta-base-ner-hrl`) is trained on ten languages and is weaker on text typed without capital letters
  (Spanish 0.36, German 0.44, French 0.46 on our lower-case test); an alternative that does better there (Otter, below) is registered but not used for any language yet.

## License

[MIT](LICENSE). Third-party models, data and code keep their own licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

IndicTrans2 and IndicTransToolkit (AI4Bharat, Varun Gumma), the rotary IndicTrans2 models (Gumma, Chitale, Bali),
Helsinki-NLP OPUS-MT, `dslim/bert-base-NER`, B-noFIN (distilroberta-base), `neuronsbyisshu/indicbert-hi-ner-naamapadam`, `Davlan/xlm-roberta-base-ner-hrl`, Hugging Face Transformers, FastAPI,
Expo / React Native, and the benchmark authors listed in `THIRD_PARTY_NOTICES.md`.
