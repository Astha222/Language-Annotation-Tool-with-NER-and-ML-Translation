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
[Translation providers](#translation-providers) · [Documents](#documents) · [Accuracy](#accuracy) · [Project layout](#project-layout) ·
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
* **English entity recognition (F1):** about 0.56-0.61 on Wikipedia text, 0.49 on noisy social media (full WNUT-2017 test set), 0.40 on Twitter and
  0.18 on financial text with the default model. Informal and financial text are hard: on Twitter most names are missed, on financial text many get the wrong type.
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
* **Other languages (strict F1 on a deliberately hard benchmark: all text lower-cased, short, no punctuation):** German 0.34, Spanish 0.31, Chinese 0.28, French 0.26.
  Ordinary text should score higher. **Not measured:** Russian, Arabic and Japanese. Chinese and Japanese entity highlighting used to cover whole clauses and was fixed.

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
* The Davlan model used for entity recognition in languages other than English and Hindi has no declared license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

[MIT](LICENSE). Third-party models, data and code keep their own licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

IndicTrans2 and IndicTransToolkit (AI4Bharat, Varun Gumma), the rotary IndicTrans2 models (Gumma, Chitale, Bali),
Helsinki-NLP OPUS-MT, `dslim/bert-base-NER`, `neuronsbyisshu/indicbert-hi-ner-naamapadam`, `Davlan/xlm-roberta-base-wikiann-ner`, Hugging Face Transformers, FastAPI,
Expo / React Native, and the benchmark authors listed in `THIRD_PARTY_NOTICES.md`.
