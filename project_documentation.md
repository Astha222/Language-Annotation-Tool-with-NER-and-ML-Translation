# Language Annotation Tool - Project Documentation

> Setup and day-to-day use are in the [README](README.md). This document explains **how the system works** and why it
> is built the way it is. Measured accuracy is in [evaluation/RESULTS.md](evaluation/RESULTS.md).

## 1. Project overview

**Language Annotation Tool** is a full-stack web application that lets a user:

- paste text or **upload a document** (PDF, Word, Excel, PowerPoint, OpenDocument, CSV, TXT, HTML, JSON, XML);
- automatically detect the source language;
- run **Named Entity Recognition (NER)** to find people, organizations and locations (dates are not detected);
- **translate** the text into Hindi, French, Spanish, German, Chinese, Arabic, Japanese, Russian or English, using
  open-source models that run on the user's own machine, or an optional cloud service;
- register / log in (JWT) so each user has a private annotation history;
- save, view, delete and **export** annotations (JSON, CSV, XLSX, TXT, XML, PDF).

It is useful for creating **NLP datasets**, studying **multilingual text** and building **annotated corpora**.

---

## 2. System architecture

```
┌──────────────────────────────────────────────────────────┐
│                 FRONTEND (Expo Web, React Native)         │
│  Login / Register   ·   Annotate (text + file upload)     │
│  History (view, delete, export)    http://localhost:8081  │
└───────────────────────────┬──────────────────────────────┘
                            │ REST (axios, Bearer JWT)
┌───────────────────────────▼──────────────────────────────┐
│                   BACKEND (FastAPI, Python)               │
│                       http://localhost:8000               │
│                                                          │
│  auth ─ annotations CRUD ─ export                         │
│  NER (run_ner) ─ language detection                       │
│  translation provider chain ── azure ── local ── …        │
│  document upload ─► text extraction ─► background job     │
└───────┬──────────────────────┬──────────────┬────────────┘
        │                      │              │
   SQL (PostgreSQL        MongoDB         Hugging Face models
   or SQLite)             entity JSON     (downloaded on first use)
   users, annotations     per annotation  NER · IndicTrans2 · Marian
```

---

## 3. Technology stack

| Layer | Technology |
|---|---|
| Frontend | React Native 0.86 + Expo 57 (web), TypeScript, axios, react-native-web, SheetJS (`xlsx`) for Excel export |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Persistence | SQLAlchemy (PostgreSQL, or SQLite for trying it out) and MongoDB (pymongo) |
| Auth | JWT (python-jose) with bcrypt password hashes |
| NER | Hugging Face `transformers` token-classification pipelines: the B-noFIN student (English, falls back to `dslim/bert-base-NER`), `neuronsbyisshu/indicbert-hi-ner-naamapadam` (Hindi), `Davlan/xlm-roberta-base-wikiann-ner` (other languages) |
| Offline translation | IndicTrans2 "rotary" 200M models (Hindi <-> English) and Helsinki-NLP OPUS-MT Marian models (other pairs) |
| Cloud translation (optional) | Azure AI Translator; unofficial Google endpoint and MyMemory are opt-in |
| Documents | pypdf, python-docx, openpyxl, python-pptx, plus the standard library for CSV, HTML, XML, OpenDocument |
| Language detection | langdetect |
| Tests | pytest with fakes for MongoDB, translators and the NER model |

Exact tested versions are in `backend/requirements.txt` (ranges) and the comments at its top.

---

## 4. Project file structure

```
.
├── backend/
│   ├── main.py                 API, auth, annotations, NER, provider chain, document jobs
│   ├── doc_extract.py          text extraction from every supported file type (+ PDF line reflow)
│   ├── local_translator.py     offline translation (IndicTrans2 + Marian), model cache, pivoting
│   ├── indic_processor.py      pure-Python port of IndicTransToolkit's pre/post-processing
│   ├── azure_translator.py     Azure AI Translator client with a token-bucket rate limiter
│   ├── check_translation.py    "does my Azure key / which providers work?" command-line check
│   ├── tests/                  pytest suite
│   ├── requirements.txt  requirements-dev.txt  pytest.ini
│   └── .env.example            every setting, documented
├── frontend/
│   ├── App.tsx  index.ts  app.json  package.json
│   └── src/
│       ├── context/AuthContext.tsx     login state, token storage, auto-logout on 401
│       ├── screens/                    Login, Register, Home (annotate), History
│       └── utils/                      pdfExport, entities, xlsxSafe
├── evaluation/                 benchmark scripts and recorded results
├── start-project.ps1           one-click start on Windows
├── LICENSE  README.md  CONTRIBUTING.md  SECURITY.md  THIRD_PARTY_NOTICES.md
└── project_documentation.md    this file
```

---

## 5. Backend

### 5.1 Startup and configuration
- Settings come from `backend/.env` (loaded with python-dotenv); `.env.example` documents all of them.
- SQL tables are created automatically; a small migration adds columns that older databases lack (`user_id`, `source_name`).
- `JWT_SECRET` is never given a hard-coded default: if it is missing or shorter than 32 characters a random one is used
  for that run (and a warning is logged).
- CORS allows only `CORS_ORIGINS` (default: the local Expo web app); credentials are not used (Bearer tokens only).
- The English NER model is **preloaded and warmed at startup** (`PRELOAD_MODELS`, default `ner_en`; translation models can be
  added, e.g. `mt:en-hi`), so the first user request is not slow (17 s -> 0.09 s on the test laptop). Other models still load on first
  use. `GET /health/models` reports what is loaded, the preload result and free disk space where models are stored.
- PyTorch's CPU thread count is fixed at startup: `TORCH_THREADS`, else the number of physical cores.
- NER and translation results are cached in a SQLite file (`result_cache.py`, `RESULT_CACHE_PATH`), keyed by text, language, model and
  settings, so re-analysing the same text or file is instant.

### 5.2 Authentication
`POST /api/auth/register` validates the username (3-30 characters: letters, digits, `. _ -`), the email and the
password (8-72 bytes), stores a bcrypt hash, and compares usernames/emails case-insensitively.
`POST /api/auth/login` returns a JWT (default 24 h). Every annotation, document and job endpoint is scoped to the
logged-in user; another user's ids return 404.

### 5.3 Entity recognition (`run_ner`)
1. Choose the model from the source language: English -> the B-noFIN student (PER, ORG, LOC, MISC; the model for each language comes from the registry `backend/ner_models.json`, see `ner_registry.py`); everything else
   -> `neuronsbyisshu/indicbert-hi-ner-naamapadam` for Hindi and eight other Indian languages (`INDIC_NER_LANGS` in `main.py`; PER, ORG, LOC; `NER_MODEL_HI` overrides it) and `Davlan/xlm-roberta-base-wikiann-ner` (PER, ORG, LOC; its output head has a DATE label but it almost never predicts it)
   for every other language. Aggregation strategy `first` keeps whole words.
2. Split the text into sentence-aware chunks of at most 800 characters (the models read 512 tokens) and run them in batches.
3. `_clean_entities` post-processes every entity: it uses the exact text from the input, trims edge punctuation,
   grows pieces cut mid-token back to the whole token (`Node` + `.js` -> `Node.js`; only a dot between letters joins),
   and merges overlapping pieces, keeping the most confident label.
4. Offsets are mapped back into the full text.

A shared lock makes concurrent callers take turns (the pipelines are not thread-safe).

Optional speed-up: `NER_ONNX_DIR` / `NER_ONNX_FILE` run the English model through ONNX Runtime (`onnx_ner.py`; export once with
`export_onnx_ner.py`). The fp32 file gives identical predictions and is about 1.7x faster on CPU; the INT8 file is smaller but lost 0.012 F1
on Twitter text. NER batch size defaults to 1 (`NER_BATCH_SIZE`) because batching was slower on CPU. Optional overlapping windows
(`NER_STRIDE_TOKENS`) and a different chunk size (`NER_CHUNK_CHARS`) exist but did not beat the defaults in our tests.

### 5.4 Translation: the provider chain
`TRANSLATION_PROVIDERS` (default `azure,local`) is an ordered list. For each piece of text the first provider that
works wins; one that fails is skipped and put on a cool-down so later pieces do not wait for it again.

| Provider | What it is | Notes |
|---|---|---|
| `azure` | Azure AI Translator REST API | Needs a key; 4,500-character chunks; rate-limited to `AZURE_CHARS_PER_MINUTE`; 429s are waited out for documents |
| `local` | Offline models via `local_translator.py` | Hindi <-> English: IndicTrans2 (pinned commits); other pairs: Marian; non-English pairs pivot through English; 1,200-character chunks |
| `google` | `deep-translator`'s unofficial Google endpoint | Opt-in; throttled by Google quickly |
| `mymemory` | MyMemory REST API | Opt-in; roughly 5,000 characters/day anonymously |

Interactive requests (`/api/translate`) fail over immediately; background documents are patient (retries and waiting).
The response says which provider was used.

### 5.5 Documents and background jobs
```
POST /api/documents            upload: stream to disk, extract text, keep the text server-side  -> doc_id, preview
POST /api/documents/{id}/analyze   start a background job (one per user at a time)               -> job_id
GET  /api/jobs/{job_id}        progress, stage, warnings, and the result when done
POST /api/jobs/{job_id}/cancel
DELETE /api/documents/{id}
```
The job runs NER (this thread) and translation (a worker pool) **concurrently**, saves the annotation
(SQL row + Mongo document, all-or-nothing) and returns a summary: translated preview, unique entities with counts,
which engines translated, and any warnings. A translation failure never discards the entity analysis.
Limits: 50 MB upload, 1,000,000 characters of text. Jobs are kept in memory while they run and saved in the `jobs` table: after a server
restart, finished jobs can still be read (`GET /api/jobs/{id}`), an interrupted job is restarted once from its saved document, and
`GET /api/jobs` lists the user's 10 most recent jobs so the page can reconnect to a running one.

`doc_extract.py` reads each format; for PDFs it additionally rebuilds paragraphs (see 5.6).

### 5.6 PDF text: layout and reflow
PDFs store positioned words, not paragraphs. The extractor prefers pypdf's *layout* mode (real line breaks and paragraph gaps)
but only when it contains exactly the same words in the same order as the plain mode (so multi-column pages are never
interleaved). The text is then **reflowed**: wrapped lines are joined, hyphenated words repaired, bullets and numbered
items kept on their own lines, and label-like short lines kept apart. Without this, justified PDFs arrive as one word per
line and a translator would treat every word as a sentence.

### 5.7 Database design
**SQL `users`:** `id`, `username` (unique), `email` (unique), `hashed_password`.
**SQL `annotations`:** `id`, `user_id` (FK), `original_text`, `translated_text`, `source_lang`, `target_lang`,
`source_name` (uploaded file name, optional).
**MongoDB `ner_tags`:** `{ annotation_id, user_id, ner_tags: [ {word, entity_group, score, start, end}, … ] }`
(plus `original_text` for pasted text). A save writes Mongo first, then commits SQL, and undoes the Mongo write if the
commit fails; a Mongo outage rolls the SQL row back (HTTP 503).

List views clip very long documents (20,000 characters, 200 entities) and flag `truncated`; single-annotation and export
endpoints return everything.

---

## 6. API endpoints

Base URL `http://localhost:8000`; interactive docs at `/docs`. 🔒 = needs `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/auth/register` · `/api/auth/login` | Create an account / get a JWT |
| `GET` | `/api/auth/me` 🔒 | Current user |
| `POST` | `/api/detect-language` 🔒 | `{text}` -> `{detected_language}` (null if detection fails) |
| `POST` | `/api/ner` 🔒 | `{text, lang?}` -> `{entities}` |
| `POST` | `/api/translate` 🔒 | `{text, source_lang, target_lang}` -> `{translated_text, provider}` |
| `POST` | `/api/annotations` 🔒 | Save an annotation (SQL + Mongo) |
| `GET` | `/api/annotations` 🔒 | Your annotations (newest first, paginated, long ones clipped) |
| `GET` `PUT` `DELETE` | `/api/annotations/{id}` 🔒 | Read / update / delete one |
| `GET` | `/api/export` 🔒 | All your annotations, complete |
| `POST` | `/api/documents` 🔒 | Upload a file (multipart) |
| `POST` | `/api/documents/{id}/analyze` 🔒 | Start the analysis job -> `{job_id}` |
| `DELETE` | `/api/documents/{id}` 🔒 | Remove an uploaded document |
| `GET` | `/api/jobs/{id}` 🔒 | Job status, progress and result |
| `POST` | `/api/jobs/{id}/cancel` 🔒 | Cancel a running job |

Errors are always `{"detail": "<readable message>"}`.

---

## 7. Frontend

- `App.tsx` shows Login/Register when logged out, otherwise the **Annotate** and **History** screens.
- `AuthContext` stores the token (AsyncStorage) and logs the user out automatically when any request returns 401.
- **Annotate** (`HomeScreen.tsx`): text box or file upload; source/target language chips; **Analyze & Translate**.
  Typed text runs detect -> NER -> translate -> save. An uploaded document becomes a file card (name, size, characters,
  preview); analysis runs as a background job with a progress bar and Cancel. Results show the translation, the translator
  used, entities (colour-coded, with counts for documents) and any warnings. Export works on the saved copy.
- **History** (`HistoryScreen.tsx`): expandable cards with file badge, translation and entities; delete with confirmation;
  export of *all* annotations through `/api/export` as JSON, CSV, XLSX, TXT, XML or PDF (print-to-PDF, so every script renders).
- Entity colours: Person blue, Organization green, Location orange, Miscellaneous purple, Date teal.

---

## 8. Design decisions

| Decision | Reason |
|---|---|
| Open-source models by default, cloud optional | No keys or quotas to get started; text can stay on the machine; Azure plugs in for speed and quality |
| Pinned, reviewed revisions for models that run their own code | `trust_remote_code` executes Python from the model repo; a pin stops later changes from altering what runs |
| Pure-Python IndicTrans2 processor | The upstream toolkit needs a C compiler to install; the port installs everywhere |
| Providers as an ordered chain with cool-downs | One throttled or failing service never blocks a whole document |
| Background jobs with progress and cancel | Translating a long document takes minutes; a single HTTP request would time out |
| Keep the entity analysis even if translation fails | The NER result is valuable on its own |
| Polyglot persistence (SQL + MongoDB) | Structured data in SQL; flexible entity JSON in MongoDB |
| Lazy model loading | The server starts instantly; models load on first use |
| Measure, don't claim | `evaluation/` scores the app on public benchmarks so changes can be judged by numbers |

---

## 9. Testing and evaluation

- `cd backend && python -m pytest` runs the suite (no network, no model downloads); `--run-slow` adds the real-model tests.
- `evaluation/eval_ner.py` and `evaluation/eval_translation.py` measure entity F1 and translation chrF++/BLEU; see
  `evaluation/README.md`.

## 10. Known limitations

- Scanned (image-only) PDFs are not supported (no OCR); neither are the old binary `.doc`, `.xls`, `.ppt` formats.
- Offline Hindi translation runs at roughly 60 characters per second on a laptop CPU: fine for a few pages, slow for
  hundreds. Azure is much faster.
- Machine translation can still translate names (for example the word "Gravity" in a heading) or choose literal wording.
- The server is built for a single machine: uploads are on local disk (`DOC_DIR`); jobs are saved in the SQL database and survive a restart.
- File upload works in the web app only.

### Per-language quality status (`GET /api/languages`)
`backend/language_status.py` holds the agreed rules (stable: strict F1 >= 0.75, lower end of the 95% range >= 0.70, >= 500 human-labelled sentences, open licence, native-speaker test; beta: F1 0.50-0.75 or a condition missing;
experimental: below 0.50; blocked: no open licence). `evaluation/make_language_status.py` computes the table from the evaluation results into `backend/language_status.json`, so labels follow the measurements. The endpoint adds live flags
(`offline_translation`, `azure_configured`). The frontend (`src/utils/languages.ts`) shows a badge on each language button and a note about quality and translation requirements. Offered: Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati,
Kannada, Malayalam, Punjabi; hidden: Odia, Assamese.
