# Contributing

Thanks for helping improve the Language Annotation Tool. Bug reports, accuracy improvements, new languages and
documentation fixes are all welcome.

## Set up

```bash
git clone <your fork>
cd Language-Annotation-Tool-with-NER-and-ML-Translation-main

# backend (Python 3.10+)
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env                                    # then edit it (see README: Configuration)

# frontend (Node 20+)
cd ../frontend
npm install
```

You need MongoDB running locally; SQL can be SQLite (`POSTGRES_URL=sqlite:///./annotationtool.db`).

## Run the checks before opening a pull request

```bash
cd backend  && python -m pytest                # about a minute; no network, no model downloads
cd frontend && npx tsc --noEmit
```

* The test suite uses a fake MongoDB, stand-in translators and a stand-in NER model, so it is fast and safe to run
  anywhere. Tests that need the real models are marked `slow`: run them with `python -m pytest --run-slow`
  (downloads roughly 1.5 GB the first time).
* Add or update a test for every behaviour you change. Bug fixes should come with a test that fails without the fix.

## Changing accuracy: measure it

Claims like "better translation" or "fewer false entities" need numbers. The `evaluation/` folder scores the app on public
benchmarks (see `evaluation/README.md`). If you change a model, a threshold or the text pre-processing, run the relevant
script before and after, and put both result tables in the pull request. With a few hundred test sentences, differences
under about 0.02 F1 are noise.

## Where things live

| Path | What it does |
|---|---|
| `backend/main.py` | FastAPI app: auth, annotations, NER, translation provider chain, document jobs |
| `backend/doc_extract.py` | Text extraction (PDF, Word, Excel, PowerPoint, OpenDocument, CSV, HTML, ...) |
| `backend/local_translator.py` | Offline translation (IndicTrans2 for Hindi, Marian for other languages) |
| `backend/indic_processor.py` | Pure-Python port of IndicTransToolkit's pre/post-processing |
| `backend/azure_translator.py` | Azure AI Translator client with rate limiting |
| `backend/tests/` | pytest suite |
| `frontend/src/screens/` | Annotate and History screens |
| `evaluation/` | Benchmark scripts and recorded results |

## Adding a language

1. NER: pick a model that covers it and check its license (`THIRD_PARTY_NOTICES.md` explains what is acceptable).
2. Translation: add the language code to `_MARIAN_TARGETS` / `_MARIAN_SOURCES` in `backend/local_translator.py` (the
   OPUS-MT model must exist for English <-> that language), to `_LANG_MAP` in `backend/main.py` for MyMemory, and to
   `SOURCE_LANGUAGES` in `frontend/src/screens/HomeScreen.tsx` and `LANG_NAMES` in `HistoryScreen.tsx`.
3. Add it to `evaluation/eval_translation.py` and record its scores.

## Adding a translation provider

Write a `_try_<name>(src, tgt, text, cancelled, patient)` function in `backend/main.py` that returns the translation or
`None`, register it in `_TRY`, `KNOWN_PROVIDERS` and `PROVIDER_LABELS`, handle its readiness in `_provider_ready`, and
document the settings in `backend/.env.example`. Add tests like the ones in `tests/test_providers.py`.

## Licensing

By contributing you agree that your contribution is released under the project's MIT license. Do not add code, models
or data whose license is incompatible (for example "non-commercial" ones such as NLLB-200) - see
`THIRD_PARTY_NOTICES.md`.

## Security

Please do not open public issues for security problems; see [SECURITY.md](SECURITY.md).
