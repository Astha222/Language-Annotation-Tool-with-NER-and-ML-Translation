# 📚 Language Annotation Tool — Project Documentation

## 1. Project Overview

**Language Annotation Tool** is a full-stack web application that allows users to:
- Enter text in any language
- Automatically detect the source language
- Run **Named Entity Recognition (NER)** using a BERT-based ML model
- **Translate** the text into 8 target languages
- Save, view, delete, and export all annotations

This tool is useful for creating **NLP datasets**, studying **multilingual text**, and building **annotated corpora** for AI/ML research.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Expo Web)                  │
│              React Native  •  TypeScript                │
│         Running on  http://localhost:8081               │
│                                                         │
│   ┌──────────────┐        ┌──────────────────┐          │
│   │  Annotate    │        │    History Tab   │          │
│   │  (HomeScreen)│        │ (HistoryScreen)  │          │
│   └──────┬───────┘        └────────┬─────────┘          │
└──────────┼──────────────────────── ┼────────────────────┘
           │   HTTP / REST API (axios)│
           ▼                         ▼
┌─────────────────────────────────────────────────────────┐
│                 BACKEND (FastAPI / Python)               │
│              Running on  http://localhost:8000           │
│                                                         │
│  ┌─────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  NER    │  │Translation │  │ Annotations CRUD     │  │
│  │  API    │  │   API      │  │ GET/POST/PUT/DELETE  │  │
│  └────┬────┘  └─────┬──────┘  └──────────┬───────────┘  │
│       │             │                    │               │
│   BERT Model   Helsinki-NLP         PostgreSQL +         │
│  (dslim/bert)  + Google Translate    MongoDB             │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### 🖥️ Frontend
| Technology | Version | Purpose |
|---|---|---|
| **React Native** | 0.86.2 | Cross-platform UI framework |
| **Expo** | ~57.0.14 | Development platform & build tooling |
| **TypeScript** | ~6.0.3 | Type-safe JavaScript |
| **Axios** | ^1.20.0 | HTTP client for API calls |
| **React Native Web** | ^0.21.2 | Run React Native in browser |

### ⚙️ Backend
| Technology | Version | Purpose |
|---|---|---|
| **FastAPI** | Latest | REST API framework |
| **Python** | 3.x | Backend language |
| **Uvicorn** | Latest | ASGI web server |
| **SQLAlchemy** | Latest | ORM for PostgreSQL |
| **Pydantic** | Latest | Request/response validation |
| **pymongo** | Latest | MongoDB driver |
| **python-dotenv** | Latest | Environment variables |

### 🤖 AI / ML Libraries
| Library | Purpose |
|---|---|
| **HuggingFace Transformers** | NER model + Helsinki-NLP translation |
| **PyTorch** | ML model inference backend |
| **dslim/bert-base-NER** | Pre-trained BERT for Named Entity Recognition |
| **Helsinki-NLP/opus-mt-\*** | Local translation models (European languages) |
| **deep-translator** | Google Translate fallback (Hindi & others) |
| **langdetect** | Automatic source language detection |

### 🗄️ Databases
| Database | Purpose |
|---|---|
| **PostgreSQL** | Stores: id, original_text, translated_text, source_lang, target_lang |
| **MongoDB** | Stores: NER tag JSON documents (linked by annotation ID) |

---

## 4. Project File Structure

```
Major Project/
├── backend/
│   ├── main.py              ← All FastAPI code (single file)
│   ├── requirements.txt     ← Python dependencies
│   ├── .env                 ← DB connection strings (not committed)
│   └── .env.example         ← Template for .env
│
├── frontend/
│   ├── App.tsx              ← Root component + custom tab navigation
│   ├── index.ts             ← Entry point
│   ├── app.json             ← Expo config
│   ├── package.json         ← Node dependencies
│   └── src/
│       ├── screens/
│       │   ├── HomeScreen.tsx      ← Annotate tab (NER + Translate)
│       │   └── HistoryScreen.tsx   ← History tab (View + Delete + Export)
│       └── navigation/
│           └── AppNavigator.tsx    ← (Legacy, not used — App.tsx handles nav)
│
├── .venv/                   ← Python virtual environment
└── start-project.ps1        ← One-click startup script
```

---

## 5. Backend — How It Works

### Startup & Configuration
- Loads environment variables from `.env` using `python-dotenv`
- Connects to **PostgreSQL** via SQLAlchemy ORM
- Connects to **MongoDB** via pymongo
- Auto-creates the `annotations` table in PostgreSQL if it doesn't exist
- ML models are loaded **lazily** (only on first API call) to keep startup fast

### ML Models (Lazy Loading)
```python
# NER — loaded once and cached globally
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Translation — hybrid engine
#   1st try: Helsinki-NLP/opus-mt-{src}-{tgt}  (local, fast, offline)
#   Fallback: GoogleTranslator via deep-translator (online, all languages)
```

### Request Flow for Analyze & Translate
```
POST /api/detect-language  →  langdetect library  →  returns "en", "hi", etc.
POST /api/ner              →  BERT model           →  returns entity list with scores
POST /api/translate        →  Helsinki-NLP / Google Translate  →  returns translated text
POST /api/annotations      →  saves to PostgreSQL + MongoDB  →  returns saved record
```

### Database Design

**PostgreSQL — `annotations` table**
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER (PK) | Auto-increment primary key |
| `original_text` | TEXT | The input text entered by user |
| `translated_text` | TEXT | Translated output (nullable) |
| `source_lang` | VARCHAR(10) | Detected source language code (e.g. `en`) |
| `target_lang` | VARCHAR(10) | Selected target language code (e.g. `hi`) |

**MongoDB — `ner_tags` collection**
```json
{
  "annotation_id": 1,
  "original_text": "Steve Jobs founded Apple in California",
  "ner_tags": [
    { "word": "Steve Jobs", "entity_group": "PER", "score": 0.9998 },
    { "word": "Apple",      "entity_group": "ORG", "score": 0.9991 },
    { "word": "California", "entity_group": "LOC", "score": 0.9987 }
  ]
}
```
> NER tags are stored in MongoDB (flexible schema) while structured data goes to PostgreSQL — a **polyglot persistence** pattern.

---

## 6. Backend — API Endpoints

### Base URL: `http://localhost:8000`
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — shows DB connection status |
| `POST` | `/api/detect-language` | Auto-detect language of input text |
| `POST` | `/api/ner` | Run Named Entity Recognition on text |
| `POST` | `/api/translate` | Translate text between language pairs |
| `POST` | `/api/annotations` | Save annotation to PostgreSQL + MongoDB |
| `GET` | `/api/annotations` | Get all annotations (paginated) |
| `GET` | `/api/annotations/{id}` | Get a single annotation by ID |
| `PUT` | `/api/annotations/{id}` | Update an existing annotation |
| `DELETE` | `/api/annotations/{id}` | Delete annotation from both databases |
| `GET` | `/api/export` | Export all annotations as JSON array |
| `GET` | `/docs` | Interactive Swagger UI (auto-generated) |

### Example Requests & Responses

**POST `/api/ner`**
```json
// Request
{ "text": "Narendra Modi visited New York last week." }

// Response
{
  "entities": [
    { "word": "Narendra Modi", "entity_group": "PER", "score": 0.9998, "start": 0,  "end": 13 },
    { "word": "New York",      "entity_group": "LOC", "score": 0.9987, "start": 22, "end": 30 }
  ]
}
```

**POST `/api/translate`**
```json
// Request
{ "text": "Hello world", "source_lang": "en", "target_lang": "hi" }

// Response
{ "translated_text": "नमस्ते दुनिया" }
```

---

## 7. Frontend — How It Works

### Navigation
`App.tsx` manages a custom **2-tab navigation** using React state (no navigation library needed):
- **✏️ Annotate tab** → `HomeScreen.tsx`
- **📋 History tab** → `HistoryScreen.tsx`

### Annotate Tab (`HomeScreen.tsx`)

**Step-by-step flow when user clicks "Analyze & Translate":**

```
Step 1 → POST /api/detect-language
         Detects source language (e.g. "en")
         Shows "🌐 Detected language: EN" below input

Step 2 → POST /api/ner
         BERT model finds named entities
         Shows color-coded tags: 🔵 Person | 🟢 Org | 🟠 Location | 🟣 Misc
         Each tag shows confidence % score

Step 3 → POST /api/translate   [Optional — won't block if fails]
         Tries Helsinki-NLP model first (fast, local)
         Falls back to Google Translate (for Hindi, Chinese, etc.)
         Shows translated text with a 📋 Copy button

Step 4 → POST /api/annotations  [Always runs, even if translation fails]
         Saves original + translation to PostgreSQL
         Saves NER JSON to MongoDB
         Shows ✅ toast notification
```

**Features:**
- Multiline text input
- 8 target languages with flag chips: 🇮🇳 Hindi, 🇫🇷 French, 🇪🇸 Spanish, 🇩🇪 German, 🇨🇳 Chinese, 🇸🇦 Arabic, 🇯🇵 Japanese, 🇷🇺 Russian
- Toast notifications (green=success, red=error, blue=info)
- Clear button to reset all state
- Loading spinner during API calls
- Copy translation to clipboard (Web Clipboard API)

### History Tab (`HistoryScreen.tsx`)

- **Auto-loads** on mount via `useEffect` → calls `GET /api/annotations`
- Shows each annotation as an **expandable card**
- Cards show: ID badge, source→target language, original text preview
- Expanded view shows: full translation, color-coded NER tags
- **Delete** button with confirmation dialog → calls `DELETE /api/annotations/{id}`
- **Load History** button for manual refresh
- **Export JSON** button → calls `GET /api/export` → triggers browser download of `.json` file
- Empty state screen with guidance when no data exists

---

## 8. NER Entity Types & Colors

| Entity Type | Meaning | Color |
|---|---|---|
| `PER` | Person name | 🔵 Blue `#1565c0` |
| `ORG` | Organization | 🟢 Green `#2e7d32` |
| `LOC` | Location / Place | 🟠 Orange `#e65100` |
| `MISC` | Miscellaneous | 🟣 Purple `#6a1b9a` |

---

## 9. Translation Engine — Hybrid Strategy

```
User selects target language
           ↓
Backend calls Helsinki-NLP/opus-mt-{src}-{tgt}
           ↓ (success)                ↓ (model not found / error)
    Return translation         Fall back to Google Translate
                               via deep-translator library
                                        ↓
                               Return translation
```

**When Helsinki-NLP is used:** French, Spanish, German (fast, local, no internet needed)
**When Google Translate is used:** Hindi, Chinese, Arabic, Japanese, Russian (online)

---

## 10. Environment Variables

**`backend/.env`**
```env
POSTGRES_URL=postgresql+psycopg2://postgres:admin123@localhost:5432/annotationtool
MONGO_URL=mongodb://localhost:27017
MONGO_DB=annotationtool
```

---

## 11. Python Dependencies (`requirements.txt`)

```
fastapi          # REST API framework
uvicorn          # ASGI server
sqlalchemy       # ORM for PostgreSQL
pydantic         # Data validation
transformers     # HuggingFace ML models (BERT NER + Helsinki translation)
torch            # PyTorch — ML inference backend
psycopg2-binary  # PostgreSQL driver
pymongo          # MongoDB driver
python-dotenv    # .env file loader
langdetect       # Source language auto-detection
deep-translator  # Google Translate fallback (no API key needed)
```

---

## 12. How to Run the Project

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL running locally
- MongoDB running locally

### One-Click Start
```powershell
# From project root:
powershell -ExecutionPolicy Bypass -File "start-project.ps1"
```

### Manual Start

**Backend:**
```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend:**
```powershell
cd frontend
npx expo start --web --port 8081
```

### Access URLs
| URL | Description |
|---|---|
| `http://localhost:8081` | Frontend web app |
| `http://localhost:8000` | Backend API |
| `http://localhost:8000/docs` | Swagger API documentation |

---

## 13. Key Design Decisions

| Decision | Reason |
|---|---|
| **Polyglot persistence** (PostgreSQL + MongoDB) | Structured data in SQL, flexible NER JSON in NoSQL |
| **Lazy ML model loading** | Keeps server startup fast; models load on first request |
| **Hybrid translation** (Helsinki + Google) | Helsinki is fast & offline; Google is universal fallback |
| **Save even if translation fails** | NER results are valuable even without translation |
| **Custom tab navigation** | Avoids react-navigation dependency issues on Expo Web |
| **Single backend file** | Simple for a project of this scope; easy to understand |
