from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import os, re, json, logging, unicodedata, secrets, shutil, threading, time, uuid, tempfile, asyncio, requests as http_requests
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from jose import JWTError, jwt
import bcrypt
import doc_extract
import azure_translator
import local_translator
import hinglish
import script_lang
import result_cache
import language_status

# --- Load Environment Variables ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POSTGRES_URL       = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/annotationtool")
MONGO_URL          = os.getenv("MONGO_URL",    "mongodb://localhost:27017")
MONGO_DB           = os.getenv("MONGO_DB",     "annotationtool")
JWT_ALGORITHM      = os.getenv("JWT_ALGORITHM","HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# Never fall back to a known, hardcoded secret: anyone could forge tokens with it.
# If no real secret is configured, use a random one for this process (logins reset on restart).
_UNSAFE_SECRETS = {"", "CHANGE_ME_TO_A_LONG_RANDOM_STRING", "supersecretjwtkey2024annotationtool"}
JWT_SECRET = os.getenv("JWT_SECRET", "")
if JWT_SECRET in _UNSAFE_SECRETS or len(JWT_SECRET) < 32:
    JWT_SECRET = secrets.token_hex(32)
    logger.warning("JWT_SECRET missing/weak in .env - using a random per-process secret; "
                   "users will be logged out on every restart. Set a 32+ char secret in backend/.env.")

# Optional: MyMemory raises the free daily limit from ~5,000 to ~50,000 characters when requests carry a contact email
MYMEMORY_EMAIL = os.getenv("MYMEMORY_EMAIL", "").strip()

# CORS: only the known frontend origins (comma-separated override via CORS_ORIGINS)
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8081,http://127.0.0.1:8081,http://localhost:19006,http://127.0.0.1:19006",
).split(",") if o.strip()]

# -------------------------------------------------------
# Auth helpers
# -------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# -------------------------------------------------------
# PostgreSQL Setup
# -------------------------------------------------------
connect_args = {"check_same_thread": False} if POSTGRES_URL.startswith("sqlite") else {}
engine       = create_engine(POSTGRES_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    annotations     = relationship("AnnotationDB", back_populates="owner", cascade="all, delete-orphan")

class AnnotationDB(Base):
    __tablename__ = "annotations"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    original_text   = Column(Text)
    translated_text = Column(Text)
    source_lang     = Column(String(10), default="en")
    target_lang     = Column(String(10), default="fr")
    source_name     = Column(String(255), nullable=True)   # uploaded file name, if any
    owner           = relationship("UserDB", back_populates="annotations")

# Add columns that older databases don't have yet (safe migration)
from sqlalchemy import inspect, text
_COLUMN_MIGRATIONS = [
    ("user_id",     "INTEGER REFERENCES users(id)"),
    ("source_name", "VARCHAR(255)"),
]
def _add_missing_columns():
    try:
        existing = [c["name"] for c in inspect(engine).get_columns("annotations")]
        for name, ddl in _COLUMN_MIGRATIONS:
            if name not in existing:
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE annotations ADD COLUMN {name} {ddl}"))
                    conn.commit()
                logger.info(f"Added {name} column to annotations table.")
    except Exception as e:
        logger.warning(f"Migration check: {e}")

Base.metadata.create_all(bind=engine)
_add_missing_columns()

# -------------------------------------------------------
# MongoDB Setup
# -------------------------------------------------------
mongo_client   = MongoClient(MONGO_URL)
mongo_db       = mongo_client[MONGO_DB]
ner_collection = mongo_db["ner_tags"]

# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
# -------------------------------------------------------
# Model preloading: load and warm the models once, at startup
# -------------------------------------------------------
# Without this the first user request pays the whole model-loading cost (about 15 s for the English NER model).
# PRELOAD_MODELS is a comma-separated list: ner_en, ner_hi, ner_multi, mt:<src>-<tgt> (e.g. mt:en-hi). "none" turns it off.
# At most NER_MAX_MODELS (default 2) NER models stay loaded, so do not preload more NER models than that.
MODEL_STATUS: dict = {}

def model_cache_dir() -> Path:
    """Where Hugging Face keeps downloaded models: HF_HOME if set (put it on a big drive), else ~/.cache/huggingface."""
    return Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser()

def disk_status() -> dict:
    """Free space on the drive that holds the model cache. Several model downloads and evaluation runs have failed when it filled."""
    path = model_cache_dir()
    while not path.exists() and path.parent != path:
        path = path.parent
    try:
        free = round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return {"model_cache": str(model_cache_dir()), "free_gb": None, "low": False}
    try:
        minimum = float(os.getenv("MIN_FREE_GB", "3"))
    except ValueError:
        minimum = 3.0
    return {"model_cache": str(model_cache_dir()), "free_gb": free, "low": free < minimum}

def _warm_up(name: str):
    if name == "ner_en":
        run_ner("Warm up call for Rahul Sharma in Delhi.", "en")
    elif name == "ner_multi":
        run_ner("Appel de mise en route pour Rahul Sharma à Paris.", "fr")
    elif name == "ner_hi":
        run_ner("राहुल शर्मा कल दिल्ली जा रहा है।", "hi")
    elif name.startswith("mt:") and "-" in name[3:]:
        src, tgt = name[3:].split("-", 1)
        local_translator.translate(src, tgt, "Warm up call.")
    else:
        raise ValueError(f"unknown model name {name!r}")

def preload_models() -> dict:
    disk = disk_status()
    if disk["low"]:
        logger.warning(f"Only {disk['free_gb']} GB free where models are stored ({disk['model_cache']}). A model download "
                       f"(0.3-1.4 GB each) may fail; free space or set HF_HOME to a larger drive.")
    spec = os.getenv("PRELOAD_MODELS", "ner_en").strip().lower()
    names = [] if spec in ("", "none", "0", "off") else [n.strip() for n in spec.split(",") if n.strip()]
    ner_names = [n for n in names if n.startswith("ner_")]
    limit = _ner_max_models()
    if limit and len(ner_names) > limit:
        logger.warning(f"PRELOAD_MODELS lists {len(ner_names)} NER models but NER_MAX_MODELS is {limit}: the first "
                       f"{len(ner_names) - limit} will be unloaded again. Raise NER_MAX_MODELS or preload fewer.")
    for name in names:
        t0 = time.perf_counter()
        try:
            _warm_up(name)
            MODEL_STATUS[name] = {"status": "ready", "seconds": round(time.perf_counter() - t0, 1)}
            logger.info(f"Preloaded {name} in {MODEL_STATUS[name]['seconds']} s")
        except Exception as e:                       # a missing model must never stop the server from starting
            MODEL_STATUS[name] = {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"}
            logger.warning(f"Could not preload {name}: {e}")
    return MODEL_STATUS

def configure_threads():
    """Fix PyTorch's CPU thread count explicitly instead of trusting a default that differs between machines.

    TORCH_THREADS=<n> wins. Otherwise the number of PHYSICAL cores is used: on a 4-core / 8-thread laptop, 8 threads made a
    single NER call about 20% faster but slowed translation by about 34%, so a document job (NER and translation at the
    same time) was about 28% slower overall. If the core count is unknown, PyTorch's own default is left alone."""
    wanted = os.getenv("TORCH_THREADS", "").strip()
    threads = int(wanted) if wanted.isdigit() and int(wanted) > 0 else None
    if threads is None:
        try:
            import psutil
            threads = psutil.cpu_count(logical=False)
        except ImportError:
            threads = None
    if not threads:
        return None
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        return None
    logger.info(f"PyTorch CPU threads: {threads}")
    return threads

@asynccontextmanager
async def lifespan(_app):
    configure_threads()
    await asyncio.to_thread(preload_models)         # requests are accepted only after this returns
    yield

app = FastAPI(title="Language Annotation Tool API", lifespan=lifespan)

MAX_UPLOAD_MB    = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """Refuse obviously oversized uploads from the Content-Length header, before reading the body."""
    if request.method == "POST" and request.url.path == "/api/documents":
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 1_000_000:
            return JSONResponse(status_code=413, content={"detail": f"File too large (max {MAX_UPLOAD_MB} MB)."})
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,   # auth uses a Bearer header, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserDB:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise exc
    except JWTError:
        raise exc
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        raise exc
    return user

# -------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,30}$")
_EMAIL_RE    = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    class Config:
        from_attributes = True

class TextRequest(BaseModel):
    text: str
    lang: Optional[str] = None   # source language; picks the NER model (auto-detected if omitted)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "fr"

class AnnotationCreate(BaseModel):
    original_text:   str
    translated_text: Optional[str] = None
    ner_tags:        Optional[str] = None
    source_lang:     Optional[str] = "en"
    target_lang:     Optional[str] = "fr"
    source_name:     Optional[str] = None

class AnnotationUpdate(BaseModel):
    original_text:   Optional[str] = None
    translated_text: Optional[str] = None
    ner_tags:        Optional[str] = None

class AnnotationResponse(BaseModel):
    id:              int
    original_text:   str
    translated_text: Optional[str] = None
    ner_tags:        Optional[str] = None
    source_lang:     Optional[str] = "en"
    target_lang:     Optional[str] = "fr"
    source_name:     Optional[str] = None
    truncated:       bool = False        # True when a list view clipped very long text/entities
    class Config:
        from_attributes = True

# -------------------------------------------------------
# Translation  (MyMemory REST → Google fallback)
# -------------------------------------------------------

# MyMemory rejects queries over ~500 chars, so stay safely below that
TRANSLATE_CHUNK_CHARS = 450
# BERT NER handles 512 tokens; ~800 chars stays under that even for non-Latin scripts
NER_CHUNK_CHARS       = 800
# Optional overlapping windows (NER_STRIDE_TOKENS > 0, e.g. 64): text goes to the model in pieces of up to NER_WINDOW_CHARS
# and Hugging Face cuts them into 512-token windows that overlap by that many tokens and merges the overlap, so a name on a
# window edge is seen whole. 0 keeps the sentence-aware 800-character chunks.
NER_WINDOW_CHARS      = 3000
MAX_TEXT_CHARS        = 20000

_SENTENCE_RE = re.compile(r"[^.!?।。！？\n]*[.!?।。！？\n]+|[^.!?।。！？\n]+", re.S)

def chunk_text(text: str, max_len: int) -> List[tuple]:
    """Split text into (offset, chunk) pairs of at most max_len chars, preferring sentence boundaries."""
    chunks: List[tuple] = []
    start = end = None
    for m in _SENTENCE_RE.finditer(text):
        # a single over-long sentence is hard-split
        for s in range(m.start(), m.end(), max_len):
            e = min(s + max_len, m.end())
            if start is None:
                start, end = s, e
            elif e - start <= max_len:
                end = e
            else:
                chunks.append((start, text[start:end]))
                start, end = s, e
    if start is not None:
        chunks.append((start, text[start:end]))
    return chunks

def _check_length(text: str):
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Text too long (max {MAX_TEXT_CHARS} characters).")

# Map short codes to full language names for MyMemory
_LANG_MAP = {
    "en": "en-GB", "hi": "hi-IN", "fr": "fr-FR", "es": "es-ES",
    "de": "de-DE", "zh": "zh-CN", "ar": "ar-SA", "ja": "ja-JP",
    "ru": "ru-RU", "auto": "en-GB",
}

def _mymemory_translate(src: str, tgt: str, text: str) -> str:
    """MyMemory free API — 5 000 words/day, no key needed."""
    src_mm = _LANG_MAP.get(src, f"{src}-{src.upper()}")
    tgt_mm = _LANG_MAP.get(tgt, f"{tgt}-{tgt.upper()}")
    langpair = f"{src_mm}|{tgt_mm}"
    params = {"q": text, "langpair": langpair}
    if MYMEMORY_EMAIL:
        params["de"] = MYMEMORY_EMAIL
    resp = http_requests.get("https://api.mymemory.translated.net/get", params=params, timeout=10)
    data = resp.json()
    if resp.status_code == 200 and data.get("responseStatus") == 200:
        translated = data["responseData"]["translatedText"]
        if translated and translated.strip():
            return translated
    raise ValueError(f"MyMemory returned: {data.get('responseDetails', 'unknown error')}")

def _mymemory_long(src: str, tgt: str, text: str) -> str:
    """MyMemory refuses requests over ~500 chars, so translate longer text in small pieces."""
    parts = [_mymemory_translate(src, tgt, c.strip()) for _, c in chunk_text(text, TRANSLATE_CHUNK_CHARS) if c.strip()]
    return " ".join(parts)

# Google wants region codes for Chinese
_GOOGLE_LANG = {"zh": "zh-CN"}

# Google throttles to ~5 requests/second per IP; space request starts out so parallel workers stay under it
GOOGLE_MIN_INTERVAL = 0.35
_google_gate_lock   = threading.Lock()
_google_next_slot   = 0.0

# When a provider reports throttling / bad credentials, every worker skips it for a while and uses the next one
GOOGLE_COOLDOWN_SECONDS = 90
AZURE_THROTTLE_COOLDOWN = 15       # seconds; Azure recovers quickly once the send rate drops
AZURE_AUTH_COOLDOWN     = 600      # wrong key/region or free quota used up: do not hammer it
_blocked_until = {"azure": 0.0, "google": 0.0, "local": 0.0}

def _provider_ok(name: str) -> bool:
    return time.monotonic() >= _blocked_until[name]

def _block_provider(name: str, seconds: float):
    _blocked_until[name] = time.monotonic() + seconds

def _azure_ready() -> bool:
    return azure_translator.is_configured() and _provider_ok("azure")

def _sleep_unless_cancelled(seconds: float, cancelled=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not (cancelled and cancelled()):
        time.sleep(0.25)

def _google_wait_turn():
    global _google_next_slot
    with _google_gate_lock:
        now  = time.monotonic()
        wait = max(0.0, _google_next_slot - now)
        _google_next_slot = max(now, _google_next_slot) + GOOGLE_MIN_INTERVAL
    if wait:
        time.sleep(wait)

def _google_translate(src: str, tgt: str, text: str) -> str:
    """Google Translate via deep-translator."""
    from deep_translator import GoogleTranslator
    _google_wait_turn()
    from deep_translator.exceptions import LanguageNotSupportedException
    s, t = _GOOGLE_LANG.get(src, src), _GOOGLE_LANG.get(tgt, tgt)
    try:
        out = GoogleTranslator(source=s, target=t).translate(text)
    except LanguageNotSupportedException:
        out = GoogleTranslator(source="auto", target=t).translate(text)   # unknown source: let Google detect it
    if not out:
        raise ValueError("Google returned an empty translation")
    return out

# ── translation providers ───────────────────────────────────────────────
# TRANSLATION_PROVIDERS is an ordered, comma-separated list. Default "azure,local": Azure when a key is configured,
# otherwise (or when Azure fails) the offline open-source models. "google" (an unofficial scraper) and "mymemory"
# (a rate-limited free API) are available but must be opted into.
KNOWN_PROVIDERS = ("azure", "local", "google", "mymemory")
PROVIDER_LABELS = {
    "azure":    "Azure Translator",
    "local":    "Local open-source models (offline)",
    "google":   "Google Translate (unofficial)",
    "mymemory": "MyMemory",
}
AZURE_CHUNK_CHARS      = 4500   # Azure accepts 50,000 per request; larger chunks mean fewer calls and better context
LOCAL_CHUNK_CHARS      = 1200   # local models are slow: small chunks keep progress and Cancel responsive
LOCAL_FAILURE_COOLDOWN = 120    # e.g. a model could not be downloaded: do not retry on every chunk

ROMAN_HINDI = "hi-Latn"        # Azure's code for Hindi written in English letters

def _provider_order() -> List[str]:
    raw   = os.getenv("TRANSLATION_PROVIDERS", "azure,local")
    order = [n for n in (x.strip().lower() for x in raw.split(",")) if n in KNOWN_PROVIDERS]
    return order or ["local"]

def _provider_ready(name: str, src: str, tgt: str) -> bool:
    if name == "azure":
        return _azure_ready()
    if name == "local":
        return local_translator.is_available() and local_translator.supports(src, tgt) and _provider_ok("local")
    if name == "google":
        return _provider_ok("google")
    return src != ROMAN_HINDI                              # mymemory (it has no code for Hindi in English letters)

def _first_ready(src: str, tgt: str) -> Optional[str]:
    return next((n for n in _provider_order() if _provider_ready(n, src, tgt)), None)

def _chunk_chars(src: str, tgt: str) -> int:
    first = _first_ready(src, tgt)
    return {"azure": AZURE_CHUNK_CHARS, "google": AZURE_CHUNK_CHARS, "local": LOCAL_CHUNK_CHARS}.get(first, TRANSLATE_CHUNK_CHARS)

# Each _try_* returns the translation, or None meaning "use the next provider".
# patient=True (background documents) retries and waits out throttling; False (a user waiting) fails over quickly.

def _try_azure(src: str, tgt: str, text: str, cancelled, patient: bool = True) -> Optional[str]:
    if not _azure_ready():
        return None
    for attempt in range(4 if patient else 1):
        if cancelled():
            return None
        try:
            return azure_translator.translate(src, tgt, text, cancelled=cancelled)
        except azure_translator.AzureAuthError as e:
            _block_provider("azure", AZURE_AUTH_COOLDOWN)
            logger.error(f"Azure rejected the request ({e}); skipping it for {AZURE_AUTH_COOLDOWN}s")
            return None
        except azure_translator.AzureThrottled as e:
            wait = min(60.0, e.retry_after or 5.0 * (attempt + 1))
            if not patient:
                _block_provider("azure", e.retry_after or AZURE_THROTTLE_COOLDOWN)
                logger.warning(f"Azure throttled ({e}); trying the next provider")
                return None
            logger.warning(f"Azure throttled (attempt {attempt + 1}); waiting {wait:.0f}s")
            _sleep_unless_cancelled(wait, cancelled)
        except Exception as e:
            logger.warning(f"Azure failed (attempt {attempt + 1}): {str(e)[:120]}")
            if patient:
                _sleep_unless_cancelled(2.0 * (attempt + 1), cancelled)
    return None

def _try_local(src: str, tgt: str, text: str, cancelled, patient: bool = True) -> Optional[str]:
    if not _provider_ready("local", src, tgt):
        return None
    try:
        return local_translator.translate(src, tgt, text, cancelled=cancelled)
    except local_translator.Cancelled:
        return None
    except Exception as e:
        logger.error(f"Local translation failed ({type(e).__name__}: {str(e)[:200]}); skipping it for {LOCAL_FAILURE_COOLDOWN}s")
        _block_provider("local", LOCAL_FAILURE_COOLDOWN)
        return None

def _try_google(src: str, tgt: str, text: str, cancelled, patient: bool = True) -> Optional[str]:
    for attempt in range(3 if patient else 1):
        if cancelled() or not _provider_ok("google"):
            return None
        try:
            return _google_translate(src, tgt, text)
        except Exception as e:
            if "too many requests" in str(e).lower() or "429" in str(e):
                _block_provider("google", GOOGLE_COOLDOWN_SECONDS)
                logger.warning(f"Google is throttling us; skipping it for {GOOGLE_COOLDOWN_SECONDS}s")
                return None
            logger.warning(f"Google failed (attempt {attempt + 1}): {str(e)[:120]}")
            if patient:
                _sleep_unless_cancelled(1.5 * (attempt + 1), cancelled)
    return None

def _try_mymemory(src: str, tgt: str, text: str, cancelled, patient: bool = True) -> Optional[str]:
    if not _provider_ready("mymemory", src, tgt):
        return None
    try:
        return _mymemory_long(src, tgt, text)
    except Exception as e:
        logger.warning(f"MyMemory failed: {str(e)[:160]}")
        return None

_TRY = {"azure": _try_azure, "local": _try_local, "google": _try_google, "mymemory": _try_mymemory}

def _translate_piece(src: str, tgt: str, text: str, cancelled=lambda: False, patient: bool = False) -> tuple:
    """Try the configured providers in order. Returns (translation, provider) or (None, None)."""
    if src == "hi" and hinglish.is_romanized_hindi(text):
        # Azure given from=hi returns Hindi typed in English letters UNCHANGED (tested 2026-11-18: no translation, no error); from=hi-Latn translates it.
        # The offline model and MyMemory cannot read it, so they are skipped for such text; Google falls back to auto-detection.
        src = ROMAN_HINDI
    preferred = _first_ready(src, tgt)
    # Only a result from the preferred provider is cached, so a temporary outage never leaves a fallback translation behind.
    key = result_cache.make_key("mt", preferred, src, tgt, text, os.getenv("LOCAL_MT_BEAMS", "2"),
                                os.getenv("LOCAL_MT_JA_MODEL", "")) if preferred else None
    if key:
        hit = result_cache.get_json(key)
        if hit:
            return hit["text"], hit["provider"]
    for name in _provider_order():
        if cancelled():
            break
        out = _TRY[name](src, tgt, text, cancelled, patient)
        if out:
            if key and name == preferred:
                result_cache.put_json(key, {"text": out, "provider": name})
            return out, name
    return None, None

def _no_provider_message(src: str, tgt: str) -> str:
    """Why nothing could translate this pair, in words a user can act on."""
    base = f"Translation failed ({src}→{tgt}): no translation provider could handle it."
    if not local_translator.supports(src, tgt) and not _azure_ready():
        return (f"{base} Offline translation is not available for this language pair, and Azure Translator is not set up. "
                f"Add an Azure key to backend/.env (see check_translation.py) to translate it.")
    return base + " Check the server log."

def perform_translation(src: str, tgt: str, text: str, used: Optional[list] = None) -> str:
    """Translate text of any length. `used`, if given, receives the names of the providers that did the work."""
    if src == tgt:
        return text
    out = []
    for _, chunk in chunk_text(text, _chunk_chars(src, tgt)):
        core = chunk.strip()
        if not core:
            continue
        if not any(ch.isalpha() for ch in core):
            out.append(chunk)
            continue
        translated, name = _translate_piece(src, tgt, core, patient=False)
        if translated is None:
            logger.error(f"No translation provider succeeded for {src}->{tgt} (order: {_provider_order()})")
            raise HTTPException(status_code=502, detail=_no_provider_message(src, tgt))
        out.append(translated + (chunk[len(chunk.rstrip()):] or " "))
        if used is not None and name not in used:
            used.append(name)
    logger.info(f"Translation ok: {src}->{tgt} via {used}")
    return "".join(out).strip()

logger.info("Translation providers (in order): " + ", ".join(
    f"{n}{'' if (n != 'azure' or azure_translator.is_configured()) else ' [no key set]'}" for n in _provider_order()))

# -------------------------------------------------------
# NER  (lazy-loaded BERT)
# -------------------------------------------------------
_ner_pipelines: dict = {}

# English: dslim/bert-base-NER (PER/ORG/LOC/MISC).
# Hindi: neuronsbyisshu/indicbert-hi-ner-naamapadam (CC0, PER/ORG/LOC; strict F1 0.827 on the Naamapadam gold test and 0.755 on HiNER
# against 0.650 and 0.527 for the previous model; it does not tag dates).
# Every other language: multilingual XLM-R fine-tuned on WikiANN (PER/ORG/LOC; no declared licence). Its head has a DATE label, but it predicted 0 dates
# on 4,472 Hindi sentences and 2 on 2,737 German/Spanish/French sentences, so dates are effectively not detected for any language.
def _default_ner_model() -> str:
    """NER_MODEL_EN may name a Hugging Face model id or a local folder (for example a distilled student); default: bert-base-NER."""
    return os.getenv("NER_MODEL_EN", "").strip() or "dslim/bert-base-NER"

NER_MODEL_EN    = _default_ner_model()
# Every language that is not English or one of the nine Indian languages. The default has NO declared licence (a release blocker); the measured open
# alternative is Davlan/xlm-roberta-base-ner-hrl (AFL-3.0, same size and labels), see evaluation/RESULTS.md "Russian, Arabic, Japanese". Switching is the owner's decision.
def _env_model(setting: str, default: str) -> str:
    """A model id or local folder from the environment; empty or unset means the default."""
    return os.getenv(setting, "").strip() or default

NER_MODEL_MULTI = _env_model("NER_MODEL_MULTI", "Davlan/xlm-roberta-base-wikiann-ner")
NER_MODEL_HI    = os.getenv("NER_MODEL_HI", "").strip() or "neuronsbyisshu/indicbert-hi-ner-naamapadam"

# Hindi and eight other Indian languages share one open (CC0) model: trained on Hindi only, it transfers to these languages through IndicBERTv2's
# pretraining (strict F1 0.73-0.85 on human-labelled test sets; see evaluation/RESULTS.md and GET /api/languages for each language's status).
# Odia (F1 0.27) and Assamese (51 test sentences) are NOT included.
INDIC_NER_LANGS = {"hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa"}

def ner_model_name(lang: str) -> str:
    """The model used for a language: English, the Indian-language model, or the multilingual one for everything else."""
    return NER_MODEL_EN if lang == "en" else NER_MODEL_HI if lang in INDIC_NER_LANGS else NER_MODEL_MULTI

def _plain_fast_tokenizer(model: str):
    """A tokenizer built from tokenizer.json, for models whose tokenizer_config names a class newer than the pinned transformers."""
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    local = Path(model)
    file = local / "tokenizer.json" if local.is_dir() else hf_hub_download(model, "tokenizer.json")
    cfg_file = local / "tokenizer_config.json" if local.is_dir() else hf_hub_download(model, "tokenizer_config.json")
    cfg = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
    specials = {k: cfg[k] for k in ("unk_token", "pad_token", "cls_token", "sep_token", "mask_token", "bos_token", "eos_token") if k in cfg}
    return PreTrainedTokenizerFast(tokenizer_file=str(file), model_max_length=512, **specials)

def _hf_ner_pipeline(model: str):
    from transformers import pipeline
    try:
        # "first" keeps whole words; "simple" splits names into ##subword fragments
        return pipeline("ner", model=model, aggregation_strategy="first")
    except ValueError as e:
        if "TokenizersBackend" not in str(e):
            raise
        return pipeline("ner", model=model, tokenizer=_plain_fast_tokenizer(model), aggregation_strategy="first")

# Memory limit: each NER model takes about 0.9-1.1 GB of RAM once loaded (measured), and the app has three (English, Indian, multilingual).
# At most NER_MAX_MODELS stay loaded; when another one is needed, the least recently used is dropped (it is loaded again, in 20-30 s, if needed later).
# 0 means no limit. The dict keeps insertion order, so "most recently used" = moved to the end.
_ner_cache_lock = threading.Lock()      # guards the dict
_ner_load_lock = threading.Lock()       # one model load at a time, so two requests never load the same model twice

def _ner_max_models() -> int:
    try:
        return max(0, int(os.getenv("NER_MAX_MODELS", "2")))
    except ValueError:
        return 2

def _evict_ner_models():
    """Drop the least recently used pipelines until the limit holds (call with _ner_cache_lock held)."""
    limit = _ner_max_models()
    dropped = []
    while limit and len(_ner_pipelines) > limit:
        oldest = next(iter(_ner_pipelines))
        del _ner_pipelines[oldest]
        dropped.append(oldest)
    if dropped:
        import gc
        gc.collect()                    # a request still using a dropped model keeps its own reference until it finishes
        logger.info(f"NER memory limit ({limit} models): unloaded {', '.join(dropped)}")

def _load_ner_pipeline(model: str, lang: str):
    onnx_dir = os.getenv("NER_ONNX_DIR", "").strip()
    onnx_file = os.getenv("NER_ONNX_FILE", "model_int8.onnx").strip()
    if lang == "en" and onnx_dir and Path(onnx_dir, onnx_file).exists():
        # exported with export_onnx_ner.py (model.onnx = fp32, model_int8.onnx = INT8)
        import onnx_ner
        logger.info(f"Loading ONNX NER model {onnx_dir}/{onnx_file} …")
        return onnx_ner.load_pipeline(onnx_dir, file=onnx_file, threads=os.getenv("TORCH_THREADS") or None)
    logger.info(f"Loading NER model {model} …")
    return _hf_ner_pipeline(model)

def get_ner_pipeline(lang: str = "en"):
    """Lazy-load one NER pipeline per model (English text uses the English model), keeping at most NER_MAX_MODELS loaded."""
    model = ner_model_name(lang)
    with _ner_cache_lock:
        if model in _ner_pipelines:
            nlp = _ner_pipelines.pop(model)
            _ner_pipelines[model] = nlp                     # mark as most recently used
            return nlp
    with _ner_load_lock:
        with _ner_cache_lock:                               # another request may have loaded it while we waited
            if model in _ner_pipelines:
                return _ner_pipelines[model]
        nlp = _load_ner_pipeline(model, lang)
        with _ner_cache_lock:
            _ner_pipelines[model] = nlp
            _evict_ner_models()
        logger.info("NER model loaded.")
        return nlp

# Chinese and Japanese are written without spaces, so "the whole word" is the whole clause. For them entities are grouped from word pieces
# (aggregation "simple") and are not grown to the nearest space or punctuation mark.
NO_SPACE_LANGS = {"zh", "ja"}

from ner_clean import _EDGE_PUNCT, _is_wordchar, _snap_to_token, _clean_entities     # noqa: E402  (shared with the evaluation scripts)

class JobCancelled(Exception):
    pass

# HF pipelines/tokenizers are not safe to call from several threads at once
_ner_lock = threading.Lock()

def run_ner(text: str, lang: str, progress=None, cancelled=None, batch_size: Optional[int] = None) -> list:
    """NER over text of any length: chunked, batched, offsets mapped back to the full text.

    progress(done, total) and cancelled() are optional hooks for long background jobs.
    """
    nlp    = get_ner_pipeline(lang)
    if batch_size is None:
        # On a CPU, batching made NER slower in our tests (20,000-character document: batch 1 = 15.3 s, batch 8 = 17.8 s with
        # PyTorch; 9.9 s against 14.6 s with ONNX), and sorting chunks by length did not help. A GPU may behave differently.
        try:
            batch_size = max(1, int(os.getenv("NER_BATCH_SIZE", "1") or 1))
        except ValueError:
            batch_size = 1
    try:
        stride = max(0, min(256, int(os.getenv("NER_STRIDE_TOKENS", "0") or 0)))
    except ValueError:
        stride = 0
    try:
        chunk_chars = max(100, int(os.getenv("NER_CHUNK_CHARS", "") or NER_CHUNK_CHARS))
    except ValueError:
        chunk_chars = NER_CHUNK_CHARS
    chunks = [(o, c) for o, c in chunk_text(text, NER_WINDOW_CHARS if stride else chunk_chars) if c.strip()]
    extra  = {"stride": stride} if stride else {}
    if lang in NO_SPACE_LANGS:
        extra["aggregation_strategy"] = "simple"
    model_id = "|".join([ner_model_name(lang), os.getenv("NER_ONNX_DIR", "") if lang == "en" else "",
                         os.getenv("NER_ONNX_FILE", "") if lang == "en" else "", str(stride)])
    results = []
    for i in range(0, len(chunks), batch_size):
        if cancelled and cancelled():
            raise JobCancelled()
        batch = chunks[i:i + batch_size]
        keys = [result_cache.make_key("ner", model_id, lang, c) for _, c in batch]
        cached = [result_cache.get_json(k) for k in keys]
        todo = [n for n, hit in enumerate(cached) if hit is None]
        if todo:
            with _ner_lock:
                outputs = nlp([batch[n][1] for n in todo], batch_size=batch_size, **extra)
            for n, entities in zip(todo, outputs):
                cached[n] = _clean_entities(batch[n][1], entities, 0, snap=lang not in NO_SPACE_LANGS)   # offsets relative to the chunk: reusable
                result_cache.put_json(keys[n], cached[n])
        for (offset, _), entities in zip(batch, cached):
            results.extend({**e, "start": e["start"] + offset, "end": e["end"] + offset} for e in entities)
        if progress:
            progress(min(i + batch_size, len(chunks)), len(chunks))
    return results

# -------------------------------------------------------
# Language detection
# -------------------------------------------------------
def normalize_lang(code: str) -> str:
    """langdetect returns e.g. 'zh-cn' / 'zh-tw'; the app and translators use the base code."""
    return (code or "").strip().lower().split("-")[0].split("_")[0]

DEVANAGARI_LANGS = {"hi", "mr", "ne"}         # the Devanagari languages langdetect knows

def detect_language(text: str) -> Optional[str]:
    """Return the detected base language code, or None if detection fails.

    Hindi typed in English letters ("Rahul kal Mumbai ja raha hai") is reported as "hi": langdetect has no profile for it (it said Indonesian, Swahili or English),
    while the Hindi NER model reads it well (strict F1 0.94 on 125 test sentences against 0.71 for the old multilingual model; evaluation/RESULTS.md).

    Indian scripts are decided by the script first (script_lang.py): Odia, Punjabi, Gujarati, Tamil, Telugu, Kannada and Malayalam each have their
    own script, Bengali script is Bengali or Assamese (Assamese letters), and Devanagari goes to langdetect (Hindi / Marathi / Nepali) with Hindi as
    the fallback. Before this, Odia made langdetect fail, so Odia text was treated as English.
    """
    if hinglish.is_romanized_hindi(text):
        return "hi"
    script = script_lang.script_language(text)
    if script and script != "deva":
        return script
    try:
        from langdetect import detect
        lang = normalize_lang(detect(text))
    except Exception:
        lang = None
    if script == "deva" and lang not in DEVANAGARI_LANGS:
        return "hi"                                 # Devanagari text that langdetect could not place (or placed outside Devanagari languages)
    return lang

# -------------------------------------------------------
# Response builder
# -------------------------------------------------------
# List views clip very long (document-sized) text/entities so 100 rows don't ship megabytes.
LIST_TEXT_CHARS   = 20000
LIST_MAX_ENTITIES = 200

def _build_response(row: AnnotationDB, mongo_doc, clip: bool = False) -> AnnotationResponse:
    tags      = mongo_doc["ner_tags"] if mongo_doc and mongo_doc.get("ner_tags") else None
    original  = row.original_text or ""
    translated = row.translated_text
    truncated = False
    if clip:
        if len(original) > LIST_TEXT_CHARS:
            original, truncated = original[:LIST_TEXT_CHARS] + " …", True
        if translated and len(translated) > LIST_TEXT_CHARS:
            translated, truncated = translated[:LIST_TEXT_CHARS] + " …", True
        if tags and len(tags) > LIST_MAX_ENTITIES:
            tags, truncated = tags[:LIST_MAX_ENTITIES], True
    return AnnotationResponse(
        id              = row.id,
        original_text   = original,
        translated_text = translated,
        ner_tags        = json.dumps(tags) if tags else None,
        source_lang     = row.source_lang or "en",
        target_lang     = row.target_lang or "fr",
        source_name     = row.source_name,
        truncated       = truncated,
    )

def _store_annotation(db: Session, user_id: int, original: str, translated: Optional[str], ner_tags: list,
                      src: str, tgt: str, source_name: Optional[str] = None, mongo_text: bool = True):
    """Write the SQL row and its Mongo NER doc so that both exist or neither does."""
    row = AnnotationDB(
        user_id=user_id, original_text=original, translated_text=translated,
        source_lang=src or "en", target_lang=tgt or "fr", source_name=source_name,
    )
    db.add(row)
    db.flush()                       # assigns row.id without committing
    mongo_doc = {"annotation_id": row.id, "user_id": user_id, "ner_tags": ner_tags}
    if mongo_text:
        mongo_doc["original_text"] = original
    try:
        ner_collection.insert_one(mongo_doc)
    except PyMongoError as e:
        db.rollback()
        logger.error(f"MongoDB write failed, annotation not saved: {e}")
        raise HTTPException(status_code=503, detail="NER store unavailable; annotation not saved.")
    try:
        db.commit()
    except Exception:
        db.rollback()
        ner_collection.delete_one({"annotation_id": mongo_doc["annotation_id"]})
        raise
    db.refresh(row)
    return row, mongo_doc

def _mongo_docs(ids: List[int]) -> dict:
    """Fetch NER docs for many annotations in one query (avoids N+1)."""
    if not ids:
        return {}
    try:
        return {d["annotation_id"]: d for d in ner_collection.find({"annotation_id": {"$in": ids}})}
    except PyMongoError as e:
        logger.error(f"MongoDB read failed: {e}")
        raise HTTPException(status_code=503, detail="NER store unavailable.")

def _parse_ner_tags(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        tags = json.loads(raw)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="ner_tags must be valid JSON.")
    if not isinstance(tags, list):
        raise HTTPException(status_code=400, detail="ner_tags must be a JSON list.")
    return tags

# =======================================================
# ROUTES
# =======================================================

@app.get("/")
def read_root():
    return {"message": "Backend is running!", "auth": "JWT enabled", "history": "per-user"}

@app.get("/api/languages")
def list_languages():
    """Quality status (stable / beta / experimental / blocked) of each Indian language's entity recognition, generated from the evaluation results
    by evaluation/make_language_status.py. Public: it describes the service, not a user's data."""
    table = language_status.load()
    languages = []
    for lang in table.get("languages", []):
        code = lang["code"]
        languages.append({**lang, "offline_translation": bool(local_translator.supports("en", code) and local_translator.supports(code, "en"))})
    return {**table, "languages": languages, "azure_configured": bool(_azure_ready())}

@app.get("/health/models")
def health_models():
    """Which models are loaded in memory right now, and the result of the startup preload."""
    return {"preload": MODEL_STATUS, "disk": disk_status(),
            "ner_loaded": list(_ner_pipelines),               # least recently used first
            "ner_max_models": _ner_max_models(),
            "translation_loaded": local_translator.loaded_engines()}

# ── Auth ────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=UserOut, status_code=201)
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    username = user.username.strip()
    email    = user.email.strip().lower()
    # plain-string 400s (not pydantic's 422 list) so the UI can show them directly
    if not _USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="Username must be 3-30 characters: letters, numbers, . _ -")
    if len(email) > 254 or not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if len(user.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password is too long (max 72 bytes)")
    existing = db.query(UserDB).filter(
        (func.lower(UserDB.username) == username.lower()) | (func.lower(UserDB.email) == email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    new_user = UserDB(
        username        = username,
        email           = email,
        hashed_password = get_password_hash(user.password),
    )
    db.add(new_user)
    try:
        db.commit()
    except IntegrityError:      # lost a race with a concurrent registration
        db.rollback()
        raise HTTPException(status_code=400, detail="Username or email already registered")
    db.refresh(new_user)
    return new_user

@app.post("/api/auth/login", response_model=Token)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": create_access_token({"sub": db_user.username}), "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserOut)
def get_me(current_user: UserDB = Depends(get_current_user)):
    return current_user

# ── Language Detection ───────────────────────────────────

@app.post("/api/detect-language")
def detect_lang(req: TextRequest, current_user: UserDB = Depends(get_current_user)):
    # detected_language is null when detection fails; the client decides on a fallback
    lang = detect_language(req.text)
    return {"detected_language": lang, "romanized": lang == "hi" and hinglish.is_romanized_hindi(req.text)}

# ── NER ─────────────────────────────────────────────────

@app.post("/api/ner")
def perform_ner(req: TextRequest, current_user: UserDB = Depends(get_current_user)):
    try:
        _check_length(req.text)
        lang    = normalize_lang(req.lang) if req.lang and req.lang != "auto" else (detect_language(req.text) or "en")
        return {"entities": run_ner(req.text, lang)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NER error: {e}")
        raise HTTPException(status_code=500, detail="Entity recognition failed.")

# ── Translation ──────────────────────────────────────────

@app.post("/api/translate")
def translate(req: TranslationRequest, current_user: UserDB = Depends(get_current_user)):
    try:
        _check_length(req.text)
        src = req.source_lang if req.source_lang != "auto" else (detect_language(req.text) or "en")
        used: list = []
        translated = perform_translation(normalize_lang(src), normalize_lang(req.target_lang), req.text, used)
        return {"translated_text": translated, "provider": PROVIDER_LABELS.get(used[0], used[0]) if used else ""}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Annotations (per-user) ───────────────────────────────

@app.post("/api/annotations", response_model=AnnotationResponse, status_code=201)
def save_annotation(
    annotation: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user),
):
    ner_tags = _parse_ner_tags(annotation.ner_tags)   # validate before touching any DB
    row, mongo_doc = _store_annotation(
        db, current_user.id, annotation.original_text, annotation.translated_text, ner_tags,
        annotation.source_lang or "en", annotation.target_lang or "fr", annotation.source_name,
    )
    return _build_response(row, mongo_doc)

@app.get("/api/annotations", response_model=List[AnnotationResponse])
def get_annotations(
    skip: int = 0, limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user),
):
    rows = (
        db.query(AnnotationDB)
        .filter(AnnotationDB.user_id == current_user.id)
        .order_by(AnnotationDB.id.desc())
        .offset(skip).limit(limit)
        .all()
    )
    docs = _mongo_docs([r.id for r in rows])
    return [_build_response(r, docs.get(r.id), clip=True) for r in rows]

@app.get("/api/annotations/{annotation_id}", response_model=AnnotationResponse)
def get_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user),
):
    row = db.query(AnnotationDB).filter(
        AnnotationDB.id == annotation_id,
        AnnotationDB.user_id == current_user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return _build_response(row, ner_collection.find_one({"annotation_id": annotation_id}))

@app.put("/api/annotations/{annotation_id}", response_model=AnnotationResponse)
def update_annotation(
    annotation_id: int,
    update: AnnotationUpdate,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user),
):
    row = db.query(AnnotationDB).filter(
        AnnotationDB.id == annotation_id,
        AnnotationDB.user_id == current_user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")
    new_tags = _parse_ner_tags(update.ner_tags) if update.ner_tags is not None else None
    if update.original_text   is not None: row.original_text   = update.original_text
    if update.translated_text is not None: row.translated_text = update.translated_text
    if new_tags is not None:
        try:
            ner_collection.update_one(
                {"annotation_id": annotation_id},
                {"$set": {"ner_tags": new_tags}},
                upsert=True,
            )
        except PyMongoError as e:
            db.rollback()
            logger.error(f"MongoDB update failed: {e}")
            raise HTTPException(status_code=503, detail="NER store unavailable; nothing was updated.")
    db.commit()
    db.refresh(row)
    return _build_response(row, ner_collection.find_one({"annotation_id": annotation_id}))

@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user),
):
    row = db.query(AnnotationDB).filter(
        AnnotationDB.id == annotation_id,
        AnnotationDB.user_id == current_user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")
    db.delete(row)
    db.commit()
    try:
        ner_collection.delete_one({"annotation_id": annotation_id})
    except PyMongoError as e:
        logger.warning(f"Orphaned NER doc for annotation {annotation_id}: {e}")
    return {"message": f"Annotation {annotation_id} deleted"}

@app.get("/api/export")
def export_annotations(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user),
):
    rows = db.query(AnnotationDB).filter(
        AnnotationDB.user_id == current_user.id
    ).order_by(AnnotationDB.id).all()
    docs   = _mongo_docs([r.id for r in rows])
    result = []
    for row in rows:
        mongo_doc = docs.get(row.id)
        result.append({
            "id":              row.id,
            "original_text":   row.original_text,
            "translated_text": row.translated_text,
            "source_lang":     row.source_lang,
            "target_lang":     row.target_lang,
            "source_name":     row.source_name,
            "ner_tags":        mongo_doc["ner_tags"] if mongo_doc and mongo_doc.get("ner_tags") else [],
        })
    return result


# =======================================================
# DOCUMENTS: upload -> extract text -> background analyze/translate job
# =======================================================
MAX_DOC_CHARS             = 1_000_000   # ~400 pages of text
DOC_TRANSLATE_CHUNK_CHARS = 4000        # Google accepts up to 5000 chars per request
TRANSLATE_WORKERS         = 4
DOC_PREVIEW_CHARS         = 3000
RESULT_PREVIEW_CHARS      = 20000
MAX_DISPLAY_ENTITIES      = 300
MAX_STORED_ENTITIES       = 50000       # keeps the Mongo document far below its 16 MB limit
DOC_TTL_SECONDS           = 6 * 3600

DOC_DIR = Path(tempfile.gettempdir()) / "annotationtool_docs"
DOC_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict = {}                 # job_id -> state (in memory; jobs end if the server restarts)
JOBS_LOCK = threading.Lock()
_ID_RE = re.compile(r"^[0-9a-f]{32}$")

class DocumentAnalyzeRequest(BaseModel):
    source_lang: str = "auto"
    target_lang: str = "fr"

def _purge_old():
    cutoff = time.time() - DOC_TTL_SECONDS
    for f in DOC_DIR.glob("*"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if v["created"] < cutoff and v["status"] not in ("queued", "running")]:
            del JOBS[jid]

def _load_doc(doc_id: str, user_id: int):
    """Return (meta, text_path) for a document this user uploaded, else 404."""
    gone = HTTPException(status_code=404, detail="Document not found or expired - please upload it again.")
    if not _ID_RE.match(doc_id):
        raise gone
    meta_path, text_path = DOC_DIR / f"{doc_id}.json", DOC_DIR / f"{doc_id}.txt"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise gone
    if meta.get("user_id") != user_id or not text_path.exists():
        raise gone
    return meta, text_path

@app.post("/api/documents")
def upload_document(file: UploadFile = File(...), current_user: UserDB = Depends(get_current_user)):
    """Receive a file, extract its text and keep it server-side for a later analyze call."""
    _purge_old()
    name = Path(file.filename or "document").name[:200]
    try:
        ext = doc_extract.check_supported(name)
    except doc_extract.ExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    doc_id   = uuid.uuid4().hex
    tmp_path = DOC_DIR / f"upload_{doc_id}{ext}"
    size = 0
    try:
        with open(tmp_path, "wb") as out:                 # stream to disk, never hold the file in memory
            while True:
                buf = file.file.read(1024 * 1024)
                if not buf:
                    break
                size += len(buf)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB).")
                out.write(buf)
        if size == 0:
            raise HTTPException(status_code=400, detail="The file is empty.")
        try:
            text, info = doc_extract.extract_text(tmp_path, name)
        except doc_extract.ExtractionError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        # On Windows a virus scanner or a sync tool can hold a just-written file for a moment; a failed delete must not replace the real answer
        # (for example "No text") with a server error. A leftover file is removed by _purge_old() later.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Could not delete the temporary upload {tmp_path.name} ({e}); it will be purged later")

    if len(text) > MAX_DOC_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"This document has {len(text):,} characters of text; the limit is {MAX_DOC_CHARS:,}. Please split it.",
        )

    (DOC_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    meta = {"user_id": current_user.id, "filename": name, "size": size, "chars": len(text), "info": info}
    (DOC_DIR / f"{doc_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    logger.info(f"Document {name} ({size} bytes) -> {len(text)} chars for user {current_user.id}")
    return {
        "doc_id":            doc_id,
        "filename":          name,
        "size":              size,
        "chars":             len(text),
        "words":             len(text.split()),
        "info":              info,
        "preview":           text[:DOC_PREVIEW_CHARS],
        "detected_language": detect_language(text[:5000]),
    }

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, current_user: UserDB = Depends(get_current_user)):
    _load_doc(doc_id, current_user.id)                    # ownership check
    for ext in (".txt", ".json"):
        (DOC_DIR / f"{doc_id}{ext}").unlink(missing_ok=True)
    return {"message": "Document removed"}

# ── background job ──────────────────────────────────────

def _translate_doc_chunk(src: str, tgt: str, chunk: str, cancelled) -> tuple:
    """Translate one chunk. Returns (text, status, provider) with status ok|skipped|failed.

    Providers are tried in TRANSLATION_PROVIDERS order, waiting out throttling. A chunk that cannot be translated
    keeps its original text so one bad request never loses the document.
    """
    core = chunk.strip()
    if not core or not any(ch.isalpha() for ch in core):
        return chunk, "skipped", None
    out, name = _translate_piece(src, tgt, core, cancelled, patient=True)
    if out is None:
        if not cancelled():
            logger.error("Chunk could not be translated by any provider")
        return chunk, "failed", None
    return out + (chunk[len(chunk.rstrip()):] or " "), "ok", name

def _summarize_entities(entities: list) -> list:
    """Unique (word, type) pairs with how often they occur - far more useful than 10,000 repeats."""
    seen: dict = {}
    for e in entities:
        key = (e["word"], e["entity_group"])
        item = seen.get(key)
        if item is None:
            seen[key] = {"word": e["word"], "entity_group": e["entity_group"], "score": e["score"], "count": 1}
        else:
            item["count"] += 1
            item["score"] = max(item["score"], e["score"])
    ranked = sorted(seen.values(), key=lambda x: (-x["count"], -x["score"]))
    return ranked

def _run_document_job(job_id: str):
    job = JOBS[job_id]
    cancelled = lambda: job["cancel"]

    def update(**kw):
        with JOBS_LOCK:
            job.update(kw)

    counters = {"ner": (0, 1), "tr": (0, 1)}
    translate_on = {"v": True}
    weight_tr = {"v": 0.5}          # share of the progress bar that translation represents

    def refresh():
        with JOBS_LOCK:
            nd, nt = counters["ner"]
            td, tt = counters["tr"]
            f_ner = nd / nt if nt else 1
            f_tr  = (td / tt if tt else 1) if translate_on["v"] else 1
            w = weight_tr["v"]
            job["progress"] = round(0.03 + 0.92 * ((1 - w) * f_ner + w * f_tr), 4)
            stage = [f"Finding entities {int(f_ner * 100)}%"]
            if translate_on["v"]:
                stage.append(f"Translating {int(f_tr * 100)}%")
            job["stage"] = " · ".join(stage)

    pool = None
    try:
        update(status="running", stage="Reading document", progress=0.01)
        meta, text_path = _load_doc(job["doc_id"], job["user_id"])
        text = text_path.read_text(encoding="utf-8")

        src = normalize_lang(job["source_lang"])
        if src in ("", "auto"):
            src = detect_language(text[:5000]) or "en"
        tgt = normalize_lang(job["target_lang"])
        translate_on["v"] = src != tgt
        warnings = []
        if not translate_on["v"]:
            warnings.append("Source and target language are the same, so translation was skipped.")

        # translation (network-bound) runs in a thread pool while NER (CPU-bound) runs here
        futures = []
        if translate_on["v"]:
            tr_chunks = chunk_text(text, _chunk_chars(src, tgt))
            counters["tr"] = (0, len(tr_chunks))
            local_first = _first_ready(src, tgt) == "local"
            weight_tr["v"] = 0.85 if local_first else 0.5          # local models take far longer than NER
            pool = ThreadPoolExecutor(max_workers=1 if local_first else TRANSLATE_WORKERS)

            def on_done(_f):
                with JOBS_LOCK:
                    counters["tr"] = (counters["tr"][0] + 1, counters["tr"][1])
                refresh()

            for _, c in tr_chunks:
                f = pool.submit(_translate_doc_chunk, src, tgt, c, cancelled)
                f.add_done_callback(on_done)
                futures.append(f)

        def ner_progress(done, total):
            with JOBS_LOCK:
                counters["ner"] = (done, total)
            refresh()

        update(stage="Loading entity model")
        entities = run_ner(text, src, progress=ner_progress, cancelled=cancelled)
        with JOBS_LOCK:
            counters["ner"] = (1, 1)

        translated = None
        failed = attempted = 0
        engines: list = []
        if futures:
            parts = []
            for f in futures:
                chunk_text_out, status, provider = f.result()
                parts.append(chunk_text_out)
                if provider and provider not in engines:
                    engines.append(provider)
                if status != "skipped":
                    attempted += 1
                if status == "failed":
                    failed += 1
            if cancelled():
                raise JobCancelled()
            if attempted and failed == attempted:
                # keep the (possibly long) entity analysis instead of failing the whole job
                warnings.append("No translation provider could translate this document (see TRANSLATION_PROVIDERS and the "
                                "server log), so the entities were saved without a translation.")
            else:
                if failed:
                    warnings.append(f"{failed} of {attempted} sections could not be translated and were left in the original language.")
                translated = "".join(parts).strip()

        if cancelled():
            raise JobCancelled()
        update(stage="Saving to history", progress=0.97)
        stored = entities
        if len(entities) > MAX_STORED_ENTITIES:
            stored = entities[:MAX_STORED_ENTITIES]
            warnings.append(f"Only the first {MAX_STORED_ENTITIES:,} of {len(entities):,} entities were stored.")
        db = SessionLocal()
        try:
            row, _ = _store_annotation(db, job["user_id"], text, translated, stored, src, tgt, meta["filename"], mongo_text=False)
            annotation_id = row.id
        finally:
            db.close()

        summary = _summarize_entities(entities)
        update(status="done", stage="Done", progress=1.0, warnings=warnings, result={
            "annotation_id":    annotation_id,
            "source_lang":      src,
            "target_lang":      tgt,
            "original_chars":   len(text),
            "translated_chars": len(translated) if translated else 0,
            "translation_engines": [PROVIDER_LABELS.get(e, e) for e in engines],
            "translated_preview": (translated or "")[:RESULT_PREVIEW_CHARS],
            "entity_total":     len(entities),
            "entity_unique":    len(summary),
            "entities":         summary[:MAX_DISPLAY_ENTITIES],
        })
        logger.info(f"Job {job_id} done: {len(text)} chars, {len(entities)} entities, annotation {annotation_id}")
    except JobCancelled:
        update(status="cancelled", stage="Cancelled")
    except HTTPException as e:
        update(status="error", stage="Failed", error=str(e.detail))
    except RuntimeError as e:                      # expected, user-facing failures (e.g. translator down)
        logger.warning(f"Job {job_id} failed: {e}")
        update(status="error", stage="Failed", error=str(e))
    except Exception:
        logger.exception(f"Job {job_id} failed")
        update(status="error", stage="Failed", error="Processing failed unexpectedly.")
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)

def _public_job(job: dict) -> dict:
    with JOBS_LOCK:
        return {k: job.get(k) for k in ("id", "doc_id", "filename", "status", "stage", "progress", "error", "warnings", "result")}

def _own_job(job_id: str, user_id: int) -> dict:
    job = JOBS.get(job_id)
    if not job or job["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Job not found (the server may have restarted). Please run it again.")
    return job

@app.post("/api/documents/{doc_id}/analyze", status_code=202)
def analyze_document(doc_id: str, req: DocumentAnalyzeRequest, current_user: UserDB = Depends(get_current_user)):
    meta, _ = _load_doc(doc_id, current_user.id)
    if not req.target_lang.strip():
        raise HTTPException(status_code=400, detail="Choose a target language.")
    with JOBS_LOCK:
        if any(j["user_id"] == current_user.id and j["status"] in ("queued", "running") for j in JOBS.values()):
            raise HTTPException(status_code=409, detail="A document is already being processed. Wait for it or cancel it first.")
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "id": job_id, "doc_id": doc_id, "user_id": current_user.id, "filename": meta["filename"],
            "source_lang": req.source_lang, "target_lang": req.target_lang,
            "status": "queued", "stage": "Queued", "progress": 0.0, "cancel": False,
            "error": None, "warnings": [], "result": None, "created": time.time(),
        }
    threading.Thread(target=_run_document_job, args=(job_id,), daemon=True, name=f"doc-job-{job_id[:6]}").start()
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, current_user: UserDB = Depends(get_current_user)):
    return _public_job(_own_job(job_id, current_user.id))

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, current_user: UserDB = Depends(get_current_user)):
    job = _own_job(job_id, current_user.id)
    with JOBS_LOCK:
        job["cancel"] = True
    return {"message": "Cancelling"}
