from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Optional
from datetime import datetime, timedelta
import os, json, logging, requests as http_requests
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- Load Environment Variables ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POSTGRES_URL       = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/annotationtool")
MONGO_URL          = os.getenv("MONGO_URL",    "mongodb://localhost:27017")
MONGO_DB           = os.getenv("MONGO_DB",     "annotationtool")
JWT_SECRET         = os.getenv("JWT_SECRET",   "supersecretjwtkey2024annotationtool")
JWT_ALGORITHM      = os.getenv("JWT_ALGORITHM","HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# -------------------------------------------------------
# Auth helpers
# -------------------------------------------------------
pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
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
    owner           = relationship("UserDB", back_populates="annotations")

# Add user_id column if it doesn't exist (safe migration)
from sqlalchemy import inspect, text
def _add_user_id_if_missing():
    try:
        inspector = inspect(engine)
        cols = [c["name"] for c in inspector.get_columns("annotations")]
        if "user_id" not in cols:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE annotations ADD COLUMN user_id INTEGER REFERENCES users(id)"))
                conn.commit()
            logger.info("Added user_id column to annotations table.")
    except Exception as e:
        logger.warning(f"Migration check: {e}")

Base.metadata.create_all(bind=engine)
_add_user_id_if_missing()

# -------------------------------------------------------
# MongoDB Setup
# -------------------------------------------------------
mongo_client   = MongoClient(MONGO_URL)
mongo_db       = mongo_client[MONGO_DB]
ner_collection = mongo_db["ner_tags"]

# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI(title="Language Annotation Tool API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    class Config:
        from_attributes = True

# -------------------------------------------------------
# Translation  (MyMemory REST → Google fallback)
# -------------------------------------------------------

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
    resp = http_requests.get(
        "https://api.mymemory.translated.net/get",
        params={"q": text, "langpair": langpair},
        timeout=10,
    )
    data = resp.json()
    if resp.status_code == 200 and data.get("responseStatus") == 200:
        translated = data["responseData"]["translatedText"]
        if translated and translated.strip():
            return translated
    raise ValueError(f"MyMemory returned: {data.get('responseDetails', 'unknown error')}")

def _google_translate(src: str, tgt: str, text: str) -> str:
    """Google Translate via deep-translator (fallback)."""
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source=src, target=tgt).translate(text)

def perform_translation(src: str, tgt: str, text: str) -> str:
    """MyMemory first → Google fallback → raise HTTPException."""
    # Try MyMemory
    try:
        result = _mymemory_translate(src, tgt, text)
        logger.info(f"MyMemory translation ok: {src}->{tgt}")
        return result
    except Exception as e1:
        logger.warning(f"MyMemory failed ({e1}), trying Google Translate...")
    # Try Google
    try:
        result = _google_translate(src, tgt, text)
        logger.info(f"Google translation ok: {src}->{tgt}")
        return result
    except Exception as e2:
        logger.error(f"Both translators failed: {e2}")
        raise HTTPException(
            status_code=400,
            detail=f"Translation failed ({src}→{tgt}). Please try again later."
        )

# -------------------------------------------------------
# NER  (lazy-loaded BERT)
# -------------------------------------------------------
_ner_pipeline = None

def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        logger.info("Loading NER model …")
        from transformers import pipeline
        _ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        logger.info("NER model loaded.")
    return _ner_pipeline

# -------------------------------------------------------
# Language detection
# -------------------------------------------------------
def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

# -------------------------------------------------------
# Response builder
# -------------------------------------------------------
def _build_response(row: AnnotationDB, mongo_doc) -> AnnotationResponse:
    ner_tags = json.dumps(mongo_doc["ner_tags"]) if mongo_doc and mongo_doc.get("ner_tags") else None
    return AnnotationResponse(
        id              = row.id,
        original_text   = row.original_text,
        translated_text = row.translated_text,
        ner_tags        = ner_tags,
        source_lang     = row.source_lang or "en",
        target_lang     = row.target_lang or "fr",
    )

# =======================================================
# ROUTES
# =======================================================

@app.get("/")
def read_root():
    return {"message": "Backend is running!", "auth": "JWT enabled", "history": "per-user"}

# ── Auth ────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=UserOut, status_code=201)
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(UserDB).filter(
        (UserDB.username == user.username) | (UserDB.email == user.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    new_user = UserDB(
        username        = user.username,
        email           = user.email,
        hashed_password = get_password_hash(user.password),
    )
    db.add(new_user)
    db.commit()
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
    return {"detected_language": detect_language(req.text)}

# ── NER ─────────────────────────────────────────────────

@app.post("/api/ner")
def perform_ner(req: TextRequest, current_user: UserDB = Depends(get_current_user)):
    try:
        nlp     = get_ner_pipeline()
        results = nlp(req.text)
        for r in results:
            r["score"] = round(float(r["score"]), 4)
        return {"entities": results}
    except Exception as e:
        logger.error(f"NER error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Translation ──────────────────────────────────────────

@app.post("/api/translate")
def translate(req: TranslationRequest, current_user: UserDB = Depends(get_current_user)):
    try:
        src = req.source_lang if req.source_lang != "auto" else detect_language(req.text)
        translated = perform_translation(src, req.target_lang, req.text)
        return {"translated_text": translated}
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
    row = AnnotationDB(
        user_id         = current_user.id,
        original_text   = annotation.original_text,
        translated_text = annotation.translated_text,
        source_lang     = annotation.source_lang or "en",
        target_lang     = annotation.target_lang or "fr",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    ner_collection.insert_one({
        "annotation_id": row.id,
        "user_id":       current_user.id,
        "original_text": annotation.original_text,
        "ner_tags":      json.loads(annotation.ner_tags) if annotation.ner_tags else [],
    })
    mongo_doc = ner_collection.find_one({"annotation_id": row.id})
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
    return [_build_response(r, ner_collection.find_one({"annotation_id": r.id})) for r in rows]

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
    if update.original_text   is not None: row.original_text   = update.original_text
    if update.translated_text is not None: row.translated_text = update.translated_text
    db.commit()
    db.refresh(row)
    if update.ner_tags is not None:
        ner_collection.update_one(
            {"annotation_id": annotation_id},
            {"$set": {"ner_tags": json.loads(update.ner_tags)}},
            upsert=True,
        )
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
    ner_collection.delete_one({"annotation_id": annotation_id})
    return {"message": f"Annotation {annotation_id} deleted"}

@app.get("/api/export")
def export_annotations(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user),
):
    rows = db.query(AnnotationDB).filter(
        AnnotationDB.user_id == current_user.id
    ).order_by(AnnotationDB.id).all()
    result = []
    for row in rows:
        mongo_doc = ner_collection.find_one({"annotation_id": row.id})
        result.append({
            "id":              row.id,
            "original_text":   row.original_text,
            "translated_text": row.translated_text,
            "source_lang":     row.source_lang,
            "target_lang":     row.target_lang,
            "ner_tags":        mongo_doc["ner_tags"] if mongo_doc and mongo_doc.get("ner_tags") else [],
        })
    return result
