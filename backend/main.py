from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Optional
import os
import json
import logging

# --- Load Environment Variables ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/annotationtool")
MONGO_URL    = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB     = os.getenv("MONGO_DB", "annotationtool")

# -------------------------------------------------------
# PostgreSQL Setup (SQLAlchemy)
# -------------------------------------------------------
connect_args = {"check_same_thread": False} if POSTGRES_URL.startswith("sqlite") else {}
engine = create_engine(POSTGRES_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnnotationDB(Base):
    __tablename__ = "annotations"
    id               = Column(Integer, primary_key=True, index=True)
    original_text    = Column(Text)
    translated_text  = Column(Text)
    source_lang      = Column(String(10), default="en")
    target_lang      = Column(String(10), default="fr")

Base.metadata.create_all(bind=engine)

# -------------------------------------------------------
# MongoDB Setup (pymongo)
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

# -------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------
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
# ML Models (Lazy Loading)
# -------------------------------------------------------
ner_pipeline         = None
translation_pipeline = None

def get_ner_pipeline():
    global ner_pipeline
    if ner_pipeline is None:
        logger.info("Loading NER model (first time — may take a moment)...")
        from transformers import pipeline
        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
        logger.info("NER model loaded.")
    return ner_pipeline

def translate_with_helsinki(src: str, tgt: str, text: str) -> str:
    """Try Helsinki-NLP local model translation."""
    from transformers import pipeline
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    logger.info(f"Trying Helsinki-NLP model: {model_name}")
    pipe = pipeline("translation", model=model_name)
    return pipe(text)[0]['translation_text']

def translate_with_google(src: str, tgt: str, text: str) -> str:
    """Fallback: Google Translate via deep-translator (no API key needed)."""
    from deep_translator import GoogleTranslator
    logger.info(f"Using Google Translate fallback: {src} → {tgt}")
    translator = GoogleTranslator(source=src, target=tgt)
    return translator.translate(text)

def perform_translation_hybrid(src: str, tgt: str, text: str) -> str:
    """
    Try Helsinki-NLP first (fast, local, offline).
    Fall back to Google Translate if Helsinki model not available.
    """
    try:
        return translate_with_helsinki(src, tgt, text)
    except Exception as helsinki_err:
        logger.warning(f"Helsinki-NLP unavailable ({helsinki_err}), falling back to Google Translate...")
        try:
            return translate_with_google(src, tgt, text)
        except Exception as google_err:
            raise HTTPException(
                status_code=400,
                detail=f"Translation failed for {src} → {tgt}. Error: {str(google_err)}"
            )

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def detect_language(text: str) -> str:
    """Auto-detect source language using langdetect."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

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

# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "Backend is running!",
        "databases": {
            "postgresql": "Connected ✅ (annotations table)",
            "mongodb":    "Connected ✅ (ner_tags collection)"
        }
    }

# --- Language Detection ---
@app.post("/api/detect-language")
def detect_lang(req: TextRequest):
    """Auto-detect the language of the input text."""
    detected = detect_language(req.text)
    return {"detected_language": detected}

# --- NER ---
@app.post("/api/ner")
def perform_ner(req: TextRequest):
    """Run NER on input text using BERT model."""
    try:
        nlp     = get_ner_pipeline()
        results = nlp(req.text)
        for res in results:
            res['score'] = round(float(res['score']), 4)
        return {"entities": results}
    except Exception as e:
        logger.error(f"NER error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Translation ---
@app.post("/api/translate")
def perform_translation(req: TranslationRequest):
    """Translate text — tries Helsinki-NLP first, falls back to Google Translate."""
    try:
        translated = perform_translation_hybrid(req.source_lang, req.target_lang, req.text)
        return {"translated_text": translated}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Create Annotation ---
@app.post("/api/annotations", response_model=AnnotationResponse, status_code=201)
def save_annotation(annotation: AnnotationCreate, db: Session = Depends(get_db)):
    """
    Save annotation:
      - original_text + translated_text + lang info → PostgreSQL
      - ner_tags (JSON) → MongoDB
    """
    db_annotation = AnnotationDB(
        original_text   = annotation.original_text,
        translated_text = annotation.translated_text,
        source_lang     = annotation.source_lang or "en",
        target_lang     = annotation.target_lang or "fr",
    )
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)

    ner_doc = {
        "annotation_id": db_annotation.id,
        "original_text": annotation.original_text,
        "ner_tags":      json.loads(annotation.ner_tags) if annotation.ner_tags else []
    }
    ner_collection.insert_one(ner_doc)

    mongo_doc = ner_collection.find_one({"annotation_id": db_annotation.id})
    return _build_response(db_annotation, mongo_doc)

# --- Get All Annotations ---
@app.get("/api/annotations", response_model=List[AnnotationResponse])
def get_annotations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Fetch all annotations (PostgreSQL + MongoDB merged)."""
    pg_rows = db.query(AnnotationDB).order_by(AnnotationDB.id.desc()).offset(skip).limit(limit).all()
    results = []
    for row in pg_rows:
        mongo_doc = ner_collection.find_one({"annotation_id": row.id})
        results.append(_build_response(row, mongo_doc))
    return results

# --- Get Single Annotation ---
@app.get("/api/annotations/{annotation_id}", response_model=AnnotationResponse)
def get_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Fetch a single annotation by ID."""
    row = db.query(AnnotationDB).filter(AnnotationDB.id == annotation_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")
    mongo_doc = ner_collection.find_one({"annotation_id": annotation_id})
    return _build_response(row, mongo_doc)

# --- Update Annotation ---
@app.put("/api/annotations/{annotation_id}", response_model=AnnotationResponse)
def update_annotation(annotation_id: int, update: AnnotationUpdate, db: Session = Depends(get_db)):
    """Update an existing annotation."""
    row = db.query(AnnotationDB).filter(AnnotationDB.id == annotation_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")

    if update.original_text is not None:
        row.original_text = update.original_text
    if update.translated_text is not None:
        row.translated_text = update.translated_text

    db.commit()
    db.refresh(row)

    if update.ner_tags is not None:
        ner_collection.update_one(
            {"annotation_id": annotation_id},
            {"$set": {"ner_tags": json.loads(update.ner_tags)}},
            upsert=True
        )

    mongo_doc = ner_collection.find_one({"annotation_id": annotation_id})
    return _build_response(row, mongo_doc)

# --- Delete Annotation ---
@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Delete an annotation from both PostgreSQL and MongoDB."""
    row = db.query(AnnotationDB).filter(AnnotationDB.id == annotation_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")

    db.delete(row)
    db.commit()
    ner_collection.delete_one({"annotation_id": annotation_id})

    logger.info(f"Deleted annotation {annotation_id}")
    return {"message": f"Annotation {annotation_id} deleted successfully"}

# --- Export All Annotations as JSON ---
@app.get("/api/export")
def export_annotations(db: Session = Depends(get_db)):
    """Export all annotations as a JSON array (for dataset download)."""
    pg_rows = db.query(AnnotationDB).order_by(AnnotationDB.id).all()
    results = []
    for row in pg_rows:
        mongo_doc = ner_collection.find_one({"annotation_id": row.id})
        ner_tags  = mongo_doc["ner_tags"] if mongo_doc and mongo_doc.get("ner_tags") else []
        results.append({
            "id":              row.id,
            "original_text":   row.original_text,
            "translated_text": row.translated_text,
            "source_lang":     row.source_lang,
            "target_lang":     row.target_lang,
            "ner_tags":        ner_tags,
        })
    return results
