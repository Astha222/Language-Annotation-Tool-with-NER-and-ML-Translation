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

# --- Load Environment Variables ---
load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/annotationtool")
MONGO_URL    = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB     = os.getenv("MONGO_DB", "annotationtool")

# -------------------------------------------------------
# PostgreSQL Setup (SQLAlchemy)
# Stores: id, original_text, translated_text
# -------------------------------------------------------
# SQLite needs check_same_thread=False; MySQL/PostgreSQL does not
connect_args = {"check_same_thread": False} if POSTGRES_URL.startswith("sqlite") else {}
engine = create_engine(POSTGRES_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnnotationDB(Base):
    __tablename__ = "annotations"
    id               = Column(Integer, primary_key=True, index=True)
    original_text    = Column(Text)   # index removed - MySQL TEXT columns need key length
    translated_text  = Column(Text)

# Auto-create table in PostgreSQL if not exists
Base.metadata.create_all(bind=engine)

# -------------------------------------------------------
# MongoDB Setup (pymongo)
# Stores: ner_tags (flexible JSON documents)
# -------------------------------------------------------
mongo_client     = MongoClient(MONGO_URL)
mongo_db         = mongo_client[MONGO_DB]
ner_collection   = mongo_db["ner_tags"]   # collection name

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

# PostgreSQL DB dependency
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
    ner_tags:        Optional[str] = None   # JSON string from frontend

class AnnotationResponse(BaseModel):
    id:              int
    original_text:   str
    translated_text: Optional[str] = None
    ner_tags:        Optional[str] = None   # fetched from MongoDB

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
        from transformers import pipeline
        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
    return ner_pipeline

def get_translation_pipeline(src, tgt):
    from transformers import pipeline
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        return pipeline("translation", model=model_name)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Translation model not found for {src} → {tgt}. Check language code."
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

@app.post("/api/ner")
def perform_ner(req: TextRequest):
    """Run NER on input text using BERT model."""
    try:
        nlp     = get_ner_pipeline()
        results = nlp(req.text)
        for res in results:
            res['score'] = float(res['score'])
        return {"entities": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate")
def perform_translation(req: TranslationRequest):
    """Translate text from source to target language."""
    try:
        translator = get_translation_pipeline(req.source_lang, req.target_lang)
        result     = translator(req.text)
        return {"translated_text": result[0]['translation_text']}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/annotations", response_model=AnnotationResponse)
def save_annotation(annotation: AnnotationCreate, db: Session = Depends(get_db)):
    """
    Save annotation:
      - original_text + translated_text → PostgreSQL
      - ner_tags (JSON) → MongoDB
    """
    # 1. Save to PostgreSQL
    db_annotation = AnnotationDB(
        original_text   = annotation.original_text,
        translated_text = annotation.translated_text,
    )
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)

    # 2. Save NER tags to MongoDB (linked by PostgreSQL id)
    ner_doc = {
        "annotation_id": db_annotation.id,
        "original_text": annotation.original_text,
        "ner_tags":      json.loads(annotation.ner_tags) if annotation.ner_tags else []
    }
    ner_collection.insert_one(ner_doc)

    return AnnotationResponse(
        id              = db_annotation.id,
        original_text   = db_annotation.original_text,
        translated_text = db_annotation.translated_text,
        ner_tags        = annotation.ner_tags,
    )

@app.get("/api/annotations", response_model=List[AnnotationResponse])
def get_annotations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Fetch all annotations:
      - Base data from PostgreSQL
      - NER tags merged from MongoDB
    """
    pg_rows = db.query(AnnotationDB).offset(skip).limit(limit).all()

    results = []
    for row in pg_rows:
        # Fetch matching NER doc from MongoDB
        mongo_doc = ner_collection.find_one({"annotation_id": row.id})
        ner_tags  = json.dumps(mongo_doc["ner_tags"]) if mongo_doc else None

        results.append(AnnotationResponse(
            id              = row.id,
            original_text   = row.original_text,
            translated_text = row.translated_text,
            ner_tags        = ner_tags,
        ))
    return results
