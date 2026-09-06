from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import List, Optional
import json

# --- Database Setup (SQLite) ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./annotations.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnnotationDB(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text, index=True)
    translated_text = Column(Text)
    ner_tags = Column(Text) # Stored as JSON string

Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(title="Language Annotation Tool API")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models ---
class TextRequest(BaseModel):
    text: str

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "fr"

class AnnotationCreate(BaseModel):
    original_text: str
    translated_text: Optional[str] = None
    ner_tags: Optional[str] = None

class AnnotationResponse(AnnotationCreate):
    id: int
    class Config:
        from_attributes = True

# --- ML Models Initialization (Lazy Loading) ---
ner_pipeline = None
translation_pipeline = None

def get_ner_pipeline():
    global ner_pipeline
    if ner_pipeline is None:
        from transformers import pipeline
        # Using a small BERT model for NER
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner_pipeline

def get_translation_pipeline(src, tgt):
    from transformers import pipeline
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        return pipeline("translation", model=model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Translation model not found for {src} to {tgt}. Make sure the language code is correct.")

# --- Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.post("/api/ner")
def perform_ner(req: TextRequest):
    try:
        nlp = get_ner_pipeline()
        results = nlp(req.text)
        for res in results:
            res['score'] = float(res['score'])
        return {"entities": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate")
def perform_translation(req: TranslationRequest):
    try:
        translator = get_translation_pipeline(req.source_lang, req.target_lang)
        result = translator(req.text)
        return {"translated_text": result[0]['translation_text']}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/annotations", response_model=AnnotationResponse)
def save_annotation(annotation: AnnotationCreate, db: Session = Depends(get_db)):
    db_annotation = AnnotationDB(**annotation.model_dump())
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)
    return db_annotation

@app.get("/api/annotations", response_model=List[AnnotationResponse])
def get_annotations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    annotations = db.query(AnnotationDB).offset(skip).limit(limit).all()
    return annotations
