from fastapi import FastAPI
from api.routes import router
from utils.legal_abbreviation import get_embedder
from utils.abbreviation_store import load_abbreviations

from services.validation_service import get_classifier, get_legal_bert, get_validator_llm
from services.preprocessing_service import get_rewriter
from services.classification_service import get_ollama_classifier
from services.system_service import get_system_llm

app = FastAPI(title="Legal RAG API")

app.include_router(router)


@app.on_event("startup")
def load_models():
    try:
        print("🔹 Loading ML models and resources...")

        # Validation / classification models
        get_classifier()
        get_legal_bert()

        # Abbreviation embedding model
        get_embedder()

        # Persistent abbreviation map
        load_abbreviations()

        # Ollama fallback config / warm-up
        get_rewriter()
        get_validator_llm()
        get_ollama_classifier()
        get_system_llm()

        print("✅ All models and resources loaded successfully")

    except Exception as e:
        print(f"❌ Startup loading failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}

