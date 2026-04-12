from fastapi import FastAPI
from api.routes import router
from utils.legal_abbreviation import get_embedder
from utils.abbreviation_store import load_abbreviations

# ✅ Import model loaders (NO logic change)
from services.validation_service import get_classifier, get_legal_bert

app = FastAPI(title="Legal RAG API")

app.include_router(router)


# ✅ NEW: Load models once at startup (NO PROCESS CHANGE)
@app.on_event("startup")
def load_models():
    try:
        print("🔹 Loading ML models...")

        get_classifier()   # zero-shot model
        get_legal_bert()   # legal BERT (if you added it)

        print("✅ Models loaded successfully")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
def load_models():
    get_embedder()
    load_abbreviations()   # ✅ ADD THIS

