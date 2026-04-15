import os


class Settings:
    APP_NAME = "Legal RAG System"
    VERSION = "v1"

    # -------------------------------
    # 🔹 Query Limits
    # -------------------------------
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 3000))   # slightly increased
    MIN_TOKENS = int(os.getenv("MIN_TOKENS", 1))      # allow short legal queries

    # -------------------------------
    # 🔹 Models
    # -------------------------------
    MODEL_NAME = os.getenv("MODEL_NAME", "facebook/bart-large-mnli")
    LEGAL_BERT_MODEL = os.getenv("LEGAL_BERT_MODEL", "nlpaueb/legal-bert-base-uncased")

    # -------------------------------
    # 🔹 Validation Thresholds
    # -------------------------------
    DOMAIN_THRESHOLD = float(os.getenv("DOMAIN_THRESHOLD", 0.4))

    # Special boost for legal abbreviations / citations
    LEGAL_STRONG_SIGNAL_THRESHOLD = float(os.getenv("LEGAL_STRONG_SIGNAL_THRESHOLD", 0.6))

    # -------------------------------
    # 🔹 Classification Confidence
    # -------------------------------
    MODEL_CONFIDENCE_MARGIN = float(os.getenv("MODEL_CONFIDENCE_MARGIN", 0.08))

    # -------------------------------
    # 🔹 Performance Controls
    # -------------------------------
    ENABLE_GPU = os.getenv("ENABLE_GPU", "true").lower() == "true"
    ENABLE_MODEL_CACHE = os.getenv("ENABLE_MODEL_CACHE", "true").lower() == "true"

    # -------------------------------
    # 🔹 Debug / Logging
    # -------------------------------
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()