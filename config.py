import os

class Settings:
    APP_NAME = "Legal RAG System"
    VERSION = "v1"

    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2000))
    MIN_TOKENS = int(os.getenv("MIN_TOKENS", 2))

    MODEL_NAME = os.getenv("MODEL_NAME", "facebook/bart-large-mnli")

settings = Settings()