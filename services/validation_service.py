from langdetect import detect
from config import settings
from utils.pii import mask_pii

LEGAL_KEYWORDS = ["law", "act", "section", "contract", "court"]

def validate_query(query: str):
    try:
        token_count = len(query.split())

        if token_count < settings.MIN_TOKENS:
            return {"error": "too_short"}

        if token_count > settings.MAX_TOKENS:
            return {"error": "too_long"}

        lang = detect(query)
        if lang != "en":
            return {"error": "unsupported_language"}

        # Domain check
        domain_conf = 1.0 if any(k in query.lower() for k in LEGAL_KEYWORDS) else 0.3
        if domain_conf < 0.4:
            return {"error": "out_of_domain"}

        masked_query, pii_flag = mask_pii(query)

        return {
            "query": masked_query,
            "pii_masked": pii_flag,
            "token_count": token_count,
            "lang": lang,
            "domain_confidence": domain_conf
        }

    except Exception as e:
        return {"error": "validation_failed", "details": str(e)}