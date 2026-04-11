from langdetect import detect
from config import settings
from utils.pii import mask_pii
from transformers import pipeline

# -------------------------------
# 🔹 Legal Keywords (FAST CHECK)
# -------------------------------
LEGAL_KEYWORDS = [
    "law", "legal", "act", "section", "statute", "code",
    "regulation", "compliance", "policy", "rule",
    "contract", "agreement", "liability", "obligation",
    "rights", "duty", "penalty", "violation",
    "court", "judge", "case", "lawsuit", "claim",
    "criminal", "civil", "constitutional", "employment",
    "tax", "intellectual property", "privacy", "data protection",
    "dispute", "settlement", "damages", "breach",

    # ✅ Added abbreviations
    "hipaa", "ada", "usc", "cfr", "gdpr"
]

# -------------------------------
# 🔹 Lazy Load DistilBART Model
# -------------------------------
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3"
        )
    return classifier


# -------------------------------
# 🔹 Hybrid Domain Detection
# -------------------------------
def is_legal_query(query: str):
    try:
        query_lower = query.lower()

        strong_keywords = [
            "rights", "law", "act", "section", "court", "legal",
            "hipaa", "ada", "usc"
        ]

        if any(k in query_lower for k in strong_keywords):
            return True, 0.7

        keyword_matches = sum(1 for k in LEGAL_KEYWORDS if k in query_lower)
        keyword_score = min(keyword_matches / 5, 1.0)

        model = get_classifier()
        labels = ["legal", "non-legal"]
        result = model(query, labels)

        top_label = result["labels"][0]
        score = result["scores"][0]

        if top_label == "legal":
            ml_score = score
        else:
            ml_score = 1 - score

        final_score = (0.5 * ml_score) + (0.5 * keyword_score)
        is_legal = final_score > 0.35

        return is_legal, round(final_score, 3)

    except Exception:
        return True, 0.5


# -------------------------------
# 🔹 Main Validation Function
# -------------------------------
def validate_query(query: str):
    try:
        # ✅ Normalize input
        query = query.strip().lower()

        token_count = len(query.split())

        # -------------------------------
        # 🔹 Length check
        # -------------------------------
        if token_count < settings.MIN_TOKENS:
            return {"error": "too_short"}

        if token_count > settings.MAX_TOKENS:
            return {"error": "too_long"}

        # -------------------------------
        # 🔹 Language detection (FIXED)
        # -------------------------------
        try:
            words = query.split()

            # ✅ Short queries → assume English
            if len(words) <= 3:
                lang = "en"
            else:
                lang = detect(query)

                # ✅ Fallback if detection is wrong
                if lang != "en":
                    if any(c.isalpha() for c in query):
                        lang = "en"
                    else:
                        return {"error": "unsupported_language"}

        except Exception:
            lang = "en"

        # -------------------------------
        # 🔹 Domain validation
        # -------------------------------
        is_legal, confidence = is_legal_query(query)

        if not is_legal:
            return {"error": "out_of_domain"}

        # -------------------------------
        # 🔹 PII masking
        # -------------------------------
        masked_query, pii_flag = mask_pii(query)

        # -------------------------------
        # 🔹 Final response
        # -------------------------------
        return {
            "query": masked_query,
            "pii_masked": pii_flag,
            "token_count": token_count,
            "lang": lang,
            "domain_confidence": confidence
        }

    except Exception as e:
        return {
            "error": "validation_failed",
            "details": str(e)
        }
        