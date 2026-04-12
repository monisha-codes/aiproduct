from langdetect import detect
from config import settings
from utils.pii import mask_pii
from transformers import pipeline
import torch
from utils.legal_abbreviation import smart_expand_abbreviations
import re

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

    # Abbreviations
    "hipaa", "ada", "usc", "cfr", "gdpr"
]

# -------------------------------
# 🔹 Lazy Load Models
# -------------------------------
classifier = None
legal_bert = None


def get_classifier():
    global classifier

    if classifier is None:
        print("🔹 Loading Zero-shot model...")

        device = 0 if torch.cuda.is_available() else -1

        classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=device
        )

        print("✅ Zero-shot model loaded")

    return classifier


# -------------------------------
# 🔹 Legal-BERT
# -------------------------------
def get_legal_bert():
    global legal_bert

    if legal_bert is None:
        print("🔹 Loading Legal-BERT model...")

        device = 0 if torch.cuda.is_available() else -1

        legal_bert = pipeline(
            "text-classification",
            model="nlpaueb/legal-bert-base-uncased",
            device=device,
            truncation=True
        )

        print("✅ Legal-BERT loaded")

    return legal_bert   # ✅ ADD THIS LINE


# -------------------------------
# 🔹 Hybrid Domain Detection (UPGRADED)
# -------------------------------
def is_legal_query(query: str):
    try:
        expanded_query, smart_abbr = smart_expand_abbreviations(query)
        query_lower = expanded_query.lower()

        # -------------------------------
        # 🔹 Strong keywords
        # -------------------------------
        strong_keywords = [
            "law", "act", "section", "court", "legal",
            "regulation", "compliance", "rights",
            "liability", "statute", "tax"
        ]

        strong_hit = any(k in query_lower for k in strong_keywords)

        # -------------------------------
        # 🔹 Abbreviation detection
        # -------------------------------
        abbr_words = re.findall(r'\b[A-Z]{2,10}\b', query)
        has_abbreviation = len(abbr_words) > 0

        # -------------------------------
        # 🔹 Keyword score
        # -------------------------------
        keyword_matches = sum(1 for k in LEGAL_KEYWORDS if k in query_lower)
        keyword_score = min(keyword_matches / 6, 1.0)

        # 🔥 preserve strong signal
        if strong_hit:
            keyword_score = max(keyword_score, 0.9)

        # -------------------------------
        # 🔹 ML scoring
        # -------------------------------
        model = get_classifier()
        result = model(expanded_query, ["legal", "non-legal"])

        ml_score = result["scores"][0] if result["labels"][0] == "legal" else 1 - result["scores"][0]
        ml_score *= 0.9

        # -------------------------------
        # 🔹 Legal-BERT
        # -------------------------------
        bert_model = get_legal_bert()
        bert_result = bert_model(expanded_query)
        bert_score = bert_result[0]["score"] * 0.95

        # -------------------------------
        # 🔹 Final score
        # -------------------------------
        final_score = (
            0.3 * keyword_score +
            0.35 * ml_score +
            0.35 * bert_score
        )

        # 🔥 STRONG SIGNAL BOOST (NEW — SAFE)
        if keyword_score >= 0.5:
            final_score += 0.2

        # 🔥 Boosts (TUNED — stronger but safe)
        if len(smart_abbr) > 0:
            final_score += 0.3   # was 0.2

        if has_abbreviation:
            final_score += 0.2   # was 0.15

        # 🔥 Extra stability boost (NEW)
        if ml_score > 0.6 and bert_score > 0.6:
            final_score += 0.1

        # normalize
        final_score = min(final_score, 1.0)

        # 🔥 smarter floor (instead of fixed 0.3)
        if final_score < 0.4 and (has_abbreviation or len(smart_abbr) > 0):
            final_score = 0.6

        # ✅ KEEP YOUR RETURN (UNCHANGED)
        return True, round(final_score, 3)

    except Exception as e:
        print("ERROR in is_legal_query:", str(e))
        return True, 0.5
    
# -------------------------------
# 🔹 Main Validation Function (UNCHANGED)
# -------------------------------
def validate_query(query: str):
    try:
        # Normalize input
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
        # 🔹 Language detection
        # -------------------------------
        try:
            words = query.split()

            if len(words) <= 3:
                lang = "en"
            else:
                lang = detect(query)

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

        if confidence < 0.2:
            return {"error": "out_of_domain"}

        # -------------------------------
        # 🔹 PII masking
        # -------------------------------
        masked_query, pii_flag = mask_pii(query)

        # -------------------------------
        # 🔹 Final response (UNCHANGED)
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
        