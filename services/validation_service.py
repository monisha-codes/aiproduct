from langdetect import detect
from config import settings
from utils.pii import mask_pii
from transformers import pipeline
import torch
from utils.legal_abbreviation import smart_expand_abbreviations
from rapidfuzz import process, fuzz
import re
import requests

# -------------------------------
# 🔹 Legal Keywords (FAST CHECK)
# -------------------------------
LEGAL_KEYWORDS = [
    "law", "legal", "act", "section", "statute", "code",
    "regulation", "compliance", "policy", "rule",
    "contract", "agreement", "liability", "obligation",
    "rights", "duty", "penalty", "penalties", "violation", "violations",
    "court", "judge", "case", "lawsuit", "claim",
    "criminal", "civil", "constitutional", "employment",
    "tax", "intellectual property", "privacy", "data protection",
    "dispute", "settlement", "damages", "breach",
    "healthcare", "medical", "patient", "insurance",
    "illegal", "offense", "offence", "fine", "punishment",
    "consumer", "workplace", "harassment", "retaliation",
    "discrimination", "wage", "overtime", "termination",
    "federal", "state", "california", "new york", "texas", "florida",
    "ssn", "social security", "personal data", "confidentiality",
    "nda", "non-disclosure", "licensing", "trademark", "copyright", "patent",

    # Abbreviations / US legal signals
    "hipaa", "ada", "usc", "u.s.c", "cfr", "c.f.r", "gdpr",
    "flsa", "fmla", "eeoc", "osha", "erisa", "rico",
    "dmca", "coppa", "ferpa", "sox", "scotus", "ftc", "doj",
    "irs", "sec", "epa", "ccpa", "fdcpa", "nlrb", "phi",
    "pii", "cwa", "fca", "aca", "hitech", "arpa", "eula"
]

KNOWN_LEGAL_ABBREVIATIONS = {
    "HIPAA", "ADA", "USC", "U.S.C", "CFR", "C.F.R",
    "GDPR", "FLSA", "FMLA", "EEOC", "OSHA", "ERISA",
    "RICO", "DMCA", "COPPA", "FERPA", "SOX", "SCOTUS",
    "FTC", "DOJ", "IRS", "SEC", "EPA", "CCPA", "FDCPA",
    "NLRB", "PHI", "PII", "ACA", "HITECH", "EULA", "NDA"
}

classifier = None
legal_bert = None
validator_llm = None


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

    return legal_bert


def get_validator_llm():
    """
    Optional Ollama fallback for borderline validation queries.
    Keeps existing flow unchanged unless needed.
    """
    global validator_llm

    if validator_llm is None:
        try:
            validator_llm = {
                "base_url": "http://localhost:11434/api/generate",
                "model": "qwen2.5:3b"
            }
        except Exception:
            validator_llm = False

    return validator_llm


# -------------------------------
# 🔹 Helpers
# -------------------------------
def normalize_query_for_domain_check(query: str) -> str:
    if not query:
        return ""

    q = query.strip()
    q = q.replace("U.S.C.", "USC").replace("u.s.c.", "usc")
    q = q.replace("C.F.R.", "CFR").replace("c.f.r.", "cfr")
    q = re.sub(r"\s+", " ", q)
    return q


def extract_uppercase_tokens(query: str):
    if not query:
        return []
    return re.findall(r"\b[A-Z][A-Z0-9\.\-]{1,12}\b", query)


def has_known_legal_abbreviation(query: str) -> bool:
    raw_tokens = extract_uppercase_tokens(query)
    lowered_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\.\-]{1,12}\b", query)

    normalized_known = {a.upper().replace(".", "") for a in KNOWN_LEGAL_ABBREVIATIONS}

    for token in raw_tokens + lowered_tokens:
        normalized = token.upper().replace(".", "")
        if normalized in normalized_known:
            return True

    return False


def correct_legal_abbreviation_typos(query: str):
    """
    Fix likely legal abbreviation spelling mistakes before domain validation.
    Examples:
    hipa -> HIPAA
    uscc -> USC
    flasaa -> FLSA
    """
    try:
        if not query:
            return query, {}

        tokens = query.split()
        corrected_tokens = []
        corrections = {}

        known_abbr_map = {
            abbr.lower().replace(".", ""): abbr
            for abbr in KNOWN_LEGAL_ABBREVIATIONS
        }
        known_abbr_keys = list(known_abbr_map.keys())

        for token in tokens:
            raw = token
            clean = re.sub(r"[^a-zA-Z\.]", "", token).lower().replace(".", "")

            if not clean or len(clean) < 3:
                corrected_tokens.append(raw)
                continue

            if clean in known_abbr_map:
                corrected_tokens.append(known_abbr_map[clean])
                continue

            match = process.extractOne(clean, known_abbr_keys, scorer=fuzz.ratio)

            if match:
                candidate, score, _ = match
                if score >= 80:
                    fixed = known_abbr_map[candidate]
                    corrections[raw] = fixed
                    corrected_tokens.append(fixed)
                    continue

            corrected_tokens.append(raw)

        return " ".join(corrected_tokens), corrections

    except Exception:
        return query, {}


def has_legal_citation(query: str) -> bool:
    patterns = [
        r"\bsection\s+\d+[a-zA-Z0-9\-\(\)]*\b",
        r"\bsec\.?\s+\d+[a-zA-Z0-9\-\(\)]*\b",
        r"\btitle\s+[ivx0-9]+\b",
        r"\b\d+\s+u\.?s\.?c\.?\s+§?\s*\d+[a-zA-Z0-9\-\(\)]*\b",
        r"\b\d+\s+c\.?f\.?r\.?\s+§?\s*\d+[a-zA-Z0-9\-\(\)]*\b",
        r"§\s*\d+[a-zA-Z0-9\-\(\)]*",
    ]
    return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)


def has_strong_legal_signal(query: str, expanded_query: str, smart_abbr: dict) -> bool:
    query_lower = expanded_query.lower()

    strong_keywords = [
        "law", "act", "section", "court", "legal",
        "regulation", "compliance", "rights",
        "liability", "statute", "tax", "rule",
        "privacy", "breach", "contract", "penalty",
        "violation", "employment", "discrimination",
        "consumer", "california", "federal", "state"
    ]

    strong_hit = any(k in query_lower for k in strong_keywords)
    abbreviation_hit = has_known_legal_abbreviation(query)
    citation_hit = has_legal_citation(query)
    smart_abbr_hit = len(smart_abbr) > 0

    return strong_hit or abbreviation_hit or citation_hit or smart_abbr_hit


def should_use_llm_validation_fallback(query: str, confidence: float) -> bool:
    """
    Only use LLM for borderline legal-domain cases.
    Existing validation flow remains the same.
    """
    try:
        q = (query or "").lower().strip()

        if not q:
            return False

        if confidence >= 0.45:
            return False

        if confidence < 0.18:
            return False

        legal_indicators = [
            "penalty", "penalties", "violation", "violations", "illegal",
            "compliance", "law", "legal", "rule", "rights", "breach",
            "contract", "employment", "privacy", "california", "state",
            "federal", "ssn", "discrimination", "retaliation", "wage",
            "termination", "court", "case"
        ]

        if any(term in q for term in legal_indicators):
            return True

        if has_known_legal_abbreviation(query) or has_legal_citation(query):
            return True

        return False

    except Exception:
        return False


def llm_validate_and_rewrite(query: str) -> tuple[bool, float, str]:
    """
    Optional Ollama fallback for borderline queries.
    Returns: (is_legal, confidence, rewritten_query)
    """
    try:
        model = get_validator_llm()
        if not model:
            return False, 0.0, query

        prompt = (
            "You are validating a user query for a legal search system.\n"
            "Decide whether the query is about law, regulation, compliance, court matters, legal rights, penalties, violations, contracts, employment law, privacy law, or similar legal topics.\n"
            "If it is legal, rewrite it into a short natural legal search query.\n"
            "Return ONLY this exact format:\n"
            "LEGAL: yes/no\n"
            "CONFIDENCE: 0.0 to 1.0\n"
            "QUERY: rewritten query\n\n"
            f"User query: {query}"
        )

        response = requests.post(
            model["base_url"],
            json={
                "model": model["model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=30
        )

        if response.status_code != 200:
            return False, 0.0, query

        data = response.json()
        text = data.get("response", "").strip()

        legal_match = re.search(r"LEGAL:\s*(yes|no)", text, flags=re.IGNORECASE)
        conf_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
        query_match = re.search(r"QUERY:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)

        is_legal = bool(legal_match and legal_match.group(1).lower() == "yes")
        confidence = float(conf_match.group(1)) if conf_match else 0.0
        rewritten_query = query_match.group(1).strip() if query_match else query

        rewritten_query = re.sub(r'^\s*(query:)\s*', '', rewritten_query, flags=re.IGNORECASE).strip()

        return is_legal, confidence, rewritten_query or query

    except Exception:
        return False, 0.0, query


# -------------------------------
# 🔹 Hybrid Domain Detection (REFINED)
# -------------------------------
def is_legal_query(query: str):
    try:
        raw_query = query or ""

        corrected_query, abbr_typos = correct_legal_abbreviation_typos(raw_query)
        normalized_query = normalize_query_for_domain_check(corrected_query)

        expanded_query, smart_abbr = smart_expand_abbreviations(normalized_query)
        query_lower = expanded_query.lower()

        if abbr_typos:
            smart_abbr.update(abbr_typos)

        # -------------------------------
        # 🔹 Strong keywords
        # -------------------------------
        strong_keywords = [
            "law", "act", "section", "court", "legal",
            "regulation", "compliance", "rights",
            "liability", "statute", "tax", "rule",
            "privacy", "breach", "contract", "penalty",
            "violation", "employment", "discrimination",
            "consumer", "california", "federal", "state"
        ]

        strong_hit = any(k in query_lower for k in strong_keywords)

        # -------------------------------
        # 🔹 Abbreviation / citation detection
        # -------------------------------
        has_abbreviation = has_known_legal_abbreviation(corrected_query)
        has_citation = has_legal_citation(corrected_query)

        # -------------------------------
        # 🔹 Keyword score
        # -------------------------------
        keyword_matches = sum(1 for k in LEGAL_KEYWORDS if k in query_lower)
        keyword_score = min(keyword_matches / 7, 1.0)

        if strong_hit:
            keyword_score = max(keyword_score, 0.82)

        if has_citation:
            keyword_score = max(keyword_score, 0.90)

        if has_abbreviation:
            keyword_score = max(keyword_score, 0.80)

        # -------------------------------
        # 🔹 Fast-path for obvious legal queries
        # -------------------------------
        if has_strong_legal_signal(corrected_query, expanded_query, smart_abbr):
            fast_path_score = max(keyword_score, 0.72)
        else:
            fast_path_score = keyword_score

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
            0.34 * fast_path_score +
            0.33 * ml_score +
            0.33 * bert_score
        )

        if keyword_score >= 0.45:
            final_score += 0.12

        if len(smart_abbr) > 0:
            final_score += 0.18

        if has_abbreviation:
            final_score += 0.18

        if has_citation:
            final_score += 0.20

        if ml_score > 0.6 and bert_score > 0.6:
            final_score += 0.08

        if any(term in query_lower for term in [
            "penalties", "violation", "violations", "illegal",
            "employment", "discrimination", "retaliation",
            "privacy", "breach", "consumer", "california"
        ]):
            final_score += 0.10

        # boost when abbreviation typo correction succeeded
        if abbr_typos:
            final_score = max(final_score, 0.68)

        final_score = min(final_score, 1.0)

        # -------------------------------
        # 🔹 Safer floor for short legal queries
        # -------------------------------
        short_query = len(raw_query.split()) <= 6
        if short_query and (has_abbreviation or has_citation or len(smart_abbr) > 0 or abbr_typos):
            final_score = max(final_score, 0.65)

        return True, round(final_score, 3)

    except Exception as e:
        print("ERROR in is_legal_query:", str(e))
        return True, 0.5


# -------------------------------
# 🔹 Main Validation Function (SAME PROCESS)
# -------------------------------
def validate_query(query: str):
    try:
        raw_query = query.strip()
        normalized_query = raw_query.lower()

        token_count = len(normalized_query.split())

        # -------------------------------
        # 🔹 Length check
        # -------------------------------
        if token_count < settings.MIN_TOKENS:
            is_legal, confidence = is_legal_query(raw_query)
            if confidence < 0.4:
                return {"error": "too_short"}

        if token_count > settings.MAX_TOKENS:
            return {"error": "too_long"}

        # -------------------------------
        # 🔹 Language detection
        # -------------------------------
        try:
            words = normalized_query.split()

            if len(words) <= 3:
                lang = "en"
            else:
                lang = detect(normalized_query)

                if lang != "en":
                    if any(c.isalpha() for c in normalized_query):
                        lang = "en"
                    else:
                        return {"error": "unsupported_language"}

        except Exception:
            lang = "en"

        # -------------------------------
        # 🔹 Domain validation
        # -------------------------------
        is_legal, confidence = is_legal_query(raw_query)
        validated_query = raw_query

        # LLM fallback only for borderline cases
        if confidence < 0.4 and should_use_llm_validation_fallback(raw_query, confidence):
            llm_is_legal, llm_confidence, llm_query = llm_validate_and_rewrite(raw_query)

            if llm_is_legal:
                validated_query = llm_query
                confidence = max(confidence, llm_confidence, 0.62)

        if confidence < 0.4:
            return {"error": "out_of_domain"}

        # -------------------------------
        # 🔹 PII masking
        # -------------------------------
        masked_query, pii_flag = mask_pii(validated_query.lower())

        # -------------------------------
        # 🔹 Final response (UNCHANGED SHAPE)
        # -------------------------------
        return {
            "query": masked_query,
            "pii_masked": pii_flag,
            "token_count": token_count,
            "lang": lang,
            "domain_confidence": round(confidence, 3)
        }

    except Exception as e:
        return {
            "error": "validation_failed",
            "details": str(e)
        }