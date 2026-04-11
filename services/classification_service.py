from transformers import pipeline
from config import settings
import re

classifier = None


# -------------------------------
# 🔹 Lazy Load Model
# -------------------------------
def get_model():
    global classifier
    if classifier is None:
        classifier = pipeline(
            "zero-shot-classification",
            model=settings.MODEL_NAME
        )
    return classifier


# -------------------------------
# 🔹 DOMAIN LABELS (RESTRICTED)
# -------------------------------
DOMAINS = [
    "constitutional law",
    "civil law",
    "criminal law"
]


# -------------------------------
# 🔹 INTENT LABELS (EXPANDED)
# -------------------------------
INTENTS = [
    "definition of law",
    "legal procedure",
    "case lookup",
    "rights and duties",
    "penalties and punishment",
    "compliance and regulation",
    "legal interpretation"
]


# -------------------------------
# 🔹 Rule-based Domain Detection
# -------------------------------
def detect_domain_rule(query: str):
    q = query.lower()

    # -------------------------------
    # 🔹 Detect Section Pattern (SCALABLE FIX)
    # -------------------------------
    has_section = bool(re.search(r"\b(section|sec\.?)\s*\d+\b", q))

    # -------------------------------
    # 🔹 Constitutional Law (HIGH PRIORITY)
    # -------------------------------
    if any(k in q for k in [
        "constitution", "constitutional", "fundamental rights", "bill of rights",
        "amendment", "first amendment", "fourth amendment", "due process",
        "equal protection", "civil rights", "liberty", "freedom of speech",
        "freedom of expression", "judicial review", "separation of powers",
        "federalism", "supreme court", "constitutional validity",
        "unconstitutional", "state action", "usc",
        "civil liberties", "government action", "habeas corpus",
        "mandamus", "certiorari", "prohibition"
    ]) or (has_section and "usc" in q):
        return "constitutional law"

    # -------------------------------
    # 🔹 Criminal Law
    # -------------------------------
    if any(k in q for k in [
        "crime", "criminal", "offense", "offence", "felony", "misdemeanor",
        "punishment", "penalty", "sentence", "imprisonment", "fine",
        "prosecution", "accused", "defendant", "conviction", "trial",
        "evidence", "burden of proof", "beyond reasonable doubt",
        "arrest", "warrant", "custody", "bail", "charge",
        "theft", "murder", "assault", "fraud", "robbery",
        "kidnapping", "homicide", "forgery", "cybercrime",
        "mens rea", "actus reus", "criminal liability"
    ]) or (has_section and any(k in q for k in ["ipc", "crpc"])):
        return "criminal law"

    # -------------------------------
    # 🔹 Civil Law
    # -------------------------------
    if any(k in q for k in [
        "contract", "agreement", "breach", "liability",
        "damages", "compensation", "injunction", "remedy",
        "specific performance", "tort", "negligence",
        "property", "ownership", "lease", "tenancy",
        "consumer protection", "family law", "divorce",
        "custody", "inheritance", "succession",
        "corporate", "company law", "employment",
        "labor", "dispute", "settlement",
        "arbitration", "mediation", "civil suit"
    ]) or (has_section and any(k in q for k in ["contract act", "cpc"])):
        return "civil law"

    return None


# -------------------------------
# 🔹 Rule-based Intent Detection
# -------------------------------
def detect_intent_rule(query: str):
    q = query.lower()

    if any(k in q for k in ["what is", "define", "meaning"]):
        return "definition of law"

    if any(k in q for k in ["how", "procedure", "steps", "process"]):
        return "legal procedure"

    if any(k in q for k in ["case", "judgment", "ruling"]):
        return "case lookup"

    if any(k in q for k in ["rights", "duties", "obligations"]):
        return "rights and duties"

    if any(k in q for k in ["penalty", "punishment", "fine", "imprisonment"]):
        return "penalties and punishment"

    if any(k in q for k in ["compliance", "regulation", "rule"]):
        return "compliance and regulation"

    return None


# -------------------------------
# 🔹 Hybrid Classification
# -------------------------------
def classify_query(data):
    try:
        model = get_model()
        query = data["expanded_query"]

        # -------------------------------
        # 🔹 DOMAIN (Hybrid)
        # -------------------------------
        domain = detect_domain_rule(query)

        if not domain:
            result = model(query, DOMAINS)
            domain = result["labels"][0]

        # -------------------------------
        # 🔹 INTENT (Hybrid)
        # -------------------------------
        intent = detect_intent_rule(query)

        if not intent:
            result = model(query, INTENTS)
            intent = result["labels"][0]

        # -------------------------------
        # 🔹 Final Output (NO COMPLEXITY)
        # -------------------------------
        return {
            "domain": domain,
            "intent": intent,
            "jurisdiction": ["US"]
        }

    except Exception:
        return {
            "domain": "unknown",
            "intent": "unknown",
            "jurisdiction": ["US"],
            "fallback": True
        }