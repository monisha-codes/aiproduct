from transformers import pipeline
from config import settings
import re
import torch
import requests

# -------------------------------
# 🔹 Final controlled taxonomy
# -------------------------------
VALID_DOMAINS = {"contract", "tort", "criminal", "ip", "family", "tax", "other"}
VALID_INTENTS = {"definition", "comparison", "procedure", "case_lookup", "advice"}

JURISDICTION_KEYWORDS = {
    "india": "IN",
    "indian": "IN",
    "us": "US",
    "usa": "US",
    "united states": "US",
    "uk": "UK",
    "united kingdom": "UK",
    "eu": "EU",
    "europe": "EU"
}

TEMPORAL_KEYWORDS = {
    "recent", "latest", "current", "today", "now",
    "update", "updated", "changes", "amendment", "amended",
    "new", "new law", "new rules", "revised",
    "recent years", "last year", "this year", "past year",
    "over time", "history", "trend", "timeline",
    "recent judgment", "latest judgment", "latest case",
    "new act", "new regulation", "latest amendment",
    "before", "after", "since", "from", "till", "until",
    "currently", "ongoing", "in force", "as of now"
}

classifier = None
ollama_classifier = None

# -------------------------------
# 🔹 Model label space
# -------------------------------
DOMAINS = [
    "contract",
    "tort",
    "criminal",
    "ip",
    "family",
    "tax",
    "other"
]

INTENTS = [
    "definition",
    "comparison",
    "procedure",
    "case_lookup",
    "advice"
]

# -------------------------------
# 🔹 Keyword support
# -------------------------------
DOMAIN_KEYWORDS = {
    "contract": [
        "contract", "agreement", "breach", "breach of contract",
        "offer", "acceptance", "consideration", "nda",
        "liability", "termination", "indemnity",
        "void contract", "voidable contract",
        "specific performance", "damages",
        "contract dispute", "service agreement", "employment contract"
    ],
    "tort": [
        "tort", "negligence", "defamation", "libel", "slander",
        "nuisance", "injury", "damages",
        "strict liability", "vicarious liability",
        "product liability", "duty of care"
    ],
    "criminal": [
        "crime", "criminal", "offense", "offence",
        "murder", "theft", "fraud", "assault", "robbery",
        "ipc", "crpc", "evidence act",
        "bail", "arrest", "fir", "charge sheet",
        "cyber crime"
    ],
    "ip": [
        "intellectual property", "copyright",
        "trademark", "patent",
        "infringement", "licensing",
        "trade secret", "royalty"
    ],
    "family": [
        "divorce", "marriage",
        "custody", "child custody",
        "alimony", "maintenance",
        "adoption", "domestic violence"
    ],
    "tax": [
        "tax", "taxation", "income tax",
        "gst", "vat", "corporate tax",
        "tax return", "tax filing",
        "deductions", "capital gains",
        "tds", "tcs"
    ]
}

DOMAIN_NORMALIZATION = {
    "contract": "contract",
    "contract law": "contract",
    "civil law": "other",

    "tort": "tort",
    "tort law": "tort",

    "criminal": "criminal",
    "criminal law": "criminal",

    "ip": "ip",
    "intellectual property": "ip",
    "intellectual property law": "ip",

    "family": "family",
    "family law": "family",

    "tax": "tax",
    "tax law": "tax",

    "constitutional law": "other",
    "employment law": "other",
    "healthcare law": "other",
    "privacy and data protection law": "other",
    "corporate and commercial law": "other",
    "other": "other"
}

INTENT_NORMALIZATION = {
    "definition": "definition",
    "definition of law": "definition",
    "legal interpretation": "definition",
    "statutory explanation": "definition",

    "comparison": "comparison",
    "compare": "comparison",

    "procedure": "procedure",
    "legal procedure": "procedure",

    "case_lookup": "case_lookup",
    "case lookup": "case_lookup",

    "advice": "advice",
    "rights and duties": "advice",
    "penalties and punishment": "advice",
    "compliance and regulation": "advice",
    "contract analysis": "advice"
}

# -------------------------------
# 🔹 Lazy Load Zero-Shot Model
# -------------------------------
def get_model():
    global classifier
    if classifier is None:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "zero-shot-classification",
            model=settings.MODEL_NAME,
            device=device
        )
    return classifier


# -------------------------------
# 🔹 Ollama Classifier Config
# -------------------------------
def get_ollama_classifier():
    global ollama_classifier

    if ollama_classifier is None:
        try:
            ollama_classifier = {
                "base_url": "http://localhost:11434/api/generate",
                "model": "qwen2.5:3b"
            }
        except Exception:
            ollama_classifier = False

    return ollama_classifier


# -------------------------------
# 🔹 Rule signals
# -------------------------------
def detect_domain_rule(query: str):
    q = (query or "").lower()

    # strong exact domain keywords first
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q:
                score = 1.0 if len(keyword.split()) > 1 else 0.94
                return domain, score

    if any(k in q for k in [
        "constitution", "constitutional", "amendment", "bill of rights",
        "due process", "equal protection", "first amendment",
        "fourth amendment", "freedom of speech", "judicial review",
        "federalism", "state action", "habeas corpus", "supreme court"
    ]):
        return "constitutional law", 0.95

    if any(k in q for k in [
        "employment", "employee", "employer", "workplace",
        "wage", "overtime", "discrimination", "harassment", "fmla",
        "flsa", "eeoc", "ada accommodation", "retaliation", "equal employment"
    ]):
        return "employment law", 0.95

    if any(k in q for k in [
        "hipaa", "patient", "medical record", "health information",
        "protected health information", "phi", "healthcare", "hitech"
    ]):
        return "healthcare law", 0.94

    if any(k in q for k in [
        "privacy", "data protection", "personal data", "gdpr",
        "ccpa", "coppa", "consent", "data sharing", "tracking", "pii",
        "confidentiality", "consumer privacy"
    ]):
        return "privacy and data protection law", 0.95

    if any(k in q for k in [
        "corporate", "company", "shareholder", "merger",
        "securities", "sec", "board", "governance"
    ]):
        return "corporate and commercial law", 0.92

    return None, 0.0


def detect_intent_rule(query: str):
    q = (query or "").lower().strip()

    if any(k in q for k in ["compare", "difference between", "vs", "versus", "distinguish"]):
        return "comparison", 0.95

    if any(k in q for k in ["how to", "procedure", "steps", "process", "filing process"]):
        return "procedure", 0.93

    if any(k in q for k in ["case", "judgment", "ruling", "precedent", "holding"]):
        return "case_lookup", 0.95

    if any(k in q for k in ["what is", "define", "meaning of", "explain"]):
        return "definition", 0.90

    if any(k in q for k in [
        "rights", "duties", "obligations", "responsibilities",
        "penalty", "punishment", "fine", "imprisonment", "sentence",
        "violation", "violations", "compliance", "regulation",
        "rule", "rules", "policy requirement", "interpret",
        "interpretation", "meaning of section", "scope of"
    ]):
        return "advice", 0.88

    if re.search(r"\b(section|sec\.?|title|usc|u\.s\.c|cfr|c\.f\.r)\b", q):
        return "definition", 0.90

    return None, 0.0


# -------------------------------
# 🔹 Helpers
# -------------------------------
def classify_with_zero_shot(model, query: str, labels: list[str]):
    result = model(query, labels, multi_label=False)
    return result["labels"][0], float(result["scores"][0])


def detect_temporal(text: str) -> bool:
    q = (text or "").lower()

    if any(word in q for word in TEMPORAL_KEYWORDS):
        return True

    if re.search(r"\b(last|past|over time|recent years|history|trend)\b", q):
        return True

    if re.search(r"\b(19|20)\d{2}\b", q):
        return True

    return False

def llm_detect_temporal(query: str) -> bool:
    try:
        model = get_ollama_classifier()
        if not model:
            return False

        prompt = (
            "Is this legal query time-sensitive?\n"
            "Return ONLY: YES or NO\n\n"
            f"Query: {query}"
        )

        response = requests.post(
            model["base_url"],
            json={
                "model": model["model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=5
        )

        if response.status_code == 200:
            result = response.json().get("response", "").strip().lower()
            return "yes" in result

        return False

    except Exception:
        return False
    
def compute_complexity(intent: str, jurisdiction: list, temporal: bool, query: str) -> str:
    multi_jurisdiction = len(jurisdiction) >= 2
    multi_intent = isinstance(intent, str) and "," in intent
    long_query = len((query or "").split()) > 15

    if multi_jurisdiction or multi_intent or temporal or long_query:
        return "complex"

    return "simple"


def normalize_domain(domain: str) -> str:
    if not domain:
        return "other"

    d = domain.lower().strip()
    d = DOMAIN_NORMALIZATION.get(d, d)

    return d if d in VALID_DOMAINS else "other"


def normalize_intent(intent: str) -> str:
    if not intent:
        return "definition"

    i = intent.lower().strip()
    i = INTENT_NORMALIZATION.get(i, i)

    return i if i in VALID_INTENTS else "definition"


def detect_jurisdiction(text: str) -> list:
    result = set()
    q = (text or "").lower()

    for key, value in JURISDICTION_KEYWORDS.items():
        if key in q:
            result.add(value)

    return list(result)


def get_best_query_text(data: dict) -> str:
    return (
        data.get("restructured_query")
        or data.get("expanded_query")
        or data.get("cleaned_query")
        or data.get("original_query")
        or ""
    )


def get_retrieved_context(data: dict) -> str:
    try:
        if not isinstance(data, dict):
            return ""

        if isinstance(data.get("retrieved_context"), str):
            return data["retrieved_context"].strip()

        chunks = data.get("retrieved_chunks") or data.get("context_chunks") or []
        if isinstance(chunks, list):
            texts = []
            for chunk in chunks[:3]:
                if isinstance(chunk, dict):
                    text = chunk.get("chunk_text") or chunk.get("text") or ""
                    if text:
                        texts.append(str(text).strip())
                elif isinstance(chunk, str):
                    texts.append(chunk.strip())
            return "\n".join(texts).strip()

        return ""
    except Exception:
        return ""


def looks_like_short_definition_query(query: str) -> bool:
    q = (query or "").lower().strip()

    if any(q.startswith(x) for x in ["what is", "explain", "define", "meaning of"]):
        return True

    if any(x in q for x in ["hipaa", "gdpr", "eeoc", "usc", "cfr"]):
        return True

    return False


def llm_classify_query(query: str, context: str = ""):
    try:
        model = get_ollama_classifier()
        if not model:
            return None, None, None, 0.0

        allowed_domains = ", ".join(DOMAINS)
        allowed_intents = ", ".join(INTENTS)

        if context:
            prompt = (
                "Classify this legal user query using both the query and retrieved legal context.\n"
                f"Return ONLY this exact format:\n"
                f"DOMAIN: one of [{allowed_domains}]\n"
                f"INTENT: one of [{allowed_intents}]\n"
                "JURISDICTION: one of [US, IN, UK, EU]\n"
                "CONFIDENCE: 0.0 to 1.0\n\n"
                f"Query: {query}\n\n"
                f"Retrieved context:\n{context}"
            )
        else:
            prompt = (
                "Classify this legal user query.\n"
                f"Return ONLY this exact format:\n"
                f"DOMAIN: one of [{allowed_domains}]\n"
                f"INTENT: one of [{allowed_intents}]\n"
                "JURISDICTION: one of [US, IN, UK, EU]\n"
                "CONFIDENCE: 0.0 to 1.0\n\n"
                f"Query: {query}"
            )

        response = requests.post(
            model["base_url"],
            json={
                "model": model["model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )

        if response.status_code != 200:
            return None, None, None, 0.0

        data = response.json()
        text = data.get("response", "").strip()

        domain_match = re.search(r"DOMAIN:\s*(.+)", text, flags=re.IGNORECASE)
        intent_match = re.search(r"INTENT:\s*(.+)", text, flags=re.IGNORECASE)
        jurisdiction_match = re.search(r"JURISDICTION:\s*(.+)", text, flags=re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)

        domain = domain_match.group(1).strip().lower() if domain_match else None
        intent = intent_match.group(1).strip().lower() if intent_match else None
        jurisdiction = jurisdiction_match.group(1).strip().upper() if jurisdiction_match else None
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0

        valid_domains = {d.lower() for d in DOMAINS}
        valid_intents = {i.lower() for i in INTENTS}

        if domain not in valid_domains:
            domain = None
        if intent not in valid_intents:
            intent = None
        if jurisdiction not in {"US", "IN", "UK", "EU"}:
            jurisdiction = None

        return domain, intent, jurisdiction, confidence

    except Exception:
        return None, None, None, 0.0


# -------------------------------
# 🔹 Hybrid Classification
# -------------------------------
def classify_query(data):
    try:
        model = get_model()
        query = get_best_query_text(data)
        retrieved_context = get_retrieved_context(data)

        rule_domain, rule_domain_score = detect_domain_rule(query)
        rule_intent, rule_intent_score = detect_intent_rule(query)

        llm_domain, llm_intent, llm_jurisdiction, llm_confidence = llm_classify_query(query)

        if retrieved_context:
            rag_domain, rag_intent, rag_jurisdiction, rag_confidence = llm_classify_query(
                query=query,
                context=retrieved_context
            )

            if rag_confidence > llm_confidence:
                llm_domain = rag_domain or llm_domain
                llm_intent = rag_intent or llm_intent
                llm_jurisdiction = rag_jurisdiction or llm_jurisdiction
                llm_confidence = rag_confidence

        model_domain, model_domain_score = classify_with_zero_shot(model, query, DOMAINS)
        model_intent, model_intent_score = classify_with_zero_shot(model, query, INTENTS)

        if rule_domain and rule_domain_score >= 0.95:
            chosen_domain = rule_domain
        elif llm_domain and llm_confidence >= 0.62:
            chosen_domain = llm_domain
        elif rule_domain and rule_domain_score > model_domain_score + 0.08:
            chosen_domain = rule_domain
        else:
            chosen_domain = model_domain

        domain = normalize_domain(chosen_domain)

        if looks_like_short_definition_query(query):
            chosen_intent = "definition"
        elif rule_intent and rule_intent_score >= 0.95:
            chosen_intent = rule_intent
        elif llm_intent and llm_confidence >= 0.62:
            chosen_intent = llm_intent
        elif rule_intent and rule_intent_score > model_intent_score + 0.08:
            chosen_intent = rule_intent
        else:
            chosen_intent = model_intent

        intent = normalize_intent(chosen_intent)

        jurisdiction = detect_jurisdiction(query)
        if not jurisdiction and llm_jurisdiction:
            jurisdiction = [llm_jurisdiction]
        if not jurisdiction:
            jurisdiction = ["US"]

        temporal = detect_temporal(query)
        temporal = detect_temporal(query)

        # 🔹 LLM fallback (only when needed)
        if not temporal and len(query.split()) > 5:
            temporal = llm_detect_temporal(query)
        
        complexity = compute_complexity(intent, jurisdiction, temporal, query)
        routing = "complex" if complexity == "complex" else "simple"

        return {
            "domain": domain,
            "intent": intent,
            "jurisdiction": jurisdiction,
            "temporal": temporal,
            "complexity": complexity,
            "routing": routing
        }

    except Exception:
        return {
            "domain": "other",
            "intent": "definition",
            "jurisdiction": ["US"],
            "temporal": False,
            "complexity": "simple",
            "routing": "simple",
            "fallback": True
        }