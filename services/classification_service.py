from transformers import pipeline
from config import settings
import re
import torch
import requests

classifier = None
ollama_classifier = None

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
# 🔹 DOMAIN LABELS (US-LEGAL FRIENDLY)
# -------------------------------
DOMAINS = [
    "constitutional law",
    "civil law",
    "criminal law",
    "employment law",
    "healthcare law",
    "privacy and data protection law",
    "corporate and commercial law",
    "tax law",
    "intellectual property law"
]

# -------------------------------
# 🔹 INTENT LABELS (MODEL-DRIVEN)
# -------------------------------
INTENTS = [
    "definition of law",
    "legal procedure",
    "case lookup",
    "rights and duties",
    "penalties and punishment",
    "compliance and regulation",
    "legal interpretation",
    "statutory explanation",
    "contract analysis"
]

# -------------------------------
# 🔹 Lightweight rule signals
# Keep rules as strong overrides only
# -------------------------------
def detect_domain_rule(query: str):
    q = query.lower()

    if any(k in q for k in [
        "constitution", "constitutional", "amendment", "bill of rights",
        "due process", "equal protection", "first amendment",
        "fourth amendment", "freedom of speech", "judicial review",
        "federalism", "state action", "habeas corpus", "supreme court"
    ]):
        return "constitutional law", 0.95

    if any(k in q for k in [
        "crime", "criminal", "felony", "misdemeanor", "arrest",
        "warrant", "bail", "conviction", "sentence", "imprisonment",
        "prosecution", "defendant", "evidence", "fraud", "robbery",
        "murder", "assault", "cybercrime"
    ]):
        return "criminal law", 0.95

    if any(k in q for k in [
        "employment", "employee", "employer", "workplace", "termination",
        "wage", "overtime", "discrimination", "harassment", "fmla",
        "flsa", "eeoc", "ada accommodation", "retaliation"
    ]):
        return "employment law", 0.95

    if any(k in q for k in [
        "hipaa", "patient", "medical record", "health information",
        "protected health information", "phi", "healthcare"
    ]):
        return "healthcare law", 0.92

    if any(k in q for k in [
        "privacy", "data protection", "personal data", "gdpr",
        "ccpa", "coppa", "consent", "data sharing", "tracking", "pii"
    ]):
        return "privacy and data protection law", 0.92

    if any(k in q for k in [
        "copyright", "trademark", "patent", "dmca",
        "intellectual property", "licensing"
    ]):
        return "intellectual property law", 0.92

    if any(k in q for k in [
        "tax", "irs", "deduction", "income tax", "withholding"
    ]):
        return "tax law", 0.92

    if any(k in q for k in [
        "corporate", "company", "shareholder", "merger",
        "securities", "sec", "board", "governance"
    ]):
        return "corporate and commercial law", 0.92

    if any(k in q for k in [
        "contract", "agreement", "breach", "liability", "damages",
        "injunction", "settlement", "arbitration", "mediation",
        "lease", "tenancy", "negligence", "tort", "remedy",
        "consumer", "civil"
    ]):
        return "civil law", 0.90

    return None, 0.0


def detect_intent_rule(query: str):
    q = query.lower()

    if any(k in q for k in ["what is", "define", "meaning of", "explain"]):
        return "definition of law", 0.88

    if any(k in q for k in ["how to", "procedure", "steps", "process"]):
        return "legal procedure", 0.92

    if any(k in q for k in ["case", "judgment", "ruling", "precedent", "holding"]):
        return "case lookup", 0.95

    if any(k in q for k in ["rights", "duties", "obligations", "responsibilities"]):
        return "rights and duties", 0.92

    if any(k in q for k in ["penalty", "punishment", "fine", "imprisonment", "sentence", "violation", "violations"]):
        return "penalties and punishment", 0.95

    if any(k in q for k in ["compliance", "regulation", "rule", "policy requirement"]):
        return "compliance and regulation", 0.92

    if any(k in q for k in ["interpret", "interpretation", "meaning of section", "scope of"]):
        return "legal interpretation", 0.90

    if re.search(r"\b(section|sec\.?|title|usc|u\.s\.c|cfr|c\.f\.r)\b", q):
        return "statutory explanation", 0.90

    if any(k in q for k in ["breach of contract", "contract clause", "agreement terms"]):
        return "contract analysis", 0.92

    return None, 0.0


# -------------------------------
# 🔹 Helpers
# -------------------------------
def classify_with_zero_shot(model, query: str, labels: list[str]):
    result = model(query, labels, multi_label=False)
    return result["labels"][0], float(result["scores"][0])


def detect_temporal(query: str) -> bool:
    q = query.lower()
    temporal_terms = [
        "latest", "recent", "current", "today", "recently",
        "updated", "amended", "new", "now"
    ]
    return any(term in q for term in temporal_terms)


def detect_complexity(query: str) -> str:
    q = query.lower()

    multi_jurisdiction = any(k in q for k in [
        "compare us and eu", "us vs eu", "federal and state", "state and federal"
    ])

    comparison = any(k in q for k in [
        "compare", "difference", "vs", "versus"
    ])

    temporal = detect_temporal(query)

    if multi_jurisdiction or comparison or temporal:
        return "complex"
    return "simple"


def normalize_domain(label: str) -> str:
    mapping = {
        "corporate and commercial law": "civil law",
        "employment law": "civil law",
        "healthcare law": "civil law",
        "privacy and data protection law": "civil law",
        "tax law": "civil law",
        "intellectual property law": "civil law",
    }
    return mapping.get(label, label)


def get_best_query_text(data: dict) -> str:
    """
    Prefer richer query variants without changing external flow.
    """
    return (
        data.get("restructured_query")
        or data.get("expanded_query")
        or data.get("cleaned_query")
        or data.get("original_query")
        or ""
    )


def get_retrieved_context(data: dict) -> str:
    """
    Optional RAG-aware context.
    Supports future retrieval output without affecting current flow.
    If no context is present, returns empty string.
    """
    try:
        if not isinstance(data, dict):
            return ""

        # direct string context
        if isinstance(data.get("retrieved_context"), str):
            return data["retrieved_context"].strip()

        # list of chunk dicts
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


def llm_classify_query(query: str, context: str = ""):
    """
    Ollama primary classifier.
    Returns: (domain, intent, jurisdiction, confidence)
    If context is provided, it performs RAG-aware classification.
    """
    try:
        model = get_ollama_classifier()
        if not model:
            return None, None, None, 0.0

        if context:
            prompt = (
                "Classify this legal user query using both the query and retrieved legal context.\n"
                "Return ONLY this exact format:\n"
                "DOMAIN: one of [constitutional law, civil law, criminal law, employment law, healthcare law, privacy and data protection law, corporate and commercial law, tax law, intellectual property law]\n"
                "INTENT: one of [definition of law, legal procedure, case lookup, rights and duties, penalties and punishment, compliance and regulation, legal interpretation, statutory explanation, contract analysis]\n"
                "JURISDICTION: US\n"
                "CONFIDENCE: 0.0 to 1.0\n\n"
                f"Query: {query}\n\n"
                f"Retrieved context:\n{context}"
            )
        else:
            prompt = (
                "Classify this legal user query.\n"
                "Return ONLY this exact format:\n"
                "DOMAIN: one of [constitutional law, civil law, criminal law, employment law, healthcare law, privacy and data protection law, corporate and commercial law, tax law, intellectual property law]\n"
                "INTENT: one of [definition of law, legal procedure, case lookup, rights and duties, penalties and punishment, compliance and regulation, legal interpretation, statutory explanation, contract analysis]\n"
                "JURISDICTION: US\n"
                "CONFIDENCE: 0.0 to 1.0\n\n"
                f"Query: {query}"
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
            return None, None, None, 0.0

        data = response.json()
        text = data.get("response", "").strip()

        domain_match = re.search(r"DOMAIN:\s*(.+)", text, flags=re.IGNORECASE)
        intent_match = re.search(r"INTENT:\s*(.+)", text, flags=re.IGNORECASE)
        jurisdiction_match = re.search(r"JURISDICTION:\s*(.+)", text, flags=re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)

        domain = domain_match.group(1).strip().lower() if domain_match else None
        intent = intent_match.group(1).strip().lower() if intent_match else None
        jurisdiction = jurisdiction_match.group(1).strip().upper() if jurisdiction_match else "US"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0

        valid_domains = {d.lower() for d in DOMAINS}
        valid_intents = {i.lower() for i in INTENTS}

        if domain not in valid_domains:
            domain = None
        if intent not in valid_intents:
            intent = None

        return domain, intent, jurisdiction, confidence

    except Exception:
        return None, None, None, 0.0


# -------------------------------
# 🔹 Hybrid Classification (FLOW-SAFE + OPTIONAL RAG-AWARE)
# -------------------------------
def classify_query(data):
    try:
        model = get_model()
        query = get_best_query_text(data)
        retrieved_context = get_retrieved_context(data)

        # -------------------------------
        # 🔹 Domain + intent from rules
        # -------------------------------
        rule_domain, rule_domain_score = detect_domain_rule(query)
        rule_intent, rule_intent_score = detect_intent_rule(query)

        # -------------------------------
        # 🔹 Domain + intent from Ollama (primary)
        # Query-only
        # -------------------------------
        llm_domain, llm_intent, llm_jurisdiction, llm_confidence = llm_classify_query(query)

        # -------------------------------
        # 🔹 Optional RAG-aware refinement
        # Only if context exists
        # -------------------------------
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

        # -------------------------------
        # 🔹 Zero-shot fallback
        # -------------------------------
        model_domain, model_domain_score = classify_with_zero_shot(model, query, DOMAINS)
        model_intent, model_intent_score = classify_with_zero_shot(model, query, INTENTS)

        # -------------------------------
        # 🔹 DOMAIN selection
        # -------------------------------
        if rule_domain and rule_domain_score >= 0.95:
            chosen_domain = rule_domain
        elif llm_domain and llm_confidence >= 0.60:
            chosen_domain = llm_domain
        elif rule_domain and rule_domain_score > model_domain_score + 0.08:
            chosen_domain = rule_domain
        else:
            chosen_domain = model_domain

        domain = normalize_domain(chosen_domain)

        # -------------------------------
        # 🔹 INTENT selection
        # -------------------------------
        if rule_intent and rule_intent_score >= 0.95:
            intent = rule_intent
        elif llm_intent and llm_confidence >= 0.60:
            intent = llm_intent
        elif rule_intent and rule_intent_score > model_intent_score + 0.08:
            intent = rule_intent
        else:
            intent = model_intent

        # -------------------------------
        # 🔹 Jurisdiction
        # Keep your current flow stable
        # -------------------------------
        jurisdiction = ["US"]

        if llm_jurisdiction and llm_jurisdiction in {"US"}:
            jurisdiction = [llm_jurisdiction]

        if any(k in query.lower() for k in ["california", "new york", "texas", "florida"]):
            jurisdiction = ["US"]

        # -------------------------------
        # 🔹 Optional internal signals
        # not returned now, so your flow is not affected
        # -------------------------------
        _temporal = detect_temporal(query)
        _complexity = detect_complexity(query)

        # -------------------------------
        # 🔹 Final Output (same shape)
        # -------------------------------
        return {
            "domain": domain,
            "intent": intent,
            "jurisdiction": jurisdiction
        }

    except Exception:
        return {
            "domain": "unknown",
            "intent": "unknown",
            "jurisdiction": ["US"],
            "fallback": True
        }