import re
import requests

system_llm = None

SYSTEM_CONTEXT = """
You are the system-explanation layer for a Legal AI platform that processes large legal documents.

System capabilities:
- validates whether a query is legal-domain
- detects and masks PII such as names, emails, phone numbers, and SSNs
- corrects spelling mistakes
- expands legal abbreviations such as HIPAA, ADA, USC, CFR, GDPR, and EEOC
- restructures weak legal queries into clearer legal search queries
- classifies legal domain, intent, and jurisdiction
- supports large legal document ingestion
- learns abbreviations from ingested documents
- supports retrieval-augmented workflows (RAG)
- supports downstream decomposition and retrieval on legal corpora
- uses rules, zero-shot models, embeddings, and Ollama-based LLM support

Pipeline overview:
1. Validation
2. Preprocessing
3. Classification
4. Optional downstream decomposition / retrieval

Important:
- Answer only about system capabilities, workflow, modules, models, ingestion, retrieval, or limitations.
- If the user is asking an actual legal question, do NOT answer it here.
- If the query is just a general unrelated consumer question, do NOT classify it as SYSTEM.
"""

LEGAL_ABBR_GUARD = {
    "ADA", "USC", "U.S.C.", "CFR", "C.F.R.", "HIPAA", "GDPR", "CCPA", "FDCPA",
    "FLSA", "EEOC", "FMLA", "OSHA", "IRS", "SEC", "EPA", "FERPA", "COPPA", "SOX",
    "ERISA", "RICO", "DMCA", "SCOTUS", "DOJ", "FTC", "NLRB", "PHI", "PII"
}

LEGAL_HINTS = {
    "law", "legal", "act", "rule", "rules", "section", "title", "usc", "u.s.c",
    "cfr", "c.f.r", "contract", "privacy", "compliance", "liability", "breach",
    "employment", "discrimination", "rights", "duties", "penalty", "penalties",
    "court", "statute", "regulation", "plaintiff", "defendant", "damages",
    "eeoc", "hipaa", "ada", "gdpr", "ccpa", "fdcpa", "irs", "sec", "ssn",
    "tax", "violation", "violations", "consumer", "agreement"
}

# only genuine system/meta/help/workflow questions
SYSTEM_HINTS = {
    "what can you do",
    "what are your capabilities",
    "how can you help",
    "how can you assist",
    "how can you assist me",
    "what support do you provide",
    "what agents can do",
    "what agents do you have",
    "what modules do you have",
    "how does your system work",
    "what is your workflow",
    "what is your pipeline",
    "what models do you use",
    "which model do you use",
    "what is rag",
    "what are your limitations",
    "can you process legal documents",
    "can you detect pii",
    "can you expand abbreviations",
    "can you classify domain and intent",
    "how do you handle legal documents internally",
    "how do you process legal documents internally",
    "what happens after i send a query",
    "what happens after i send query",
    "what happens after validation",
    "what happens after preprocessing",
    "what happens after classification",
    "how do you process queries",
    "how do you handle queries",
    "what do you do after i send a query",
}

CLEAR_NON_SYSTEM_NON_LEGAL = {
    "movie", "movies", "recipe", "cook", "cooking", "biryani",
    "laptop", "mobile", "phone", "weather", "temperature",
    "cricket", "football", "music", "song", "restaurant",
    "shopping", "travel", "vacation", "game", "games"
}


def get_system_llm():
    global system_llm

    if system_llm is None:
        try:
            system_llm = {
                "base_url": "http://localhost:11434/api/generate",
                "model": "qwen2.5:3b"
            }
        except Exception:
            system_llm = False

    return system_llm


def _looks_clearly_unrelated_general_query(query: str) -> bool:
    q = (query or "").lower().strip()
    if not q:
        return False

    return any(term in q for term in CLEAR_NON_SYSTEM_NON_LEGAL)


def _looks_like_explicit_system_query(query: str) -> bool:
    q_lower = (query or "").lower().strip()
    if not q_lower:
        return False

    if any(hint in q_lower for hint in SYSTEM_HINTS):
        return True

    system_patterns = [
        r"\bhow\b.*\b(help|assist|support)\b",
        r"\bwhat\b.*\b(can you do|do you do|support|capabilities?)\b",
        r"\bwhat\b.*\bmodules?\b",
        r"\bwhat\b.*\bagents?\b",
        r"\bhow\b.*\b(system|workflow|pipeline)\b",
        r"\bwhich\b.*\bmodel\b",
        r"\bwhat\b.*\bmodel\b",
        r"\bcan you\b.*\b(process|detect|expand|classify)\b",
        r"\bhow\b.*\b(handle|process)\b.*\b(documents?|queries?)\b",
        r"\bwhat happens\b.*\b(query|queries|validation|preprocessing|classification)\b",
        r"\bafter i send\b.*\bquery\b",
        r"\bafter sending\b.*\bquery\b",
    ]

    return any(re.search(pattern, q_lower, flags=re.IGNORECASE) for pattern in system_patterns)


def _looks_like_obvious_legal_query(query: str) -> bool:
    q = query.strip()
    q_lower = q.lower()

    tokens = {
        re.sub(r"[^A-Za-z\.]", "", w).upper()
        for w in q.split()
        if w.strip()
    }

    normalized_legal_abbr = {abbr.replace(".", "").upper() for abbr in LEGAL_ABBR_GUARD}
    normalized_tokens = {t.replace(".", "").upper() for t in tokens}

    if any(t in normalized_legal_abbr for t in normalized_tokens):
        return True

    legal_reference_patterns = [
        r"\bsection\s+\d+",
        r"\btitle\s+\d+",
        r"\b\d+\s+u\.?s\.?c\.?",
        r"\b\d+\s+c\.?f\.?r\.?",
        r"§\s*\d+",
    ]
    if any(re.search(pattern, q_lower, flags=re.IGNORECASE) for pattern in legal_reference_patterns):
        return True

    if any(hint in q_lower for hint in LEGAL_HINTS):
        return True

    return False


def detect_system_query(query: str) -> bool:
    """
    Detect if query is about system capability or workflow.
    Keeps routing safe:
    1. explicit system/workflow questions -> True
    2. obvious legal-content queries -> False
    3. clearly unrelated consumer/general queries -> False
    4. LLM fallback only for ambiguous meta/system wording
    """
    try:
        if not query:
            return False

        q = query.strip()

        # 1. explicit system/meta/workflow queries should win first
        if _looks_like_explicit_system_query(q):
            return True

        # 2. obvious legal-content queries must stay in legal flow
        if _looks_like_obvious_legal_query(q):
            return False

        # 3. clearly unrelated general queries should NOT become system_response
        if _looks_clearly_unrelated_general_query(q):
            return False

        # 4. LLM fallback only for ambiguous capability/workflow wording
        model = get_system_llm()
        if not model:
            return False

        prompt = (
            f"{SYSTEM_CONTEXT}\n\n"
            "Classify the user query.\n"
            "Rules:\n"
            "- If the query is asking about the system, its workflow, modules, models, pipeline, document handling, query handling, or capabilities, return SYSTEM.\n"
            "- If the query is asking about a legal topic, statute, act, section, compliance issue, contract, privacy issue, right, duty, or legal meaning, return LEGAL.\n"
            "- If the query is just an unrelated general consumer question, return OTHER.\n"
            "Return ONLY one word:\n"
            "SYSTEM\n"
            "LEGAL\n"
            "or\n"
            "OTHER\n\n"
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
            result = re.sub(r"[^a-z]", "", result)
            return result.startswith("system")

        return False

    except Exception:
        return False


def get_system_response(query: str) -> str:
    """
    Project-level system answer.
    Gives common answers about the overall Legal AI platform,
    not just the three current modules.
    """
    try:
        q_lower = (query or "").lower().strip()

        # common capability answer
        if any(x in q_lower for x in [
            "what can you do", "your capabilities", "how can you help",
            "how can you assist", "how can you assist me", "what support do you provide"
        ]):
            return (
                "This Legal AI platform is designed to process large legal documents and legal queries. "
                "It can validate whether a query is legal, detect and mask sensitive information, correct legal spelling errors, "
                "expand legal abbreviations, restructure weak legal queries, classify legal domain and intent, "
                "and support downstream legal retrieval and document-based workflows."
            )

        # workflow / pipeline answer
        if any(x in q_lower for x in [
            "how does your system work", "workflow", "pipeline",
            "what happens after i send a query", "what happens after validation",
            "what happens after preprocessing", "what happens after classification",
            "how do you process queries", "how do you handle queries"
        ]):
            return (
                "After a query is received, the platform first validates whether it belongs to the legal domain and masks sensitive information if needed. "
                "It then preprocesses the query by cleaning it, correcting legal spelling issues, expanding abbreviations, and extracting legal signals. "
                "Next, it classifies the query by domain, intent, and jurisdiction, and this enriched output can then be used by downstream legal retrieval and document analysis components."
            )

        # legal document handling answer
        if any(x in q_lower for x in [
            "how do you handle legal documents internally",
            "how do you process legal documents internally",
            "can you process legal documents"
        ]):
            return (
                "The platform is built for large legal document workflows. It can ingest legal content, normalize and understand legal terminology, "
                "learn document-specific abbreviations, and prepare structured signals that can support downstream legal retrieval, chunking, and document analysis."
            )

        # model/technology answer
        if any(x in q_lower for x in ["what models do you use", "which model do you use", "llm"]):
            return (
                "The platform uses a hybrid approach that combines rules, regex-based extraction, semantic classification models, legal-domain language models, "
                "abbreviation handling, and Ollama-based LLM support for difficult query understanding tasks."
            )

        # RAG answer
        if any(x in q_lower for x in ["what is rag", "retrieval"]):
            return (
                "RAG stands for Retrieval-Augmented Generation. In this platform, it means using relevant legal document chunks and structured legal context "
                "to improve downstream legal understanding and retrieval."
            )

        # limitations answer
        if any(x in q_lower for x in ["what are your limitations", "limitations"]):
            return (
                "The platform is specialized for legal-domain processing. It is designed to work best on legal queries and legal documents, "
                "and it does not aim to answer unrelated general-purpose consumer questions."
            )

        # LLM fallback for broader system/meta queries
        model = get_system_llm()
        if not model:
            return default_system_response()

        prompt = (
            f"{SYSTEM_CONTEXT}\n\n"
            "You are answering a question about the Legal AI platform itself.\n"
            "Rules:\n"
            "- Keep the answer concise and clear.\n"
            "- Answer at project/platform level, not only one module.\n"
            "- Mention large legal document processing when relevant.\n"
            "- Do not answer legal advice questions.\n"
            "- Do not answer unrelated general consumer questions.\n"
            "- Return only the answer.\n\n"
            f"User: {query}"
        )

        response = requests.post(
            model["base_url"],
            json={
                "model": model["model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=10
        )

        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            if answer:
                answer = re.sub(r"^\s*(answer:|response:)\s*", "", answer, flags=re.IGNORECASE).strip()
                return answer

        return default_system_response()

    except Exception:
        return default_system_response()


def default_system_response():
    return (
        "This Legal AI platform is built to process large legal documents and legal queries. "
        "It supports validation, sensitive-data masking, legal query preprocessing, abbreviation expansion, "
        "legal classification, and downstream legal retrieval-oriented workflows."
    )