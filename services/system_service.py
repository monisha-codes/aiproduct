import re
import requests

system_llm = None

SYSTEM_CONTEXT = """
You are the system-explanation layer for a Legal AI platform that processes large legal documents.

System capabilities:
- validates whether a query is legal-domain
- detects and masks PII such as names, emails, phone numbers, and SSNs
- corrects spelling mistakes
- expands legal abbreviations such as HIPAA, ADA, USC, CFR
- restructures weak legal queries into clearer legal search queries
- classifies legal domain, intent, and jurisdiction
- supports large legal document ingestion
- learns abbreviations from ingested documents
- supports retrieval-augmented workflows (RAG)
- uses rules, zero-shot models, embeddings, and Ollama-based LLM fallback

Pipeline overview:
1. Validation
2. Preprocessing
3. Classification
4. Optional retrieval-aware refinement

Important:
- Answer only about system capabilities, workflow, modules, models, ingestion, retrieval, or limitations.
- If the user is asking an actual legal question, do NOT answer it here.
"""

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


def detect_system_query(query: str) -> bool:
    """
    Detect if query is about system capability using LLM (fallback-safe)
    """
    try:
        prompt = (
            "Classify this query.\n"
            "Return ONLY 'system' or 'legal'.\n\n"
            f"Query: {query}"
        )

        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=5
        )

        if response.status_code == 200:
            result = response.json().get("response", "").strip().lower()
            return "system" in result

        return False

    except Exception:
        return False


def get_system_response(query: str) -> str:
    """
    LLM-based system answer (safe + short)
    """
    try:
        prompt = (
            "You are a legal AI assistant.\n"
            "Explain your capabilities clearly and briefly.\n"
            "Do NOT answer legal questions.\n\n"
            f"User: {query}"
        )

        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=10
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()

        return default_system_response()

    except Exception:
        return default_system_response()


def default_system_response():
    return (
        "I can process legal queries, extract entities, expand abbreviations, "
        "classify legal intent, and assist in retrieving relevant legal information."
    )