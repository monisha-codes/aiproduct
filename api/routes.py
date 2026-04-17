from fastapi import APIRouter, Request
from services.validation_service import validate_query
from services.preprocessing_service import preprocess_query
from services.classification_service import classify_query
from services.system_service import detect_system_query, get_system_response
from logger import logger
import json
import re

router = APIRouter()


def extract_query_from_request_body(raw_text: str) -> str:
    """
    Supports:
    - JSON: {"query": "..."}
    - plain text: breach of contract
    - malformed JSON-like input: { "query": explain hipaa rule }
    """
    if not raw_text:
        return ""

    raw_text = raw_text.strip()

    # Valid JSON
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return str(parsed.get("query", "")).strip()
        if isinstance(parsed, str):
            return parsed.strip()
    except Exception:
        pass

    # Malformed JSON-like fallback
    match = re.search(r'["\']?query["\']?\s*:\s*(.+)', raw_text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip()

        # remove wrapping braces / quotes
        value = value.rstrip("}").strip()
        value = value.strip('"').strip("'").strip()

        return value

    # Plain text fallback
    return raw_text


@router.post(
    "/v1/process",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"],
                        "example": {"query": "breach of contract"}
                    }
                },
                "text/plain": {
                    "schema": {
                        "type": "string"
                    },
                    "example": "breach of contract"
                }
            }
        }
    }
)
async def process_query(request: Request):
    trace_id = "req-" + str(id(request))

    try:
        logger.info(f"Processing request {trace_id}")

        raw_body = await request.body()
        raw_text = raw_body.decode("utf-8", errors="ignore")

        query = extract_query_from_request_body(raw_text)

        # Safe normalization only
        query = str(query).strip().replace("\n", " ").replace("\t", " ")
        query = " ".join(query.split())

        if not query:
            return {
                "status": "error",
                "error": {"error": "empty_query"}
            }

        # -------------------------------
        # 🔹 System Q&A Layer (SAFE)
        # -------------------------------
        try:
            if detect_system_query(query):
                system_answer = get_system_response(query)

                if system_answer and str(system_answer).strip():
                    return {
                        "status": "success",
                        "data": {
                            "type": "system_response",
                            "query": query,
                            "answer": system_answer
                        }
                    }

        except Exception as system_error:
            logger.warning(f"System layer failed for {trace_id}: {str(system_error)}")

        # -------------------------------
        # Existing flow (UNCHANGED)
        # -------------------------------
        val = validate_query(query)
        if "error" in val:
            return {"status": "error", "error": val}

        pre = preprocess_query(val)
        if "error" in pre:
            return {"status": "error", "error": pre}

        cls = classify_query(pre)

        return {
            "status": "success",
            "data": {
                "validation": val,
                "preprocessing": pre,
                "classification": cls
            }
        }

    except Exception as e:
        logger.error(f"Error {trace_id}: {str(e)}")
        return {
            "status": "error",
            "error": {"message": "internal_error"}
        }
