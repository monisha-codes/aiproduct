from fastapi import APIRouter, Request, Body
from services.validation_service import validate_query
from services.preprocessing_service import preprocess_query
from services.classification_service import classify_query
from logger import logger
import json

router = APIRouter()


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

        # -------------------------------
        # 🔹 Universal Input Handler (KEY FIX)
        # -------------------------------
        raw_body = await request.body()
        raw_text = raw_body.decode("utf-8").strip()

        query = ""

        # Try JSON parse
        try:
            parsed = json.loads(raw_text)

            if isinstance(parsed, dict):
                query = parsed.get("query", "")
            else:
                query = raw_text

        except Exception:
            # Plain text fallback
            query = raw_text

        # -------------------------------
        # 🔹 Clean query (UNCHANGED)
        # -------------------------------
        query = str(query).strip().replace("\n", "").replace("\\", "")

        if query.startswith("{") and "query" in query:
            try:
                parsed = json.loads(query)
                if isinstance(parsed, dict):
                    query = parsed.get("query", query)
            except Exception:
                pass

        if "query" in query and ":" in query and "{" in query:
            query = query.split(":")[-1].replace("}", "").strip()

        # -------------------------------
        # 🔹 Existing flow (UNCHANGED)
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