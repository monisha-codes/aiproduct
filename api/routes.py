from fastapi import APIRouter, Request, Body
from pydantic import BaseModel
from services.validation_service import validate_query
from services.preprocessing_service import preprocess_query
from services.classification_service import classify_query
from logger import logger
import json

router = APIRouter()


# ✅ Schema for Swagger UI
class QueryRequest(BaseModel):
    query: str


@router.post("/v1/process")
async def process_query(
    payload: QueryRequest = Body(...),   # ✅ FIX: Explicit body
    request: Request = None
):
    trace_id = "req-" + str(id(request))

    try:
        logger.info(f"Processing request {trace_id}")

        # -------------------------------
        # 🔹 Primary query (Swagger input)
        # -------------------------------
        query = payload.query if payload else ""

        # -------------------------------
        # 🔹 Fallback (NO CHANGE)
        # -------------------------------
        if not query and request:
            try:
                body = await request.json()

                if isinstance(body, dict):
                    query = body.get("query", "")

                elif isinstance(body, str):
                    try:
                        parsed = json.loads(body)
                        if isinstance(parsed, dict):
                            query = parsed.get("query", "")
                        else:
                            query = body
                    except Exception:
                        query = body

                else:
                    query = ""

            except Exception:
                raw_body = await request.body()
                query = raw_body.decode("utf-8")

        # -------------------------------
        # 🔹 CLEAN QUERY (UNCHANGED)
        # -------------------------------
        query = str(query).strip()
        query = query.replace("\n", "").replace("\\", "")

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