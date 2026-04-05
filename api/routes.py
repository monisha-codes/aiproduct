from fastapi import APIRouter, Request
from models.schemas import QueryRequest
from services.validation_service import validate_query
from services.preprocessing_service import preprocess_query
from services.classification_service import classify_query
from logger import logger

router = APIRouter()

@router.post("/v1/process")
async def process_query(request: QueryRequest):
    trace_id = "req-" + str(id(request))

    try:
        logger.info(f"Processing request {trace_id}")

        val = validate_query(request.query)
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