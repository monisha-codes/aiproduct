from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class QueryRequest(BaseModel):
    # Keep schema permissive; let validation_service handle legal-specific checks
    query: str = Field(..., min_length=1, max_length=5000)


class APIResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None