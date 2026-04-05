from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)

class APIResponse(BaseModel):
    status: str
    data: dict | None = None
    error: dict | None = None