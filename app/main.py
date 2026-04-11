from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Legal RAG API")

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}