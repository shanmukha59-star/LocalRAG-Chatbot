# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import ask_rag

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    query: str
    history: list[dict] | None = None


class QueryResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    answer = ask_rag(request.query, request.history)
    return QueryResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print("✅ FastAPI app loaded successfully — starting server...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)