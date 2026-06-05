import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from pizza_graph import graph

app = FastAPI(title="Helper — Pizza Assistant")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/v1/ask")
async def ask(req: AskRequest) -> AskResponse:
    result = graph.invoke({"question": req.question})
    return AskResponse(answer=result["answer"])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8082"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
