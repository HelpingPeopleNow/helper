import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from pizza_graph import build_graph
from db import get_system_prompt

app = FastAPI(title="Helper — Pizza Assistant")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup():
    """On startup, load the system prompt from the database and build the graph."""
    prompt = get_system_prompt()
    app.state.graph = build_graph(prompt)
    app.state.system_prompt = prompt
    print(f"✅ System prompt loaded ({len(prompt)} chars)")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/ask")
async def ask(req: AskRequest) -> AskResponse:
    result = app.state.graph.invoke({"question": req.question})
    return AskResponse(answer=result["answer"])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8082"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
