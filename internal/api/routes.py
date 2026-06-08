"""
HTTP routes. Thin layer — translate HTTP <-> domain, nothing more.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from internal.api.dependencies import get_assistant
from internal.core.helper_agent import Answer, HelperAgent, Question
from internal.ports.llm import Message


class HistoryItem(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    history: list[HistoryItem] = Field(default_factory=list, description="Previous conversation messages")


class AskResponse(BaseModel):
    answer: str


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    assistant: HelperAgent = Depends(get_assistant),
) -> AskResponse:
    history = tuple(Message(role=h.role, content=h.content) for h in req.history)
    result: Answer = assistant.answer(Question(text=req.question), history=history)
    return AskResponse(answer=result.text)
