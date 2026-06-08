"""
HTTP routes. Thin layer — translate HTTP <-> domain, nothing more.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from internal.api.dependencies import get_assistant
from internal.core.helper_agent import Answer, HelperAgent, Question


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")


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
    result: Answer = assistant.answer(Question(text=req.question))
    return AskResponse(answer=result.text)
