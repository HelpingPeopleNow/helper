"""
HTTP routes. Thin layer — translate HTTP <-> domain, nothing more.
"""
import logging
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from internal.api.auth_middleware import validate_session
from internal.api.dependencies import get_assistant
from internal.core.helper_agent import Answer, HelperAgent, Question
from internal.ports.llm import Message

logger = logging.getLogger(__name__)


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
    _session=Depends(validate_session),
) -> AskResponse:
    start = time.monotonic()
    history_len = len(req.history)
    logger.info("ask request: q_len=%d history=%d", len(req.question), history_len)

    try:
        history = tuple(Message(role=h.role, content=h.content) for h in req.history)
        result: Answer = assistant.answer(Question(text=req.question), history=history)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "ask response: answer_len=%d elapsed_ms=%.0f",
            len(result.text), elapsed_ms,
        )
        return AskResponse(answer=result.text)
    except Exception:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.exception("ask failed after %.0fms", elapsed_ms)
        raise
