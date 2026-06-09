"""
Domain core: the HelperAgent aggregate.

Pure business logic. No framework, no I/O, no LLM, no DB.
Depends only on the port protocol (LLM interface), not on adapters.
The system prompt is received from the backend via gRPC.
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from internal.ports.llm import LLMPort, Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Question:
    """A user's question, normalized."""
    text: str

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("Question text cannot be empty")


@dataclass(frozen=True)
class Answer:
    """The assistant's answer to a question."""
    text: str
    detected_role: str = ""  # "worker" | "client" | "" (unclear)

    def __post_init__(self) -> None:
        if self.detected_role not in ("worker", "client", ""):
            raise ValueError(f"Invalid role: {self.detected_role!r}")


@dataclass(frozen=True)
class SystemPrompt:
    """A system prompt read from storage."""
    text: str


class HelperAgent:
    """
    Domain service: orchestrates answering a question using an LLM
    with the provided system prompt.

    Holds multiple LLM adapters; the backend selects which one to use
    per-request via the llm_provider field. Empty = env default fallback.
    """

    def __init__(self, adapters: dict[str, LLMPort], default_provider: str = "opencode") -> None:
        self._adapters = adapters
        self._default_provider = default_provider
        logger.info("HelperAgent: %d adapters loaded, default=%s", len(adapters), default_provider)

    def answer(self, question: Question, system_prompt: str, history: tuple[Message, ...] = (), llm_provider: str = "") -> Answer:
        # Pick adapter: explicit override or env default
        provider = llm_provider or self._default_provider
        llm = self._adapters.get(provider)
        if llm is None:
            logger.warning("Unknown llm_provider=%r, falling back to default=%s", llm_provider, self._default_provider)
            provider = self._default_provider
            llm = self._adapters[provider]

        logger.info("HelperAgent.answer: provider=%s (request=%r default=%s) sp_len=%d q_len=%d history=%d",
                     provider, llm_provider, self._default_provider, len(system_prompt), len(question.text), len(history))
        if logger.isEnabledFor(logging.DEBUG) and system_prompt:
            logger.debug("system_prompt[:150]: %s", system_prompt[:150])

        raw = ""
        try:
            # Append format instruction to the user message directly — some
            # providers are less reliable at following system prompt formatting.
            user_text = question.text + (
                "\n\nIMPORTANT — You MUST respond with valid JSON ONLY in this exact format: "
                '{"answer": "your response here", "role": "worker|client|"}\n'
                'Choose role="worker" if they offer services, role="client" if they need help, '
                'or role="" if unclear. Use double quotes only.'
            )
            raw = llm.complete(
                system_prompt=system_prompt,
                user=user_text,
                history=history,
            )
            return self._parse_response(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, falling back to raw text", exc_info=True)
            return Answer(text=raw)
        except Exception:
            logger.exception("LLM call failed")
            raise

    def _parse_response(self, raw: str) -> Answer:
        """Parse the LLM's JSON response into Answer + role."""
        # Try to extract JSON from the response
        text = raw.strip()
        # Handle markdown-wrapped JSON
        if text.startswith("```"):
            # Remove opening ```json / ``` and closing ```
            for prefix in ("```json", "```"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        answer_text = data.get("answer", "")
        role = data.get("role", "")

        if not answer_text:
            logger.warning("LLM response missing 'answer' field, using raw")
            return Answer(text=raw)

        if role not in ("worker", "client", ""):
            logger.warning("LLM returned unexpected role %r, ignoring", role)
            role = ""

        return Answer(text=answer_text, detected_role=role)
