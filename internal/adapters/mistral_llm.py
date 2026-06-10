"""Mistral LLM adapter.

Uses langchain_openai.ChatOpenAI because Mistral's API is
OpenAI-compatible. The API key and base URL come from env vars.
"""
import logging
import os
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from internal.ports.llm import LLMPort, Message

logger = logging.getLogger(__name__)


class MistralLLMAdapter:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        model_name = model or os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        base = base_url or os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        key = api_key or os.getenv("MISTRAL_API_KEY", "")
        logger.info(
            "Mistral adapter init: model=%s base_url=%s key_set=%s",
            model_name, base, "yes" if key else "no",
        )
        self._llm = ChatOpenAI(
            model=model_name,
            base_url=base,
            api_key=key,
            temperature=temperature,
            timeout=20,  # 20s — fail-fast if Mistral is slow
        )

    def complete(self, system_prompt: str, user: str, history: tuple[Message, ...] = ()) -> str:
        messages = [SystemMessage(content=system_prompt)]

        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        messages.append(HumanMessage(content=user))

        msg_count = len(messages)
        prompt_chars = sum(len(m.content) for m in messages)
        logger.info("Mistral call: msgs=%d prompt_chars=%d", msg_count, prompt_chars)

        start = time.monotonic()
        try:
            response = self._llm.invoke(messages)
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "Mistral response: elapsed_ms=%.0f response_chars=%d",
                elapsed_ms, len(response.content),
            )
            return response.content
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("Mistral call failed after %.0fms", elapsed_ms)
            raise
