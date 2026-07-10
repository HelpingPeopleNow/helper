"""
Mistral LLM adapter.

Uses langchain_openai.ChatOpenAI because Mistral's API is
OpenAI-compatible. The API key and base URL come from env vars.

P1-5: populates `self.last_usage` from LangChain's `usage_metadata` after
each call. Mistral-large's backend populates `response.usage` with true
counts through the OpenAI-compatible API.
"""
import logging
import os
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from internal.adapters.token_counting import (
    TokenUsage,
    extract_usage_from_langchain,
    langchain_content_to_text,
)
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
        # P1-5: side-channel populated after each complete(); agent reads it.
        self.last_usage: TokenUsage | None = None

    def complete(self, system_prompt: str, user: str, history: tuple[Message, ...] = ()) -> str:
        # Reset per-call so a failed call doesn't leak prior usage.
        self.last_usage = None

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
            # P1-5: Mistral's backend populates response.usage with real
            # tokens via the OpenAI-compatible API.
            self.last_usage = extract_usage_from_langchain(response)
            # P3-4: langchain content can be str | list-of-blocks | None;
            # flatten to a plain string before logging length and returning.
            text = langchain_content_to_text(response.content)
            logger.info(
                "Mistral response: elapsed_ms=%.0f response_chars=%d in_t=%s out_t=%s",
                elapsed_ms, len(text),
                self.last_usage.input_tokens,
                self.last_usage.output_tokens,
            )
            return text
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("Mistral call failed after %.0fms", elapsed_ms)
            raise
