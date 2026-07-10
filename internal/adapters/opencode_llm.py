"""
OpenCode / OpenAI-compatible LLM adapter.

Uses langchain_openai.ChatOpenAI because OpenCode Zen exposes an
OpenAI-compatible /chat/completions endpoint. Env-driven config.

P1-5: populates `self.last_usage` from LangChain's `usage_metadata` after
each call. The agent reads it to record real token counts.
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


class OpenCodeLLMAdapter:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        model_name = model or os.getenv("LLM_MODEL", "deepseek-v4-flash-free")
        base = base_url or os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        logger.info(
            "LLM adapter init: model=%s base_url=%s temperature=%.1f",
            model_name, base, temperature,
        )
        self._llm = ChatOpenAI(
            model=model_name,
            base_url=base,
            api_key=api_key or os.getenv("LLM_API_KEY", ""),
            temperature=temperature,
            timeout=20,  # 20s — fail-fast, fall through quickly
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
        logger.info("LLM call: msgs=%d prompt_chars=%d user_chars=%d", msg_count, prompt_chars, len(user))

        start = time.monotonic()
        try:
            response = self._llm.invoke(messages)
            elapsed_ms = (time.monotonic() - start) * 1000
            # P1-5: capture real or fallback usage from LangChain.
            self.last_usage = extract_usage_from_langchain(response)
            # P3-4: langchain content can be str | list-of-blocks | None;
            # flatten to a plain string before logging length and returning.
            text = langchain_content_to_text(response.content)
            logger.info(
                "LLM response: elapsed_ms=%.0f response_chars=%d in_t=%s out_t=%s",
                elapsed_ms,
                len(text),
                self.last_usage.input_tokens,
                self.last_usage.output_tokens,
            )
            return text
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("LLM call failed after %.0fms", elapsed_ms)
            raise
