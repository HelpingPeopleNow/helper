"""
OpenCode / OpenAI-compatible LLM adapter.

Uses langchain_openai.ChatOpenAI because OpenCode Zen exposes an
OpenAI-compatible /chat/completions endpoint. Env-driven config.
"""
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from internal.ports.llm import LLMPort, Message


class OpenCodeLLMAdapter:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        self._llm = ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "deepseek-v4-flash-free"),
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", ""),
            temperature=temperature,
        )

    def complete(self, system_prompt: str, user: str, history: tuple[Message, ...] = ()) -> str:
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # Add current question
        messages.append(HumanMessage(content=user))

        response = self._llm.invoke(messages)
        return response.content
