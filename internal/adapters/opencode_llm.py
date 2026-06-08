"""
OpenCode / OpenAI-compatible LLM adapter.

Uses langchain_openai.ChatOpenAI because OpenCode Zen exposes an
OpenAI-compatible /chat/completions endpoint. Env-driven config.
"""
import os

from langchain_openai import ChatOpenAI

from internal.ports.llm import LLMPort


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

    def complete(self, system_prompt: str, user: str) -> str:
        response = self._llm.invoke([("system", system_prompt), ("user", user)])
        return response.content
