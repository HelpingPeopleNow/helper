"""Ollama LLM adapter.

Uses the local qwen3:4b model via the Ollama API.
No rate limits, slower but reliable.
"""
import logging
import os
import requests

from internal.ports.llm import LLMPort, Message

logger = logging.getLogger(__name__)


class OllamaLLMAdapter:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "")).rstrip("/")
        logger.info("Ollama init: model=%s base_url=%s", self._model, self._base_url)

    def complete(self, system_prompt: str, user: str, history: tuple[Message, ...] = ()) -> str:
        # Build the full prompt with conversation history
        full_prompt = f"{system_prompt}\n\n"
        for msg in history:
            role = "User" if msg.role == "user" else "Assistant"
            full_prompt += f"{role}: {msg.content}\n"
        full_prompt += f"User: {user}\n\nAssistant:"

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "think": False,
        }

        logger.info("Ollama call: model=%s prompt_chars=%d", self._model, len(full_prompt))

        resp = requests.post(url, json=payload, stream=False, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("response", "")
        logger.info("Ollama response: response_chars=%d", len(result))
        return result