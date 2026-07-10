"""
Ollama LLM adapter.

Uses the local qwen3:4b model via the Ollama API.
No rate limits, slower but reliable.

P1-5: Ollama's /api/generate (stream=false) returns prompt_eval_count and
eval_count — the daemon's actual token counts. We capture and store them
on `self.last_usage`. Falls back to tiktoken estimates if the daemon
omits those fields.
"""
import logging
import os
import requests

from internal.adapters.token_counting import (
    TokenUsage,
    extract_usage_from_ollama,
)
from internal.ports.llm import LLMPort, Message

logger = logging.getLogger(__name__)


class OllamaLLMAdapter:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "")).rstrip("/")
        # P1-5: side-channel populated after each complete(); agent reads it.
        self.last_usage: TokenUsage | None = None
        logger.info("Ollama init: model=%s base_url=%s", self._model, self._base_url)

    def complete(self, system_prompt: str, user: str, history: tuple[Message, ...] = ()) -> str:
        # Reset per-call so a failed call doesn't leak prior usage.
        self.last_usage = None

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

        try:
            resp = requests.post(url, json=payload, stream=False, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Ollama adapter error url=%s model=%s", url, self._model)
            raise

        content = data.get("response", "")
        # P1-5: capture real prompt_eval_count / eval_count from the daemon.
        self.last_usage = extract_usage_from_ollama(data, fallback_output_text=content)
        logger.info(
            "Ollama response: response_chars=%d in_t=%s out_t=%s",
            len(content),
            self.last_usage.input_tokens,
            self.last_usage.output_tokens,
        )
        return content
