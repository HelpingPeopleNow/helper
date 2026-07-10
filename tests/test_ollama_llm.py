"""Tests for the Ollama LLM adapter (raw requests.post)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from internal.adapters.ollama_llm import OllamaLLMAdapter
from internal.adapters.token_counting import TokenUsage
from internal.ports.llm import Message


class TestConstructor:
    def test_defaults_from_env(self, isolated_env) -> None:
        os.environ["OLLAMA_MODEL"] = "env-model"
        os.environ["OLLAMA_BASE_URL"] = "http://env:11434"
        adapter = OllamaLLMAdapter()
        assert adapter._model == "env-model"
        assert adapter._base_url == "http://env:11434"

    def test_explicit_overrides_env(self, isolated_env) -> None:
        os.environ["OLLAMA_MODEL"] = "ignored"
        os.environ["OLLAMA_BASE_URL"] = "http://ignored:11434"
        adapter = OllamaLLMAdapter(model="explicit-model", base_url="http://explicit:11434")
        assert adapter._model == "explicit-model"
        assert adapter._base_url == "http://explicit:11434"

    def test_defaults_when_env_unset(self, isolated_env) -> None:
        # P3-2: adapter default is the canonical production model
        # (qwen2.5:1.5b, matching main.py + infra/docker-compose.yml).
        adapter = OllamaLLMAdapter()
        assert adapter._model == "qwen2.5:1.5b"
        assert adapter._base_url == ""

    def test_base_url_strips_trailing_slash(self, isolated_env) -> None:
        adapter = OllamaLLMAdapter(base_url="http://local:11434/")
        assert adapter._base_url == "http://local:11434"


class TestComplete:
    def test_success(self) -> None:
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "hello world"}
        with patch("requests.post", return_value=mock_response) as mock_post:
            result = adapter.complete("system prompt", "user message")
        assert result == "hello world"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "m"
        assert kwargs["json"]["prompt"].startswith("system prompt")
        assert kwargs["json"]["stream"] is False
        assert kwargs["timeout"] == 60

    def test_with_history(self) -> None:
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "ans"}
        with patch("requests.post", return_value=mock_response) as mock_post:
            history = (Message("user", "hi"), Message("assistant", "hey"))
            result = adapter.complete("sys", "current", history=history)
        assert result == "ans"
        _, kwargs = mock_post.call_args
        prompt = kwargs["json"]["prompt"]
        assert "hi" in prompt
        assert "hey" in prompt
        assert "current" in prompt

    def test_empty_history(self) -> None:
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "ans"}
        with patch("requests.post", return_value=mock_response):
            result = adapter.complete("sys", "user")
        assert result == "ans"

    def test_missing_response_key_returns_empty(self) -> None:
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}
        with patch("requests.post", return_value=mock_response):
            result = adapter.complete("sys", "user")
        assert result == ""

    def test_raises_on_http_error(self) -> None:
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = RuntimeError("500 error")
        with patch("requests.post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="500 error"):
                adapter.complete("sys", "user")

    def test_raises_on_connection_error(self) -> None:
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        with patch("requests.post", side_effect=RuntimeError("Connection refused")):
            with pytest.raises(RuntimeError, match="Connection refused"):
                adapter.complete("sys", "user")


# ── P1-5: real token usage via Ollama prompt_eval_count / eval_count ──


class TestTokenUsage:
    def test_last_usage_populated_from_eval_count(self) -> None:
        """Ollama /api/generate returns real token counts when stream=false."""
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "response": "ok",
            "prompt_eval_count": 42,
            "eval_count": 7,
        }
        with patch("requests.post", return_value=mock_response):
            adapter.complete("sys", "user")
        assert adapter.last_usage is not None
        assert adapter.last_usage.input_tokens == 42
        assert adapter.last_usage.output_tokens == 7

    def test_last_usage_none_on_failure(self) -> None:
        """A failed call must reset last_usage so we don't leak prior data."""
        adapter = OllamaLLMAdapter(model="m", base_url="http://local:11434")
        # Pre-pollute
        adapter.last_usage = TokenUsage(input_tokens=999, output_tokens=999)
        with patch("requests.post", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                adapter.complete("sys", "user")
        # Reset on entry to complete(); failure must keep it None.
        assert adapter.last_usage is None
