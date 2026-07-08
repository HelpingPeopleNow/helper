"""Tests for the Mistral LLM adapter (langchain_openai.ChatOpenAI)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from internal.adapters.mistral_llm import MistralLLMAdapter
from internal.ports.llm import Message


class TestConstructor:
    def test_defaults_from_env(self, isolated_env) -> None:
        os.environ["MISTRAL_MODEL"] = "env-model"
        os.environ["MISTRAL_BASE_URL"] = "https://env.example.com"
        os.environ["MISTRAL_API_KEY"] = "env-key"
        adapter = MistralLLMAdapter()
        assert adapter._llm.model == "env-model"
        assert "env.example.com" in adapter._llm.openai_api_base

    def test_explicit_overrides_env(self, isolated_env) -> None:
        os.environ["MISTRAL_MODEL"] = "ignored"
        os.environ["MISTRAL_BASE_URL"] = "https://ignored.example.com"
        os.environ["MISTRAL_API_KEY"] = "ignored"
        adapter = MistralLLMAdapter(
            model="explicit-model",
            base_url="https://explicit.example.com",
            api_key="explicit-key",
        )
        assert adapter._llm.model == "explicit-model"
        assert "explicit.example.com" in adapter._llm.openai_api_base

    def test_defaults_when_env_unset(self, isolated_env) -> None:
        os.environ["MISTRAL_API_KEY"] = "dummy"
        adapter = MistralLLMAdapter()
        assert adapter._llm.model == "mistral-large-latest"
        assert adapter._llm.temperature == 0.3


class TestComplete:
    def test_success(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(type(adapter._llm), "invoke", return_value=MagicMock(content="bonjour")):
            result = adapter.complete("system prompt", "user message")
        assert result == "bonjour"

    def test_with_history(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        mock_invoke = MagicMock(return_value=MagicMock(content="answer"))
        with patch.object(type(adapter._llm), "invoke", mock_invoke):
            history = (Message("user", "previous q"), Message("assistant", "previous a"))
            result = adapter.complete("sys", "current q", history=history)
        assert result == "answer"
        call_args, _ = mock_invoke.call_args
        messages = call_args[0]
        assert len(messages) == 4
        assert messages[0].content == "sys"
        assert messages[1].content == "previous q"
        assert messages[2].content == "previous a"
        assert messages[3].content == "current q"

    def test_empty_history(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(type(adapter._llm), "invoke", return_value=MagicMock(content="ans")):
            result = adapter.complete("sys", "user")
        assert result == "ans"

    def test_raises_on_api_error(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(type(adapter._llm), "invoke", side_effect=RuntimeError("API down")):
            with pytest.raises(RuntimeError, match="API down"):
                adapter.complete("sys", "user")
