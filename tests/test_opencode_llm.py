"""Tests for the OpenCode LLM adapter (langchain_openai.ChatOpenAI)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from internal.adapters.opencode_llm import OpenCodeLLMAdapter
from internal.adapters.token_counting import TokenUsage
from internal.ports.llm import Message


class TestConstructor:
    def test_defaults_from_env(self, isolated_env) -> None:
        os.environ["LLM_MODEL"] = "env-model"
        os.environ["LLM_BASE_URL"] = "https://env.example.com"
        os.environ["LLM_API_KEY"] = "env-key"
        adapter = OpenCodeLLMAdapter()
        assert adapter._llm.model == "env-model"
        assert "env.example.com" in adapter._llm.openai_api_base

    def test_explicit_overrides_env(self, isolated_env) -> None:
        os.environ["LLM_MODEL"] = "ignored-model"
        os.environ["LLM_BASE_URL"] = "https://ignored.example.com"
        os.environ["LLM_API_KEY"] = "ignored-key"
        adapter = OpenCodeLLMAdapter(
            model="explicit-model",
            base_url="https://explicit.example.com",
            api_key="explicit-key",
        )
        assert adapter._llm.model == "explicit-model"
        assert "explicit.example.com" in adapter._llm.openai_api_base

    def test_defaults_when_env_unset(self, isolated_env) -> None:
        os.environ["LLM_API_KEY"] = "dummy"
        adapter = OpenCodeLLMAdapter()
        assert adapter._llm.model == "deepseek-v4-flash-free"
        assert adapter._llm.temperature == 0.3


class TestComplete:
    def test_success(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        # P1-5: explicit usage_metadata=None so the dataclass doesn't
        # pick up a sub-MagicMock as a token count.
        with patch.object(
            type(adapter._llm), "invoke",
            return_value=MagicMock(content="hello", usage_metadata=None),
        ):
            result = adapter.complete("system prompt", "user message")
        assert result == "hello"

    def test_with_history(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        mock_invoke = MagicMock(
            return_value=MagicMock(content="answer", usage_metadata=None),
        )
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
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(
            type(adapter._llm), "invoke",
            return_value=MagicMock(content="ans", usage_metadata=None),
        ):
            result = adapter.complete("sys", "user")
        assert result == "ans"

    def test_raises_on_api_error(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(type(adapter._llm), "invoke", side_effect=RuntimeError("API down")):
            with pytest.raises(RuntimeError, match="API down"):
                adapter.complete("sys", "user")


# ── P1-5: real token usage extraction from langchain.usage_metadata ──


class TestTokenUsage:
    def test_last_usage_from_usage_metadata_dict(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content="ok",
            usage_metadata={
                "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
            },
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            adapter.complete("sys", "user")
        assert adapter.last_usage is not None
        assert adapter.last_usage.input_tokens == 100
        assert adapter.last_usage.output_tokens == 50

    def test_last_usage_from_response_metadata_legacy(self) -> None:
        """Older LangChain versions store usage under response_metadata."""
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content="ok",
            usage_metadata=None,
            response_metadata={
                "token_usage": {"prompt_tokens": 12, "completion_tokens": 8},
            },
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            adapter.complete("sys", "user")
        assert adapter.last_usage is not None
        assert adapter.last_usage.input_tokens == 12
        assert adapter.last_usage.output_tokens == 8

    def test_last_usage_none_on_api_error(self) -> None:
        """A failed call must reset last_usage."""
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        adapter.last_usage = TokenUsage(input_tokens=999, output_tokens=999)
        with patch.object(type(adapter._llm), "invoke", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                adapter.complete("sys", "user")
        assert adapter.last_usage is None


# ── P3-4: langchain AIMessage.content may be a list of content blocks ──


class TestContentBlocks:
    """P3-4: tool / multimodal OpenCode-compatible models return content as
    a list of content blocks instead of a plain str. Adapter must normalize
    to text before returning so the domain core's parsers stay safe."""

    def test_returns_joined_text_blocks(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content=[
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "world!"},
            ],
            usage_metadata=None,
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            result = adapter.complete("sys", "user")
        assert result == "Hello, world!"

    def test_drops_non_text_blocks(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content=[
                {"type": "text", "text": "Before "},
                {"type": "tool_use", "id": "abc", "name": "do_thing", "input": {"x": 1}},
                {"type": "text", "text": "after."},
            ],
            usage_metadata=None,
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            result = adapter.complete("sys", "user")
        assert result == "Before after."

    def test_empty_when_only_tool_blocks(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content=[{"type": "tool_use", "id": "abc", "name": "do", "input": {}}],
            usage_metadata=None,
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            result = adapter.complete("sys", "user")
        assert result == ""

    def test_empty_when_content_none(self) -> None:
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content=None,
            usage_metadata=None,
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            result = adapter.complete("sys", "user")
        assert result == ""

    def test_simple_str_passthrough(self) -> None:
        # Sanity: a plain-string content (the common case) is unchanged.
        adapter = OpenCodeLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(
            type(adapter._llm), "invoke",
            return_value=MagicMock(content="just text", usage_metadata=None),
        ):
            assert adapter.complete("sys", "user") == "just text"
