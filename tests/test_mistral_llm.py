"""Tests for the Mistral LLM adapter (langchain_openai.ChatOpenAI)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from internal.adapters.mistral_llm import MistralLLMAdapter
from internal.adapters.token_counting import TokenUsage
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
        with patch.object(
            type(adapter._llm), "invoke",
            return_value=MagicMock(content="bonjour", usage_metadata=None),
        ):
            result = adapter.complete("system prompt", "user message")
        assert result == "bonjour"

    def test_with_history(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
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
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(
            type(adapter._llm), "invoke",
            return_value=MagicMock(content="ans", usage_metadata=None),
        ):
            result = adapter.complete("sys", "user")
        assert result == "ans"

    def test_raises_on_api_error(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(type(adapter._llm), "invoke", side_effect=RuntimeError("API down")):
            with pytest.raises(RuntimeError, match="API down"):
                adapter.complete("sys", "user")


# ── P1-5: Mistral-large backend populates response.usage via langchain ──


class TestTokenUsage:
    def test_last_usage_from_usage_metadata_dict(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content="ok",
            usage_metadata={
                "input_tokens": 200, "output_tokens": 80, "total_tokens": 280,
            },
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            adapter.complete("sys", "user")
        assert adapter.last_usage is not None
        assert adapter.last_usage.input_tokens == 200
        assert adapter.last_usage.output_tokens == 80

    def test_last_usage_none_on_api_error(self) -> None:
        """A failed call must reset last_usage so we don't leak prior data."""
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        adapter.last_usage = TokenUsage(input_tokens=999, output_tokens=999)
        with patch.object(type(adapter._llm), "invoke", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                adapter.complete("sys", "user")
        assert adapter.last_usage is None


# ── P3-4: langchain AIMessage.content may be a list of content blocks ──


class TestContentBlocks:
    """P3-4: Mistral (and other OpenAI-compatible langchain adapters) may
    return content as a list of content blocks for tool / multimodal calls.
    Adapter must normalize to text before returning."""

    def test_returns_joined_text_blocks(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content=[{"type": "text", "text": "Hi "}, {"type": "text", "text": "there!"}],
            usage_metadata=None,
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            assert adapter.complete("sys", "user") == "Hi there!"

    def test_drops_non_text_blocks(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        fake_response = MagicMock(
            content=[
                {"type": "text", "text": "Answer: "},
                {"type": "function_call", "name": "lookup", "arguments": "{}"},
                {"type": "text", "text": "Done."},
            ],
            usage_metadata=None,
            response_metadata={},
        )
        with patch.object(type(adapter._llm), "invoke", return_value=fake_response):
            assert adapter.complete("sys", "user") == "Answer: Done."

    def test_empty_when_content_none(self) -> None:
        adapter = MistralLLMAdapter(model="m", api_key="k", base_url="https://example.com")
        with patch.object(
            type(adapter._llm), "invoke",
            return_value=MagicMock(content=None, usage_metadata=None),
        ):
            assert adapter.complete("sys", "user") == ""


# ── P3-4: direct helper coverage ──────────────────────────────────────


class TestLangchainContentNormalizer:
    """Direct coverage for the helper used by both langchain adapters."""

    def test_str_passthrough(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        assert langchain_content_to_text("hello") == "hello"

    def test_none_to_empty(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        assert langchain_content_to_text(None) == ""

    def test_list_of_text_blocks_joined(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        assert langchain_content_to_text(
            [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}],
        ) == "foobar"

    def test_list_mixed_text_and_tool_blocks(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        content = [
            {"type": "text", "text": "a"},
            {"type": "tool_use", "id": "x", "name": "do", "input": {}},
            {"type": "text", "text": "b"},
        ]
        assert langchain_content_to_text(content) == "ab"

    def test_list_of_strings_joined(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        assert langchain_content_to_text(["x", "y"]) == "xy"

    def test_empty_list(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        assert langchain_content_to_text([]) == ""

    def test_unknown_object_uses_str_fallback(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        class Weird:
            def __str__(self): return "weird-repr"
        assert langchain_content_to_text(Weird()) == "weird-repr"

    def test_text_block_missing_text_field_doesnt_crash(self) -> None:
        from internal.adapters.token_counting import langchain_content_to_text
        # Malformed text block without a 'text' key should contribute "" —
        # not raise TypeError on indexing.
        assert (
            langchain_content_to_text(
                [{"type": "text"}, {"type": "text", "text": "ok"}],
            )
            == "ok"
        )
