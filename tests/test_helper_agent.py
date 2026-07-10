"""
Tests for `HelperAgent` — the pure domain core.

Covers:
- Fallback chain construction (R5: cheap-first order)
- Adapter iteration and error handling
- Response parsing
- Dataclass invariants
- Deadline / budget propagation (R4)
- Token recording (P1-5 — input AND output directions via tiktoken fallback)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proto"))

from internal.adapters.metrics import llm_tokens_total
from internal.adapters.token_counting import TokenUsage
from internal.core.helper_agent import Answer, HelperAgent, Question


# ── §3.1 Fallback chain construction (R5) ───────────────────────────


class TestFallbackChainOrder:
    def test_default_chain_is_cheap_first(self) -> None:
        """R5: Mistral (premium) is no longer first. OpenCode leads."""
        chain = list(HelperAgent.FALLBACK_CHAIN)
        assert chain[0] == "opencode0"
        assert "mistral" in chain
        assert chain.index("mistral") > 0
        assert chain[-1] == "ollama"

    def test_fallback_chain_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """#16a: FALLBACK_CHAIN env var reshapes the chain."""
        monkeypatch.setenv("FALLBACK_CHAIN", "ollama,mistral")
        result = os.getenv("FALLBACK_CHAIN", "opencode0,opencode1,opencode2,mistral,ollama").split(",")
        assert result == ["ollama", "mistral"]

    def test_explicit_provider_promoted_to_front(self) -> None:
        agent = HelperAgent(adapters={"opencode0": _Stub(), "ollama": _Stub(), "mistral": _Stub()})
        order: list[str] = []
        for name in ("opencode0", "ollama", "mistral"):
            adapter = _Stub(on_call=lambda *a, _n=name, **kw: order.append(_n) or "ok")
            agent._adapters[name] = adapter
        agent.answer(Question("hi"), system_prompt="sp", llm_provider="ollama")
        assert order[0] == "ollama"
        assert order.count("ollama") == 1

    def test_explicit_provider_not_in_tail(self) -> None:
        agent = HelperAgent(adapters={"opencode0": _Stub(), "ollama": _Stub(), "mistral": _Stub()})
        order: list[str] = []
        for name in ("opencode0", "ollama", "mistral"):
            agent._adapters[name] = _Stub(on_call=lambda *a, _n=name, **kw: order.append(_n) or "ok")
        agent.answer(Question("hi"), system_prompt="sp", llm_provider="opencode0")
        assert order.count("opencode0") == 1

    def test_explicit_provider_not_loaded_still_continues(self) -> None:
        agent = HelperAgent(adapters={"ollama": _Stub()})
        agent._adapters["ollama"] = _Stub(on_call=lambda *a, **kw: "ok")
        result = agent.answer(Question("hi"), system_prompt="sp", llm_provider="opencode0")
        assert result.text == "ok"

    def test_empty_provider_treated_as_auto(self) -> None:
        agent = HelperAgent(adapters={"ollama": _Stub()})
        agent._adapters["ollama"] = _Stub(on_call=lambda *a, **kw: "ok")
        result = agent.answer(Question("hi"), system_prompt="sp", llm_provider="")
        assert result.text == "ok"


# ── §3.2 Adapter iteration & error handling ─────────────────────────


class TestAdapterIteration:
    def test_first_adapter_success_breaks_loop(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "first ok")
        b = _Stub(on_call=lambda *a, **kw: "second ok", last_usage=TokenUsage(input_tokens=5, output_tokens=5))
        agent = HelperAgent(adapters={"opencode0": a, "ollama": b})
        result = agent.answer(Question("hi"), system_prompt="sp")
        assert result.text == "first ok"
        assert len(b.calls) == 0

    def test_first_adapter_fails_second_succeeds(self) -> None:
        a = _Stub(raises=RuntimeError("boom"))
        b = _Stub(on_call=lambda *a, **kw: "second ok", last_usage=TokenUsage(input_tokens=10, output_tokens=10))
        agent = HelperAgent(adapters={"opencode0": a, "ollama": b})
        result = agent.answer(Question("hi"), system_prompt="sp")
        assert result.text == "second ok"
        assert len(b.calls) == 1

    def test_all_adapters_fail_raises_last_error(self) -> None:
        a = _Stub(raises=ValueError("opencode0 err"))
        b = _Stub(raises=RuntimeError("ollama err"))
        agent = HelperAgent(adapters={"opencode0": a, "ollama": b})
        with pytest.raises(RuntimeError, match="ollama err"):
            agent.answer(Question("hi"), system_prompt="sp")

    def test_empty_adapters_raises(self) -> None:
        agent = HelperAgent(adapters={})
        with pytest.raises(RuntimeError, match="No LLM providers available"):
            agent.answer(Question("hi"), system_prompt="sp")


# ── §3.3 skip_role_detection ───────────────────────────────────────


class TestSkipRoleDetection:
    def test_skip_role_detection_false_appends_json(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "ok")
        agent = HelperAgent(adapters={"ollama": a})
        agent.answer(Question("hello"), system_prompt="sp", skip_role_detection=False)
        assert '"answer"' in a.calls[0]["user"]
        assert '"role"' in a.calls[0]["user"]

    def test_skip_role_detection_true_preserves_user_text(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "ok")
        agent = HelperAgent(adapters={"ollama": a})
        agent.answer(Question("hello world"), system_prompt="sp", skip_role_detection=True)
        assert a.calls[0]["user"] == "hello world"

    def test_json_instruction_not_in_system_prompt(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "ok")
        agent = HelperAgent(adapters={"ollama": a})
        agent.answer(Question("hi"), system_prompt="clean sp", skip_role_detection=False)
        assert a.calls[0]["system_prompt"] == "clean sp"


# ── §3.4 Response parsing ───────────────────────────────────────────


class TestResponseParsing:
    def test_pure_json(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: '{"answer":"hi","role":"worker"}')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == "hi"
        assert result.detected_role == "worker"

    def test_json_without_role(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: '{"answer":"hi"}')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == "hi"
        assert result.detected_role == ""

    def test_json_with_invalid_role_resets(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: '{"answer":"hi","role":"manager"}')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.detected_role == ""

    def test_markdown_fenced_json(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: '```json\n{"answer":"hi","role":"client"}\n```')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == "hi"
        assert result.detected_role == "client"

    def test_markdown_fenced_no_lang(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: '```\n{"answer":"hi","role":"worker"}\n```')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == "hi"
        assert result.detected_role == "worker"

    def test_malformed_json_falls_through_to_raw(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: '{"answer":"hi",')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == '{"answer":"hi",'
        assert result.detected_role == ""

    def test_plain_text(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "just plain text")
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == "just plain text"
        assert result.detected_role == ""

    def test_nested_braces(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: '{"answer":"foo {bar} baz","role":""}')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == "foo {bar} baz"

    def test_leading_text_before_json(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: 'Sure! {"answer":"hi","role":"client"}')
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == "hi"
        assert result.detected_role == "client"

    def test_empty_string(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "")
        agent = HelperAgent(adapters={"ollama": a})
        result = agent.answer(Question("q"), system_prompt="sp", skip_role_detection=True)
        assert result.text == ""
        assert result.detected_role == ""


# ── §3.5 Dataclass invariants ───────────────────────────────────────


class TestDataclassInvariants:
    def test_question_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Question text cannot be empty"):
            Question("")

    def test_question_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="Question text cannot be empty"):
            Question("   ")

    def test_answer_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid role"):
            Answer(text="hi", detected_role="admin")

    def test_answer_uppercase_role_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid role"):
            Answer(text="hi", detected_role="WORKER")

    def test_answer_frozen(self) -> None:
        a = Answer(text="hi", detected_role="worker")
        with pytest.raises(Exception):
            a.text = "modified"  # type: ignore[misc]

    def test_answer_valid_roles(self) -> None:
        for role in ("worker", "client", ""):
            assert Answer(text="hi", detected_role=role).detected_role == role


# ── §3.6 Deadline / budget propagation (R4) ─────────────────────────


class TestDeadlineBudget:
    def test_no_deadline_uses_default_budget(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "ok")
        agent = HelperAgent(adapters={"opencode0": a})
        result = agent.answer(Question("hi"), system_prompt="sp", deadline_s=None)
        assert result.text == "ok"

    def test_tight_deadline_stops_chain_after_first_fail(self) -> None:
        original = HelperAgent.REQUEST_BUDGET_S
        HelperAgent.REQUEST_BUDGET_S = 10.0
        try:
            def _slow_fail(*args, **kwargs):
                time.sleep(0.05)
                raise RuntimeError("slow fail")
            a = _Stub(on_call=_slow_fail)
            b = _Stub(on_call=lambda *a, **kw: "second ok")
            agent = HelperAgent(adapters={"opencode0": a, "ollama": b})
            with pytest.raises(RuntimeError):
                agent.answer(Question("hi"), system_prompt="sp", deadline_s=0.001)
            assert len(a.calls) == 1
            assert len(b.calls) == 0
        finally:
            HelperAgent.REQUEST_BUDGET_S = original

    def test_zero_deadline_breaks_immediately(self) -> None:
        a = _Stub(on_call=lambda *a, **kw: "ok")
        agent = HelperAgent(adapters={"opencode0": a})
        with pytest.raises(RuntimeError, match="No LLM providers available"):
            agent.answer(Question("hi"), system_prompt="sp", deadline_s=0.0)

    def test_budget_check_happens_before_complete_call(self) -> None:
        original = HelperAgent.REQUEST_BUDGET_S
        HelperAgent.REQUEST_BUDGET_S = 0.05
        try:
            a_calls = []
            a = _Stub(
                on_call=lambda *a, _c=a_calls, **kw: (
                    _c.append(1) or time.sleep(0.2) or "ok"
                ),
            )
            b_calls = []
            b = _Stub(
                on_call=lambda *a, _c=b_calls, **kw: _c.append(1) or "ok",
            )
            agent = HelperAgent(adapters={"opencode0": a, "ollama": b})
            result = agent.answer(Question("hi"), system_prompt="sp")
            assert result.text == "ok"
            assert len(a.calls) == 1
            assert len(b.calls) == 0
        finally:
            HelperAgent.REQUEST_BUDGET_S = original


# ── §3.7 P1-5: token recording into llm_tokens_total ─────────────────


class TestTokenRecording:
    def test_real_usage_recorded_in_both_directions(self) -> None:
        """Adapter with last_usage=TokenUsage(50, 25): counters get +50 input and +25 output."""
        a = _Stub(
            on_call=lambda *a, **kw: "ok",
            last_usage=TokenUsage(input_tokens=50, output_tokens=25),
        )
        agent = HelperAgent(adapters={"opencode0": a})

        # Capture counter before/after via Prometheus internal API.
        before_in = llm_tokens_total.labels(
            provider="opencode0", direction="input",
        )._value.get()
        before_out = llm_tokens_total.labels(
            provider="opencode0", direction="output",
        )._value.get()

        agent.answer(Question("hi"), system_prompt="sp", skip_role_detection=True)

        after_in = llm_tokens_total.labels(
            provider="opencode0", direction="input",
        )._value.get()
        after_out = llm_tokens_total.labels(
            provider="opencode0", direction="output",
        )._value.get()
        assert after_in - before_in == 50
        assert after_out - before_out == 25

    def test_tiktoken_fallback_when_usage_input_only(self) -> None:
        """Output direction falls back to tiktoken if adapter didn't return it."""
        a = _Stub(
            on_call=lambda *a, **kw: "ok",
            last_usage=TokenUsage(input_tokens=10, output_tokens=None),
        )
        agent = HelperAgent(adapters={"opencode0": a})
        before_out = llm_tokens_total.labels(
            provider="opencode0", direction="output",
        )._value.get()
        agent.answer(Question("hi"), system_prompt="sp", skip_role_detection=True)
        after_out = llm_tokens_total.labels(
            provider="opencode0", direction="output",
        )._value.get()
        # "ok" via tiktoken: small but non-zero
        assert after_out >= before_out

    def test_tiktoken_fallback_when_usage_missing_completely(self) -> None:
        """Adapter with no last_usage: both directions still recorded via tiktoken."""
        a = _Stub(on_call=lambda *a, **kw: "ok", last_usage=None)
        agent = HelperAgent(adapters={"opencode0": a})
        before_in = llm_tokens_total.labels(
            provider="opencode0", direction="input",
        )._value.get()
        before_out = llm_tokens_total.labels(
            provider="opencode0", direction="output",
        )._value.get()
        agent.answer(
            Question("hello world"), system_prompt="sp", skip_role_detection=True,
        )
        after_in = llm_tokens_total.labels(
            provider="opencode0", direction="input",
        )._value.get()
        after_out = llm_tokens_total.labels(
            provider="opencode0", direction="output",
        )._value.get()
        assert after_in >= before_in + 1
        assert after_out >= before_out + 1


# ── Helpers ─────────────────────────────────────────────────────────


class _Stub:
    """Minimal LLMPort stub. Records calls, returns canned value or raises.

    P1-5: exposes `last_usage` so the agent can read real or estimated token
    counts after each call (matches the protocol note in
    `internal/ports/llm.py`).
    """

    def __init__(
        self,
        on_call: "callable | None" = None,
        raises: "Exception | None" = None,
        delay_s: float = 0.0,
        last_usage: "TokenUsage | None" = None,
    ) -> None:
        self._on_call = on_call
        self._raises = raises
        self._delay_s = delay_s
        self.last_usage = last_usage
        self.calls: list[dict] = []

    def complete(self, system_prompt: str, user: str, history: tuple = ()) -> str:
        self.calls.append({"system_prompt": system_prompt, "user": user, "history": history})
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._raises:
            raise self._raises
        if self._on_call:
            return self._on_call(system_prompt, user, history)
        return "ok"
