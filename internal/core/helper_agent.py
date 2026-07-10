"""
Domain core: the HelperAgent aggregate.

Pure business logic. No framework, no I/O, no LLM, no DB.
Depends only on the port protocol (LLM interface), not on adapters.
The system prompt is received from the backend via gRPC.

P1-5: after each successful adapter call, the agent records token usage
into `llm_tokens_total{provider,direction}`. The agent first reads the
adapter's `last_usage` attribute (set by the adapter from LangChain
usage_metadata or Ollama prompt_eval_count); if either direction is
missing, it falls back to tiktoken estimates from the assembled message
list (input) and the response text (output). This gives us real
billing-accurate counts where the vendor cooperates, and a much
better-than-chars/4 estimate everywhere else.
"""
import json
import logging
import os
import time
from dataclasses import dataclass

from internal.adapters.metrics import (
    active_requests,
    classify_error,
    llm_errors_total,
    llm_request_duration_seconds,
    llm_requests_total,
)
from internal.adapters.token_counting import (
    TokenUsage,
    count_messages_tokens,
    count_text_tokens,
    record_usage,
)
from internal.ports.llm import LLMPort, Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Question:
    """A user's question, normalized."""
    text: str

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("Question text cannot be empty")


@dataclass(frozen=True)
class Answer:
    """The assistant's answer to a question."""
    text: str
    detected_role: str = ""  # "worker" | "client" | "" (unclear)

    def __post_init__(self) -> None:
        if self.detected_role not in ("worker", "client", ""):
            raise ValueError(f"Invalid role: {self.detected_role!r}")


@dataclass(frozen=True)
class SystemPrompt:
    """A system prompt read from storage."""
    text: str


class HelperAgent:
    """
    Domain service: orchestrates answering a question using an LLM
    with the provided system prompt.

    Holds multiple LLM adapters; the backend selects which one to use
    per-request via the llm_provider field. Empty = auto fallback chain.
    """

    # R5: cheap-first by default. Premium (Mistral) is promoted only when
    # the backend explicitly sets llm_provider="mistral". Override via
    # FALLBACK_CHAIN env (comma-separated) without a code change.
    FALLBACK_CHAIN = (
        os.getenv("FALLBACK_CHAIN", "opencode0,opencode1,opencode2,mistral,ollama").split(",")
    )

    # R4: overall wall-clock budget for an Ask across the whole chain.
    REQUEST_BUDGET_S = float(os.getenv("REQUEST_BUDGET_S", "45.0"))

    def __init__(self, adapters: dict[str, LLMPort]) -> None:
        self._adapters = adapters
        logger.info("HelperAgent: %d adapters loaded", len(adapters))

    def answer(self, question: Question, system_prompt: str, history: tuple[Message, ...] = (), llm_provider: str = "", skip_role_detection: bool = False, deadline_s: float | None = None) -> Answer:
        active_requests.inc()
        try:
            return self._answer_inner(question, system_prompt, history, llm_provider, skip_role_detection, deadline_s)
        finally:
            active_requests.dec()

    def _answer_inner(self, question: Question, system_prompt: str, history: tuple[Message, ...], llm_provider: str, skip_role_detection: bool, deadline_s: float | None = None) -> Answer:
        # R4: compute per-request budget from the tighter of the global budget
        # and the client gRPC deadline. `deadline_s=0` means "no time left" (break
        # immediately), so check `is not None` (not truthiness) to avoid the
        # 0.0-falsey pitfall.
        budget = min(self.REQUEST_BUDGET_S, deadline_s) if deadline_s is not None else self.REQUEST_BUDGET_S
        started = time.monotonic()

        if llm_provider:
            providers_chain = [llm_provider] + [p for p in self.FALLBACK_CHAIN if p != llm_provider]
        else:
            providers_chain = self.FALLBACK_CHAIN

        if skip_role_detection:
            user_text = question.text
        else:
            user_text = question.text + (
                "\n\nIMPORTANT -- You MUST respond with valid JSON ONLY in this exact format: "
                '{"answer": "your response here", "role": "worker"}'
                ' Choose role="worker" if they offer services, role="client" if they need help, '
                'or role="" if unclear. Use double quotes only.'
            )

        # P1-5: assemble the input token estimate once per request. Used as
        # fallback when an adapter's `last_usage` lacks input_tokens. Per
        # provider in the loop we still re-derive to keep callers flexible
        # (some sanity-check providers return different content).
        history_parts = [m.content for m in history]
        fallback_input_parts = [system_prompt] + history_parts + [user_text]

        # P1-3 (R3): track that all of these providers failed, so the final
        # exception reflects the last adapter that actually raised (rather
        # than silently masking context).
        last_error: Exception | None = None

        for i, provider in enumerate(providers_chain):
            # R4: stop the chain when the budget is exhausted.
            elapsed = time.monotonic() - started
            if elapsed >= budget:
                logger.warning("Ask budget %.1fs exhausted before provider=%s -- stopping chain (R4)", budget, provider)
                break

            llm = self._adapters.get(provider)
            if not llm:
                logger.debug("No adapter for provider %r, skipping", provider)
                continue
            logger.info("Trying LLM provider=%s (attempt %d/%d) sp_len=%d q_len=%d history=%d",
                        provider, i + 1, len(providers_chain),
                        len(system_prompt), len(question.text), len(history))
            llm_requests_total.labels(provider=provider, mode="worker_intake").inc()
            llm_start = time.monotonic()
            try:
                raw = llm.complete(
                    system_prompt=system_prompt,
                    user=user_text,
                    history=history,
                )
                llm_request_duration_seconds.labels(provider=provider, mode="worker_intake").observe(time.monotonic() - llm_start)

                # P1-5: capture the adapter-populated usage, fall back to
                # tiktoken estimates for any missing direction.
                usage = getattr(llm, "last_usage", None) or TokenUsage()
                if usage.input_tokens is None:
                    usage = TokenUsage(
                        input_tokens=count_messages_tokens(fallback_input_parts),
                        output_tokens=usage.output_tokens,
                    )
                if usage.output_tokens is None:
                    usage = TokenUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=count_text_tokens(raw),
                    )
                record_usage(provider, usage)

                return self._parse_response(raw)
            except Exception as e:
                llm_request_duration_seconds.labels(provider=provider, mode="worker_intake").observe(time.monotonic() - llm_start)
                llm_errors_total.labels(provider=provider, error_type=classify_error(e)).inc()
                last_error = e
                logger.warning("LLM provider %s failed: %s", provider, e)
                continue

        logger.error("All LLM providers failed")
        raise last_error or RuntimeError("No LLM providers available")

    def _parse_response(self, raw: str) -> Answer:
        text = raw.strip()
        if text.startswith("```"):
            for prefix in ("```json", "```"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_part = text[brace_start:i + 1]
                        try:
                            data = json.loads(json_part)
                            answer_text = data.get("answer", "")
                            role = data.get("role", "")
                            if answer_text:
                                if role not in ("worker", "client", ""):
                                    logger.warning("LLM returned unexpected role %r, ignoring", role)
                                    role = ""
                                return Answer(text=answer_text, detected_role=role)
                        except json.JSONDecodeError as e:
                            logger.warning("LLM returned malformed JSON error=%s raw_text=%s", str(e), text[:200])
                        break

        return Answer(text=raw)
