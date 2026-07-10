"""
Token counting for LLM adapters (P1-5).

Centralizes:
- tiktoken cl100k_base encoder singleton (no model-name lookup since tiktoken
  natively supports only OpenAI; for Mistral/OpenCode/Ollama we use cl100k_base
  as an industry-standard cross-model approximation. When LangChain's
  `AIMessage.usage_metadata` is populated by the vendor, that wins.)
- TokenUsage dataclass (input + output token counts)
- CompletionResult dataclass wrapping (text, TokenUsage)
- `record_usage(adapter, provider, usage)` helper that updates the
  `llm_tokens_total{provider,direction=...}` Prometheus counters with safe
  fallbacks if usage is missing in any direction.

Why tiktoken cl100k_base and not p50k_base / o200k_base:
- p50k_base is for codex; cl100k_base is the GPT-3.5/4 base and is what
  tiktoken recommends for unknown models when the vendor doesn't expose
  a served-model encoding. LangChain adapters decode exactly; Ollama has
  no decoder at all. cl100k_base is the de-facto cross-model fallback
  in the Python ecosystem.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    import tiktoken  # noqa: WPS433 (intentional optional import)
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken always imports successfully
    _ENCODER = None


# Lazily imported here to avoid an import cycle (metrics.py is the only
# consumer at module init time). The counters are module globals in metrics.
def _counters():
    from internal.adapters.metrics import (
        llm_tokens_total,
        classify_error,
        llm_errors_total,
    )
    return llm_tokens_total, classify_error, llm_errors_total


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """Real or estimated LLM token usage for a single call.

    `input_tokens` is what the assistant was asked to consider; `output_tokens`
    is what the model returned. Either may be None if the vendor didn't return
    them — the agent falls back to tiktoken estimates in that case.
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def is_complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


@dataclass(frozen=True)
class CompletionResult:
    """Adapters return this instead of bare str so the agent can record
    exact token usage when the vendor cooperates (LangChain usage_metadata
    for OpenAI/Mistral; Ollama eval_count for /api/generate)."""
    text: str
    usage: TokenUsage


def count_text_tokens(text: str) -> int:
    """Encode `text` with tiktoken cl100k_base and return the token count.

    Falls back to `len(text) // 4` if tiktoken failed to load (defensive).
    The chars/4 fallback is the *previous* behavior; we keep it only as the
    last line of defense, but emit a one-shot warning so it's discoverable.
    """
    if _ENCODER is None:
        if not count_text_tokens._warned:
            logger.warning("tiktoken unavailable; falling back to chars/4 token estimate")
            count_text_tokens._warned = True
        return max(len(text) // 4, 0)
    return len(_ENCODER.encode(text))


count_text_tokens._warned = False  # type: ignore[attr-defined]


def count_messages_tokens(parts: list[str]) -> int:
    """Sum token counts across multiple message parts (system + history +
    user). Used as the input-direction estimate when the vendor didn't
    return usage.
    """
    return sum(count_text_tokens(p) for p in parts)


def extract_usage_from_langchain(response) -> TokenUsage:
    """Best-effort extraction of token usage from a LangChain AIMessage.

    LangChain ≥0.1 exposes `usage_metadata` as a typed dict (input_tokens,
    output_tokens, total_tokens) for OpenAI-compatible endpoints that
    populate `response.usage`. Older / free-tier endpoints may leave it
    None — we return `TokenUsage(None, None)` and the agent falls back to
    tiktoken estimates.

    Also inspects `response_metadata.get("token_usage")` as a secondary
    fallback for older LangChain versions.
    """
    if response is None:
        return TokenUsage()

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        in_t = usage_metadata.get("input_tokens")
        out_t = usage_metadata.get("output_tokens")
    elif usage_metadata is not None:
        # Some versions return an object with attributes, not a dict.
        in_t = getattr(usage_metadata, "input_tokens", None)
        out_t = getattr(usage_metadata, "output_tokens", None)
    else:
        in_t = None
        out_t = None

    if in_t is None and out_t is None:
        # Older LangChain fallback: response_metadata["token_usage"]
        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") if isinstance(response_metadata, dict) else None
        if isinstance(token_usage, dict):
            in_t = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
            out_t = token_usage.get("completion_tokens") or token_usage.get("output_tokens")

    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def langchain_content_to_text(content: object) -> str:
    """P3-4: normalize langchain `AIMessage.content` into a plain string.

    Multimodal / tool-calling models (Claude-via-OpenAI, Groq, OpenCode
    tool endpoints) return content as a list of content blocks instead
    of a plain string. The agent's `_parse_response` (regex / JSON
    block extract) only handles strings, so each adapter MUST flatten
    to text before returning. Non-text blocks (tool_use, image, etc.)
    are intentionally dropped — tool calls are an out-of-band contract
    and leaving their JSON in the answer string would poison the parser.

    Accepted shapes:

        None              -> ""
        str               -> unchanged
        list[str]         -> elements joined
        list[dict]        -> every {"type": "text", ...} block's "text"
                            field, joined; everything else dropped
        anything else     -> str(content), defensive

    Modules outside `internal.adapters.*` should never need this; the
    `LLMPort` contract is that `complete()` returns `str`. This helper
    exists purely to keep that contract honest when langchain returns a
    richer shape internally.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # LangChain content block: {"type": "text", "text": "...", ...}.
                # Other types (tool_use / function_call / image / reasoning)
                # carry no user-facing prose — drop them.
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)
    # Last-line-of-defense: encode anything else as its string repr.
    try:
        return str(content)
    except Exception:  # noqa: BLE001 — defensive, never raise here
        return ""


def extract_usage_from_ollama(data: dict, fallback_output_text: str) -> TokenUsage:
    """Extract real prompt_eval_count / eval_count from Ollama /api/generate
    response (returned when `stream=False`). These are the daemon's actual
    counts, much more accurate than tiktoken for Qwen models.
    """
    in_t = data.get("prompt_eval_count")
    out_t = data.get("eval_count")
    if in_t is None and out_t is None:
        return TokenUsage()
    if out_t is None:
        out_t = count_text_tokens(fallback_output_text)
    if in_t is None:
        in_t = 0
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def record_usage(provider: str, usage: TokenUsage) -> None:
    """Record input + output token counts into llm_tokens_total counter.

    Called from helper_agent._answer_inner after each successful LLM call.
    If usage is incomplete, the caller (agent) is expected to compute
    tiktoken fallback estimates *before* calling this and pass them in.
    """
    llm_tokens_total, _, _ = _counters()
    if usage.input_tokens is not None and usage.input_tokens >= 0:
        llm_tokens_total.labels(provider=provider, direction="input").inc(usage.input_tokens)
    if usage.output_tokens is not None and usage.output_tokens >= 0:
        llm_tokens_total.labels(provider=provider, direction="output").inc(usage.output_tokens)
