"""
LLM port: abstract interface for any chat-completion provider.

The domain depends on this protocol, not on any specific vendor SDK.

P1-5: adapters also populate `self.last_usage: TokenUsage | None` after each
call. The agent reads it to record real (or tiktoken-estimated) token counts
into `llm_tokens_total{provider,direction}`. Keeping this out of `complete()`
preserves a clean str return type and tests that compare response text
directly.
"""
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from internal.adapters.token_counting import CompletionResult, TokenUsage  # noqa: F401  (re-export)


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    content: str


class LLMPort(Protocol):
    """Anything that can answer a chat-completion call.

    Implementations SHOULD set `self.last_usage = TokenUsage(...)` on the
    instance AFTER each successful call. The attribute MUST be reset to
    None at the start of each call (so failed calls don't leak prior
    successes). The agent reads `adapter.last_usage` after `complete()`
    returns; uses it if both directions are populated, else falls back to
    tiktoken estimates from `token_counting.count_messages_tokens(...)`.
    """

    last_usage: Optional[TokenUsage]

    def complete(self, system_prompt: str, user: str, history: Sequence[Message] = ()) -> str:
        """
        Return the assistant's text.

        Args:
            system_prompt: The system prompt to constrain the model.
            user: The current user question.
            history: Previous messages in the conversation (optional).
        """
        ...
