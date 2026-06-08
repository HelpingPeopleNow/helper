"""
LLM port: abstract interface for any chat-completion provider.

The domain depends on this protocol, not on any specific vendor SDK.
"""
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    content: str


class LLMPort(Protocol):
    """Anything that can answer a chat-completion call."""

    def complete(self, system_prompt: str, user: str, history: Sequence[Message] = ()) -> str:
        """
        Return the assistant's text.

        Args:
            system_prompt: The system prompt to constrain the model.
            user: The current user question.
            history: Previous messages in the conversation (optional).
        """
        ...
