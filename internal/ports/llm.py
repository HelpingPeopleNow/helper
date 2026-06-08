"""
LLM port: abstract interface for any chat-completion provider.

The domain depends on this protocol, not on any specific vendor SDK.
"""
from typing import Protocol


class LLMPort(Protocol):
    """Anything that can answer a chat-completion call."""

    def complete(self, system_prompt: str, user: str) -> str:
        """Return the assistant's text for the given prompt pair."""
        ...
