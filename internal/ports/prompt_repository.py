"""
Prompt repository port: abstract interface for system-prompt storage.

Lets the domain read its own system prompts without knowing whether they
come from a hardcoded constant, a Postgres table, or a config file.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SystemPrompt:
    """A system prompt read from storage."""
    text: str


class PromptRepository(Protocol):
    """Anything that can return the configured system prompt."""

    def get_system_prompt(self) -> SystemPrompt:
        ...
