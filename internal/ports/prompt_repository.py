"""
Prompt repository port: abstract interface for system-prompt storage.

Lets the domain read its own system prompts without knowing whether they
come from a hardcoded constant, a Postgres table, or a config file.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SystemPrompt:
    """A system prompt with metadata about the policy it enforces."""
    text: str
    enforces_pizza_only: bool


class PromptRepository(Protocol):
    """Anything that can return the configured pizza system prompt."""

    def get_pizza_system_prompt(self) -> SystemPrompt:
        ...
