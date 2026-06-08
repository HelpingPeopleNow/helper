"""
Domain core: the PizzaAssistant aggregate.

Pure business logic. No framework, no I/O, no LLM, no DB.
Depends only on the port protocols (interfaces), not on adapters.
"""
from dataclasses import dataclass
from typing import Protocol

from internal.ports.llm import LLMPort
from internal.ports.prompt_repository import PromptRepository


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


class PizzaOnlyViolation(Exception):
    """Raised when a prompt template does not enforce the pizza-only policy."""


class PizzaAssistant:
    """
    Domain service: orchestrates answering a question about pizza.

    Uses injected ports (LLM + prompt repository). The domain itself
    doesn't know whether the LLM is OpenAI, OpenCode, or a mock — that's
    the adapters' concern.
    """

    def __init__(self, llm: LLMPort, prompts: PromptRepository) -> None:
        self._llm = llm
        self._prompts = prompts

    def answer(self, question: Question) -> Answer:
        system_prompt = self._prompts.get_pizza_system_prompt()
        if not system_prompt.enforces_pizza_only:
            raise PizzaOnlyViolation(
                "Configured system prompt must enforce the pizza-only policy"
            )
        text = self._llm.complete(system_prompt=system_prompt.text, user=question.text)
        return Answer(text=text)
