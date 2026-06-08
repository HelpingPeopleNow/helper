"""
Graph runner port: thin abstraction around the orchestration graph.

In a real product, this would let you swap LangGraph for Haystack,
LlamaIndex, or a hand-rolled state machine. For the current single-node
flow, the abstraction still keeps the domain clean and future-proof.
"""
from typing import Any, Protocol, TypedDict


class GraphState(TypedDict, total=False):
    question: str
    answer: str


class GraphRunner(Protocol):
    """Anything that can run a question through the assistant pipeline."""

    def run(self, question: str) -> dict[str, Any]:
        ...
