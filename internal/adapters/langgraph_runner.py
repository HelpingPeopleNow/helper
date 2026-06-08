"""
LangGraph-based graph runner adapter.

Implements the GraphRunner port using the PizzaAssistant domain service
through a tiny StateGraph. LangGraph is overkill for a single-node flow,
but it gives us a place to add nodes (intent detection, response guard,
logging) without restructuring the domain.
"""
from typing import Any

from langgraph.graph import END, START, StateGraph

from internal.core.pizza_assistant import PizzaAssistant
from internal.ports.graph import GraphRunner, GraphState


def _make_answer_node(assistant: PizzaAssistant):
    def answer_node(state: GraphState) -> dict[str, Any]:
        from internal.core.pizza_assistant import Question
        result = assistant.answer(Question(text=state["question"]))
        return {"answer": result.text, "question": state["question"]}
    return answer_node


class LangGraphRunner(GraphRunner):
    def __init__(self, assistant: PizzaAssistant) -> None:
        builder = StateGraph(GraphState)
        builder.add_node("answer", _make_answer_node(assistant))
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        self._graph = builder.compile()

    def run(self, question: str) -> dict[str, Any]:
        return self._graph.invoke({"question": question})
