from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# ── State ──────────────────────────────────────────────────
class PizzaState(TypedDict):
    question: str
    answer: str


# ── Node: call LLM with dynamic system prompt ──────────────
def create_call_llm(system_prompt: str):
    """Factory that returns an LLM node bound to the given system prompt."""

    def call_llm(state: PizzaState) -> PizzaState:
        from langchain_openai import ChatOpenAI
        import os

        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-v4-flash-free"),
            base_url=os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=0.3,
        )

        messages = [
            ("system", system_prompt),
            ("user", state["question"]),
        ]

        response = llm.invoke(messages)
        return {"answer": response.content, "question": state["question"]}

    return call_llm


def build_graph(system_prompt: str):
    """Build the LangGraph state machine with the given system prompt."""
    builder = StateGraph(PizzaState)
    builder.add_node("llm", create_call_llm(system_prompt))
    builder.add_edge(START, "llm")
    return builder.compile()
