from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# ── State ──────────────────────────────────────────────────
class PizzaState(TypedDict):
    question: str
    answer: str

# ── Node: call LLM with pizza-only system prompt ───────────
def call_llm(state: PizzaState) -> PizzaState:
    from langchain_openai import ChatOpenAI
    import os

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash-free"),
        base_url=os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        temperature=0.3,
    )

    system_prompt = (
        "You are a strict pizza-only assistant. "
        "You ONLY answer questions that are about pizza — its ingredients, "
        "history, recipes, cultural variations, preparation techniques, or anything "
        "pizza-adjacent. If the question is NOT about pizza, politely refuse to answer "
        "and explain that you can only discuss pizza."
    )

    messages = [
        ("system", system_prompt),
        ("user", state["question"]),
    ]

    response = llm.invoke(messages)
    return {"answer": response.content, "question": state["question"]}

# ── Build graph ────────────────────────────────────────────
builder = StateGraph(PizzaState)
builder.add_node("llm", call_llm)
builder.add_edge(START, "llm")

graph = builder.compile()
