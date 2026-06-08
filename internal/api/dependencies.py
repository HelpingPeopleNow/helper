"""
Composition root: the ONLY place in the app where adapters are wired
to the domain. FastAPI's dependency-injection system calls this once
per request, but the underlying instances are cached at app startup.
"""
from functools import lru_cache

from internal.adapters.langgraph_runner import LangGraphRunner
from internal.adapters.opencode_llm import OpenCodeLLMAdapter
from internal.adapters.postgres_repo import PostgresPromptRepository
from internal.core.helper_agent import HelperAgent
from internal.ports.graph import GraphRunner
from internal.ports.llm import LLMPort
from internal.ports.prompt_repository import PromptRepository


@lru_cache
def _build_llm() -> LLMPort:
    return OpenCodeLLMAdapter()


@lru_cache
def _build_prompts() -> PromptRepository:
    return PostgresPromptRepository()


@lru_cache
def _build_assistant() -> HelperAgent:
    return HelperAgent(llm=_build_llm(), prompts=_build_prompts())


@lru_cache
def _build_graph() -> GraphRunner:
    return LangGraphRunner(assistant=_build_assistant())


# Exposed for FastAPI Depends()
def get_assistant() -> HelperAgent:
    return _build_assistant()


def get_graph() -> GraphRunner:
    return _build_graph()
