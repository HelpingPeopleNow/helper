"""
Pytest fixtures for the helper test suite.

The helper is a stateless adapter; tests in this suite use only:
- In-memory mocks for LLMs and embeddings (no real services)
- `unittest.mock` patches to verify call sites
- An in-process gRPC server bound to localhost:0 for R1/R3/R8 tests

No fixtures talk to real LLM APIs or a real Ollama daemon.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Make the helper package importable as `from internal...` and `from proto...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))


class FakeAdapter:
    """Test double for the LLMPort protocol.

    Records every call and returns canned responses or raises canned errors
    in declared order. Tests can inspect `calls` to assert call shape and
    order.
    """

    def __init__(
        self,
        responses: tuple[str, ...] = (),
        errors: tuple[Exception, ...] = (),
        delay_s: float = 0.0,
    ) -> None:
        self.responses = list(responses)
        self.errors = list(errors)
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []

    def complete(self, system_prompt: str, user: str, history: tuple = ()) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user": user,
                "history": history,
            }
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.errors:
            raise self.errors.pop(0)
        if not self.responses:
            raise AssertionError("FakeAdapter exhausted")
        return self.responses.pop(0)


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def fake_openai_compat() -> MagicMock:
    """MagicMock for `_check_openai_compat` style helpers."""
    m = MagicMock(return_value=("ok", "ok"))
    m.__name__ = "_check_openai_compat"
    return m


@pytest.fixture
def fake_ollama() -> MagicMock:
    m = MagicMock(return_value=("ok", "ok"))
    m.__name__ = "_check_ollama"
    return m


@pytest.fixture
def fake_ollama_embedding() -> MagicMock:
    m = MagicMock(return_value=("ok", "ok"))
    m.__name__ = "_check_ollama_embedding"
    return m


@pytest.fixture
def reset_health_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level health cache between tests."""
    import internal.adapters.grpc_server as gs

    gs._health_cache.clear()
    gs._health_cache_detail.clear()
    gs._health_cache_ts = 0.0
    yield


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip all HELPER_* / LLM_* / OLLAMA_* env vars for a clean slate."""
    for key in list(os.environ.keys()):
        if any(
            key.startswith(p)
            for p in ("HELPER_", "LLM_", "MISTRAL_", "OLLAMA_", "GRPC_", "EMBEDDING_", "HEALTH_", "MAX_")
        ):
            monkeypatch.delenv(key, raising=False)
    yield
