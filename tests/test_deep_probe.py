"""Tests for admin-scoped deep probe targeting."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proto"))

from internal.adapters import enabled_providers as ep
from internal.adapters import grpc_server as gs


class TestResolveDeepProbeTargets:
    def test_uses_admin_list_when_set(self) -> None:
        adapters = {"groq": object(), "openrouter": object(), "opencode0": object()}
        assert ep.resolve_deep_probe_targets(adapters, ["groq", "openrouter"]) == [
            "groq",
            "openrouter",
        ]

    def test_falls_back_when_admin_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALLBACK_CHAIN", "groq,ollama")
        adapters = {"groq": object(), "ollama": object(), "mistral": object()}
        assert ep.resolve_deep_probe_targets(adapters, []) == ["groq", "ollama"]
        assert ep.resolve_deep_probe_targets(adapters, None) == ["groq", "ollama"]

    def test_skips_providers_not_loaded(self) -> None:
        adapters = {"groq": object(), "ollama": object()}
        assert ep.resolve_deep_probe_targets(adapters, ["groq", "openrouter"]) == ["groq"]


class TestRunDeepProbeScoping:
    def test_skips_disabled_providers_and_marks_them_ok(self) -> None:
        ok = MagicMock()
        bad = MagicMock(side_effect=RuntimeError("down"))
        adapters = {
            "groq": ok,
            "openrouter": ok,
            "opencode0": bad,
        }
        source = ep.EnabledProvidersSource("")
        with patch.object(source, "fetch", return_value=["groq", "openrouter"]):
            results = gs._run_deep_probe(adapters, source)

        assert results == {"groq": True, "openrouter": True}
        ok.assert_called()
        bad.assert_not_called()
        assert gs.helper_deep_probe_success.labels(provider="opencode0")._value.get() == 1.0

    def test_probes_fallback_chain_when_admin_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALLBACK_CHAIN", "groq,ollama")
        ok = MagicMock()
        adapters = {"groq": ok, "opencode0": MagicMock(side_effect=RuntimeError("down"))}
        source = ep.EnabledProvidersSource("")
        with patch.object(source, "fetch", return_value=[]):
            results = gs._run_deep_probe(adapters, source)

        assert results == {"groq": True}
        adapters["opencode0"].assert_not_called()
