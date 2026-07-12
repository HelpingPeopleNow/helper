"""Tests for main.py — require_env and adapter loading logic."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the helper package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proto"))


class TestRequireEnv:
    """Tests for require_env() function."""

    def test_returns_value_when_env_var_is_set(self, isolated_env, monkeypatch):
        monkeypatch.setenv("TEST_VAR_123", "hello")
        main_mod = importlib.import_module("main")
        main_mod.require_env = main_mod.__dict__.get("require_env", lambda k: "")
        # Reimport to get fresh function restored by viper
        # Actually just call directly from the module
        from main import require_env

        result = require_env("TEST_VAR_123")
        assert result == "hello"

    def test_strips_whitespace_from_value(self, isolated_env, monkeypatch):
        monkeypatch.setenv("TEST_VAR_456", "  spaced  ")
        from main import require_env

        result = require_env("TEST_VAR_456")
        assert result == "spaced"

    def test_raises_system_exit_when_env_var_missing(self, isolated_env):
        os.environ.pop("NONEXISTENT_VAR_789", None)
        from main import require_env

        with pytest.raises(SystemExit) as exc_info:
            require_env("NONEXISTENT_VAR_789")
        assert exc_info.value.code == 1

    def test_raises_system_exit_when_env_var_is_empty_string(self, isolated_env, monkeypatch):
        monkeypatch.setenv("EMPTY_VAR", "")
        from main import require_env

        with pytest.raises(SystemExit) as exc_info:
            require_env("EMPTY_VAR")
        assert exc_info.value.code == 1

    def test_raises_system_exit_when_env_var_is_whitespace_only(self, isolated_env, monkeypatch):
        monkeypatch.setenv("WHITESPACE_VAR", "   ")
        from main import require_env

        with pytest.raises(SystemExit) as exc_info:
            require_env("WHITESPACE_VAR")
        assert exc_info.value.code == 1

    def test_logs_fatal_message_on_missing_var(self, isolated_env, monkeypatch):
        os.environ.pop("MISSING_LOG_VAR_123", None)
        import main as main_mod
        logger = main_mod.logger

        mock_logger = MagicMock()

        # Patch the logger attribute on the module
        with patch.object(main_mod, "logger", mock_logger):
            with pytest.raises(SystemExit):
                main_mod.require_env("MISSING_LOG_VAR_123")
            # Check the call was made
            mock_logger.error.assert_called()
            call_args = str(mock_logger.error.call_args)
            assert "FATAL" in call_args
            assert "MISSING_LOG_VAR_123" in call_args


class TestAdapterLoading:
    """Tests for adapter loading logic in main() function."""

    def test_opencode_adapters_always_present(self, isolated_env, monkeypatch):
        """OpenCode adapters dict always has opencode0, opencode1, opencode2 keys."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("GRPC_PORT", "50051")
        monkeypatch.setenv("HEALTH_PORT", "8084")

        with patch("main.serve_grpc"), \
             patch("main.serve_health"), \
             patch("main.configure_health_handler"), \
             patch("main.signal.signal"), \
             patch("main.threading.Event"):
            main_mod = importlib.import_module("main")
            importlib.reload(main_mod)

            from main import OpenCodeLLMAdapter
            # Verify the mapping keys in the local variable
            # We read it indirectly: re-run and capture adapter names via configure_health_handler
            with patch.object(main_mod, "OllamaEmbeddingProvider", return_value=MagicMock(model="test-model")), \
                 patch.object(main_mod, "OllamaLLMAdapter", return_value=MagicMock()), \
                 patch.object(main_mod, "OpenCodeLLMAdapter", return_value=MagicMock()), \
                 patch.object(main_mod, "MistralLLMAdapter", return_value=MagicMock()), \
                 patch("main.HelperAgent") as mock_agent, \
                 patch("main.serve_grpc") as mock_grpc, \
                 patch("main.serve_health") as mock_health, \
                 patch("main.configure_health_handler") as mock_health_cfg, \
                 patch("main.signal.signal"), \
                 patch("main.threading.Event") as mock_event:
                mock_grpc.return_value = MagicMock()
                mock_health.return_value = MagicMock()
                mock_event.return_value.wait = MagicMock(side_effect=SystemExit(0))

                try:
                    main_mod.main()
                except SystemExit:
                    pass

                call_kwargs = mock_health_cfg.call_args
                adapter_names = call_kwargs[1].get("adapter_names", [])
                for expected in ("opencode0", "opencode1", "opencode2"):
                    assert expected in adapter_names, f"Expected {expected!r} in adapter_names, got {adapter_names}"

    def test_mistral_adapter_skipped_when_key_missing(self, isolated_env, monkeypatch):
        """Mistral adapter should be skipped when MISTRAL_API_KEY is not set."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("GRPC_PORT", "50051")
        monkeypatch.setenv("HEALTH_PORT", "8084")
        # MISTRAL_API_KEY intentionally not set

        main_mod = importlib.import_module("main")
        importlib.reload(main_mod)

        with patch.object(main_mod, "OllamaEmbeddingProvider", return_value=MagicMock(model="test-model")), \
             patch.object(main_mod, "OpenCodeLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "OllamaLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "MistralLLMAdapter", return_value=MagicMock()), \
             patch("main.HelperAgent") as mock_agent, \
             patch("main.serve_grpc") as mock_grpc, \
             patch("main.serve_health") as mock_health, \
             patch("main.configure_health_handler") as mock_health_cfg, \
             patch("main.signal.signal"), \
             patch("main.threading.Event") as mock_event:
            mock_grpc.return_value = MagicMock()
            mock_health.return_value = MagicMock()
            mock_event.return_value.wait = MagicMock(side_effect=SystemExit(0))

            try:
                main_mod.main()
            except SystemExit:
                pass

            call_kwargs = mock_health_cfg.call_args
            adapter_names = call_kwargs[1].get("adapter_names", [])
            assert "mistral" not in adapter_names

    def test_mistral_adapter_included_when_key_set(self, isolated_env, monkeypatch):
        """Mistral adapter should be loaded when MISTRAL_API_KEY is set."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("GRPC_PORT", "50051")
        monkeypatch.setenv("HEALTH_PORT", "8084")
        monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key-123")

        main_mod = importlib.import_module("main")
        importlib.reload(main_mod)

        with patch.object(main_mod, "OllamaEmbeddingProvider", return_value=MagicMock(model="test-model")), \
             patch.object(main_mod, "OpenCodeLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "OllamaLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "MistralLLMAdapter", return_value=MagicMock()), \
             patch("main.HelperAgent") as mock_agent, \
             patch("main.serve_grpc") as mock_grpc, \
             patch("main.serve_health") as mock_health, \
             patch("main.configure_health_handler") as mock_health_cfg, \
             patch("main.signal.signal"), \
             patch("main.threading.Event") as mock_event:
            mock_grpc.return_value = MagicMock()
            mock_health.return_value = MagicMock()
            mock_event.return_value.wait = MagicMock(side_effect=SystemExit(0))

            try:
                main_mod.main()
            except SystemExit:
                pass

            call_kwargs = mock_health_cfg.call_args
            adapter_names = call_kwargs[1].get("adapter_names", [])
            assert "mistral" in adapter_names

    def test_ollama_always_in_adapter_names(self, isolated_env, monkeypatch):
        """Ollama adapter should always appear in adapter_names."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("GRPC_PORT", "50051")
        monkeypatch.setenv("HEALTH_PORT", "8084")

        main_mod = importlib.import_module("main")
        importlib.reload(main_mod)

        with patch.object(main_mod, "OllamaEmbeddingProvider", return_value=MagicMock(model="test-model")), \
             patch.object(main_mod, "OpenCodeLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "OllamaLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "MistralLLMAdapter", return_value=MagicMock()), \
             patch("main.HelperAgent") as mock_agent, \
             patch("main.serve_grpc") as mock_grpc, \
             patch("main.serve_health") as mock_health, \
             patch("main.configure_health_handler") as mock_health_cfg, \
             patch("main.signal.signal"), \
             patch("main.threading.Event") as mock_event:
            mock_grpc.return_value = MagicMock()
            mock_health.return_value = MagicMock()
            mock_event.return_value.wait = MagicMock(side_effect=SystemExit(0))

            try:
                main_mod.main()
            except SystemExit:
                pass

            call_kwargs = mock_health_cfg.call_args
            adapter_names = call_kwargs[1].get("adapter_names", [])
            assert "ollama" in adapter_names

    def test_embedding_always_in_adapter_names(self, isolated_env, monkeypatch):
        """Embedding provider should always appear in adapter_names."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("GRPC_PORT", "50051")
        monkeypatch.setenv("HEALTH_PORT", "8084")

        main_mod = importlib.import_module("main")
        importlib.reload(main_mod)

        with patch.object(main_mod, "OllamaEmbeddingProvider", return_value=MagicMock(model="test-model")), \
             patch.object(main_mod, "OpenCodeLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "OllamaLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "MistralLLMAdapter", return_value=MagicMock()), \
             patch("main.HelperAgent") as mock_agent, \
             patch("main.serve_grpc") as mock_grpc, \
             patch("main.serve_health") as mock_health, \
             patch("main.configure_health_handler") as mock_health_cfg, \
             patch("main.signal.signal"), \
             patch("main.threading.Event") as mock_event:
            mock_grpc.return_value = MagicMock()
            mock_health.return_value = MagicMock()
            mock_event.return_value.wait = MagicMock(side_effect=SystemExit(0))

            try:
                main_mod.main()
            except SystemExit:
                pass

            call_kwargs = mock_health_cfg.call_args
            adapter_names = call_kwargs[1].get("adapter_names", [])
            assert "embedding" in adapter_names

    def test_main_exits_on_missing_required_env(self, isolated_env):
        """main() should exit when required env vars are missing."""
        for key in ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "GRPC_PORT", "HEALTH_PORT"]:
            os.environ.pop(key, None)

        # Cache a fresh main module
        main_mod = importlib.import_module("main")
        importlib.reload(main_mod)

        with pytest.raises(SystemExit) as exc_info:
            main_mod.main()
        assert exc_info.value.code == 1

    def test_grpc_port_parsed_from_env(self, isolated_env, monkeypatch):
        """GRPC_PORT should be parsed as int from env."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("GRPC_PORT", "9999")
        monkeypatch.setenv("HEALTH_PORT", "8085")

        main_mod = importlib.import_module("main")
        importlib.reload(main_mod)

        with patch.object(main_mod, "OllamaEmbeddingProvider", return_value=MagicMock(model="test-model")), \
             patch.object(main_mod, "OpenCodeLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "OllamaLLMAdapter", return_value=MagicMock()), \
             patch.object(main_mod, "MistralLLMAdapter", return_value=MagicMock()), \
             patch("main.HelperAgent") as mock_agent, \
             patch("main.serve_grpc") as mock_grpc, \
             patch("main.serve_health") as mock_health, \
             patch("main.configure_health_handler"), \
             patch("main.signal.signal"), \
             patch("main.threading.Event") as mock_event:
            mock_grpc.return_value = MagicMock()
            mock_health.return_value = MagicMock()
            mock_event.return_value.wait = MagicMock(side_effect=SystemExit(0))

            try:
                main_mod.main()
            except SystemExit:
                pass

            call_kwargs = mock_grpc.call_args
            assert call_kwargs[1].get("port") == 9999
