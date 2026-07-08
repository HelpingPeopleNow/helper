"""Tests for the Ollama embedding provider (httpx)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from internal.adapters.embedding_provider import (
    EXPECTED_DIMENSIONS,
    DimensionMismatchError,
    OllamaEmbeddingProvider,
    sha256_hex,
)

_FAKE_VEC = [0.1] * EXPECTED_DIMENSIONS


class TestConstructor:
    def test_defaults_from_env(self, isolated_env) -> None:
        os.environ["OLLAMA_BASE_URL"] = "http://env:11434"
        os.environ["EMBEDDING_MODEL"] = "env-model"
        p = OllamaEmbeddingProvider()
        assert p._base_url == "http://env:11434"
        assert p._model == "env-model"

    def test_explicit_overrides_env(self, isolated_env) -> None:
        os.environ["OLLAMA_BASE_URL"] = "http://ignored:11434"
        os.environ["EMBEDDING_MODEL"] = "ignored"
        p = OllamaEmbeddingProvider(
            base_url="http://explicit:11434",
            model="explicit-model",
        )
        assert p._base_url == "http://explicit:11434"
        assert p._model == "explicit-model"

    def test_defaults_when_env_unset(self, isolated_env) -> None:
        p = OllamaEmbeddingProvider()
        assert p._base_url == "http://localhost:11434"
        assert p._model == "granite-embedding:278m"
        assert p._timeout_s == 30.0

    def test_model_property(self) -> None:
        p = OllamaEmbeddingProvider(model="test-model")
        assert p.model == "test-model"

    def test_base_url_strips_trailing_slash(self, isolated_env) -> None:
        p = OllamaEmbeddingProvider(base_url="http://local:11434/")
        assert p._base_url == "http://local:11434"


class TestEmbed:
    @respx.mock
    def test_success(self) -> None:
        respx.post("http://local:11434/api/embeddings").respond(
            200, json={"embedding": _FAKE_VEC}
        )
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        result = p.embed("hello world")
        assert result == _FAKE_VEC
        assert len(result) == EXPECTED_DIMENSIONS

    @respx.mock
    def test_dimension_mismatch_raises(self) -> None:
        bad_vec = [0.1, 0.2, 0.3]
        respx.post("http://local:11434/api/embeddings").respond(
            200, json={"embedding": bad_vec}
        )
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        with pytest.raises(DimensionMismatchError) as exc:
            p.embed("text")
        assert "768" in str(exc.value)

    @respx.mock
    def test_http_error_raises(self) -> None:
        respx.post("http://local:11434/api/embeddings").respond(500)
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            p.embed("text")

    @respx.mock
    def test_network_error_raises(self) -> None:
        respx.post("http://local:11434/api/embeddings").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        with pytest.raises(RuntimeError, match="Connection refused"):
            p.embed("text")

    @respx.mock
    def test_empty_vector_raises(self) -> None:
        respx.post("http://local:11434/api/embeddings").respond(
            200, json={"embedding": []}
        )
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        with pytest.raises(RuntimeError, match="empty vector"):
            p.embed("text")

    @respx.mock
    def test_missing_embedding_key_raises(self) -> None:
        respx.post("http://local:11434/api/embeddings").respond(200, json={})
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        with pytest.raises(RuntimeError, match="empty vector"):
            p.embed("text")


class TestEmbedBatch:
    @respx.mock
    def test_sequential_batch(self) -> None:
        route = respx.post("http://local:11434/api/embeddings").respond(
            200, json={"embedding": _FAKE_VEC}
        )
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        results = p.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(r == _FAKE_VEC for r in results)
        assert route.call_count == 3

    @respx.mock
    def test_batch_stops_on_first_error(self) -> None:
        respx.post("http://local:11434/api/embeddings").respond(503)
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        with pytest.raises(RuntimeError, match="HTTP 503"):
            p.embed_batch(["a", "b", "c"])

    @respx.mock
    def test_empty_batch(self) -> None:
        p = OllamaEmbeddingProvider(base_url="http://local:11434")
        results = p.embed_batch([])
        assert results == []


class TestHealth:
    @staticmethod
    def _mock_get(json_data: dict, status_code: int = 200) -> MagicMock:
        """Build a mock httpx response for GET /api/tags."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json.return_value = json_data
        return resp

    def test_ok_when_model_available(self) -> None:
        with patch("internal.adapters.embedding_provider.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = self._mock_get(
                {"models": [{"name": "granite-embedding:278m"}]}
            )
            p = OllamaEmbeddingProvider(
                base_url="http://local:11434", model="granite-embedding:278m"
            )
            status, detail = p.health()
        assert status == "ok"
        assert "available" in detail

    def test_ok_with_latest_suffix(self) -> None:
        with patch("internal.adapters.embedding_provider.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = self._mock_get(
                {"models": [{"name": "granite-embedding:278m:latest"}]}
            )
            p = OllamaEmbeddingProvider(
                base_url="http://local:11434", model="granite-embedding:278m"
            )
            status, detail = p.health()
        assert status == "ok"

    def test_down_when_model_not_pulled(self) -> None:
        with patch("internal.adapters.embedding_provider.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = self._mock_get(
                {"models": [{"name": "other-model"}]}
            )
            p = OllamaEmbeddingProvider(
                base_url="http://local:11434", model="granite-embedding:278m"
            )
            status, detail = p.health()
        assert status == "down"
        assert "not pulled" in detail

    def test_down_when_tags_http_error(self) -> None:
        with patch("internal.adapters.embedding_provider.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = self._mock_get(
                {"models": []}, status_code=500
            )
            p = OllamaEmbeddingProvider(
                base_url="http://local:11434", model="granite-embedding:278m"
            )
            status, detail = p.health()
        assert status == "down"
        assert "500" in detail

    def test_down_on_network_error(self) -> None:
        with patch("internal.adapters.embedding_provider.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("refused")
            p = OllamaEmbeddingProvider(
                base_url="http://local:11434", model="granite-embedding:278m"
            )
            status, detail = p.health()
        assert status == "down"
        assert "refused" in detail

    def test_down_when_ollama_unreachable(self, isolated_env) -> None:
        """Real HTTP call to a non-existent server — verifies the error fallback."""
        p = OllamaEmbeddingProvider(base_url="http://127.0.0.1:19999")
        status, detail = p.health()
        assert status == "down"


class TestSha256Hex:
    def test_known_hash(self) -> None:
        result = sha256_hex("hello")
        assert result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_empty_string(self) -> None:
        result = sha256_hex("")
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_unicode(self) -> None:
        result = sha256_hex("héllo")
        assert isinstance(result, str)
        assert len(result) == 64
