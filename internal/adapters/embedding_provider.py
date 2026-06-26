"""
Embedding provider for the helper service.

Implements the EmbeddingProvider port (see internal/ports/embedding.py if/when
extracted) as an Ollama adapter. Mirrors the LLM-adapter hexagonal pattern
already used by OpenCode/Mistral/Ollama LLM adapters.

Reference: VECTOR_SEARCH_PLAN §7.2, §7.3, §7.5.
"""
import hashlib
import logging
import os
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

# Default embedding model — overridable via EMBEDDING_MODEL env var.
# See plan §4.4 / §7.5. 278m params, 768 dims, multilingual.
DEFAULT_EMBEDDING_MODEL = "granite-embedding:278m"
EXPECTED_DIMENSIONS = 768


class DimensionMismatchError(Exception):
    """Embedding returned the wrong dimensionality. Caller should NOT persist
    this row — silently storing a mismatching vector corrupts cosine search."""


class EmbeddingProvider:
    """Port contract. Concrete adapter is OllamaEmbeddingProvider.

    Kept here as a simple class rather than an ABC to match the codebase's
    informal pattern (ollama_llm.py doesn't use ABC either).
    """

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError

    def health(self) -> tuple[str, str]:
        """Returns (status, detail). status in {"ok", "down", "skipped"}."""
        raise NotImplementedError


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama-backed embedding adapter.

    Same OLLAMA_BASE_URL env var the LLM adapters consume. Compatible with
    the `embeddings` API shape (POST /api/embeddings).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        # Resolve env at construction time so health() can report the
        # configured model even when the daemon is unreachable.
        self._base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self._model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._timeout_s = timeout_s
        logger.info(
            "ollama embedding provider: model=%s base_url=%s timeout_s=%s",
            self._model,
            self._base_url,
            self._timeout_s,
        )

    @property
    def model(self) -> str:
        return self._model

    def _post_embeddings(self, text: str) -> list[float]:
        """Single-text Ollama call. Returns a float list (Python floats —
        gRPC marshals to repeated float which maps to Go []float32)."""
        url = f"{self._base_url}/api/embeddings"
        payload = {"model": self._model, "prompt": text}
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.warning(
                        "ollama embed http error: status=%s body=%s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    raise RuntimeError(
                        f"Ollama embeddings returned HTTP {resp.status_code}"
                    )
                data = resp.json()
                embedding = data.get("embedding") or []
        except httpx.HTTPError as exc:
            logger.warning("ollama embed network error: %s", exc)
            raise RuntimeError(f"Ollama embeddings request failed: {exc}") from exc
        if not embedding:
            raise RuntimeError(
                "Ollama embeddings returned an empty vector (model not loaded?)"
            )
        return embedding

    def embed(self, text: str) -> list[float]:
        vec = self._post_embeddings(text)
        if len(vec) != EXPECTED_DIMENSIONS:
            raise DimensionMismatchError(
                f"Embedding returned dim={len(vec)}, expected "
                f"{EXPECTED_DIMENSIONS} for model={self._model}. "
                "Refusing to return — do not persist mismatched-dim vectors."
            )
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama doesn't expose a true batched /api/embeddings endpoint
        # without using ollama-python client; loop sequentially. NUM_PARALLEL=1
        # at the daemon-side means parallelism here buys nothing — keep
        # sequential to match daemon's load model.
        out: list[list[float]] = []
        for i, t in enumerate(texts):
            out.append(self.embed(t))
            if (i + 1) % 10 == 0:
                logger.info("ollama embed_batch progress: %d/%d", i + 1, len(texts))
        return out

    def health(self) -> tuple[str, str]:
        """Check that Ollama is reachable AND has the embedding model pulled."""
        if not self._base_url:
            return "down", "no OLLAMA_BASE_URL configured"
        try:
            with httpx.Client(timeout=3.0) as client:
                r = client.get(f"{self._base_url}/api/tags")
                if r.status_code != 200:
                    return "down", f"http {r.status_code}: {r.text[:100]}"
                tags = r.json().get("models", [])
                names = {m.get("name") for m in tags}
                # Ollama tags often return tagless names with ":latest" suffix
                # stripped. Match either prefix.
                if any(n == self._model or n == f"{self._model}:latest" for n in names):
                    return "ok", f"model {self._model} available"
                return "down", f"model {self._model} not pulled"
        except Exception as exc:
            logger.warning("ollama embedding health error: %s", exc)
            return "down", str(exc)


def sha256_hex(text: str) -> str:
    """Hex SHA-256 digests for the text_hash column. Same algorithm used by
    the backend's reembedWorker so hash-match skip works end-to-end."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
