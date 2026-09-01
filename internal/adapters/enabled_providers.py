"""Admin-selected LLM providers for deep-probe scoping.

The helper probes only providers enabled in the admin panel (same list
used for chat). Fetched from the backend internal endpoint; empty admin
selection falls back to FALLBACK_CHAIN.
"""
from __future__ import annotations

import json
import logging
import os
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def fallback_chain() -> list[str]:
    """Return the env-configured fallback chain (read at call time for tests)."""
    return [
        p.strip()
        for p in os.getenv(
            "FALLBACK_CHAIN",
            "groq,openrouter,opencode0,opencode1,opencode2,mistral,ollama",
        ).split(",")
        if p.strip()
    ]


class EnabledProvidersSource:
    """Fetches and caches the admin provider list from the backend."""

    def __init__(self, url: str = "") -> None:
        self._url = url.strip()
        self._last_good: list[str] | None = None

    def fetch(self) -> list[str] | None:
        """Return admin providers, [] for auto-chain, or None on fetch failure."""
        if not self._url:
            return None
        try:
            req = Request(self._url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            providers = data.get("providers")
            if not isinstance(providers, list):
                raise ValueError("providers must be a JSON array")
            cleaned = [str(p).strip() for p in providers if str(p).strip()]
            self._last_good = cleaned
            return cleaned
        except Exception as exc:
            logger.warning("enabled providers fetch failed: %s", exc)
            return self._last_good


def resolve_deep_probe_targets(
    adapters: dict[str, object],
    admin_providers: list[str] | None,
) -> list[str]:
    """Providers to deep-probe: admin list, or FALLBACK_CHAIN when unset/empty."""
    if admin_providers:
        chain = admin_providers
    else:
        chain = fallback_chain()
    return [name for name in chain if name in adapters]
