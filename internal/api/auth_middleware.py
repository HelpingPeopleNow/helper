"""Session validation middleware/dependency.

Intercepts requests and validates the `better-auth-session` cookie
against the auth service at http://auth:8083/api/auth/session.
Returns 401 if the session is invalid.
Skips validation for the /health endpoint.
"""
import logging

import httpx
from fastapi import Cookie, HTTPException, Request
from httpx import AsyncClient

logger = logging.getLogger(__name__)

AUTH_SESSION_URL = "http://auth:8083/api/auth/session"

_session_client: AsyncClient | None = None


async def get_session_client() -> AsyncClient:
    global _session_client
    if _session_client is None:
        _session_client = AsyncClient(timeout=httpx.Timeout(5.0))
    return _session_client


async def validate_session(
    request: Request,
    better_auth_session: str | None = Cookie(default=None),
) -> None:
    """FastAPI dependency: validates the session cookie against the auth service.

    Attach this as a dependency to any route that needs session protection.
    The /health endpoint is excluded from validation.
    """
    # Skip validation for health endpoint
    if request.url.path == "/health":
        return

    if not better_auth_session:
        logger.warning("missing better-auth-session cookie")
        raise HTTPException(status_code=401, detail="Unauthorized: no session cookie")

    client = await get_session_client()
    try:
        logger.debug("validating session %s...", better_auth_session[:10])
        resp = await client.get(
            AUTH_SESSION_URL,
            cookies={"better-auth-session": better_auth_session},
        )
        if resp.status_code != 200:
            logger.warning(
                "session invalid: auth returned %d", resp.status_code,
            )
            raise HTTPException(status_code=401, detail="Unauthorized: invalid session")

        logger.info("session validated successfully")
    except httpx.RequestError as exc:
        logger.error("auth service unreachable: %s", exc)
        # Fail closed — return 401 if auth is unreachable
        raise HTTPException(status_code=401, detail="Unauthorized: auth service unavailable") from exc
