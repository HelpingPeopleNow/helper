"""
gRPC server adapter for the helper service.

Runs alongside FastAPI on a separate port (50051).
"""
import http.server
import json
import logging
import os
import threading
import time
from concurrent import futures
from typing import Callable, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import grpc
from grpc import ServicerContext

from internal.core.helper_agent import Answer, HelperAgent, Question
from internal.ports.llm import Message
from proto import helper_pb2, helper_pb2_grpc

logger = logging.getLogger(__name__)


class HelperServicer(helper_pb2_grpc.HelperServiceServicer):
    """Implements the gRPC HelperService protocol."""

    def __init__(self, assistant: HelperAgent) -> None:
        self._assistant = assistant

    def Ask(self, request: helper_pb2.AskRequest, context: ServicerContext) -> helper_pb2.AskResponse:
        start = time.monotonic()
        history_len = len(request.history)
        system_prompt = request.system_prompt  # received from backend
        llm_provider = request.llm_provider  # "ollama" | "opencode" | ""
        skip_role_detection = request.skip_role_detection
        logger.info("gRPC Ask: q_len=%d history=%d sp_len=%d provider=%s skip_role=%s",
                     len(request.question), history_len, len(system_prompt), llm_provider or "(env default)", skip_role_detection)
        if logger.isEnabledFor(logging.DEBUG) and system_prompt:
            logger.debug("gRPC system_prompt[:150]: %s", system_prompt[:150])

        try:
            history = tuple(
                Message(role=m.role, content=m.content)
                for m in request.history
            )
            result: Answer = self._assistant.answer(
                Question(text=request.question),
                system_prompt=system_prompt,  # passed through from backend
                history=history,
                llm_provider=llm_provider,  # "ollama" | "opencode" | "" (falls back to env default)
                skip_role_detection=skip_role_detection,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info("gRPC Ask done: answer_len=%d role=%s elapsed_ms=%.0f", len(result.text), result.detected_role or "none", elapsed_ms)
            return helper_pb2.AskResponse(answer=result.text, detected_role=result.detected_role)
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("gRPC Ask failed after %.0fms", elapsed_ms)
            raise


def _check_http_url(url: str, timeout: int = 3) -> tuple[str, str]:
    """Check if a URL is reachable. Returns (status, detail)."""
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return "ok", f"HTTP {resp.status}"
    except Exception as exc:
        logger.warning("health check http_url=%s error=%s", url, exc)
        return "down", str(exc)


def _check_openai_compat(base_url: str, api_key: str, model: str, timeout: int = 5) -> tuple[str, str]:
    """Lightweight check for OpenAI-compatible API endpoints."""
    if not api_key:
        return "down", "missing api_key"
    try:
        import requests
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return "ok", f"HTTP {resp.status_code}"
        return "down", f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as exc:
        logger.warning("health check openai_compat url=%s error=%s", base_url, exc)
        return "down", str(exc)


def _check_ollama(base_url: str, model: str, timeout: int = 5) -> tuple[str, str]:
    """Check if Ollama is reachable and has the model."""
    if not base_url:
        return "down", "missing base_url"
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return "down", f"HTTP {resp.status_code}: {resp.text[:100]}"
        data = resp.json()
        models = data.get("models", [])
        if any(m.get("name") == model for m in models):
            return "ok", f"model {model} found"
        return "down", f"model {model} not found"
    except Exception as exc:
        logger.warning("health check ollama url=%s error=%s", base_url, exc)
        return "down", str(exc)


class HealthHandler(http.server.BaseHTTPRequestHandler):
    """Health check endpoint with dependency checks.

    Reports on: gRPC server, LLM adapters.
    """
    # Set via configure_health_handler() before server starts
    _adapter_names: list[str] = []
    _grpc_server: Optional[grpc.Server] = None
    _adapter_details: dict[str, dict[str, str]] = {}

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        else:
            logger.warning("health unknown_path=%s", self.path)
            self.send_response(404)
            self.end_headers()

    def _handle_health(self) -> None:
        status = {"status": "ok", "grpc": "ok", "adapters": "ok"}

        # Check gRPC server — if this handler is serving, the process is alive.
        # The gRPC server object can tell us if it's been stopped.
        if self._grpc_server is not None:
            # grpc.Server doesn't expose a clean "is_running" check,
            # but if we got here the process is alive and the server started.
            status["grpc"] = "ok"
        else:
            status["grpc"] = "unknown"

        # Check adapters — report which ones are loaded and their connectivity
        adapter_results: dict[str, str] = {}
        adapter_details: dict[str, str] = {}
        for name, details in self._adapter_details.items():
            # Skip Ollama — it's optional/local-only and shouldn't block health
            if details.get("kind") == "ollama":
                adapter_results[name] = "skipped"
                adapter_details[name] = "optional/local"
                continue
            kind = details.get("kind", "")
            if kind == "openai_compat":
                check_status, check_detail = _check_openai_compat(
                    details.get("base_url", ""),
                    details.get("api_key", ""),
                    details.get("model", ""),
                )
            elif kind == "ollama":
                check_status, check_detail = _check_ollama(
                    details.get("base_url", ""),
                    details.get("model", ""),
                )
            else:
                check_status, check_detail = "unknown", "unknown adapter kind"
            adapter_results[name] = check_status
            adapter_details[name] = check_detail

        status["adapters"] = "ok" if all(v == "ok" for v in adapter_results.values() if v != "skipped") else "degraded"
        status["adapter_results"] = adapter_results
        status["adapter_details"] = adapter_details
        status["loaded_adapters"] = self._adapter_names

        # Determine overall status:
        # - 503 only if critical deps are down (grpc, no adapters)
        # - 200 if service is usable but optional deps are degraded
        has_any_healthy_adapter = any(v == "ok" for v in adapter_results.values())
        critical_ok = status["grpc"] == "ok" and has_any_healthy_adapter

        if not critical_ok:
            status["status"] = "degraded"

        body = json.dumps(status).encode()
        self.send_response(200 if status["status"] == "ok" else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:
        logger.debug("health: %s", fmt % args)


def serve_grpc(assistant: HelperAgent, port: int = 50051) -> grpc.Server:
    """Start the gRPC server in a background thread."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helper_pb2_grpc.add_HelperServiceServicer_to_server(
        HelperServicer(assistant), server,
    )
    bound = server.add_insecure_port(f"[::]:{port}")
    logger.info("gRPC server bound on :%d (port_result=%d)", port, bound)
    server.start()
    logger.info("gRPC server listening on :%d", port)
    return server


def configure_health_handler(adapter_names: list[str], grpc_server: Optional[grpc.Server] = None,
                              adapter_details: Optional[dict[str, dict[str, str]]] = None) -> None:
    """Set health check dependencies before starting the server."""
    HealthHandler._adapter_names = adapter_names
    HealthHandler._grpc_server = grpc_server
    HealthHandler._adapter_details = adapter_details or {}


def serve_health(port: int = 8084) -> http.server.HTTPServer:
    """Start a lightweight health HTTP server in a daemon thread."""
    server = http.server.HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("health HTTP server on :%d", port)
    return server
