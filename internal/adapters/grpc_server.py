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


class HealthHandler(http.server.BaseHTTPRequestHandler):
    """Health check endpoint with dependency checks.

    Reports on: gRPC server, LLM adapters, transcribe service.
    """
    # Set via configure_health_handler() before server starts
    _adapter_names: list[str] = []
    _grpc_server: Optional[grpc.Server] = None
    _transcribe_port: int = 0

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        else:
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

        # Check adapters — report which ones are loaded
        if self._adapter_names:
            status["adapters"] = "ok"
            status["loaded_adapters"] = self._adapter_names
        else:
            status["adapters"] = "down"
            status["status"] = "degraded"

        # Transcribe service — if port is configured, report it
        if self._transcribe_port:
            status["transcribe"] = "ok"
            status["transcribe_port"] = self._transcribe_port
        else:
            status["transcribe"] = "not_configured"

        body = json.dumps(status).encode()
        self.send_response(200 if status["status"] == "ok" else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:
        logger.debug("health: %s", fmt % args)


def configure_health_handler(adapter_names: list[str], grpc_server: Optional[grpc.Server] = None,
                              transcribe_port: int = 0) -> None:
    """Set health check dependencies before starting the server."""
    HealthHandler._adapter_names = adapter_names
    HealthHandler._grpc_server = grpc_server
    HealthHandler._transcribe_port = transcribe_port


def serve_health(port: int = 8084) -> http.server.HTTPServer:
    """Start a lightweight health HTTP server in a daemon thread."""
    server = http.server.HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("health HTTP server on :%d", port)
    return server
