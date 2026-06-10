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
    """Minimal health check endpoint on a separate HTTP port."""
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args) -> None:
        logger.debug("health: %s", fmt % args)


def serve_health(port: int = 8084) -> http.server.HTTPServer:
    """Start a lightweight health HTTP server in a daemon thread."""
    server = http.server.HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("health HTTP server on :%d", port)
    return server
