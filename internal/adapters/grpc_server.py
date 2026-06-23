"""
gRPC server adapter for the helper service.

Runs alongside FastAPI on a separate port (50051).
"""
import http.server
import json
import logging
import threading
import time
from concurrent import futures
from typing import Callable, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import grpc
from grpc import ServicerContext
from internal.adapters.embedding_provider import (
    EmbeddingProvider,
    DimensionMismatchError,
)
from internal.adapters.metrics import (
    active_requests,
    classify_error,
    estimate_tokens,
    grpc_request_duration_seconds,
    grpc_requests_total,
    health_check_total,
    llm_errors_total,
    llm_request_duration_seconds,
    llm_requests_total,
    llm_tokens_total,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from internal.core.helper_agent import Answer, HelperAgent, Question
from internal.ports.llm import Message
from proto import helper_pb2, helper_pb2_grpc

logger = logging.getLogger(__name__)


class HelperServicer(helper_pb2_grpc.HelperServiceServicer):
    """Implements the gRPC HelperService protocol."""

    def __init__(
        self,
        assistant: HelperAgent,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._assistant = assistant
        self._embedding = embedding_provider

    def Ask(self, request: helper_pb2.AskRequest, context: ServicerContext) -> helper_pb2.AskResponse:
        start = time.monotonic()
        history_len = len(request.history)
        system_prompt = request.system_prompt
        llm_provider = request.llm_provider
        skip_role_detection = request.skip_role_detection
        logger.info("gRPC Ask: q_len=%d history=%d sp_len=%d provider=%s skip_role=%s",
                     len(request.question), history_len, len(system_prompt), llm_provider or "(env default)", skip_role_detection)

        try:
            history = tuple(
                Message(role=m.role, content=m.content)
                for m in request.history
            )
            result: Answer = self._assistant.answer(
                Question(text=request.question),
                system_prompt=system_prompt,
                history=history,
                llm_provider=llm_provider,
                skip_role_detection=skip_role_detection,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info("gRPC Ask done: answer_len=%d role=%s elapsed_ms=%.0f", len(result.text), result.detected_role or "none", elapsed_ms)
            grpc_requests_total.labels(method="Ask", status="ok").inc()
            grpc_request_duration_seconds.labels(method="Ask").observe(time.monotonic() - start)
            return helper_pb2.AskResponse(answer=result.text, detected_role=result.detected_role)
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("gRPC Ask failed after %.0fms", elapsed_ms)
            grpc_requests_total.labels(method="Ask", status="error").inc()
            grpc_request_duration_seconds.labels(method="Ask").observe(time.monotonic() - start)
            raise

    # ── Embedding RPCs (VECTOR_SEARCH_PLAN §7.3) ──────────────────

    def Embed(
        self,
        request: helper_pb2.EmbedRequest,
        context: ServicerContext,
    ) -> helper_pb2.EmbedResponse:
        start = time.monotonic()
        text_len = len(request.text)
        requested_model = request.model or ""
        logger.info("gRPC Embed: text_len=%d model=%s", text_len, requested_model or "(default)")

        if self._embedding is None:
            grpc_requests_total.labels(method="Embed", status="error").inc()
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("embedding provider not configured")
            raise RuntimeError("embedding provider not configured")

        try:
            vec = self._embedding.embed(request.text)
            used_model = requested_model or self._embedding.model
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info("gRPC Embed done: dim=%d model=%s elapsed_ms=%.0f",
                        len(vec), used_model, elapsed_ms)
            grpc_requests_total.labels(method="Embed", status="ok").inc()
            grpc_request_duration_seconds.labels(method="Embed").observe(time.monotonic() - start)
            return helper_pb2.EmbedResponse(
                embedding=vec, model=used_model, dimensions=len(vec),
            )
        except DimensionMismatchError as exc:
            logger.error("gRPC Embed dim mismatch: %s", exc)
            grpc_requests_total.labels(method="Embed", status="dim_mismatch").inc()
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            raise
        except Exception:
            logger.exception("gRPC Embed failed")
            grpc_requests_total.labels(method="Embed", status="error").inc()
            raise

    def EmbedBatch(
        self,
        request: helper_pb2.EmbedBatchRequest,
        context: ServicerContext,
    ) -> helper_pb2.EmbedBatchResponse:
        start = time.monotonic()
        n_texts = len(request.texts)
        requested_model = request.model or ""
        logger.info("gRPC EmbedBatch: n=%d model=%s", n_texts, requested_model or "(default)")

        if self._embedding is None:
            grpc_requests_total.labels(method="EmbedBatch", status="error").inc()
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("embedding provider not configured")
            raise RuntimeError("embedding provider not configured")

        try:
            vectors = self._embedding.embed_batch(list(request.texts))
            used_model = requested_model or self._embedding.model
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info("gRPC EmbedBatch done: n=%d model=%s elapsed_ms=%.0f",
                        n_texts, used_model, elapsed_ms)
            grpc_requests_total.labels(method="EmbedBatch", status="ok").inc()
            grpc_request_duration_seconds.labels(method="EmbedBatch").observe(
                time.monotonic() - start,
            )
            return helper_pb2.EmbedBatchResponse(
                embeddings=[
                    helper_pb2.EmbedResponse(
                        embedding=v, model=used_model, dimensions=len(v),
                    )
                    for v in vectors
                ]
            )
        except DimensionMismatchError as exc:
            logger.error("gRPC EmbedBatch dim mismatch: %s", exc)
            grpc_requests_total.labels(method="EmbedBatch", status="dim_mismatch").inc()
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            raise
        except Exception:
            logger.exception("gRPC EmbedBatch failed")
            grpc_requests_total.labels(method="EmbedBatch", status="error").inc()
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
        health_check_total.labels(target="openai_compat", status="fail").inc()
        return "down", "missing api_key"
    try:
        import requests
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            health_check_total.labels(target="openai_compat", status="ok").inc()
            return "ok", f"HTTP {resp.status_code}"
        health_check_total.labels(target="openai_compat", status="fail").inc()
        return "down", f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as exc:
        logger.warning("health check openai_compat url=%s error=%s", base_url, exc)
        health_check_total.labels(target="openai_compat", status="fail").inc()
        return "down", str(exc)


def _check_ollama(base_url: str, model: str, timeout: int = 5) -> tuple[str, str]:
    """Check if Ollama is reachable and has the model."""
    if not base_url:
        health_check_total.labels(target="ollama", status="fail").inc()
        return "down", "missing base_url"
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            health_check_total.labels(target="ollama", status="fail").inc()
            return "down", f"HTTP {resp.status_code}: {resp.text[:100]}"
        data = resp.json()
        models = data.get("models", [])
        if any(m.get("name") == model for m in models):
            health_check_total.labels(target="ollama", status="ok").inc()
            return "ok", f"model {model} found"
        health_check_total.labels(target="ollama", status="fail").inc()
        return "down", f"model {model} not found"
    except Exception as exc:
        logger.warning("health check ollama url=%s error=%s", base_url, exc)
        health_check_total.labels(target="ollama", status="fail").inc()
        return "down", str(exc)


def _check_ollama_embedding(base_url: str, model: str, timeout: int = 5) -> tuple[str, str]:
    """Embedding-specific Ollama check (VECTOR_SEARCH_PLAN §7.6 / §4.4).

    Verifies that the embedding model is pulled. Optional — like
    the LLM Ollama check, doesn't block critical status.
    """
    if not base_url:
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        return "skipped", "no OLLAMA_BASE_URL configured"
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            health_check_total.labels(target="ollama_embed", status="fail").inc()
            return "down", f"http {resp.status_code}"
        tags = resp.json().get("models", [])
        names = {m.get("name") for m in tags}
        if any(n == model or n == f"{model}:latest" for n in names):
            health_check_total.labels(target="ollama_embed", status="ok").inc()
            return "ok", f"model {model} available"
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        return "down", f"model {model} not pulled yet — run ollama pull {model}"
    except Exception as exc:
        logger.warning("health check ollama_embed url=%s error=%s", base_url, exc)
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        return "down", str(exc)


class HealthHandler(http.server.BaseHTTPRequestHandler):
    """Health check endpoint with dependency checks.

    Reports on: gRPC server, LLM adapters, embedding provider.
    """
    _adapter_names: list[str] = []
    _grpc_server: Optional[grpc.Server] = None
    _adapter_details: dict[str, dict[str, str]] = {}

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            logger.warning("health unknown_path=%s", self.path)
            self.send_response(404)
            self.end_headers()
    def _handle_metrics(self) -> None:
        """Return Prometheus metrics in text format."""
        body = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_health(self) -> None:
        status = {"status": "ok", "grpc": "ok", "adapters": "ok"}

        if self._grpc_server is not None:
            status["grpc"] = "ok"
        else:
            status["grpc"] = "unknown"

        adapter_results: dict[str, str] = {}
        adapter_details: dict[str, str] = {}
        for name, details in self._adapter_details.items():
            # Skip Ollama — it's optional/local-only and shouldn't block health.
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

        # Embedding provider (VECTOR_SEARCH_PLAN §7.6)
        embed_status, embed_detail = _check_ollama_embedding(
            self._adapter_details.get("embedding", {}).get("base_url", ""),
            self._adapter_details.get("embedding", {}).get("model", ""),
        )
        # "skipped" / "down" both surface but don't downgrade status to degraded
        # — embedding is optional for the chat path. JSON response includes the
        # raw status so operators can see.
        adapter_results["embedding"] = embed_status
        adapter_details["embedding"] = embed_detail

        status["adapters"] = "ok" if all(v == "ok" for v in adapter_results.values() if v not in ("skipped", "down")) else "degraded"
        status["adapter_results"] = adapter_results
        status["adapter_details"] = adapter_details
        status["loaded_adapters"] = self._adapter_names

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


def serve_grpc(
    assistant: HelperAgent,
    embedding_provider: Optional[EmbeddingProvider] = None,
    port: int = 50051,
) -> grpc.Server:
    """Start the gRPC server in a background thread."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helper_pb2_grpc.add_HelperServiceServicer_to_server(
        HelperServicer(assistant, embedding_provider=embedding_provider), server,
    )
    bound = server.add_insecure_port(f"[::]:{port}")
    logger.info("gRPC server bound on :%d (port_result=%d)", port, bound)
    server.start()
    logger.info("gRPC server listening on :%d", port)
    return server


def configure_health_handler(
    adapter_names: list[str],
    grpc_server: Optional[grpc.Server] = None,
    adapter_details: Optional[dict[str, dict[str, str]]] = None,
) -> None:
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
