"""
gRPC server adapter for the helper service.

Runs alongside the HTTP health/metrics sidecar on a separate port (50051).

This module groups four audit-driven hardening passes that landed together
because they share the same file and the same request-lifecycle:

- P2-4 (JSON logging + trace_id): a gRPC `_TraceIdInterceptor` extracts
  `x-trace-id` / `traceparent` from request metadata and binds it to the
  `trace_id_var` ContextVar for the duration of the RPC handler. The
  `finally:` branch always calls `reset_trace_id(token)` so the variable
  cannot leak across requests handled by the same ThreadPoolExecutor
  worker (critical; without reset, pooled threads would carry the previous
  request's trace id into the next call's logs).
- The root logger has a `_TraceIdFilter` that reads `trace_id_var` on every
  record, plus a `JsonFormatter` that emits single-line JSON. Installed at
  import time so adapters' `logger.info(...)` calls benefit automatically.
- P2-3 (EmbedBatch partial failure): the response is now
  `repeated EmbedBatchItem` (see proto). The server iterates the provider's
  per-item results and translates each into the proto message — including
  failures — so a partial outage never voids the whole batch.
- P2-5 (strip upstream /health bodies): `resp.text[:100]` is moved from
  the public `/health` JSON into server-side logs only; the response now
  exposes only `{adapter, status_code, fingerprint}` so an unauthenticated
  observer can't scrape provider-specific error messages.
"""
import hashlib
import hmac
import http.server
import json
import logging
import os
import re
import threading
import time
from concurrent import futures
from typing import Optional
import grpc
from grpc import ServicerContext

# Install structured JSON logging + trace_id filter as the very first side
# effect of this module so every adapter (which imports logging at top of
# file) gets the JSON handler regardless of import order.
from internal.adapters.logging_config import (  # noqa: E402
    install as install_logging,
    reset_trace_id,
    set_trace_id,
)
install_logging(level=os.getenv("LOG_LEVEL", "INFO"))

from internal.adapters.embedding_provider import (  # noqa: E402
    EmbeddingProvider,
    DimensionMismatchError,
)
from internal.adapters.metrics import (  # noqa: E402
    active_requests,
    auth_errors_total,
    classify_error,
    grpc_request_duration_seconds,
    grpc_requests_total,
    health_check_total,
    llm_errors_total,
    llm_request_duration_seconds,
    llm_requests_total,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from internal.core.helper_agent import Answer, HelperAgent, Question  # noqa: E402
from internal.ports.llm import Message  # noqa: E402
from proto import helper_pb2, helper_pb2_grpc  # noqa: E402

# P2-3 server-side defence: providers SHOULD validate embedding dims, but
# bad rows could still slip through (model evicted mid-batch, future adapter
# bug). We refuse to forward an empty / wrong-dim "success" item — it would
# otherwise silently upsert a zero or wrong-dim vector into worker_embeddings
# and corrupt cosine search. We downgrade to dim_mismatch on size mismatch.
EXPECTED_EMBEDDING_DIM = 768

logger = logging.getLogger(__name__)

# Max question length in characters (R8): reject oversized prompts
# before forwarding to any LLM adapter (cost/memory protection).
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "32000"))

# Maximum input bytes for gRPC messages (R3).
MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))


# ── Trace ID extraction (P2-4) ──────────────────────────────────────
#
# Header priority order (loose de-facto standard):
#   1. `x-trace-id`           — common in microservices, short hex/uuid
#   2. `x-request-id`         — fall-back used by some proxies
#   3. `traceparent` (W3C)    — last because parsing it for just the id
#                                 adds complexity we don't need yet
# Empty / missing → trace_id is ""; logs still serialize cleanly.
_TRACE_ID_HEADERS = ("x-trace-id", "x-request-id", "traceparent")


def _extract_trace_id(invocation_metadata) -> str:
    """Pull the trace id from gRPC invocation metadata (case-insensitive)."""
    if not invocation_metadata:
        return ""
    md = {k.lower(): v for k, v in invocation_metadata}
    for h in _TRACE_ID_HEADERS:
        v = md.get(h, "")
        if v:
            # W3C traceparent is `version-traceid-spanid-flags`; the
            # middle segment is the trace id we want.
            if h == "traceparent":
                parts = v.split("-")
                if len(parts) >= 2:
                    return parts[1]
            return v
    return ""


class _TraceIdInterceptor(grpc.ServerInterceptor):
    """P2-4: bind per-RPC trace id into the `trace_id_var` ContextVar.

    Critical: `contextvars.Token` returned from `set()` is reset in `finally`.
    grpcio reuses ThreadPoolExecutor worker threads across RPCs, so without
    this reset, the trace id would leak into the next request processed on
    the same worker.
    """

    def intercept_service(self, continuation, handler_call_details):
        trace_id = _extract_trace_id(handler_call_details.invocation_metadata)
        token = set_trace_id(trace_id)
        try:
            return continuation(handler_call_details)
        finally:
            reset_trace_id(token)


class _AuthInterceptor(grpc.ServerInterceptor):
    """R1: require a shared-secret bearer token in call metadata.

    Constant-time compare; no token configured => open in local dev
    (HELPER_AUTH_TOKEN unset).
    """

    def __init__(self, token: str) -> None:
        self._token = token
        self._deny = grpc.unary_unary_rpc_method_handler(
            lambda req, ctx: ctx.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
        )

    def intercept_service(self, continuation, handler_call_details):
        if not self._token:
            return continuation(handler_call_details)
        md = dict(handler_call_details.invocation_metadata or ())
        presented = md.get("authorization", "").removeprefix("Bearer ").strip()
        if hmac.compare_digest(presented, self._token):
            return continuation(handler_call_details)
        auth_errors_total.labels(reason="bad_token").inc()
        return self._deny


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
        # R8: reject oversized questions before forwarding to any LLM adapter.
        if len(request.question) > MAX_QUESTION_LENGTH:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"question too long: {len(request.question)} > {MAX_QUESTION_LENGTH}")
            raise RuntimeError(f"question too long: {len(request.question)}")

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
            # R4: propagate client gRPC deadline to the domain chain budget.
            deadline_s = context.time_remaining()
            result: Answer = self._assistant.answer(
                Question(text=request.question),
                system_prompt=system_prompt,
                history=history,
                llm_provider=llm_provider,
                skip_role_detection=skip_role_detection,
                deadline_s=deadline_s,
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
        """P2-3: per-item status. One bad row never voids the batch.

        The provider's `embed_batch` returns `list[EmbedBatchResultItem]`
        with `status` ∈ {"success", "dim_mismatch", "fail"}. We translate
        each to the proto `EmbedBatchItem` so the caller (backend grid /
        backfill script) can persist successes and skip failures.
        """
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
            items = self._embedding.embed_batch(list(request.texts))
        except Exception as exc:
            # The provider is expected to NEVER bubble now (P2-3: it returns
            # per-item statuses). This except is the safety net.
            logger.exception("gRPC EmbedBatch unexpected provider-level failure")
            grpc_requests_total.labels(method="EmbedBatch", status="error").inc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"embed_batch raised unexpectedly: {exc}")
            raise

        used_model = requested_model or self._embedding.model
        elapsed_ms = (time.monotonic() - start) * 1000
        success_count = sum(1 for x in items if x.status == "success")
        logger.info(
            "gRPC EmbedBatch done: n=%d success=%d fail=%d dim_mismatch=%d model=%s elapsed_ms=%.0f",
            n_texts, success_count,
            sum(1 for x in items if x.status == "fail"),
            sum(1 for x in items if x.status == "dim_mismatch"),
            used_model, elapsed_ms,
        )

        # Promote the metric when ANY item failed so dashboards can flag
        # the partial outage without us echoing per-item labels.
        if success_count == n_texts:
            grpc_requests_total.labels(method="EmbedBatch", status="ok").inc()
        else:
            grpc_requests_total.labels(method="EmbedBatch", status="partial").inc()
        grpc_request_duration_seconds.labels(method="EmbedBatch").observe(
            time.monotonic() - start,
        )

        proto_items = []
        for i, item in enumerate(items):
            # P2-3 + Q3 reviewer fix: defensive dim check on the server side.
            # If a provider claims "success" but the embedding is None, empty,
            # or wrong-dim, we down-grade to "dim_mismatch" so the caller
            # never persists a zero or out-of-shape vector.
            if (
                item.status == "success"
                and item.embedding is not None
                and len(item.embedding) == EXPECTED_EMBEDDING_DIM
            ):
                proto_items.append(helper_pb2.EmbedBatchItem(
                    index=i,
                    status="success",
                    embedding=item.embedding,
                    model=item.model or used_model,
                    dimensions=EXPECTED_EMBEDDING_DIM,
                    error="",
                ))
            elif item.status == "success":
                # Provider bug: claimed success but bad payload. Down-grade.
                bad_dim = (
                    len(item.embedding)
                    if item.embedding is not None
                    else 0
                )
                logger.warning(
                    "EmbedBatch item %d claimed success but dim=%d → downgrading to dim_mismatch",
                    i, bad_dim,
                )
                proto_items.append(helper_pb2.EmbedBatchItem(
                    index=i,
                    status="dim_mismatch",
                    embedding=[],
                    model=item.model or used_model,
                    dimensions=0,
                    error=(
                        f"provider returned status=success but embedding dim={bad_dim} "
                        f"(expected {EXPECTED_EMBEDDING_DIM}) — downgraded server-side"
                    ),
                ))
            else:
                # dim_mismatch or fail as reported by the provider.
                proto_items.append(helper_pb2.EmbedBatchItem(
                    index=i,
                    status=item.status,
                    embedding=[],
                    model=item.model or used_model,
                    dimensions=0,
                    error=item.error or "unknown error",
                ))
        return helper_pb2.EmbedBatchResponse(items=proto_items)


# ── Health check helpers (R2: cached) ──────────────────────────────

_health_cache_lock = threading.Lock()
_health_cache: dict[str, str] = {}
_health_cache_detail: dict[str, str] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL_S = float(os.getenv("HEALTH_CACHE_TTL_S", "20"))

# P2-5 helper: recognise our own short sanitised status tokens (Q4 fix).
# Anything that doesn't strictly look like one of these gets replaced
# with a sha256 fingerprint so operators can grep logs for the match.
_SAFE_DETAIL_RX = re.compile(
    r"^(HTTP\s+\d+|"           # "HTTP 503"
    r"model\s+\S+\s+(found|not|not\s+pulled|not\s+found\s+yet)|"
    r"missing\s+\S+|"         # "missing api_key"
    r"no\s+\S+|"               # "no OLLAMA_BASE_URL configured"
    r"\S+\s+available)$"       # "model X available"
)


def _is_safe_detail_token(detail: str) -> bool:
    """True iff the detail is one of our own short sanitised strings.

    Anything else (e.g. an upstream error body) is replaced by a
    sha256-fingerprint marker in `_handle_health`.
    """
    return bool(_SAFE_DETAIL_RX.match(detail))


def _check_openai_compat(base_url: str, api_key: str, model: str, timeout: int = 5) -> tuple[str, str]:
    """P2-5: log full upstream body, expose only status_code in /health.

    Previously `resp.text[:100]` was served verbatim on the /health JSON
    endpoint. Any unauthenticated caller could scrape provider-specific
    error messages. The body is now logged in full (with adapter name +
    status code prefix) and only `HTTP <status_code>` appears in
    `/health` (full bodies go to the JSON logs).
    """
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
        # P2-5: log full body for ops; serve only the status code.
        logger.warning(
            "openai_compat health non-200 url=%s status=%s body=%s",
            base_url, resp.status_code, resp.text[:300],
        )
        health_check_total.labels(target="openai_compat", status="fail").inc()
        return "down", f"HTTP {resp.status_code}"
    except Exception as exc:
        logger.warning("health check openai_compat url=%s error=%s", base_url, exc)
        health_check_total.labels(target="openai_compat", status="fail").inc()
        return "down", "unreachable"


def _check_ollama(base_url: str, model: str, timeout: int = 5) -> tuple[str, str]:
    """P2-5: strip upstream error body from the response detail."""
    if not base_url:
        health_check_total.labels(target="ollama", status="fail").inc()
        return "down", "missing base_url"
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            logger.warning(
                "ollama health non-200 url=%s status=%s body=%s",
                base_url, resp.status_code, resp.text[:300],
            )
            health_check_total.labels(target="ollama", status="fail").inc()
            return "down", f"HTTP {resp.status_code}"
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
        return "down", "unreachable"


def _check_ollama_embedding(base_url: str, model: str, timeout: int = 5) -> tuple[str, str]:
    """P2-5: strip upstream error body from the response detail."""
    if not base_url:
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        return "skipped", "no OLLAMA_BASE_URL configured"
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            logger.warning(
                "ollama embedding health non-200 url=%s status=%s body=%s",
                base_url, resp.status_code, resp.text[:300],
            )
            health_check_total.labels(target="ollama_embed", status="fail").inc()
            return "down", f"http {resp.status_code}"
        tags = resp.json().get("models", [])
        names = {m.get("name") for m in tags}
        if any(n == model or n == f"{model}:latest" for n in names):
            health_check_total.labels(target="ollama_embed", status="ok").inc()
            return "ok", f"model {model} available"
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        return "down", f"model {model} not pulled yet -- run ollama pull {model}"
    except Exception as exc:
        logger.warning("health check ollama_embed url=%s error=%s", base_url, exc)
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        return "down", "unreachable"


def _refresh_health_cache(
    adapter_details: dict[str, dict[str, str]],
) -> None:
    """Run all upstream health checks and write results under lock (R2)."""
    global _health_cache, _health_cache_detail, _health_cache_ts
    results: dict[str, str] = {}
    details: dict[str, str] = {}
    for name, info in adapter_details.items():
        kind = info.get("kind", "")
        if kind == "openai_compat":
            status, detail = _check_openai_compat(
                info.get("base_url", ""),
                info.get("api_key", ""),
                info.get("model", ""),
            )
        elif kind == "ollama":
            status, detail = _check_ollama(
                info.get("base_url", ""),
                info.get("model", ""),
            )
        elif kind == "embedding":
            status, detail = _check_ollama_embedding(
                info.get("base_url", ""),
                info.get("model", ""),
            )
        else:
            status, detail = "unknown", "unknown kind"
        results[name] = status
        details[name] = detail
    with _health_cache_lock:
        _health_cache = results
        _health_cache_detail = details
        _health_cache_ts = time.monotonic()


class HealthHandler(http.server.BaseHTTPRequestHandler):
    _adapter_names: list[str] = []
    _grpc_server: Optional[grpc.Server] = None
    _adapter_details: dict[str, dict[str, str]] = {}

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_ready()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            logger.warning("health unknown_path=%s", self.path)
            self.send_response(404)
            self.end_headers()

    def _handle_ready(self) -> None:
        """Cheap readiness: gRPC up + at least one adapter OK in last cache."""
        with _health_cache_lock:
            healthy = any(v == "ok" for v in _health_cache.values())
        ready = self._grpc_server is not None and (healthy or not _health_cache)
        body = json.dumps({"ready": ready}).encode()
        self.send_response(200 if ready else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_metrics(self) -> None:
        body = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_health(self) -> None:
        """Serve health from cache; never calls upstream inline (R2).

        P2-5: `adapter_details` in the response no longer carries upstream
        error body fragments — only sanitised status tokens like
        `HTTP 503` or `unreachable`. Full upstream bodies are in logs.
        """
        status = {"status": "ok", "grpc": "ok", "adapters": "ok"}

        if self._grpc_server is not None:
            status["grpc"] = "ok"
        else:
            status["grpc"] = "unknown"

        # Read cached results under lock.
        with _health_cache_lock:
            cached = dict(_health_cache)
            cached_detail = dict(_health_cache_detail)

        adapter_results: dict[str, str] = {}
        adapter_details: dict[str, str] = {}

        for name, result in cached.items():
            adapter_results[name] = result
            adapter_details[name] = cached_detail.get(name, "")

        # P2-5: any detail string that doesn't strictly look like one of
        # our own sanitized status tokens is replaced by a sha256 fingerprint
        # prefix — operators can grep logs for `fp=XXXXXXXX` to find the
        # matching full body. This closes the short-JSON-body info-disclosure
        # hole the length-only heuristic left open. (Q4 reviewer fix.)
        sanitized_details: dict[str, str] = {}
        for name, detail in adapter_details.items():
            if not detail:
                sanitized_details[name] = ""
                continue
            if _is_safe_detail_token(detail):
                sanitized_details[name] = detail
                continue
            fp = hashlib.sha256(detail.encode("utf-8")).hexdigest()[:8]
            sanitized_details[name] = f"upstream error (fp={fp})"

        status["adapters"] = "ok" if all(v == "ok" for v in adapter_results.values() if v not in ("skipped",)) else "degraded"
        status["adapter_results"] = adapter_results
        status["adapter_details"] = sanitized_details
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
    """Start the gRPC server in a background thread.

    R1/R3/P2-4: supports auth interceptor, bounded workers, concurrent RPC
    cap, configurable message size, trace_id propagation, and optional TLS.
    """
    max_workers = int(os.getenv("GRPC_MAX_WORKERS", "16"))
    max_concurrent = int(os.getenv("GRPC_MAX_CONCURRENT_RPCS", "32"))
    token = os.getenv("HELPER_AUTH_TOKEN", "").strip()

    # P2-4: install trace-id interceptor; auth interceptor is conditional
    # (only when a token env is configured) but trace-id is always on.
    interceptors: list[grpc.ServerInterceptor] = [_TraceIdInterceptor()]
    if token:
        interceptors.append(_AuthInterceptor(token))

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        maximum_concurrent_rpcs=max_concurrent,
        interceptors=interceptors,
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
        ],
    )
    helper_pb2_grpc.add_HelperServiceServicer_to_server(
        HelperServicer(assistant, embedding_provider=embedding_provider), server,
    )
    cert = os.getenv("GRPC_TLS_CERT_PATH")
    key = os.getenv("GRPC_TLS_KEY_PATH")
    if cert and key:
        with open(cert, "rb") as c, open(key, "rb") as k:
            creds = grpc.ssl_server_credentials([(k.read(), c.read())])
        bound = server.add_secure_port(f"[::]:{port}", creds)
        logger.info("gRPC TLS enabled (cert=%s key=%s)", cert, key)
    else:
        logger.warning("gRPC TLS not configured -- using insecure port (dev only)")
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
    """Set health check dependencies and start the background refresh loop (R2)."""
    HealthHandler._adapter_names = adapter_names
    HealthHandler._grpc_server = grpc_server
    HealthHandler._adapter_details = adapter_details or {}
    # Initial cache populate.
    if adapter_details:
        _refresh_health_cache(adapter_details)
    # Background refresh loop.
    def _background_refresh():
        while True:
            time.sleep(_HEALTH_CACHE_TTL_S)
            try:
                _refresh_health_cache(adapter_details or {})
            except Exception:
                logger.exception("health cache refresh failed")
    t = threading.Thread(target=_background_refresh, daemon=True)
    t.start()
    logger.info("health cache background refresh started (ttl=%.0fs)", _HEALTH_CACHE_TTL_S)


def serve_health(port: int = 8084) -> http.server.HTTPServer:
    """Start a lightweight health HTTP server in a daemon thread.

    R6: uses ThreadingHTTPServer so a slow /health never blocks /metrics scrapes.
    """
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("health HTTP server on :%d", port)
    return server
