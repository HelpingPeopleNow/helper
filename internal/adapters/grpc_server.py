"""
gRPC server adapter for the helper service.

Runs alongside FastAPI on a separate port (50051).
"""
import hmac
import http.server
import json
import logging
import os
import threading
import time
from concurrent import futures
from typing import Optional
from urllib.request import Request, urlopen
import grpc
from grpc import ServicerContext
from internal.adapters.embedding_provider import (
    EmbeddingProvider,
    DimensionMismatchError,
)
from internal.adapters.enabled_providers import EnabledProvidersSource, resolve_deep_probe_targets
from internal.adapters.metrics import (
    active_requests,
    auth_errors_total,
    classify_error,
    estimate_tokens,
    grpc_request_duration_seconds,
    grpc_requests_total,
    health_check_total,
    helper_deep_probe_success,
    llm_errors_total,
    llm_request_duration_seconds,
    llm_requests_total,
    llm_tokens_total,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from internal.core.helper_agent import Answer, HelperAgent, Question
from internal.ports.llm import LLMPort, Message
from proto import helper_pb2, helper_pb2_grpc

logger = logging.getLogger(__name__)

# Max question length in characters (R8): reject oversized prompts
# before forwarding to any LLM adapter (cost/memory protection).
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "32000"))

# Maximum input bytes for gRPC messages (R3).
MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))


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
        logger.warning("RPC rejected: bad token, method=%s", handler_call_details.method)
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
            logger.warning("Ask rejected: question too long (%d chars)", len(request.question))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"question too long: {len(request.question)} > {MAX_QUESTION_LENGTH}")
            raise RuntimeError(f"question too long: {len(request.question)}")

        start = time.monotonic()
        history_len = len(request.history)
        system_prompt = request.system_prompt
        enabled_providers = list(request.enabled_providers)  # repeated string → list
        skip_role_detection = request.skip_role_detection
        logger.info("gRPC Ask: q_len=%d history=%d sp_len=%d providers=%s skip_role=%s",
                     len(request.question), history_len, len(system_prompt), enabled_providers or "(env default)", skip_role_detection)

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
                enabled_providers=enabled_providers,
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
            logger.warning("Embed rejected: no embedding provider configured")
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
            logger.warning("EmbedBatch rejected: no embedding provider configured")
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


# ── Health check helpers (R2: cached) ──────────────────────────────

_health_cache_lock = threading.Lock()
_health_cache: dict[str, str] = {}
_health_cache_detail: dict[str, str] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL_S = float(os.getenv("HEALTH_CACHE_TTL_S", "20"))


def _check_http_url(url: str, timeout: int = 3) -> tuple[str, str]:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return "ok", f"HTTP {resp.status}"
    except Exception as exc:
        logger.warning("health check http_url=%s error=%s", url, exc)
        return "down", str(exc)


def _check_openai_compat(base_url: str, api_key: str, model: str, timeout: int = 5) -> tuple[str, str]:
    if not api_key:
        health_check_total.labels(target="openai_compat", status="fail").inc()
        logger.warning("health: OpenAI-compat down: %s", "missing api_key")
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
        logger.warning("health: OpenAI-compat down: HTTP %s: %s", resp.status_code, resp.text[:100])
        return "down", f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as exc:
        logger.warning("health check openai_compat url=%s error=%s", base_url, exc)
        health_check_total.labels(target="openai_compat", status="fail").inc()
        logger.warning("health: OpenAI-compat down: %s", exc)
        return "down", str(exc)


def _check_ollama(base_url: str, model: str, timeout: int = 5) -> tuple[str, str]:
    if not base_url:
        health_check_total.labels(target="ollama", status="fail").inc()
        logger.warning("health: Ollama down: %s", "missing base_url")
        return "down", "missing base_url"
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            health_check_total.labels(target="ollama", status="fail").inc()
            logger.warning("health: Ollama down: HTTP %s: %s", resp.status_code, resp.text[:100])
            return "down", f"HTTP {resp.status_code}: {resp.text[:100]}"
        data = resp.json()
        models = data.get("models", [])
        if any(m.get("name") == model for m in models):
            health_check_total.labels(target="ollama", status="ok").inc()
            return "ok", f"model {model} found"
        health_check_total.labels(target="ollama", status="fail").inc()
        logger.warning("health: Ollama down: %s", f"model {model} not found")
        return "down", f"model {model} not found"
    except Exception as exc:
        logger.warning("health check ollama url=%s error=%s", base_url, exc)
        health_check_total.labels(target="ollama", status="fail").inc()
        logger.warning("health: Ollama down: %s", exc)
        return "down", str(exc)


def _check_ollama_embedding(base_url: str, model: str, timeout: int = 5) -> tuple[str, str]:
    if not base_url:
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        logger.warning("health: Ollama embedding skipped: %s", "no OLLAMA_BASE_URL configured")
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
        return "down", f"model {model} not pulled yet -- run ollama pull {model}"
    except Exception as exc:
        logger.warning("health check ollama_embed url=%s error=%s", base_url, exc)
        health_check_total.labels(target="ollama_embed", status="fail").inc()
        return "down", str(exc)


# ── Deep health probe (OBSERVABILITY_AUDIT_REPORT.md §3.2) ─────────────
#
# The checks above (_check_openai_compat, _check_ollama) only hit cheap
# metadata endpoints (/models, /api/tags). A Cloudflare WAF rule that
# fingerprints/blocks POST-heavy chat-completion traffic while leaving GET
# metadata endpoints open would defeat them entirely (this is what happened
# during the incident this mechanism was designed to catch). This probe
# instead performs a real, cheap (1-token) completion call per adapter, on
# its own longer-interval cache lane so it never runs on every /health scrape.
_DEEP_PROBE_PROMPT = "Reply with only the single word OK."
_DEEP_PROBE_USER = "ping"
_HEALTH_DEEP_PROBE_INTERVAL_S = float(os.getenv("HEALTH_DEEP_PROBE_INTERVAL_S", "60"))

_deep_probe_lock = threading.Lock()
_deep_probe_results: dict[str, bool] = {}


def _deep_probe_rate_limited(exc: Exception) -> bool:
    """True when the provider answered but rejected the call due to quota.

    A 429 means the completion route is reachable (the probe's goal); it must
    not page as HelperDeepProbeFailed or the 60s probe burns free-tier RPD.
    """
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
        or "resource_exhausted" in msg
    )


def _run_deep_probe(
    adapters: dict[str, "LLMPort"],
    provider_source: EnabledProvidersSource | None = None,
) -> dict[str, bool]:
    """Run a synthetic 1-token completion against admin-enabled chat
    adapters and record pass/fail into helper_deep_probe_success{provider}.

    Providers disabled in the admin panel are marked healthy (1) so
    HelperDeepProbeFailed only pages for models the app actually uses.
    When the backend list is empty/unset, FALLBACK_CHAIN is used (same
    as chat). Ollama is skipped when OLLAMA_BASE_URL is unset.
    """
    admin_providers = provider_source.fetch() if provider_source else None
    probe_targets = set(resolve_deep_probe_targets(adapters, admin_providers))
    results: dict[str, bool] = {}
    for name, adapter in adapters.items():
        if name == "ollama" and not os.getenv("OLLAMA_BASE_URL", "").strip():
            continue
        if name not in probe_targets:
            helper_deep_probe_success.labels(provider=name).set(1)
            continue
        try:
            adapter.complete(system_prompt=_DEEP_PROBE_PROMPT, user=_DEEP_PROBE_USER)
            results[name] = True
            helper_deep_probe_success.labels(provider=name).set(1)
        except Exception as exc:
            if _deep_probe_rate_limited(exc):
                logger.warning(
                    "deep probe rate-limited provider=%s (treating as ok): %s",
                    name,
                    exc,
                )
                results[name] = True
                helper_deep_probe_success.labels(provider=name).set(1)
            else:
                logger.warning("deep probe failed provider=%s error=%s", name, exc)
                results[name] = False
                helper_deep_probe_success.labels(provider=name).set(0)
    with _deep_probe_lock:
        _deep_probe_results.clear()
        _deep_probe_results.update(results)
    return results


def _deep_probe_loop(
    adapters: dict[str, "LLMPort"],
    provider_source: EnabledProvidersSource | None = None,
) -> None:
    while True:
        time.sleep(_HEALTH_DEEP_PROBE_INTERVAL_S)
        try:
            _run_deep_probe(adapters, provider_source)
        except Exception:
            logger.exception("deep probe loop iteration failed")


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
            logger.warning("health: unknown adapter kind=%s", kind)
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
        """Serve health from cache; never calls upstream inline (R2)."""
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
            # R2: skip the inline check entirely — use cached value.
            adapter_results[name] = result
            adapter_details[name] = cached_detail.get(name, "")

        status["adapters"] = "ok" if all(v == "ok" for v in adapter_results.values() if v not in ("skipped",)) else "degraded"
        status["adapter_results"] = adapter_results
        status["adapter_details"] = adapter_details
        status["loaded_adapters"] = self._adapter_names

        # OBSERVABILITY_AUDIT_REPORT.md §3.2 — surface the deep-probe layer
        # alongside the shallow adapter_results so a glance at /health shows
        # both "metadata endpoint reachable" and "real inference path works".
        # Read-only from cache; the background _deep_probe_loop is the only
        # writer (never runs inline on a scrape, same R2 pattern as above).
        with _deep_probe_lock:
            deep_probe_results = dict(_deep_probe_results)
        status["deep_probe_results"] = {
            name: "ok" if ok else "down" for name, ok in deep_probe_results.items()
        }
        status["deep_probe"] = (
            "ok" if all(deep_probe_results.values()) else "degraded"
        ) if deep_probe_results else "unknown"

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

    R1/R3: supports auth interceptor, bounded workers, concurrent RPC cap,
    configurable message size, and optional TLS.
    """
    max_workers = int(os.getenv("GRPC_MAX_WORKERS", "16"))
    max_concurrent = int(os.getenv("GRPC_MAX_CONCURRENT_RPCS", "32"))
    token = os.getenv("HELPER_AUTH_TOKEN", "").strip()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        maximum_concurrent_rpcs=max_concurrent,
        interceptors=[_AuthInterceptor(token)] if token else [],
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
    llm_adapters: Optional[dict[str, "LLMPort"]] = None,
) -> None:
    """Set health check dependencies and start the background refresh loop (R2).

    llm_adapters (OBSERVABILITY_AUDIT_REPORT.md §3.2) — when provided, also
    starts the deep-probe background loop (separate cache lane, separate
    interval from the shallow adapter_details checks above).
    """
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

    if llm_adapters:
        providers_url = os.getenv("BACKEND_LLM_PROVIDERS_URL", "").strip()
        provider_source = EnabledProvidersSource(providers_url)
        dt = threading.Thread(
            target=_deep_probe_loop,
            args=(llm_adapters, provider_source),
            daemon=True,
        )
        dt.start()
        logger.info(
            "deep probe background loop started (interval=%.0fs, loaded=%s, admin_url=%s)",
            _HEALTH_DEEP_PROBE_INTERVAL_S,
            list(llm_adapters.keys()),
            providers_url or "(fallback chain only)",
        )


def serve_health(port: int = 8084) -> http.server.HTTPServer:
    """Start a lightweight health HTTP server in a daemon thread.

    R6: uses ThreadingHTTPServer so a slow /health never blocks /metrics scrapes.
    """
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("health HTTP server on :%d", port)
    return server
