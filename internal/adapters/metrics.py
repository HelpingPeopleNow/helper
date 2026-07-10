"""
Prometheus metrics for the helper service.

Exposes counters, histograms, and gauges for gRPC, LLM, health check,
and auth observability. All metrics are defined as module-level singletons so they
can be imported and used from any part of the application.
"""
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# 1. gRPC request metrics
# ---------------------------------------------------------------------------
grpc_requests_total = Counter(
    "grpc_requests_total",
    "Total gRPC requests",
    ["method", "status"],
)

grpc_request_duration_seconds = Histogram(
    "grpc_request_duration_seconds",
    "Duration of gRPC requests in seconds",
    ["method"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)


# ---------------------------------------------------------------------------
# 2. LLM call metrics
# ---------------------------------------------------------------------------
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "mode"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "Duration of LLM requests in seconds",
    ["provider", "mode"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

llm_errors_total = Counter(
    "llm_errors_total",
    "Total LLM errors",
    ["provider", "error_type"],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Estimated LLM tokens (chars / 4 as rough proxy)",
    ["provider", "direction"],
)


# ---------------------------------------------------------------------------
# 3. Health check metrics
# ---------------------------------------------------------------------------
health_check_total = Counter(
    "health_check_total",
    "Total health checks performed",
    ["target", "status"],
)


# ---------------------------------------------------------------------------
# 4. Auth error metric (R10)
# ---------------------------------------------------------------------------
auth_errors_total = Counter(
    "auth_errors_total",
    "Total rejected gRPC calls due to auth failure",
    ["reason"],
)


# ---------------------------------------------------------------------------
# 5. Active requests gauge
# ---------------------------------------------------------------------------
active_requests = Gauge(
    "active_requests",
    "Current number of in-flight LLM requests",
)


# ---------------------------------------------------------------------------
# Helper: classify exceptions into error_type labels (R10)
# ---------------------------------------------------------------------------
def classify_error(exc: Exception) -> str:
    """Return a Prometheus-friendly error_type label for the given exception.

    Uses explicit exception-type matching first (R10), then falls back
    to substring heuristics for third-party/unknown exception classes.
    """
    name = type(exc).__name__
    import httpx as _httpx
    import grpc as _grpc

    # Explicit type matches (R10)
    if isinstance(exc, _httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, _grpc.RpcError):
        code = getattr(exc, "code", lambda: None)()
        if code is not None:
            mapping = {
                _grpc.StatusCode.DEADLINE_EXCEEDED: "timeout",
                _grpc.StatusCode.UNAVAILABLE: "connection_error",
                _grpc.StatusCode.UNAUTHENTICATED: "auth_error",
                _grpc.StatusCode.INVALID_ARGUMENT: "invalid_argument",
                _grpc.StatusCode.RESOURCE_EXHAUSTED: "rate_limited",
            }
            return mapping.get(code, "rpc_error")

    # Fallback: substring heuristics for other exception types
    msg = str(exc).lower()
    n = name.lower()
    if "timeout" in n or "timeout" in msg:
        return "timeout"
    if "connection" in n or "connection" in msg:
        return "connection_error"
    if "json" in n or "jsondecode" in n or "parse" in msg:
        return "parse_error"
    if "decode" in msg or "expect" in msg:
        return "parse_error"
    if any(h in msg for h in ("429", "500", "502", "503", "504")):
        return "http_error"
    return "unknown"


# ---------------------------------------------------------------------------
# Helper: output estimation
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(len(text) // 4, 0)
