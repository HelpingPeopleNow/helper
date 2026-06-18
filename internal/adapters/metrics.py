"""
Prometheus metrics for the helper service.

Exposes counters, histograms, and gauges for gRPC, LLM, and health check
observability. All metrics are defined as module-level singletons so they
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
# 4. Active requests gauge
# ---------------------------------------------------------------------------
active_requests = Gauge(
    "active_requests",
    "Current number of in-flight LLM requests",
)


# ---------------------------------------------------------------------------
# Helper: classify exceptions into error_type labels
# ---------------------------------------------------------------------------
def classify_error(exc: Exception) -> str:
    """Return a Prometheus-friendly error_type label for the given exception."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg:
        return "timeout"
    if "connection" in name or "connectionerror" in name or "connection" in msg:
        return "connection_error"
    if "http" in msg and ("5" in msg[:3] or "4" in msg[:3]):
        return "http_error"
    if "json" in name or "jsondecode" in name or "parse" in msg:
        return "parse_error"
    # Fallback: check common substrings
    if "timed out" in msg or "deadline" in msg:
        return "timeout"
    if "connect" in msg:
        return "connection_error"
    if "status" in msg and ("5" in msg or "4" in msg):
        return "http_error"
    if "decode" in msg or "expect" in msg:
        return "parse_error"
    return "http_error"


# ---------------------------------------------------------------------------
# Helper: output estimation
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(len(text) // 4, 0)
