"""
Structured JSON logging + trace_id propagation (P2-4).

Three pieces:
1. `trace_id_var` — `contextvars.ContextVar[str]` holding the current
   request's trace id. Default is "" so logs without active trace still
   serialize cleanly.
2. `_TraceIdFilter` — `logging.Filter` that attaches the current trace_id
   onto every record that doesn't already have one set.
3. `JsonFormatter` — `logging.Formatter` subclass emitting one JSON object
   per record on a single line (no embedded newlines). Includes ts, level,
   logger, msg, exc_info as JSON-stringified traceback (if present), and
   any extras (request_id, trace_id, method, etc.) that were attached.

Why contextvars and not threading.local: grpcio's server dispatches RPC
handlers onto its `ThreadPoolExecutor` pool. The worker thread is reused
across requests. A naive `threading.local.set(trace_id)` inside the handler
would leak the value to the next request handled by that same worker until
the next handler re-set it. `ContextVar.set()` returns a `Token`; we MUST
`reset(token)` in a `finally`. The `_TraceIdInterceptor` below does this
centrally so individual handlers don't have to.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Default to empty string — empty trace_id is fine in JSON output.
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)


def set_trace_id(value: str) -> contextvars.Token:
    """Set the trace id for the current execution context.

    Returns a Token that MUST be passed to `reset_trace_id` in a finally
    block to prevent leak across thread-pool reuse. Use only via the
    `_TraceIdInterceptor` in grpc_server.py — direct callers must know
    what they're doing.
    """
    return trace_id_var.set(value)


def reset_trace_id(token: contextvars.Token) -> None:
    trace_id_var.reset(token)


def current_trace_id() -> str:
    return trace_id_var.get()


class _TraceIdFilter(logging.Filter):
    """Inject trace_id onto the record if not already set.

    Attached to the root logger in `install()`. Cheap (one var.get call per
    record), safe under all `Handler` postfix routing.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            try:
                record.trace_id = trace_id_var.get()
            except LookupError:
                record.trace_id = ""
        return True


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Keys: `ts`, `level`, `logger`, `msg`, `trace_id`, `module`, `lineno`,
    plus any user-set extras (record.__dict__ attrs not in the stdlib
    filter set — we whitelist by skipping known stdlib attrs).
    """

    # Skip these standard LogRecord attributes in the extras set.
    _STDLIB_KEYS = frozenset({
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        # P2-4 portability fix: `time.strftime("%f", ...)` is glibc-only and
        # throws ValueError on macOS / musl / Windows. Build the ISO timestamp
        # from `record.created` (epoch seconds, set by logging at record time)
        # and ALSO emit epoch_ms so Loki / Datadog / OTel can index cheaply.
        iso_ts = datetime.fromtimestamp(
            record.created, tz=timezone.utc,
        ).isoformat(timespec="milliseconds")
        epoch_ms = int(record.created * 1000)
        payload: dict[str, Any] = {
            "ts": iso_ts,
            "epoch_ms": epoch_ms,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", ""),
            "module": record.module,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        # Carry over safe extras (request_id, method, provider, etc.).
        for key, value in record.__dict__.items():
            if key in self._STDLIB_KEYS or key in payload:
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)  # ensure serializable
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        try:
            return json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            # Fall back to a minimal payload so we never raise from logging.
            return json.dumps({
                "ts": payload["ts"],
                "level": "ERROR",
                "logger": "logging_config",
                "msg": f"json-format-failed: original={payload['msg'][:200]}",
                "trace_id": payload.get("trace_id", ""),
            })


def install(level: str = "INFO") -> None:
    """Install the JSON formatter + trace_id filter on the root logger.

    Idempotent: if a previous install replaced the handlers, calling again
    reuses them (we mutate in place rather than add new handlers). This
    keeps pytest `caplog` happy because caplog hooks the root logger pre-
    handler-mutation.
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Add the filter (idempotent — propagate so child loggers inherit).
    if not any(isinstance(f, _TraceIdFilter) for f in root.filters):
        root.addFilter(_TraceIdFilter())

    formatter = JsonFormatter()
    # Update existing handlers in-place. Don't replace; tests may have
    # attached caplog on top.
    for handler in root.handlers:
        handler.setFormatter(formatter)
    # If no handlers, attach a default stream→stderr one (the canonical
    # `logging.basicConfig` behavior).
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)
