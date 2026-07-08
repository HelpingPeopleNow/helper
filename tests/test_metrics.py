"""
Tests for the metrics module (R10).

Source: `helper/internal/adapters/metrics.py`.

These verify:
- `auth_errors_total` counter is exported and labels work (R10)
- `classify_error` uses explicit isinstance checks for known types
- Substring fallback labels are stable across calls
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import grpc
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from internal.adapters import metrics
from internal.adapters.metrics import (
    auth_errors_total,
    classify_error,
    estimate_tokens,
)


class TestAuthErrorsCounter:
    """Test #103-104: `auth_errors_total` is a Counter with a `reason` label."""

    def test_auth_errors_total_exists(self) -> None:
        assert hasattr(auth_errors_total, "labels")
        assert hasattr(auth_errors_total, "inc")

    def test_auth_errors_total_increments_with_reason(self) -> None:
        before = auth_errors_total.labels(reason="bad_token")._value.get()
        auth_errors_total.labels(reason="bad_token").inc()
        after = auth_errors_total.labels(reason="bad_token")._value.get()
        assert after == before + 1

    def test_distinct_reason_labels_are_independent(self) -> None:
        a_before = auth_errors_total.labels(reason="test_a")._value.get()
        b_before = auth_errors_total.labels(reason="test_b")._value.get()
        auth_errors_total.labels(reason="test_a").inc()
        auth_errors_total.labels(reason="test_b").inc(2)
        assert auth_errors_total.labels(reason="test_a")._value.get() == a_before + 1
        assert auth_errors_total.labels(reason="test_b")._value.get() == b_before + 2


class TestClassifyErrorExplicitTypes:
    """Test #104-111: explicit isinstance-based classification."""

    def test_requests_timeout(self) -> None:
        assert classify_error(requests.exceptions.Timeout()) == "timeout"

    def test_requests_connection_error(self) -> None:
        assert classify_error(requests.exceptions.ConnectionError()) == "connection_error"

    def test_grpc_deadline_exceeded(self) -> None:
        err = _fake_grpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
        assert classify_error(err) == "timeout"

    def test_grpc_unavailable(self) -> None:
        err = _fake_grpc_error(grpc.StatusCode.UNAVAILABLE)
        assert classify_error(err) == "connection_error"

    def test_grpc_unauthenticated(self) -> None:
        err = _fake_grpc_error(grpc.StatusCode.UNAUTHENTICATED)
        assert classify_error(err) == "auth_error"

    def test_grpc_invalid_argument(self) -> None:
        err = _fake_grpc_error(grpc.StatusCode.INVALID_ARGUMENT)
        assert classify_error(err) == "invalid_argument"

    def test_grpc_resource_exhausted(self) -> None:
        err = _fake_grpc_error(grpc.StatusCode.RESOURCE_EXHAUSTED)
        assert classify_error(err) == "rate_limited"

    def test_grpc_unknown_status(self) -> None:
        err = _fake_grpc_error(grpc.StatusCode.INTERNAL)
        assert classify_error(err) == "rpc_error"


class TestClassifyErrorSubstringFallback:
    """Test #112-113: substring fallback for non-typed exceptions."""

    def test_429_in_message_maps_to_http_error(self) -> None:
        assert classify_error(RuntimeError("upstream 429 too many requests")) == "http_error"

    def test_untyped_exception_maps_to_unknown(self) -> None:
        # No recognised tokens, no recognised type -> "unknown"
        result = classify_error(Exception("the quick brown fox"))
        assert result == "unknown"

    def test_decode_in_message_maps_to_parse_error(self) -> None:
        assert classify_error(ValueError("could not decode token")) == "parse_error"


class TestEstimateTokens:
    """Test #114-116."""

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_string(self) -> None:
        # "hello world" is 11 chars -> 11 // 4 = 2
        assert estimate_tokens("hello world") == 2

    def test_long_string(self) -> None:
        assert estimate_tokens("a" * 1000) == 250

    def test_whitespace_only(self) -> None:
        # 4 spaces -> 4 // 4 = 1
        assert estimate_tokens("    ") == 1


def _fake_grpc_error(code: grpc.StatusCode) -> grpc.RpcError:
    """Build a grpc.RpcError that reports the given status code.

    `grpc.RpcError` is an ABC; we subclass it directly because the concrete
    gRPC error types are not part of the public API. For classification
    tests we only need `.code()` to return the right value.
    """

    class _FakeError(grpc.RpcError):
        def code(self):
            return code

        def details(self):
            return ""

    return _FakeError()
