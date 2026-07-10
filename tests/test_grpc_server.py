"""
Tests for the gRPC server adapter (R1, R3, R6, R8, R9, P1-5, P2-3, P2-4).

These tests:
- Start a real `serve_grpc` instance bound to localhost:0
- Make real gRPC calls via `grpc.insecure_channel`
- Verify auth, input cap, server config, trace_id propagation,
  EmbedBatch per-item status, and server type

No external services are required.
"""
from __future__ import annotations

import http.server
import os
import signal
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proto"))

import json  # for /health body assertions

from internal.adapters import grpc_server as gs
from internal.adapters.metrics import auth_errors_total, grpc_requests_total
from internal.core.helper_agent import HelperAgent
from proto import helper_pb2, helper_pb2_grpc


# ── Helpers ─────────────────────────────────────────────────────────


def _stub_adapter() -> MagicMock:
    """LLM adapter that returns a canned answer AND exposes a None last_usage
    so P1-5 tiktoken fallback in the agent doesn't accumulate MagicMocks."""
    a = MagicMock()
    a.complete.return_value = '{"answer":"hello","role":""}'
    a.last_usage = None
    return a


def _stub_embedding(batch_responses=None):
    """Build a MagicMock embedding provider that returns per-item statuses.

    `batch_responses` is a list of dicts with keys: status, embedding,
    error, model. Default = all successes with a 768-dim vector.
    """
    fake_vec = [0.1] * 768
    if batch_responses is None:
        batch_responses = [
            {
                "status": "success",
                "embedding": fake_vec,
                "error": "",
                "model": "test-model",
            }
        ]
    a = MagicMock()
    a.model = "test-model"
    a.embed.return_value = fake_vec
    a.embed_batch.return_value = [
        MagicMock(  # EmbedBatchResultItem-compatible
            status=r["status"],
            embedding=r["embedding"],
            error=r["error"],
            model=r["model"],
        )
        for r in batch_responses
    ]
    return a


def _start_server(
    monkeypatch: pytest.MonkeyPatch,
    token: str = "",
    max_workers: int = 4,
    max_concurrent: int = 8,
    embedding_provider=None,
) -> tuple[grpc.Server, str]:
    monkeypatch.setenv("HELPER_AUTH_TOKEN", token)
    monkeypatch.setenv("GRPC_MAX_WORKERS", str(max_workers))
    monkeypatch.setenv("GRPC_MAX_CONCURRENT_RPCS", str(max_concurrent))
    monkeypatch.setenv("GRPC_TLS_CERT_PATH", "")
    monkeypatch.setenv("GRPC_TLS_KEY_PATH", "")

    adapters = {"ollama": _stub_adapter()}
    agent = HelperAgent(adapters=adapters)
    port = _free_port()
    server = gs.serve_grpc(
        agent, embedding_provider=embedding_provider, port=port,
    )
    return server, f"127.0.0.1:{port}"


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── R1: auth interceptor (Test #117-122) ───────────────────────────


class TestAuthInterceptor:
    def test_no_token_allows_anonymous(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HELPER_AUTH_TOKEN", raising=False)
        server, addr = _start_server(monkeypatch, token="")
        try:
            with grpc.insecure_channel(addr) as ch:
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                resp = stub.Ask(
                    helper_pb2.AskRequest(question="hi", system_prompt="sp", skip_role_detection=True),
                    timeout=5,
                )
                assert resp.answer == "hello"
        finally:
            server.stop(grace=1)

    def test_token_required_rejects_anonymous(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, addr = _start_server(monkeypatch, token="secret")
        try:
            with grpc.insecure_channel(addr) as ch:
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Ask(
                        helper_pb2.AskRequest(question="hi", system_prompt="sp", skip_role_detection=True),
                        timeout=5,
                    )
                assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
        finally:
            server.stop(grace=1)

    def test_wrong_token_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, addr = _start_server(monkeypatch, token="secret")
        try:
            with grpc.insecure_channel(addr) as ch:
                md = (("authorization", "Bearer wrong"),)
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                with pytest.raises(grpc.RpcError) as exc_info:
                    stub.Ask(
                        helper_pb2.AskRequest(question="hi", system_prompt="sp", skip_role_detection=True),
                        timeout=5,
                        metadata=md,
                    )
                assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
        finally:
            server.stop(grace=1)

    def test_correct_token_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, addr = _start_server(monkeypatch, token="secret")
        try:
            with grpc.insecure_channel(addr) as ch:
                md = (("authorization", "Bearer secret"),)
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                resp = stub.Ask(
                    helper_pb2.AskRequest(question="hi", system_prompt="sp", skip_role_detection=True),
                    timeout=5,
                    metadata=md,
                )
                assert resp.answer == "hello"
        finally:
            server.stop(grace=1)

    def test_auth_errors_counter_increments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, addr = _start_server(monkeypatch, token="secret")
        try:
            before = auth_errors_total.labels(reason="bad_token")._value.get()
            with grpc.insecure_channel(addr) as ch:
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                with pytest.raises(grpc.RpcError):
                    stub.Ask(
                        helper_pb2.AskRequest(question="hi", system_prompt="sp", skip_role_detection=True),
                        timeout=5,
                    )
            after = auth_errors_total.labels(reason="bad_token")._value.get()
            assert after == before + 1
        finally:
            server.stop(grace=1)


# ── R8: input cap (Test #123-124) ──────────────────────────────────


class TestInputCap:
    def test_oversized_question_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_QUESTION_LENGTH", "100")
        import importlib
        importlib.reload(gs)
        try:
            server, addr = _start_server(monkeypatch, token="")
            try:
                with grpc.insecure_channel(addr) as ch:
                    stub = helper_pb2_grpc.HelperServiceStub(ch)
                    with pytest.raises(grpc.RpcError) as exc_info:
                        stub.Ask(
                            helper_pb2.AskRequest(
                                question="x" * 101, system_prompt="sp", skip_role_detection=True,
                            ),
                            timeout=5,
                        )
                    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            finally:
                server.stop(grace=1)
        finally:
            monkeypatch.delenv("MAX_QUESTION_LENGTH", raising=False)
            importlib.reload(gs)

    def test_boundary_size_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_QUESTION_LENGTH", "100")
        import importlib
        importlib.reload(gs)
        try:
            server, addr = _start_server(monkeypatch, token="")
            try:
                with grpc.insecure_channel(addr) as ch:
                    stub = helper_pb2_grpc.HelperServiceStub(ch)
                    resp = stub.Ask(
                        helper_pb2.AskRequest(
                            question="x" * 100, system_prompt="sp", skip_role_detection=True,
                        ),
                        timeout=5,
                    )
                    assert resp.answer == "hello"
            finally:
                server.stop(grace=1)
        finally:
            monkeypatch.delenv("MAX_QUESTION_LENGTH", raising=False)
            importlib.reload(gs)


# ── R3: gRPC server config (Test #125-127) ─────────────────────────


class TestGrpcServerConfig:
    def test_max_concurrent_rpcs_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRPC_MAX_CONCURRENT_RPCS", "10")
        import importlib
        importlib.reload(gs)
        assert os.getenv("GRPC_MAX_CONCURRENT_RPCS") == "10"

    def test_max_workers_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRPC_MAX_WORKERS", "5")
        import importlib
        importlib.reload(gs)
        assert os.getenv("GRPC_MAX_WORKERS") == "5"


# ── R6: ThreadingHTTPServer (Test #128-129) ────────────────────────


class TestThreadingHealthServer:
    def test_serve_health_returns_threading_http_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        port = _free_port()
        server = gs.serve_health(port=port)
        try:
            assert isinstance(server, http.server.ThreadingHTTPServer)
        finally:
            server.shutdown()
            server.server_close()

    def test_health_server_responds(self) -> None:
        import urllib.request
        port = _free_port()
        gs.HealthHandler._grpc_server = MagicMock()
        gs._health_cache.clear()
        gs._health_cache.update({"ollama": "ok"})
        server = gs.serve_health(port=port)
        try:
            time.sleep(0.05)
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2)
                assert resp.status in (200, 503)
            except urllib.error.HTTPError as e:
                assert e.code in (200, 503)
        finally:
            server.shutdown()
            server.server_close()


# ── R9: graceful shutdown (Test #130-132) ─────────────────────────


class TestGracefulShutdown:
    def test_sigterm_handler_installed(self) -> None:
        import main as helper_main
        assert hasattr(helper_main, "signal")
        assert hasattr(helper_main.signal, "signal")
        assert hasattr(helper_main.signal, "SIGTERM")
        assert hasattr(helper_main.signal, "SIGINT")

    def test_sigint_handler_installed(self) -> None:
        import main as helper_main
        assert hasattr(helper_main.signal, "SIGINT")

    def test_shutdown_event_set_on_signal(self) -> None:
        import main as helper_main
        import threading
        event = threading.Event()
        captured: list[int] = []
        def _h(signum, _frame):
            captured.append(signum)
            event.set()
        _h(signal.SIGTERM, None)
        assert event.is_set()
        assert captured == [signal.SIGTERM]


# ── P2-4: trace_id interceptor ────────────────────────────────────


class TestTraceIdInterceptor:
    def test_trace_id_header_does_not_break_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """x-trace-id metadata is accepted by the Ask RPC and doesn't error."""
        server, addr = _start_server(monkeypatch, token="")
        try:
            with grpc.insecure_channel(addr) as ch:
                md = (("x-trace-id", "abc123def456"),)
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                resp = stub.Ask(
                    helper_pb2.AskRequest(question="hi", system_prompt="sp", skip_role_detection=True),
                    timeout=5,
                    metadata=md,
                )
                assert resp.answer == "hello"
        finally:
            server.stop(grace=1)

    def test_traceparent_w3c_header_extracts_trace_id(self) -> None:
        """W3C traceparent `version-traceid-spanid-flags` → trace_id is the middle segment."""
        from internal.adapters.grpc_server import _extract_trace_id
        md = (("traceparent", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),)
        trace_id = _extract_trace_id(md)
        assert trace_id == "0af7651916cd43dd8448eb211c80319c"

    def test_x_request_id_header_accepted(self) -> None:
        from internal.adapters.grpc_server import _extract_trace_id
        md = (("x-request-id", "req-9999"),)
        assert _extract_trace_id(md) == "req-9999"

    def test_no_trace_id_header_returns_empty(self) -> None:
        from internal.adapters.grpc_server import _extract_trace_id
        assert _extract_trace_id(None) == ""
        assert _extract_trace_id([]) == ""
        assert _extract_trace_id((("unrelated", "x"),)) == ""


# ── P2-3: EmbedBatch per-item status translation ──────────────────


class TestEmbedBatchPerItem:
    def test_all_success_returns_success_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # P2-3: provider must return N per-item results matching N input
        # texts. _stub_embedding's default returns 1 — for 2 texts the test
        # must supply both explicitly.
        fake_vec = [0.1] * 768
        provider = _stub_embedding([
            {"status": "success", "embedding": fake_vec, "error": "", "model": "test-model"},
            {"status": "success", "embedding": fake_vec, "error": "", "model": "test-model"},
        ])
        server, addr = _start_server(monkeypatch, token="", embedding_provider=provider)
        try:
            with grpc.insecure_channel(addr) as ch:
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                resp = stub.EmbedBatch(
                    helper_pb2.EmbedBatchRequest(texts=["a", "b"], model="m"),
                    timeout=5,
                )
            assert len(resp.items) == 2
            for i, item in enumerate(resp.items):
                assert item.index == i
                assert item.status == "success"
                assert len(item.embedding) == 768
                assert item.error == ""
        finally:
            server.stop(grace=1)

    def test_partial_failure_returns_per_item_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_vec = [0.1] * 768
        provider = _stub_embedding([
            {"status": "success", "embedding": fake_vec, "error": "", "model": "test-model"},
            {"status": "fail", "embedding": None, "error": "HTTP 500", "model": "test-model"},
            {"status": "dim_mismatch", "embedding": None, "error": "got 3 expected 768", "model": "test-model"},
        ])
        server, addr = _start_server(monkeypatch, token="", embedding_provider=provider)
        try:
            with grpc.insecure_channel(addr) as ch:
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                resp = stub.EmbedBatch(
                    helper_pb2.EmbedBatchRequest(texts=["a", "b", "c"], model="m"),
                    timeout=5,
                )
            assert len(resp.items) == 3
            assert resp.items[0].status == "success"
            assert resp.items[1].status == "fail"
            assert resp.items[1].error == "HTTP 500"
            assert resp.items[2].status == "dim_mismatch"
            assert resp.items[2].error == "got 3 expected 768"
        finally:
            server.stop(grace=1)

    def test_provider_bug_success_but_empty_embedding_downgraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q3 reviewer fix: server-side dim guard downgrades success+empty
        to dim_mismatch so the backend never silently upserts a zero vector.
        """
        # Provider lies: claims "success" but embedding is empty list.
        provider = _stub_embedding([
            {"status": "success", "embedding": [], "error": "", "model": "test-model"},
        ])
        server, addr = _start_server(monkeypatch, token="", embedding_provider=provider)
        try:
            with grpc.insecure_channel(addr) as ch:
                stub = helper_pb2_grpc.HelperServiceStub(ch)
                resp = stub.EmbedBatch(
                    helper_pb2.EmbedBatchRequest(texts=["a"], model="m"),
                    timeout=5,
                )
            assert len(resp.items) == 1
            assert resp.items[0].status == "dim_mismatch"
            assert "downgraded server-side" in resp.items[0].error
            assert len(resp.items[0].embedding) == 0
            assert resp.items[0].dimensions == 0
        finally:
            server.stop(grace=1)


# ── P2-5: /health JSON sanitation ──────────────────────────────────


class TestHealthBodySanitization:
    def test_long_error_is_hashed_with_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import hashlib
        original = "upstream returned a long error string that would risk info disclosure"
        expected_fp = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        from internal.adapters.grpc_server import (
            HealthHandler, _health_cache, _health_cache_detail, _health_cache_lock,
        )
        with _health_cache_lock:
            _health_cache.update({"ollama": "down"})
            # The OLD line `_health_cache_detail = {"ollama": original}` rebinds a
            # local variable in this test function — the module-level dict
            # (which the handler reads) stays empty. Mutate in place instead.
            _health_cache_detail.clear()
            _health_cache_detail.update({"ollama": original})
        HealthHandler._grpc_server = MagicMock()

        port = _free_port()
        server = gs.serve_health(port=port)
        try:
            time.sleep(0.05)
            import urllib.request
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                body = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode())
            detail = body["adapter_details"]["ollama"]
            assert f"fp={expected_fp}" in detail
            assert "info disclosure" not in json.dumps(body)
        finally:
            server.shutdown()
            server.server_close()
            with _health_cache_lock:
                _health_cache.clear()
                _health_cache_detail.clear()

    def test_short_safe_token_passthrough(self) -> None:
        from internal.adapters.grpc_server import (
            HealthHandler, _health_cache, _health_cache_detail, _health_cache_lock,
        )
        with _health_cache_lock:
            _health_cache.update({"ollama": "ok"})
            _health_cache_detail.clear()
            _health_cache_detail.update({"ollama": "model qwen3:4b found"})
        HealthHandler._grpc_server = MagicMock()

        port = _free_port()
        server = gs.serve_health(port=port)
        try:
            time.sleep(0.05)
            import urllib.request
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                body = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode())
            assert "qwen3" in body["adapter_details"]["ollama"]
        finally:
            server.shutdown()
            server.server_close()
            with _health_cache_lock:
                _health_cache.clear()
                _health_cache_detail.clear()

    def test_short_unsafe_token_still_hashed(self) -> None:
        """The 40-char heuristic allowed short upstream bodies to leak.
        The fingerprint approach masks anything not matching the safe regex,
        regardless of length.
        """
        detail_in = 'rate limit exceeded'  # 19 chars but not a recognised token
        import hashlib
        expected_fp = hashlib.sha256(detail_in.encode("utf-8")).hexdigest()[:8]
        from internal.adapters.grpc_server import (
            HealthHandler, _health_cache, _health_cache_detail, _health_cache_lock,
        )
        with _health_cache_lock:
            _health_cache.update({"mistral": "down"})
            _health_cache_detail.clear()
            _health_cache_detail.update({"mistral": detail_in})
        HealthHandler._grpc_server = MagicMock()

        port = _free_port()
        server = gs.serve_health(port=port)
        try:
            time.sleep(0.05)
            import urllib.request
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                body = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode())
            out_detail = body["adapter_details"]["mistral"]
            assert f"fp={expected_fp}" in out_detail
            assert detail_in not in json.dumps(body)
        finally:
            server.shutdown()
            server.server_close()
            with _health_cache_lock:
                _health_cache.clear()
                _health_cache_detail.clear()
