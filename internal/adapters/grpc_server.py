"""
gRPC server adapter for the helper service.

Runs alongside FastAPI on a separate port (50051).
"""
import logging
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
        history = tuple(
            Message(role=m.role, content=m.content)
            for m in request.history
        )
        result: Answer = self._assistant.answer(
            Question(text=request.question),
            history=history,
        )
        return helper_pb2.AskResponse(answer=result.text)


def serve_grpc(assistant: HelperAgent, port: int = 50051) -> grpc.Server:
    """Start the gRPC server in a background thread."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helper_pb2_grpc.add_HelperServiceServicer_to_server(
        HelperServicer(assistant), server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server listening on :%d", port)
    return server
