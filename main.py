"""
Application entry point. Composition root for the FastAPI app.

Wires the API router to the dependency container and starts uvicorn
alongside a gRPC server on :50051.
"""
import logging
import os

import uvicorn
from fastapi import FastAPI

from internal.adapters.grpc_server import serve_grpc
from internal.api.dependencies import get_assistant, get_graph
from internal.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reference to keep gRPC server alive
_grpc_server = None

def create_app() -> FastAPI:
    app = FastAPI(title="Helper — AI Assistant (hexagonal)")
    app.include_router(router)

    @app.on_event("startup")
    async def start_grpc():
        global _grpc_server
        assistant = get_assistant()
        _grpc_server = serve_grpc(assistant)
        logger.info("gRPC server started and referenced")

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8082"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)