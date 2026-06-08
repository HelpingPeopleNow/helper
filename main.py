"""
Application entry point. Composition root for the FastAPI app.

Wires the API router to the dependency container and starts uvicorn.
"""
import os

import uvicorn
from fastapi import FastAPI

from internal.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Helper — Pizza Assistant (hexagonal)")
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8082"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
