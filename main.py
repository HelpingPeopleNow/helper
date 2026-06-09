"""
Application entry point — starts only the gRPC server.

Loads all LLM adapters at startup. The backend selects which one to use
per-request via the llm_provider gRPC field. Empty = fall back to env default.
"""
import logging
import os

from internal.adapters.grpc_server import serve_grpc, serve_health
from internal.adapters.opencode_llm import OpenCodeLLMAdapter
from internal.adapters.ollama_llm import OllamaLLMAdapter
from internal.core.helper_agent import HelperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Helper Service Starting ===")
    logger.info("LLM model=%s", os.getenv("LLM_MODEL", "deepseek-v4-flash-free"))

    # Load ALL adapters so the backend can switch per-request
    adapters = {
        "opencode": OpenCodeLLMAdapter(),
        "ollama": OllamaLLMAdapter(),
    }
    logger.info("Loaded adapters: %s", list(adapters.keys()))

    # Determine env-based default
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
    default_provider = "ollama" if use_ollama else "opencode"
    logger.info("Default provider (from env USE_OLLAMA): %s", default_provider)

    assistant = HelperAgent(adapters=adapters, default_provider=default_provider)
    logger.info("HelperAgent initialized (no DB dependency, multi-adapter)")

    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    logger.info("Starting gRPC server on port %d", grpc_port)
    server = serve_grpc(assistant, port=grpc_port)
    logger.info("=== Helper Service Ready ===")

    # Start lightweight health HTTP endpoint
    health_port = int(os.getenv("HEALTH_PORT", "8084"))
    serve_health(port=health_port)
    logger.info("Health HTTP server on :%d", health_port)

    server.wait_for_termination()


if __name__ == "__main__":
    main()
